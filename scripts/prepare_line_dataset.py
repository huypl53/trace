#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert canvas JSON annotations to line segmentation training format.

Features:
- Extract lines from canvas JSON (horizontal/vertical)
- Use original color images (from image_path in canvas JSON) as training input
- Crop each table using explicit width/height from table properties
- Split data into train/val/test sets
- Generate line JSON annotations or line mask images

The training pipeline uses:
- INPUT: Original color image (cropped to table region)
- GROUND TRUTH: Line JSON annotations (converted to heatmaps during training)

Input folder structure (JSON and images in same folder):
    data/raw_canvas/
        canvas_001.json      # contains "image_path": "page1.png"
        page1.png            # original color image
        ...

Usage:
    python scripts/prepare_line_dataset.py \
        --input_dir data/raw_canvas \
        --output_dir data/line_dataset \
        --padding 5 \
        --split 0.8 0.1 0.1
"""

import argparse
import copy
import json
import os
import random
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

from parsers.canvas_parser import extract_single_table_data


def _get_table_dim(props, item, key):
    value = props.get(key)
    if value is None:
        return item.get(key)
    return value


def _build_sizes(size_map, count, total):
    if count <= 0:
        return []
    size_map = size_map or {}
    if total is None:
        total = sum(float(v) for v in size_map.values()) if size_map else 0.0
    default = (total / count) if total else 0.0
    sizes = []
    for i in range(count):
        key = str(i)
        if key in size_map:
            sizes.append(float(size_map[key]))
        else:
            sizes.append(default)
    return sizes


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_item_bounds(item):
    props = item.get("properties", {}) or {}
    x = _to_float(item.get("x"))
    y = _to_float(item.get("y"))
    w = _to_float(_get_table_dim(props, item, "width"))
    h = _to_float(_get_table_dim(props, item, "height"))
    if x is None or y is None or w is None or h is None:
        return None
    x1 = x + w
    y1 = y + h
    return x, y, x1, y1


def _collect_overlap_cuts(table_bounds, blocker_bounds, min_overlap_ratio=0.8):
    tx0, ty0, tx1, ty1 = table_bounds
    table_w = max(0.0, tx1 - tx0)
    table_h = max(0.0, ty1 - ty0)
    left_cut = right_cut = top_cut = bottom_cut = 0.0
    for bounds in blocker_bounds:
        ox0, oy0, ox1, oy1 = bounds
        if ox0 <= tx0 and oy0 <= ty0 and ox1 >= tx1 and oy1 >= ty1:
            continue
        overlap_x = max(0.0, min(tx1, ox1) - max(tx0, ox0))
        overlap_y = max(0.0, min(ty1, oy1) - max(ty0, oy0))
        if overlap_x <= 0.0 or overlap_y <= 0.0:
            continue
        if ox0 <= tx0 < ox1 and overlap_y >= table_h * min_overlap_ratio:
            left_cut = max(left_cut, min(tx1, ox1) - tx0)
        if ox0 < tx1 <= ox1 and overlap_y >= table_h * min_overlap_ratio:
            right_cut = max(right_cut, tx1 - max(tx0, ox0))
        if oy0 <= ty0 < oy1 and overlap_x >= table_w * min_overlap_ratio:
            top_cut = max(top_cut, min(ty1, oy1) - ty0)
        if oy0 < ty1 <= oy1 and overlap_x >= table_w * min_overlap_ratio:
            bottom_cut = max(bottom_cut, ty1 - max(ty0, oy0))
    return left_cut, right_cut, top_cut, bottom_cut


def _apply_table_edge_cuts(table_item, blockers, min_edge_size=1.0):
    bounds = _get_item_bounds(table_item)
    if bounds is None:
        return table_item
    left_cut, right_cut, top_cut, bottom_cut = _collect_overlap_cuts(bounds, blockers)
    if left_cut <= 0 and right_cut <= 0 and top_cut <= 0 and bottom_cut <= 0:
        return table_item

    item = copy.deepcopy(table_item)
    props = item.get("properties", {}) or {}
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))
    table_x = _to_float(item.get("x", 0)) or 0.0
    table_y = _to_float(item.get("y", 0)) or 0.0
    table_w = _to_float(_get_table_dim(props, item, "width"))
    table_h = _to_float(_get_table_dim(props, item, "height"))

    if cols > 0 and table_w is not None and (left_cut > 0 or right_cut > 0):
        col_widths = _build_sizes(props.get("columnWidths", {}), cols, table_w)
        if col_widths:
            if left_cut > 0:
                cut = min(left_cut, max(0.0, col_widths[0] - min_edge_size))
                col_widths[0] -= cut
                table_x += cut
                table_w -= cut
            if right_cut > 0:
                cut = min(right_cut, max(0.0, col_widths[-1] - min_edge_size))
                col_widths[-1] -= cut
                table_w -= cut
            props["columnWidths"] = {str(i): w for i, w in enumerate(col_widths)}
    elif table_w is not None and (left_cut > 0 or right_cut > 0):
        cut_left = min(left_cut, max(0.0, table_w - min_edge_size))
        cut_right = min(right_cut, max(0.0, table_w - cut_left - min_edge_size))
        table_x += cut_left
        table_w -= (cut_left + cut_right)

    if rows > 0 and table_h is not None and (top_cut > 0 or bottom_cut > 0):
        row_heights = _build_sizes(props.get("rowHeights", {}), rows, table_h)
        if row_heights:
            if top_cut > 0:
                cut = min(top_cut, max(0.0, row_heights[0] - min_edge_size))
                row_heights[0] -= cut
                table_y += cut
                table_h -= cut
            if bottom_cut > 0:
                cut = min(bottom_cut, max(0.0, row_heights[-1] - min_edge_size))
                row_heights[-1] -= cut
                table_h -= cut
            props["rowHeights"] = {str(i): h for i, h in enumerate(row_heights)}
    elif table_h is not None and (top_cut > 0 or bottom_cut > 0):
        cut_top = min(top_cut, max(0.0, table_h - min_edge_size))
        cut_bottom = min(bottom_cut, max(0.0, table_h - cut_top - min_edge_size))
        table_y += cut_top
        table_h -= (cut_top + cut_bottom)

    item["x"] = table_x
    item["y"] = table_y
    if "width" in item:
        item["width"] = table_w
    if "height" in item:
        item["height"] = table_h
    if table_w is not None:
        props["width"] = table_w
    if table_h is not None:
        props["height"] = table_h
    item["properties"] = props
    return item




def crop_table_from_image(image, table_data, padding=5):
    """Crop a table region from the original color image.

    Args:
        image: Original color image (H, W, 3)
        table_data: Table data with bounds
        padding: Padding around table bounds

    Returns:
        cropped_image: numpy array (H, W, 3)
        offset: (x_offset, y_offset) for coordinate adjustment
    """
    x_min, y_min, x_max, y_max = table_data["bounds"]
    img_h, img_w = image.shape[:2]

    # Apply padding with bounds checking
    crop_x0 = max(0, x_min - padding)
    crop_y0 = max(0, y_min - padding)
    crop_x1 = min(img_w, x_max + padding)
    crop_y1 = min(img_h, y_max + padding)

    # Crop the image
    cropped = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    # Offset for coordinate transformation
    x_offset = crop_x0
    y_offset = crop_y0

    return cropped, (x_offset, y_offset)


def render_table_image(table_data, padding=5, bg_color=(255, 255, 255)):
    """Render a table as an image with white background and black borders.

    This is a fallback when no original image is available.

    Returns:
        image: numpy array (H, W, 3)
        offset: (x_offset, y_offset) for coordinate adjustment
    """
    x_min, y_min, x_max, y_max = table_data["bounds"]

    # Add padding
    width = (x_max - x_min) + 2 * padding
    height = (y_max - y_min) + 2 * padding

    # Create white background
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Offset for coordinate transformation
    x_offset = x_min - padding
    y_offset = y_min - padding

    # Draw cell backgrounds (optional - all white for now)
    for cell in table_data["cells"]:
        cx0 = cell["x0"] - x_offset
        cy0 = cell["y0"] - y_offset
        cx1 = cell["x1"] - x_offset
        cy1 = cell["y1"] - y_offset
        cv2.rectangle(image, (cx0, cy0), (cx1, cy1), bg_color, -1)

    # Draw horizontal lines
    for start, end, thickness in table_data["lines_h"]:
        p1 = (start[0] - x_offset, start[1] - y_offset)
        p2 = (end[0] - x_offset, end[1] - y_offset)
        cv2.line(image, p1, p2, (0, 0, 0), thickness)

    # Draw vertical lines
    for start, end, thickness in table_data["lines_v"]:
        p1 = (start[0] - x_offset, start[1] - y_offset)
        p2 = (end[0] - x_offset, end[1] - y_offset)
        cv2.line(image, p1, p2, (0, 0, 0), thickness)

    return image, (x_offset, y_offset)


def adjust_lines_to_crop(lines_h, lines_v, offset):
    """Adjust line coordinates relative to cropped region."""
    x_off, y_off = offset
    adjusted = []

    for start, end, thickness in lines_h:
        adjusted.append(
            {
                "type": "horizontal",
                "points": [
                    [start[0] - x_off, start[1] - y_off],
                    [end[0] - x_off, end[1] - y_off],
                ],
                "thickness": thickness,
            }
        )

    for start, end, thickness in lines_v:
        adjusted.append(
            {
                "type": "vertical",
                "points": [
                    [start[0] - x_off, start[1] - y_off],
                    [end[0] - x_off, end[1] - y_off],
                ],
                "thickness": thickness,
            }
        )

    return adjusted


def draw_line_masks(lines, width, height):
    mask_h = np.zeros((height, width), dtype=np.uint8)
    mask_v = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        line_type = line.get("type")
        points = line.get("points", [])
        if len(points) != 2:
            continue
        start = tuple(map(int, points[0]))
        end = tuple(map(int, points[1]))
        thickness = line.get("thickness", 1)
        try:
            thickness = int(round(float(thickness)))
        except (TypeError, ValueError):
            thickness = 1
        thickness = max(1, thickness)
        if line_type == "horizontal":
            cv2.line(mask_h, start, end, color=255, thickness=thickness)
        elif line_type == "vertical":
            cv2.line(mask_v, start, end, color=255, thickness=thickness)
    return mask_h, mask_v


def process_canvas_file(json_path, output_dir, padding=5, show_all_borders=True, output_mode="json"):
    """Process a single canvas JSON file, generating separate output for each table.

    Args:
        json_path: Path to canvas JSON file
        output_dir: Output directory for processed files
        padding: Padding around table when cropping

    Returns:
        List of generated file basenames
    """
    with open(json_path, "r", encoding="utf-8") as f:
        canvas_data = json.load(f)

    items = canvas_data.get("items", [])
    tables = [item for item in items if item.get("type") == "table"]
    blockers = []
    for item in items:
        item_type = item.get("type")
        if item_type == "table" or item_type == "highlight":
            continue
        bounds = _get_item_bounds(item)
        if bounds is not None:
            blockers.append(bounds)

    if not tables:
        print(f"Warning: No tables found in {json_path}")
        return []

    # Try to load original image from same folder as JSON
    original_image = None
    image_path = canvas_data.get("image_path")
    json_dir = os.path.dirname(json_path)
    json_base = os.path.splitext(os.path.basename(json_path))[0]

    # Build list of possible image paths
    possible_paths = []

    # Try image with same name as JSON file (e.g., 4781.json -> 4781.png)
    for ext in [".png", ".jpg", ".jpeg"]:
        possible_paths.append(os.path.join(json_dir, json_base + ext))

    # Try image_path from JSON if available
    if image_path:
        possible_paths.append(os.path.join(json_dir, image_path))
        possible_paths.append(os.path.join(json_dir, os.path.basename(image_path)))

    for path in possible_paths:
        if os.path.exists(path):
            original_image = cv2.imread(path)
            if original_image is not None:
                break

    if original_image is None:
        print(f"Warning: Could not load image for {json_path}, using rendered fallback")

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    generated = []

    # print(os.path.basename(json_path))
    for idx, table_item in enumerate(tables):
        # Extract table data
        adjusted_item = _apply_table_edge_cuts(table_item, blockers)

        table_data = extract_single_table_data(adjusted_item, show_all_borders=show_all_borders)

        if not table_data["lines_h"] and not table_data["lines_v"]:
            continue

        # Get image: crop from original or render fallback
        if original_image is not None:
            image, offset = crop_table_from_image(original_image, table_data, padding)
        else:
            image, offset = render_table_image(table_data, padding=padding)

        # Adjust line coordinates
        lines = adjust_lines_to_crop(
            table_data["lines_h"], table_data["lines_v"], offset
        )

        # Generate output name (with table index if multiple tables)
        if len(tables) > 1:
            out_name = f"{base_name}_table{idx}"
        else:
            out_name = base_name

        if output_mode == "mask":
            image_dir = os.path.join(output_dir, "images")
            mask_dir = os.path.join(output_dir, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
        else:
            image_dir = output_dir
            mask_dir = None

        # Save image
        out_img_path = os.path.join(image_dir, out_name + ".png")
        cv2.imwrite(out_img_path, image)

        if output_mode == "json":
            # Save line JSON
            out_json_path = os.path.join(output_dir, out_name + ".json")
            output_data = {"filename": out_name + ".png", "lines": lines}
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        else:
            mask_h, mask_v = draw_line_masks(lines, image.shape[1], image.shape[0])
            out_mask_h = os.path.join(mask_dir, out_name + "_mask_h.png")
            out_mask_v = os.path.join(mask_dir, out_name + "_mask_v.png")
            cv2.imwrite(out_mask_h, mask_h)
            cv2.imwrite(out_mask_v, mask_v)

        generated.append(out_name)

    return generated


def split_files(file_list, ratios):
    """Split file list into train/val/test sets."""
    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    return {
        "train": file_list[:train_end],
        "val": file_list[train_end:val_end],
        "test": file_list[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert canvas JSON to line segmentation training format"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing canvas JSON files (images should be in same folder)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--output_mode",
        choices=["json", "mask"],
        default="json",
        help="Output format: json (image+json) or mask (images/masks)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding around table when cropping (default: 5)",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splits"
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Don't split data, process all to single 'all' directory",
    )
    parser.add_argument(
        "--show_all_borders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw all table borders regardless of border width visibility",
    )
    args = parser.parse_args()

    # Validate split ratios
    if not args.no_split and abs(sum(args.split) - 1.0) > 0.01:
        print(f"Error: Split ratios must sum to 1.0, got {sum(args.split)}")
        return

    random.seed(args.seed)

    # Find all canvas JSON files
    json_files = list(Path(args.input_dir).rglob("*.json"))
    json_files = [str(p) for p in json_files]

    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} canvas JSON files")

    # Split source files first
    if args.no_split:
        splits = {"all": json_files}
    else:
        splits = split_files(json_files, tuple(args.split))
        print(
            f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    # Process each split
    total_generated = 0
    for split_name, files in splits.items():
        if not files:
            continue

        output_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(output_dir, exist_ok=True)

        split_count = 0
        for json_path in tqdm(files):
            generated = process_canvas_file(
                json_path,
                output_dir,
                args.padding,
                show_all_borders=args.show_all_borders,
                output_mode=args.output_mode,
            )
            split_count += len(generated)

        print(
            f"{split_name}: generated {split_count} samples from {len(files)} files -> {output_dir}"
        )
        total_generated += split_count

    print(f"\nTotal: {total_generated} samples generated")


if __name__ == "__main__":
    main()
