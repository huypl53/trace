#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert canvas JSON annotations to line segmentation training format.

Features:
- Extract lines from canvas JSON (horizontal/vertical)
- Use original color images (from image_path in canvas JSON) as training input
- Crop each table using explicit width/height from table properties
- Split data into train/val/test sets
- Generate corresponding line JSON annotations

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
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from parsers.canvas_parser import extract_single_table_data




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


def process_canvas_file(json_path, output_dir, padding=5):
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

    for idx, table_item in enumerate(tables):
        # Extract table data
        table_data = extract_single_table_data(table_item)

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

        # Save image
        out_img_path = os.path.join(output_dir, out_name + ".png")
        cv2.imwrite(out_img_path, image)

        # Save line JSON
        out_json_path = os.path.join(output_dir, out_name + ".json")
        output_data = {"filename": out_name + ".png", "lines": lines}
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

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
        for json_path in files:
            generated = process_canvas_file(json_path, output_dir, args.padding)
            split_count += len(generated)

        print(
            f"{split_name}: generated {split_count} samples from {len(files)} files -> {output_dir}"
        )
        total_generated += split_count

    print(f"\nTotal: {total_generated} samples generated")


if __name__ == "__main__":
    main()
