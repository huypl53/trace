#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tile RGB + line mask datasets into near-square patches.

Smart tiling features:
- Multi-scale tiling (--scales): Generate tiles at multiple scales for better model generalization
- Skip tiles with no/few line pixels (--min_line_pixels)
- Restrict tiling to content bounding box (--content_bbox)

Expected input structure:
  <input_dir>/(train|val|test|all)/images
  <input_dir>/(train|val|test|all)/masks
or a single dataset:
  <input_dir>/images
  <input_dir>/masks

Usage:
  # Single scale (default)
  python -m scripts.tile_line_mask_dataset --input_dir data/raw --output_dir data/tiled

  # Multi-scale for diverse training data
  python -m scripts.tile_line_mask_dataset --input_dir data/raw --output_dir data/tiled \\
      --scales 0.5,0.75,1.0,1.25,1.5

  # Wide range with smaller output tiles
  python -m scripts.tile_line_mask_dataset --input_dir data/raw --output_dir data/tiled \\
      --scales 0.25,0.5,0.75,1.0,1.25,1.5,2.0 --tile_size 512

Output naming:
  {base}_s{scale}_tile_{idx}.png
  Example: doc001_s050_tile_000.png (scale=0.5), doc001_s150_tile_003.png (scale=1.5)
"""

import argparse
import math
import os

import cv2
import numpy as np
from tqdm import tqdm

import file_utils


SPLIT_NAMES = ("train", "val", "test", "all")


def scale_to_str(scale):
    """Convert scale factor to filename-safe string (e.g., 0.5 -> '050', 1.5 -> '150')."""
    return "{:03d}".format(int(round(scale * 100)))


def resize_with_masks(image, mask_h, mask_v, scale):
    """Resize image and masks by scale factor.

    Uses INTER_AREA for downscaling (better quality), INTER_LINEAR for upscaling.
    Uses INTER_NEAREST for masks to preserve binary values.
    """
    if scale == 1.0:
        return image, mask_h, mask_v

    new_w = int(round(image.shape[1] * scale))
    new_h = int(round(image.shape[0] * scale))

    if new_w <= 0 or new_h <= 0:
        return None, None, None

    interp_image = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    interp_mask = cv2.INTER_NEAREST

    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interp_image)
    scaled_mask_h = cv2.resize(mask_h, (new_w, new_h), interpolation=interp_mask)
    scaled_mask_v = cv2.resize(mask_v, (new_w, new_h), interpolation=interp_mask)

    return scaled_image, scaled_mask_h, scaled_mask_v


def get_content_bbox(mask_h, mask_v, padding=0):
    """Get bounding box of non-zero pixels in combined masks.

    Returns (x1, y1, x2, y2) or None if masks are empty.
    """
    combined = cv2.bitwise_or(mask_h, mask_v)
    non_zero = cv2.findNonZero(combined)
    if non_zero is None:
        return None
    x, y, w, h = cv2.boundingRect(non_zero)
    height, width = mask_h.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    return (x1, y1, x2, y2)


def count_line_pixels(mask_h_tile, mask_v_tile):
    """Count non-zero pixels in mask tiles."""
    return np.count_nonzero(mask_h_tile) + np.count_nonzero(mask_v_tile)


def compute_tile_size(width, height, tile_size, max_aspect_ratio):
    tile_w = min(width, tile_size)
    tile_h = min(height, tile_size)
    if tile_w <= 0 or tile_h <= 0:
        return tile_w, tile_h
    ratio = max(tile_w, tile_h) / max(1, min(tile_w, tile_h))
    if ratio > max_aspect_ratio:
        if tile_w < tile_h:
            tile_w = int(math.ceil(tile_h / max_aspect_ratio))
        else:
            tile_h = int(math.ceil(tile_w / max_aspect_ratio))
    return tile_w, tile_h


def compute_positions(total, tile, stride):
    if total <= tile:
        return [0]
    stride = max(1, stride)
    if stride > tile:
        stride = tile
    positions = list(range(0, total - tile + 1, stride))
    last = total - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def crop_with_pad(img, x, y, tile_w, tile_h, pad_value):
    height, width = img.shape[:2]
    x_end = x + tile_w
    y_end = y + tile_h

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x_end)
    y1 = min(height, y_end)

    crop = img[y0:y1, x0:x1].copy()

    pad_left = x0 - x
    pad_top = y0 - y
    pad_right = x_end - x1
    pad_bottom = y_end - y1

    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        if crop.ndim == 2:
            pad_color = pad_value
        else:
            pad_color = [pad_value, pad_value, pad_value]
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )
    return crop


def find_split_dirs(input_dir, recursive=False):
    """Find image/mask directory pairs under input_dir.

    Looks for directories containing both 'images/' and 'masks/' subdirs.
    Checks for split structure (train/val/test/all) first, then flat layout.

    Args:
        input_dir: Root directory to search.
        recursive: Walk the entire tree to find all images/masks pairs.

    Returns:
        List of (label, images_dir, masks_dir) tuples where label is used
        for output subdirectory naming (e.g. 'train', 'subdir/train', or None).
    """
    # Check for standard split structure at top level
    split_dirs = []
    for split in SPLIT_NAMES:
        images_dir = os.path.join(input_dir, split, "images")
        masks_dir = os.path.join(input_dir, split, "masks")
        if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
            split_dirs.append((split, images_dir, masks_dir))
    if split_dirs:
        return split_dirs

    # Check for flat layout at top level
    images_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")
    if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        return [(None, images_dir, masks_dir)]

    if not recursive:
        raise ValueError("Expected images/ and masks/ under {}".format(input_dir))

    # Recursive: walk tree to find all dirs containing images/ + masks/
    found = []
    for dirpath, dirnames, _filenames in os.walk(input_dir):
        imgs = os.path.join(dirpath, "images")
        msks = os.path.join(dirpath, "masks")
        if os.path.isdir(imgs) and os.path.isdir(msks):
            label = os.path.relpath(dirpath, input_dir)
            if label == ".":
                label = None
            found.append((label, imgs, msks))
            # Don't descend into images/ or masks/
            dirnames[:] = [d for d in dirnames if d not in ("images", "masks")]

    if not found:
        raise ValueError("No images/ + masks/ pairs found under {} (recursive)".format(input_dir))

    return found


def tile_dataset(
    images_dir,
    masks_dir,
    output_dir,
    tile_size,
    tile_stride,
    max_aspect_ratio,
    mask_suffix_h,
    mask_suffix_v,
    min_line_pixels=0,
    use_content_bbox=False,
    bbox_padding=50,
    scales=(1.0,),
):
    """Tile dataset with smart filtering and multi-scale support.

    Args:
        min_line_pixels: Skip tiles with fewer line pixels than this (0=keep all).
            Threshold is scaled proportionally with scale factor.
        use_content_bbox: Only tile within the bounding box of mask content
        bbox_padding: Padding around content bounding box (pixels)
        scales: Tuple of scale factors for multi-scale tiling (e.g., (0.5, 1.0, 1.5))
    """
    image_files = sorted(file_utils.get_image_list(images_dir))
    total_tiles = 0
    skipped_tiles = 0

    for img_path in tqdm(image_files):
        rel_path = os.path.relpath(img_path, images_dir)
        rel_dir = os.path.dirname(rel_path)
        base = os.path.splitext(os.path.basename(rel_path))[0]

        mask_h_path = os.path.join(masks_dir, os.path.splitext(rel_path)[0] + mask_suffix_h)
        mask_v_path = os.path.join(masks_dir, os.path.splitext(rel_path)[0] + mask_suffix_v)
        if not os.path.exists(mask_h_path) or not os.path.exists(mask_v_path):
            print("No mask files found for {}".format(img_path))
            continue

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_h = cv2.imread(mask_h_path, cv2.IMREAD_GRAYSCALE)
        mask_v = cv2.imread(mask_v_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask_h is None or mask_v is None:
            print("Failed to read image/masks for {}".format(img_path))
            continue

        orig_height, orig_width = image.shape[:2]
        if mask_h.shape[:2] != (orig_height, orig_width):
            mask_h = cv2.resize(mask_h, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        if mask_v.shape[:2] != (orig_height, orig_width):
            mask_v = cv2.resize(mask_v, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

        out_images_dir = os.path.join(output_dir, "images", rel_dir)
        out_masks_dir = os.path.join(output_dir, "masks", rel_dir)
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)

        # Process each scale
        for scale in scales:
            scaled_image, scaled_mask_h, scaled_mask_v = resize_with_masks(
                image, mask_h, mask_v, scale
            )
            if scaled_image is None:
                continue

            height, width = scaled_image.shape[:2]

            # Skip if scaled image is too small for tiling
            if width < tile_size // 4 or height < tile_size // 4:
                continue

            # Determine tiling region on scaled image
            if use_content_bbox:
                # Scale bbox_padding proportionally
                scaled_padding = max(1, int(round(bbox_padding * scale)))
                bbox = get_content_bbox(scaled_mask_h, scaled_mask_v, padding=scaled_padding)
                if bbox is None:
                    continue
                region_x1, region_y1, region_x2, region_y2 = bbox
                region_w = region_x2 - region_x1
                region_h = region_y2 - region_y1
            else:
                region_x1, region_y1 = 0, 0
                region_w, region_h = width, height

            tile_w, tile_h = compute_tile_size(region_w, region_h, tile_size, max_aspect_ratio)
            if tile_w <= 0 or tile_h <= 0:
                continue

            xs = [region_x1 + x for x in compute_positions(region_w, tile_w, tile_stride)]
            ys = [region_y1 + y for y in compute_positions(region_h, tile_h, tile_stride)]

            # Scale min_line_pixels threshold proportionally (area scales as scale^2)
            scaled_min_pixels = int(round(min_line_pixels * (scale ** 2)))

            scale_str = scale_to_str(scale)
            tile_idx = 0

            for row_idx, y in enumerate(ys):
                for col_idx, x in enumerate(xs):
                    img_tile = crop_with_pad(scaled_image, x, y, tile_w, tile_h, pad_value=255)
                    mask_h_tile = crop_with_pad(scaled_mask_h, x, y, tile_w, tile_h, pad_value=0)
                    mask_v_tile = crop_with_pad(scaled_mask_v, x, y, tile_w, tile_h, pad_value=0)

                    # Skip tiles with insufficient line content
                    line_pixels = count_line_pixels(mask_h_tile, mask_v_tile)
                    if line_pixels < scaled_min_pixels:
                        skipped_tiles += 1
                        continue

                    tile_name = "{}_s{}_tile_{:03d}".format(base, scale_str, tile_idx)
                    tile_idx += 1

                    out_img_path = os.path.join(out_images_dir, tile_name + ".png")
                    out_mask_h_path = os.path.join(out_masks_dir, tile_name + mask_suffix_h)
                    out_mask_v_path = os.path.join(out_masks_dir, tile_name + mask_suffix_v)

                    cv2.imwrite(out_img_path, img_tile)
                    cv2.imwrite(out_mask_h_path, mask_h_tile)
                    cv2.imwrite(out_mask_v_path, mask_v_tile)
                    total_tiles += 1

    return total_tiles, skipped_tiles


def main():
    parser = argparse.ArgumentParser(
        description="Tile RGB + line mask datasets into near-square patches (smart tiling)"
    )
    parser.add_argument("--input_dir", required=True, help="Input dataset root")
    parser.add_argument("--output_dir", required=True, help="Output dataset root")
    parser.add_argument("--tile_size", type=int, default=1280, help="Tile size (square)")
    parser.add_argument(
        "--tile_stride",
        type=int,
        default=1024,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--max_aspect_ratio",
        type=float,
        default=1.2,
        help="Max allowed aspect ratio for small images",
    )
    parser.add_argument(
        "--mask_suffix_h", default="_mask_h.png", help="Horizontal mask suffix"
    )
    parser.add_argument(
        "--mask_suffix_v", default="_mask_v.png", help="Vertical mask suffix"
    )
    # Smart tiling options
    parser.add_argument(
        "--min_line_pixels",
        type=int,
        default=100,
        help="Skip tiles with fewer line pixels (0=keep all, default=100)",
    )
    parser.add_argument(
        "--content_bbox",
        action="store_true",
        help="Only tile within the bounding box of mask content",
    )
    parser.add_argument(
        "--bbox_padding",
        type=int,
        default=50,
        help="Padding around content bounding box (default=50)",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="1.0",
        help="Comma-separated scale factors for multi-scale tiling (default='1.0'). "
        "Example: '0.5,0.75,1.0,1.25,1.5' for wide range multi-scale.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find all images/masks directory pairs under input_dir",
    )
    args = parser.parse_args()

    # Parse scales
    scales = tuple(float(s.strip()) for s in args.scales.split(",") if s.strip())
    if not scales:
        scales = (1.0,)
    print("Scales: {}".format(scales))

    split_dirs = find_split_dirs(args.input_dir, recursive=args.recursive)
    if args.recursive:
        print("Found {} dataset(s): {}".format(
            len(split_dirs),
            [s[0] or "." for s in split_dirs],
        ))

    total_tiles = 0
    total_skipped = 0
    for split_name, images_dir, masks_dir in split_dirs:
        split_output = args.output_dir if split_name is None else os.path.join(args.output_dir, split_name)
        os.makedirs(split_output, exist_ok=True)
        split_tiles, split_skipped = tile_dataset(
            images_dir,
            masks_dir,
            split_output,
            args.tile_size,
            args.tile_stride,
            args.max_aspect_ratio,
            args.mask_suffix_h,
            args.mask_suffix_v,
            min_line_pixels=args.min_line_pixels,
            use_content_bbox=args.content_bbox,
            bbox_padding=args.bbox_padding,
            scales=scales,
        )
        label = split_name or "all"
        print("{}: {} tiles (skipped {} empty)".format(label, split_tiles, split_skipped))
        total_tiles += split_tiles
        total_skipped += split_skipped

    print("\nTotal: {} tiles generated, {} skipped".format(total_tiles, total_skipped))
    print("Settings: scales={}, min_line_pixels={}, tile_size={}".format(
        scales, args.min_line_pixels, args.tile_size
    ))


if __name__ == "__main__":
    main()
