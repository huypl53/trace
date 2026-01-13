#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tile RGB + line mask datasets into near-square patches.

Expected input structure:
  <input_dir>/(train|val|test|all)/images
  <input_dir>/(train|val|test|all)/masks
or a single dataset:
  <input_dir>/images
  <input_dir>/masks
"""

import argparse
import math
import os

import cv2
from tqdm import tqdm

import file_utils


SPLIT_NAMES = ("train", "val", "test", "all")


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


def find_split_dirs(input_dir):
    split_dirs = []
    for split in SPLIT_NAMES:
        images_dir = os.path.join(input_dir, split, "images")
        masks_dir = os.path.join(input_dir, split, "masks")
        if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
            split_dirs.append((split, images_dir, masks_dir))
    if split_dirs:
        return split_dirs

    images_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")
    if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        return [(None, images_dir, masks_dir)]

    raise ValueError("Expected images/ and masks/ under {}".format(input_dir))


def tile_dataset(
    images_dir,
    masks_dir,
    output_dir,
    tile_size,
    tile_stride,
    max_aspect_ratio,
    mask_suffix_h,
    mask_suffix_v,
):
    image_files = sorted(file_utils.get_image_list(images_dir))
    total_tiles = 0
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

        height, width = image.shape[:2]
        if mask_h.shape[:2] != (height, width):
            mask_h = cv2.resize(mask_h, (width, height), interpolation=cv2.INTER_NEAREST)
        if mask_v.shape[:2] != (height, width):
            mask_v = cv2.resize(mask_v, (width, height), interpolation=cv2.INTER_NEAREST)

        tile_w, tile_h = compute_tile_size(width, height, tile_size, max_aspect_ratio)
        if tile_w <= 0 or tile_h <= 0:
            continue

        xs = compute_positions(width, tile_w, tile_stride)
        ys = compute_positions(height, tile_h, tile_stride)

        out_images_dir = os.path.join(output_dir, "images", rel_dir)
        out_masks_dir = os.path.join(output_dir, "masks", rel_dir)
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)

        for row_idx, y in enumerate(ys):
            for col_idx, x in enumerate(xs):
                tile_name = "{}_tile_{:03d}_{:03d}".format(base, row_idx, col_idx)
                img_tile = crop_with_pad(image, x, y, tile_w, tile_h, pad_value=255)
                mask_h_tile = crop_with_pad(mask_h, x, y, tile_w, tile_h, pad_value=0)
                mask_v_tile = crop_with_pad(mask_v, x, y, tile_w, tile_h, pad_value=0)

                out_img_path = os.path.join(out_images_dir, tile_name + ".png")
                out_mask_h_path = os.path.join(out_masks_dir, tile_name + mask_suffix_h)
                out_mask_v_path = os.path.join(out_masks_dir, tile_name + mask_suffix_v)

                cv2.imwrite(out_img_path, img_tile)
                cv2.imwrite(out_mask_h_path, mask_h_tile)
                cv2.imwrite(out_mask_v_path, mask_v_tile)
                total_tiles += 1

    return total_tiles


def main():
    parser = argparse.ArgumentParser(
        description="Tile RGB + line mask datasets into near-square patches"
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
    args = parser.parse_args()

    split_dirs = find_split_dirs(args.input_dir)

    total_tiles = 0
    for split_name, images_dir, masks_dir in split_dirs:
        split_output = args.output_dir if split_name is None else os.path.join(args.output_dir, split_name)
        os.makedirs(split_output, exist_ok=True)
        split_tiles = tile_dataset(
            images_dir,
            masks_dir,
            split_output,
            args.tile_size,
            args.tile_stride,
            args.max_aspect_ratio,
            args.mask_suffix_h,
            args.mask_suffix_v,
        )
        label = split_name or "all"
        print("{}: generated {} tiles".format(label, split_tiles))
        total_tiles += split_tiles

    print("Total tiles generated: {}".format(total_tiles))


if __name__ == "__main__":
    main()
