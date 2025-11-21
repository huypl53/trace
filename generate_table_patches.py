# -*- coding: utf-8 -*-
"""
Generate cropped table patches and mask pairs from JSON annotations.

The script walks an input directory that contains images and TRACE-style JSON
labels, crops each table region with optional random padding, generates the
corresponding mask using GTTransform, and optionally produces additional random
patches sampled from the cropped table area. The output directory mirrors the
input structure so downstream tooling can map patches back to their sources.
"""

import argparse
import os
import random

import cv2
import numpy as np
import tqdm

import imgproc
from loader import GTTransform
from parsers.json_parser import ParserTRACEJSON

"""
uv run python generate_table_patches.py \
  --input_dir /path/to/input \
  --output_dir /path/to/output \
  --scale_down 2 \
  --pad_min 4 --pad_max 32 \
  --patch_min 256 --patch_max 512 \
  --num_patches 5 \
  --seed 42
"""


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _sample_padding(pad_min, pad_max):
    if pad_max < pad_min:
        pad_max = pad_min
    if pad_max <= 0:
        return 0, 0, 0, 0
    return (
        random.randint(pad_min, pad_max),
        random.randint(pad_min, pad_max),
        random.randint(pad_min, pad_max),
        random.randint(pad_min, pad_max),
    )


def _table_bbox(table_bounds, quads_arr, img_width, img_height):
    if table_bounds:
        min_x = min(int(np.floor(tb["x1"])) for tb in table_bounds)
        min_y = min(int(np.floor(tb["y1"])) for tb in table_bounds)
        max_x = max(int(np.ceil(tb["x2"])) for tb in table_bounds)
        max_y = max(int(np.ceil(tb["y2"])) for tb in table_bounds)
    elif quads_arr.size > 0:
        xs = quads_arr[:, 0::2]
        ys = quads_arr[:, 1::2]
        min_x = int(np.floor(xs.min()))
        min_y = int(np.floor(ys.min()))
        max_x = int(np.ceil(xs.max()))
        max_y = int(np.ceil(ys.max()))
    else:
        return None

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(img_width, max_x)
    max_y = min(img_height, max_y)

    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x, max_y


def _crop_table(img, quads_arr, table_bounds, pad_min, pad_max):
    height, width = img.shape[:2]
    bbox = _table_bbox(table_bounds, quads_arr, width, height)
    if bbox is None:
        return None

    min_x, min_y, max_x, max_y = bbox
    pad_left, pad_right, pad_top, pad_bottom = _sample_padding(pad_min, pad_max)

    crop_x1 = max(0, min_x - pad_left)
    crop_y1 = max(0, min_y - pad_top)
    crop_x2 = min(width, max_x + pad_right)
    crop_y2 = min(height, max_y + pad_bottom)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    quads_shifted = quads_arr.copy()
    quads_shifted[:, 0::2] -= crop_x1
    quads_shifted[:, 1::2] -= crop_y1

    crop_box = {
        "x1": int(crop_x1),
        "y1": int(crop_y1),
        "x2": int(crop_x2),
        "y2": int(crop_y2),
    }

    return cropped_img, quads_shifted, crop_box


def _normalize_quads(quads_arr, width, height):
    num_pt = int(quads_arr.shape[1] / 2)
    quads_norm = quads_arr.copy()
    if width <= 0 or height <= 0:
        raise ValueError("Invalid crop size for normalization.")
    quads_norm[:, : 2 * num_pt] /= [width, height] * num_pt
    return quads_norm


def _generate_mask(quads_norm, lines, width, height, scale_down):
    mask_width = width / scale_down
    mask_height = height / scale_down
    gt_gathered = []
    for attr, quad in zip(lines, quads_norm):
        gt_gathered.append({"quad": quad, "line": attr})
    gt_image, gt_weight = GTTransform(
        gt_gathered, mask_width, mask_height, hide_invisible=True
    )
    return gt_image, gt_weight


def _crop_mask_tensor(tensor, x, y, size, scale_down, is_weight=False):
    if tensor is None:
        return None

    mask_h, mask_w = tensor.shape[:2]
    y1 = int(np.floor(y / scale_down))
    x1 = int(np.floor(x / scale_down))
    y2 = int(np.ceil((y + size) / scale_down))
    x2 = int(np.ceil((x + size) / scale_down))

    y1 = max(0, min(mask_h - 1, y1))
    x1 = max(0, min(mask_w - 1, x1))
    y2 = max(y1 + 1, min(mask_h, y2))
    x2 = max(x1 + 1, min(mask_w, x2))

    region = tensor[y1:y2, x1:x2]
    target_dim = max(1, int(round(size / scale_down)))

    if region.shape[0] == target_dim and region.shape[1] == target_dim:
        return region

    if is_weight:
        return cv2.resize(
            region, (target_dim, target_dim), interpolation=cv2.INTER_LINEAR
        )

    resized = np.zeros((target_dim, target_dim, tensor.shape[2]), dtype=tensor.dtype)
    for ch in range(tensor.shape[2]):
        resized[:, :, ch] = cv2.resize(
            region[:, :, ch],
            (target_dim, target_dim),
            interpolation=cv2.INTER_LINEAR,
        )
    return resized


def _save_table_outputs(
    base_dir,
    img_crop,
    mask_data,
    crop_box,
    scale_down,
    rel_path,
):
    _ensure_dir(base_dir)
    img_path = os.path.join(base_dir, "table.png")
    mask_path = os.path.join(base_dir, "table_mask.npy")

    cv2.imwrite(img_path, img_crop)
    np.save(
        mask_path,
        {
            "mask": mask_data["mask"],
            "weight": mask_data["weight"],
            "scale_down": scale_down,
            "crop_box": crop_box,
            "source_relpath": rel_path,
        },
    )


def _save_patch_outputs(patch_dir, idx, img_patch, mask_patch, weight_patch, info):
    _ensure_dir(patch_dir)
    img_path = os.path.join(patch_dir, f"patch_{idx:02d}.png")
    mask_path = os.path.join(patch_dir, f"patch_{idx:02d}.npy")

    cv2.imwrite(img_path, img_patch)
    np.save(
        mask_path,
        {
            "mask": mask_patch,
            "weight": weight_patch,
            "scale_down": info["scale_down"],
            "crop_box": info["patch_box"],
            "source_relpath": info["source_relpath"],
            "parent_crop": info["parent_crop"],
        },
    )


def generate_table_patches(
    input_dir,
    output_dir,
    phase="",
    scale_down=2,
    pad_min=0,
    pad_max=0,
    patch_min=0,
    patch_max=0,
    num_patches=0,
):
    parser = ParserTRACEJSON(input_dir, "", phase or "")
    print(f"Found {len(parser.gt)} labeled images.")

    for gt_entry in tqdm.tqdm(parser.gt):
        img_file = gt_entry["file_name"]
        rel_path = os.path.relpath(img_file, input_dir)
        if rel_path.startswith(".."):
            print(f"Skipping {img_file} (outside input_dir).")
            continue

        rel_without_ext = os.path.splitext(rel_path)[0]
        base_out_dir = os.path.join(output_dir, rel_without_ext)

        quads = gt_entry["quads"]
        lines = gt_entry["lines"]
        table_bounds = gt_entry.get("table_bounds") or []

        if not quads:
            print(f"Warning: no cells for {img_file}")
            continue

        img = imgproc.loadImage(img_file)
        quads_arr = np.array(quads, dtype=np.float32).reshape(-1, 8)
        crop_result = _crop_table(img, quads_arr, table_bounds, pad_min, pad_max)

        if crop_result is None:
            print(f"Warning: unable to crop table for {img_file}")
            continue

        img_crop, quads_shifted, crop_box = crop_result
        crop_h, crop_w = img_crop.shape[:2]

        quads_norm = _normalize_quads(quads_shifted, crop_w, crop_h)
        gt_image, gt_weight = _generate_mask(
            quads_norm, lines, crop_w, crop_h, scale_down
        )

        mask_data = {"mask": gt_image, "weight": gt_weight}
        _save_table_outputs(
            base_out_dir, img_crop, mask_data, crop_box, scale_down, rel_path
        )

        if num_patches <= 0:
            continue

        patches_dir = os.path.join(base_out_dir, "patches")
        max_dim = min(crop_w, crop_h)
        if max_dim <= 0:
            continue

        effective_patch_min = max(1, min(patch_min, max_dim))
        effective_patch_max = max(effective_patch_min, min(patch_max, max_dim))

        for idx in range(1, num_patches + 1):
            patch_size = random.randint(effective_patch_min, effective_patch_max)
            if patch_size > crop_w or patch_size > crop_h:
                continue

            x_max = crop_w - patch_size
            y_max = crop_h - patch_size
            if x_max < 0 or y_max < 0:
                continue

            patch_x = 0 if x_max == 0 else random.randint(0, x_max)
            patch_y = 0 if y_max == 0 else random.randint(0, y_max)

            img_patch = img_crop[
                patch_y : patch_y + patch_size, patch_x : patch_x + patch_size
            ]
            mask_patch = _crop_mask_tensor(
                gt_image, patch_x, patch_y, patch_size, scale_down, is_weight=False
            )
            weight_patch = _crop_mask_tensor(
                gt_weight, patch_x, patch_y, patch_size, scale_down, is_weight=True
            )

            patch_info = {
                "scale_down": scale_down,
                "patch_box": {
                    "x1": int(patch_x),
                    "y1": int(patch_y),
                    "x2": int(patch_x + patch_size),
                    "y2": int(patch_y + patch_size),
                },
                "source_relpath": rel_path,
                "parent_crop": crop_box,
            }
            _save_patch_outputs(
                patches_dir, idx, img_patch, mask_patch, weight_patch, patch_info
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate cropped table patches with masks."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with images and JSON labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cropped tables and patches.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="",
        help="Optional phase name (if input_dir contains sub-phases).",
    )
    parser.add_argument(
        "--scale_down", type=int, default=2, help="Scale down factor for masks."
    )
    parser.add_argument(
        "--pad_min", type=int, default=0, help="Minimum random padding around tables."
    )
    parser.add_argument(
        "--pad_max", type=int, default=0, help="Maximum random padding around tables."
    )
    parser.add_argument(
        "--patch_min",
        type=int,
        default=256,
        help="Minimum patch size for augmentation.",
    )
    parser.add_argument(
        "--patch_max",
        type=int,
        default=512,
        help="Maximum patch size for augmentation.",
    )
    parser.add_argument(
        "--num_patches", type=int, default=0, help="Number of random patches per table."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    generate_table_patches(
        input_dir=os.path.abspath(args.input_dir),
        output_dir=os.path.abspath(args.output_dir),
        phase=args.phase,
        scale_down=args.scale_down,
        pad_min=args.pad_min,
        pad_max=args.pad_max,
        patch_min=args.patch_min,
        patch_max=args.patch_max,
        num_patches=args.num_patches,
    )


if __name__ == "__main__":
    main()
