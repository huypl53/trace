# -*- coding: utf-8 -*-
"""
Utility script to visualize mask `.npy` artifacts produced by
`generate_masks_from_json.py` or `generate_table_patches.py`.

For every `.npy` under `--input_dir`, the script loads the stored mask tensor,
optionally infers the target display size from metadata, renders per-channel
binary images (corner heatmap, visible/invisible borders), and writes them to
an output directory that mirrors the input structure.
"""

import argparse
import os

import cv2
import numpy as np
import tqdm


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _infer_display_size(mask_data):
    """Return (width, height) for visualization, if metadata provides it."""
    scale_down = mask_data.get("scale_down")
    crop_box = mask_data.get("crop_box")
    patch_box = mask_data.get("patch_box")

    # Prefer patch box for patches, otherwise crop box (table)
    box = patch_box or crop_box
    if box:
        width = int(box["x2"] - box["x1"])
        height = int(box["y2"] - box["y1"])
        if width > 0 and height > 0:
            return width, height, scale_down

    if scale_down is not None and "mask" in mask_data:
        mask = mask_data["mask"]
        return int(mask.shape[1] * scale_down), int(mask.shape[0] * scale_down), scale_down

    return None, None, scale_down


def _resize_mask(gt_image, width, height):
    if gt_image is None:
        return None
    mask_height, mask_width = gt_image.shape[:2]
    if mask_width == width and mask_height == height:
        return gt_image

    resized = np.zeros((height, width, gt_image.shape[2]), dtype=gt_image.dtype)
    for ch in range(gt_image.shape[2]):
        resized[:, :, ch] = cv2.resize(
            gt_image[:, :, ch], (width, height), interpolation=cv2.INTER_LINEAR
        )
    return resized


def _visualize_mask(gt_image, output_path):
    """Save per-channel visualizations for the provided mask tensor."""
    _ensure_dir(output_path)
    corner_heatmap = gt_image[:, :, 0]
    hor_visible = gt_image[:, :, 1]
    ver_visible = gt_image[:, :, 2]
    hor_invisible = gt_image[:, :, 3]
    ver_invisible = gt_image[:, :, 4]

    def to_binary(channel):
        return (channel > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(output_path, "corner_heatmap.png"), to_binary(corner_heatmap))
    cv2.imwrite(os.path.join(output_path, "horizontal_visible.png"), to_binary(hor_visible))
    cv2.imwrite(os.path.join(output_path, "vertical_visible.png"), to_binary(ver_visible))
    cv2.imwrite(
        os.path.join(output_path, "horizontal_invisible.png"), to_binary(hor_invisible)
    )
    cv2.imwrite(
        os.path.join(output_path, "vertical_invisible.png"), to_binary(ver_invisible)
    )

    combined = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
    combined[:, :, 2] = to_binary(hor_invisible + ver_invisible)  # Blue: invisible
    combined[:, :, 0] = to_binary(hor_visible)  # Red: horizontal visible
    combined[:, :, 1] = to_binary(ver_visible)  # Green: vertical visible

    corner_binary = to_binary(corner_heatmap)
    combined = np.maximum(
        combined, np.stack([corner_binary, corner_binary, corner_binary], axis=-1)
    )
    cv2.imwrite(os.path.join(output_path, "combined.png"), combined)


def visualize_directory(input_dir, output_dir):
    npy_files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for file in filenames:
            if file.lower().endswith(".npy"):
                npy_files.append(os.path.join(dirpath, file))

    if not npy_files:
        print("No .npy files found in the specified input directory.")
        return

    for npy_path in tqdm.tqdm(npy_files, desc="Visualizing masks"):
        try:
            mask_data = np.load(npy_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Failed to load {npy_path}: {e}")
            continue

        gt_image = mask_data.get("mask")
        if gt_image is None:
            print(f"No 'mask' key in {npy_path}, skipping.")
            continue

        target_w, target_h, _ = _infer_display_size(mask_data)
        if target_w and target_h:
            gt_image_vis = _resize_mask(gt_image, target_w, target_h)
        else:
            gt_image_vis = gt_image

        rel_path = os.path.relpath(npy_path, input_dir)
        rel_base = os.path.splitext(rel_path)[0]
        out_dir = os.path.join(output_dir, rel_base)
        _visualize_mask(gt_image_vis, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize mask .npy files.")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing mask .npy files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write visualization images (mirrors input structure).",
    )

    args = parser.parse_args()
    visualize_directory(os.path.abspath(args.input_dir), os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()

