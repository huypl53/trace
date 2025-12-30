# -*- coding: utf-8 -*-
import argparse
import json
import os

import cv2
import numpy as np

from parsers.canvas_parser import extract_table_lines, infer_canvas_size


def draw_line_masks(lines_h, lines_v, width, height):
    mask_h = np.zeros((height, width), dtype=np.uint8)
    mask_v = np.zeros((height, width), dtype=np.uint8)

    for start, end, thickness in lines_h:
        cv2.line(mask_h, start, end, color=255, thickness=thickness)
    for start, end, thickness in lines_v:
        cv2.line(mask_v, start, end, color=255, thickness=thickness)

    return mask_h, mask_v


def process_file(path, out_dir, width_override=None, height_override=None, preview_only=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines_h, lines_v = extract_table_lines(data)
    width, height = infer_canvas_size(data)
    if width_override is not None:
        width = width_override
    if height_override is not None:
        height = height_override

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid canvas size inferred or provided for {path}.")

    mask_h, mask_v = draw_line_masks(lines_h, lines_v, width, height)

    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, f"{stem}_mask_h.png")
    v_path = os.path.join(out_dir, f"{stem}_mask_v.png")
    if not preview_only:
        cv2.imwrite(h_path, mask_h)
        cv2.imwrite(v_path, mask_v)

    preview = np.zeros((height, width, 3), dtype=np.uint8)
    preview[:, :, 1] = mask_h
    preview[:, :, 2] = mask_v
    preview_path = os.path.join(out_dir, f"{stem}_mask_preview.png")
    cv2.imwrite(preview_path, preview)

    print("Saved:")
    if not preview_only:
        print(h_path)
        print(v_path)
    print(preview_path)


def list_json_files(input_path, recursive=False):
    if os.path.isfile(input_path):
        return [input_path]
    json_files = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for name in files:
                if name.endswith(".json"):
                    json_files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_path):
            if name.endswith(".json"):
                json_files.append(os.path.join(input_path, name))
    return sorted(json_files)


def main():
    parser = argparse.ArgumentParser(description="Draw line masks from canvas JSON")
    parser.add_argument("--input", required=True, help="Path to canvas JSON file or directory")
    parser.add_argument("--out_dir", default=".", help="Output directory for masks")
    parser.add_argument("--width", type=int, default=None, help="Override output width")
    parser.add_argument("--height", type=int, default=None, help="Override output height")
    parser.add_argument("--recursive", action="store_true", help="Process JSONs in subdirectories")
    parser.add_argument("--preview-only", action="store_true", help="Only write preview image")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    json_files = list_json_files(args.input, args.recursive)
    if not json_files:
        raise ValueError(f"No JSON files found in {args.input}")

    for path in json_files:
        process_file(path, args.out_dir, args.width, args.height, args.preview_only)


if __name__ == "__main__":
    main()
