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


def main():
    parser = argparse.ArgumentParser(description="Draw line masks from canvas JSON")
    parser.add_argument("--input", required=True, help="Path to canvas JSON file")
    parser.add_argument("--out_dir", default=".", help="Output directory for masks")
    parser.add_argument("--width", type=int, default=None, help="Override output width")
    parser.add_argument("--height", type=int, default=None, help="Override output height")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines_h, lines_v = extract_table_lines(data)
    width, height = infer_canvas_size(data)
    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height

    if width <= 0 or height <= 0:
        raise ValueError("Invalid canvas size inferred or provided.")

    mask_h, mask_v = draw_line_masks(lines_h, lines_v, width, height)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    h_path = os.path.join(args.out_dir, f"{stem}_mask_h.png")
    v_path = os.path.join(args.out_dir, f"{stem}_mask_v.png")
    cv2.imwrite(h_path, mask_h)
    cv2.imwrite(v_path, mask_v)

    preview = np.zeros((height, width, 3), dtype=np.uint8)
    preview[:, :, 1] = mask_h
    preview[:, :, 2] = mask_v
    preview_path = os.path.join(args.out_dir, f"{stem}_mask_preview.png")
    cv2.imwrite(preview_path, preview)

    print("Saved:")
    print(h_path)
    print(v_path)
    print(preview_path)


if __name__ == "__main__":
    main()
