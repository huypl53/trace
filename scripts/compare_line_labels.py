# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os

import cv2
import numpy as np

import file_utils


GT_H_COLOR = (255, 0, 0)      # blue (BGR)
GT_V_COLOR = (255, 255, 0)    # cyan
PRED_H_COLOR = (0, 0, 255)    # red
PRED_V_COLOR = (0, 165, 255)  # orange

TP_COLOR = (0, 255, 0)        # green
FP_COLOR = (0, 0, 255)        # red
FN_COLOR = (255, 0, 0)        # blue


def load_line_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h_lines = []
    v_lines = []
    for line in data.get("lines", []):
        pts = line.get("points", [])
        if len(pts) != 2:
            continue
        start = (float(pts[0][0]), float(pts[0][1]))
        end = (float(pts[1][0]), float(pts[1][1]))
        if line.get("type") == "horizontal":
            h_lines.append((start, end))
        elif line.get("type") == "vertical":
            v_lines.append((start, end))
    return h_lines, v_lines, data


def scale_lines(lines, scale_x, scale_y):
    scaled = []
    for (x1, y1), (x2, y2) in lines:
        scaled.append(((x1 * scale_x, y1 * scale_y), (x2 * scale_x, y2 * scale_y)))
    return scaled


def draw_lines(img, lines, color, thickness):
    for (x1, y1), (x2, y2) in lines:
        cv2.line(
            img,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            thickness,
        )


def draw_lines_mask(lines, shape, thickness):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(
            mask,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            255,
            thickness,
        )
    return mask


def find_image_path(gt_path, image_dir, gt_data):
    search_dirs = []
    if image_dir:
        search_dirs.append(image_dir)
    search_dirs.append(os.path.dirname(gt_path))

    filename = gt_data.get("filename")
    if filename:
        for d in search_dirs:
            candidate = os.path.join(d, filename)
            if os.path.exists(candidate):
                return candidate

    base = os.path.splitext(os.path.basename(gt_path))[0]
    for d in search_dirs:
        for ext in file_utils.IMAGE_EXTENTIONS:
            candidate = os.path.join(d, f"{base}.{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def parse_image_size(value):
    if not value:
        return None
    sep = "x" if "x" in value else ","
    parts = value.split(sep)
    if len(parts) != 2:
        raise ValueError("image_size must be in W,H or WxH format")
    return int(parts[0]), int(parts[1])


def build_pred_map(pred_dir, recursive):
    pattern = "**/*.json" if recursive else "*.json"
    pred_paths = glob.glob(os.path.join(pred_dir, pattern), recursive=recursive)
    pred_map = {}
    duplicates = []
    for path in pred_paths:
        rel = os.path.relpath(path, pred_dir)
        key = os.path.splitext(rel)[0]
        if key in pred_map:
            duplicates.append(key)
            continue
        pred_map[key] = path
    if duplicates:
        print(f"Warning: duplicate pred relpaths, keeping first: {sorted(set(duplicates))}")
    return pred_map


def add_legend(img, show_diff):
    legend1 = "GT-H blue  GT-V cyan  Pred-H red  Pred-V orange"
    legend2 = "Diff: TP green  FP red  FN blue"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    x, y = 10, 20
    cv2.putText(img, legend1, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, legend1, (x, y), font, scale, (255, 255, 255), thickness)
    if show_diff:
        y2 = y + 20
        cv2.putText(img, legend2, (x + 1, y2 + 1), font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, legend2, (x, y2), font, scale, (255, 255, 255), thickness)


def main():
    parser = argparse.ArgumentParser(description="Compare GT and predicted line labels")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT line JSONs")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted line JSONs")
    parser.add_argument("--out_dir", required=True, help="Output directory for comparison images")
    parser.add_argument("--image_dir", default=None, help="Optional directory with original images")
    parser.add_argument("--pred_size", type=int, default=1280, help="Predicted line coordinate size")
    parser.add_argument("--line_thickness", type=int, default=2, help="Line thickness for drawing")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha for diff overlay")
    parser.add_argument("--recursive", action="store_true", help="Search GT/pred dirs recursively")
    parser.add_argument("--no_diff", action="store_true", help="Disable TP/FP/FN overlay")
    parser.add_argument("--image_size", default=None, help="Fallback size W,H if image missing")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of outputs (0 = no limit)")
    args = parser.parse_args()

    gt_pattern = "**/*.json" if args.recursive else "*.json"
    gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, gt_pattern), recursive=args.recursive))
    if not gt_paths:
        raise ValueError(f"No GT JSONs found in {args.gt_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    pred_map = build_pred_map(args.pred_dir, args.recursive)
    fallback_size = parse_image_size(args.image_size) if args.image_size else None

    processed = 0
    for gt_path in gt_paths:
        rel = os.path.relpath(gt_path, args.gt_dir)
        key = os.path.splitext(rel)[0]
        base = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = pred_map.get(key) or pred_map.get(base)
        if not pred_path:
            print(f"Missing pred for {key}, skipping")
            continue

        gt_h, gt_v, gt_data = load_line_json(gt_path)
        pred_h, pred_v, _ = load_line_json(pred_path)

        image_path = find_image_path(gt_path, args.image_dir, gt_data)
        image = cv2.imread(image_path) if image_path else None

        if image is None:
            if fallback_size:
                width, height = fallback_size
            else:
                all_pts = [p for line in gt_h + gt_v for p in line]
                if not all_pts:
                    print(f"No image or GT points for {base}, skipping")
                    continue
                max_x = int(max(p[0] for p in all_pts)) + 1
                max_y = int(max(p[1] for p in all_pts)) + 1
                width, height = max_x, max_y
            image = np.full((height, width, 3), 255, dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]
        if args.pred_size > 0:
            scale_x = orig_w / float(args.pred_size)
            scale_y = orig_h / float(args.pred_size)
            pred_h = scale_lines(pred_h, scale_x, scale_y)
            pred_v = scale_lines(pred_v, scale_x, scale_y)

        overlay = image.copy()
        draw_lines(overlay, gt_h, GT_H_COLOR, args.line_thickness)
        draw_lines(overlay, gt_v, GT_V_COLOR, args.line_thickness)
        draw_lines(overlay, pred_h, PRED_H_COLOR, args.line_thickness)
        draw_lines(overlay, pred_v, PRED_V_COLOR, args.line_thickness)

        if not args.no_diff:
            gt_mask = draw_lines_mask(gt_h + gt_v, overlay.shape, args.line_thickness)
            pred_mask = draw_lines_mask(pred_h + pred_v, overlay.shape, args.line_thickness)
            tp = (gt_mask > 0) & (pred_mask > 0)
            fp = (gt_mask == 0) & (pred_mask > 0)
            fn = (gt_mask > 0) & (pred_mask == 0)
            diff = np.zeros_like(overlay)
            diff[tp] = TP_COLOR
            diff[fp] = FP_COLOR
            diff[fn] = FN_COLOR
            overlay = cv2.addWeighted(overlay, 1.0, diff, args.alpha, 0)

        add_legend(overlay, not args.no_diff)
        out_rel = f"{key}_compare.png"
        out_path = os.path.join(args.out_dir, out_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, overlay)
        processed += 1
        if args.max_images and processed >= args.max_images:
            break

    print(f"Saved {processed} comparisons to {args.out_dir}")


if __name__ == "__main__":
    main()
