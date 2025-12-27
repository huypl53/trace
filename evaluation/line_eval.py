#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Line Segmentation Evaluation

Metrics:
- Pixel-level: Precision, Recall, F1 on line pixels
- Line-level: Matched lines based on endpoint distance threshold
- Separate metrics for horizontal and vertical lines

Usage:
    python evaluation/line_eval.py \
        --gt_path data/line_dataset/test \
        --pred_path results/predictions \
        --threshold 10
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def load_line_json(path):
    """Load line annotation JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h_lines = []
    v_lines = []
    for line in data.get("lines", []):
        pts = line["points"]
        start = tuple(pts[0])
        end = tuple(pts[1])
        if line["type"] == "horizontal":
            h_lines.append((start, end))
        else:
            v_lines.append((start, end))
    return h_lines, v_lines


def draw_lines_mask(lines, shape, thickness=2):
    """Draw lines as binary mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for start, end in lines:
        cv2.line(mask, (int(start[0]), int(start[1])),
                 (int(end[0]), int(end[1])), 255, thickness)
    return mask


def pixel_metrics(gt_mask, pred_mask):
    """Calculate pixel-level metrics."""
    gt_binary = gt_mask > 0
    pred_binary = pred_mask > 0

    tp = np.sum(gt_binary & pred_binary)
    fp = np.sum(~gt_binary & pred_binary)
    fn = np.sum(gt_binary & ~pred_binary)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": f1}


def line_distance(line1, line2):
    """Calculate distance between two lines (sum of endpoint distances)."""
    s1, e1 = line1
    s2, e2 = line2

    # Try both orderings
    d1 = np.sqrt((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2) + \
         np.sqrt((e1[0]-e2[0])**2 + (e1[1]-e2[1])**2)
    d2 = np.sqrt((s1[0]-e2[0])**2 + (s1[1]-e2[1])**2) + \
         np.sqrt((e1[0]-s2[0])**2 + (e1[1]-s2[1])**2)

    return min(d1, d2)


def line_metrics(gt_lines, pred_lines, threshold=10):
    """Calculate line-level metrics based on endpoint matching."""
    matched_gt = set()
    matched_pred = set()

    for i, gt_line in enumerate(gt_lines):
        for j, pred_line in enumerate(pred_lines):
            if j in matched_pred:
                continue
            dist = line_distance(gt_line, pred_line)
            if dist < threshold:
                matched_gt.add(i)
                matched_pred.add(j)
                break

    tp = len(matched_gt)
    fp = len(pred_lines) - len(matched_pred)
    fn = len(gt_lines) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_file(gt_path, pred_path, img_shape, threshold=10, line_thickness=2):
    """Evaluate a single file pair."""
    gt_h, gt_v = load_line_json(gt_path)
    pred_h, pred_v = load_line_json(pred_path)

    results = {}

    # Pixel-level metrics
    gt_mask_h = draw_lines_mask(gt_h, img_shape, line_thickness)
    gt_mask_v = draw_lines_mask(gt_v, img_shape, line_thickness)
    pred_mask_h = draw_lines_mask(pred_h, img_shape, line_thickness)
    pred_mask_v = draw_lines_mask(pred_v, img_shape, line_thickness)

    results["pixel_h"] = pixel_metrics(gt_mask_h, pred_mask_h)
    results["pixel_v"] = pixel_metrics(gt_mask_v, pred_mask_v)
    results["pixel_all"] = pixel_metrics(
        np.maximum(gt_mask_h, gt_mask_v),
        np.maximum(pred_mask_h, pred_mask_v)
    )

    # Line-level metrics
    results["line_h"] = line_metrics(gt_h, pred_h, threshold)
    results["line_v"] = line_metrics(gt_v, pred_v, threshold)
    results["line_all"] = line_metrics(gt_h + gt_v, pred_h + pred_v, threshold)

    return results


def aggregate_results(all_results):
    """Aggregate results across all files."""
    agg = defaultdict(lambda: defaultdict(float))

    for result in all_results:
        for metric_type, metrics in result.items():
            for key, value in metrics.items():
                agg[metric_type][key] += value

    # Calculate final precision/recall/f1
    summary = {}
    for metric_type in agg:
        tp = agg[metric_type]["tp"]
        fp = agg[metric_type]["fp"]
        fn = agg[metric_type]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary[metric_type] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": f1
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Line segmentation evaluation")
    parser.add_argument("--gt_path", required=True, help="Ground truth directory")
    parser.add_argument("--pred_path", required=True, help="Predictions directory")
    parser.add_argument("--threshold", type=float, default=10,
                        help="Line matching threshold in pixels")
    parser.add_argument("--line_thickness", type=int, default=2,
                        help="Line thickness for pixel metrics")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Find all ground truth files
    gt_files = glob.glob(os.path.join(args.gt_path, "*.json"))

    if not gt_files:
        print(f"No JSON files found in {args.gt_path}")
        return

    print(f"Found {len(gt_files)} ground truth files")

    all_results = []
    missing_count = 0
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        pred_file = os.path.join(args.pred_path, basename)

        if not os.path.exists(pred_file):
            print(f"Warning: No prediction for {basename}")
            missing_count += 1
            continue

        # Get image shape from corresponding image
        img_base = os.path.splitext(gt_file)[0]
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            if os.path.exists(img_base + ext):
                img_path = img_base + ext
                break

        if img_path:
            img = cv2.imread(img_path)
            img_shape = img.shape
        else:
            # Default shape if no image found
            img_shape = (1000, 1000, 3)

        result = evaluate_file(gt_file, pred_file, img_shape,
                               args.threshold, args.line_thickness)
        all_results.append(result)

    if not all_results:
        print("No valid file pairs to evaluate")
        return

    # Aggregate and print results
    summary = aggregate_results(all_results)

    print("\n" + "="*60)
    print("LINE SEGMENTATION EVALUATION RESULTS")
    print("="*60)
    print(f"Evaluated: {len(all_results)} files (missing: {missing_count})")

    for metric_type in ["line_all", "line_h", "line_v", "pixel_all", "pixel_h", "pixel_v"]:
        if metric_type in summary:
            m = summary[metric_type]
            print(f"\n{metric_type.upper()}:")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1:        {m['f1']:.4f}")
            print(f"  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
