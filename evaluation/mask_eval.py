#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mask-based Line Segmentation Evaluation

Compares ground truth masks (*_mask_h.png, *_mask_v.png) against predicted masks.
Supports both binary masks and probability maps (with thresholding).

Metrics:
- IoU (Intersection over Union / Jaccard Index)
- Dice coefficient (F1 score)
- Precision, Recall

Usage:
    python evaluation/mask_eval.py \
        --gt_dir data/line_mask_dataset/test \
        --pred_dir results/predictions \
        --threshold 0.5

    # Find optimal threshold
    python evaluation/mask_eval.py \
        --gt_dir data/line_mask_dataset/test \
        --pred_dir results/predictions \
        --sweep_threshold
"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def load_mask(path, threshold=None, normalize=True):
    """Load mask image and optionally binarize.

    Args:
        path: Path to mask image
        threshold: If provided, binarize at this threshold (0-1 range)
        normalize: If True, normalize to 0-1 range before thresholding

    Returns:
        Binary mask (0 or 1) as uint8
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    if normalize and mask.max() > 0:
        mask = mask.astype(np.float32) / mask.max()
    else:
        mask = mask.astype(np.float32) / 255.0

    if threshold is not None:
        mask = (mask >= threshold).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    return mask


def compute_metrics(gt_mask, pred_mask):
    """Compute pixel-level metrics.

    Returns dict with: tp, fp, fn, tn, iou, dice, precision, recall, f1
    """
    gt_binary = gt_mask > 0
    pred_binary = pred_mask > 0

    tp = np.sum(gt_binary & pred_binary)
    fp = np.sum(~gt_binary & pred_binary)
    fn = np.sum(gt_binary & ~pred_binary)
    tn = np.sum(~gt_binary & ~pred_binary)

    # IoU (Jaccard Index)
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0

    # Dice coefficient
    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_mask_pairs(gt_dir, pred_dir, mask_suffix_h="_mask_h.png", mask_suffix_v="_mask_v.png"):
    """Find matching GT and prediction mask pairs.

    Expected structure:
        gt_dir/masks/*_mask_h.png, *_mask_v.png
        pred_dir/*_mask_h.png, *_mask_v.png  (or pred_dir/masks/...)

    Returns list of (name, gt_h_path, gt_v_path, pred_h_path, pred_v_path)
    """
    # Check if masks are in a 'masks' subdirectory
    gt_masks_dir = os.path.join(gt_dir, "masks")
    if not os.path.isdir(gt_masks_dir):
        gt_masks_dir = gt_dir

    pred_masks_dir = os.path.join(pred_dir, "masks")
    if not os.path.isdir(pred_masks_dir):
        pred_masks_dir = pred_dir

    # Find all GT horizontal masks
    pairs = []
    for fname in os.listdir(gt_masks_dir):
        if not fname.endswith(mask_suffix_h):
            continue

        name = fname[: -len(mask_suffix_h)]
        gt_h = os.path.join(gt_masks_dir, fname)
        gt_v = os.path.join(gt_masks_dir, name + mask_suffix_v)

        pred_h = os.path.join(pred_masks_dir, fname)
        pred_v = os.path.join(pred_masks_dir, name + mask_suffix_v)

        if os.path.exists(gt_v) and os.path.exists(pred_h) and os.path.exists(pred_v):
            pairs.append((name, gt_h, gt_v, pred_h, pred_v))

    return sorted(pairs, key=lambda x: x[0])


def evaluate_masks(gt_dir, pred_dir, threshold=0.5, mask_suffix_h="_mask_h.png", mask_suffix_v="_mask_v.png"):
    """Evaluate all mask pairs in directories.

    Returns:
        results: dict with aggregated metrics for h, v, and combined
        per_file: list of per-file results
    """
    pairs = find_mask_pairs(gt_dir, pred_dir, mask_suffix_h, mask_suffix_v)

    if not pairs:
        print(f"No matching mask pairs found in {gt_dir} and {pred_dir}")
        return None, []

    # Accumulators
    acc = {
        "h": defaultdict(int),
        "v": defaultdict(int),
        "all": defaultdict(int),
    }

    per_file = []

    for name, gt_h_path, gt_v_path, pred_h_path, pred_v_path in tqdm(pairs, desc="Evaluating"):
        gt_h = load_mask(gt_h_path, threshold=None)
        gt_v = load_mask(gt_v_path, threshold=None)
        pred_h = load_mask(pred_h_path, threshold=threshold)
        pred_v = load_mask(pred_v_path, threshold=threshold)

        if gt_h is None or gt_v is None or pred_h is None or pred_v is None:
            print(f"Warning: Failed to load masks for {name}")
            continue

        # Resize predictions if needed
        if pred_h.shape != gt_h.shape:
            pred_h = cv2.resize(pred_h, (gt_h.shape[1], gt_h.shape[0]), interpolation=cv2.INTER_NEAREST)
        if pred_v.shape != gt_v.shape:
            pred_v = cv2.resize(pred_v, (gt_v.shape[1], gt_v.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Compute per-channel metrics
        metrics_h = compute_metrics(gt_h, pred_h)
        metrics_v = compute_metrics(gt_v, pred_v)

        # Combined metrics
        gt_combined = np.maximum(gt_h, gt_v)
        pred_combined = np.maximum(pred_h, pred_v)
        metrics_all = compute_metrics(gt_combined, pred_combined)

        # Accumulate
        for key in ["tp", "fp", "fn", "tn"]:
            acc["h"][key] += metrics_h[key]
            acc["v"][key] += metrics_v[key]
            acc["all"][key] += metrics_all[key]

        per_file.append({
            "name": name,
            "h": metrics_h,
            "v": metrics_v,
            "all": metrics_all,
        })

    # Compute aggregated metrics
    results = {}
    for channel in ["h", "v", "all"]:
        tp = acc[channel]["tp"]
        fp = acc[channel]["fp"]
        fn = acc[channel]["fn"]
        tn = acc[channel]["tn"]

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        dice_denom = 2 * tp + fp + fn
        dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[channel] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return results, per_file


def sweep_thresholds(gt_dir, pred_dir, thresholds=None, mask_suffix_h="_mask_h.png", mask_suffix_v="_mask_v.png"):
    """Find optimal threshold by sweeping values.

    Returns:
        best_threshold: threshold with highest F1
        all_results: dict mapping threshold to results
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    all_results = {}
    best_f1 = -1
    best_threshold = 0.5

    for thresh in thresholds:
        print(f"\nEvaluating threshold={thresh:.2f}")
        results, _ = evaluate_masks(gt_dir, pred_dir, threshold=thresh, mask_suffix_h=mask_suffix_h, mask_suffix_v=mask_suffix_v)

        if results is None:
            continue

        all_results[thresh] = results
        f1 = results["all"]["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, all_results


def print_results(results, num_files):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("MASK-BASED LINE SEGMENTATION EVALUATION")
    print("=" * 60)
    print(f"Evaluated: {num_files} image pairs")

    for channel, label in [("all", "COMBINED"), ("h", "HORIZONTAL"), ("v", "VERTICAL")]:
        if channel not in results:
            continue
        m = results[channel]
        print(f"\n{label}:")
        print(f"  IoU:       {m['iou']:.4f}")
        print(f"  Dice:      {m['dice']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1:        {m['f1']:.4f}")
        print(f"  (TP={m['tp']:,}, FP={m['fp']:,}, FN={m['fn']:,})")


def print_worst_cases(per_file, n=10, sort_by="iou", channel="all"):
    """Print worst performing cases sorted by metric.

    Args:
        per_file: List of per-file results
        n: Number of worst cases to print
        sort_by: Metric to sort by (iou, dice, precision, recall)
        channel: Channel to use for sorting (all, h, v)
    """
    if not per_file:
        return

    # Sort by metric (ascending = worst first)
    sorted_files = sorted(per_file, key=lambda x: x[channel][sort_by])

    print(f"\n{'=' * 60}")
    print(f"WORST {n} CASES (sorted by {sort_by.upper()} on {channel.upper()})")
    print("=" * 60)

    for i, item in enumerate(sorted_files[:n]):
        m = item[channel]
        print(f"\n{i+1}. {item['name']}")
        print(f"   IoU={m['iou']:.4f}, Dice={m['dice']:.4f}, "
              f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}")
        print(f"   TP={m['tp']:,}, FP={m['fp']:,}, FN={m['fn']:,}")


def save_worst_case_visualizations(
    per_file, gt_dir, pred_dir, output_dir, n=10, sort_by="iou", channel="all",
    threshold=0.5, mask_suffix_h="_mask_h.png", mask_suffix_v="_mask_v.png"
):
    """Save visualization images for worst cases.

    Creates side-by-side comparison images showing:
    - Original image (if available)
    - GT mask (green=horizontal, red=vertical)
    - Pred mask
    - Diff (green=TP, red=FN, blue=FP)
    """
    if not per_file:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Sort by metric (ascending = worst first)
    sorted_files = sorted(per_file, key=lambda x: x[channel][sort_by])

    # Check if masks are in a 'masks' subdirectory
    gt_masks_dir = os.path.join(gt_dir, "masks")
    if not os.path.isdir(gt_masks_dir):
        gt_masks_dir = gt_dir

    gt_images_dir = os.path.join(gt_dir, "images")
    if not os.path.isdir(gt_images_dir):
        gt_images_dir = gt_dir

    pred_masks_dir = os.path.join(pred_dir, "masks")
    if not os.path.isdir(pred_masks_dir):
        pred_masks_dir = pred_dir

    print(f"\nSaving {n} worst case visualizations to {output_dir}...")

    for i, item in enumerate(sorted_files[:n]):
        name = item["name"]
        m = item[channel]

        # Load masks
        gt_h = cv2.imread(os.path.join(gt_masks_dir, name + mask_suffix_h), cv2.IMREAD_GRAYSCALE)
        gt_v = cv2.imread(os.path.join(gt_masks_dir, name + mask_suffix_v), cv2.IMREAD_GRAYSCALE)
        pred_h = load_mask(os.path.join(pred_masks_dir, name + mask_suffix_h), threshold=threshold)
        pred_v = load_mask(os.path.join(pred_masks_dir, name + mask_suffix_v), threshold=threshold)

        if gt_h is None or gt_v is None or pred_h is None or pred_v is None:
            continue

        # Binarize GT
        gt_h_bin = (gt_h > 127).astype(np.uint8)
        gt_v_bin = (gt_v > 127).astype(np.uint8)

        height, width = gt_h.shape

        # Resize predictions if needed
        if pred_h.shape != gt_h.shape:
            pred_h = cv2.resize(pred_h, (width, height), interpolation=cv2.INTER_NEAREST)
        if pred_v.shape != gt_v.shape:
            pred_v = cv2.resize(pred_v, (width, height), interpolation=cv2.INTER_NEAREST)

        # Try to load original image
        orig_img = None
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(gt_images_dir, name + ext)
            if os.path.exists(img_path):
                orig_img = cv2.imread(img_path)
                break

        # Create visualization panels
        panels = []

        # Panel 1: Original image (or blank)
        if orig_img is not None:
            orig_resized = cv2.resize(orig_img, (width, height))
            panels.append(orig_resized)
        else:
            panels.append(np.ones((height, width, 3), dtype=np.uint8) * 200)

        # Panel 2: GT mask (green=horizontal, red=vertical)
        gt_vis = np.zeros((height, width, 3), dtype=np.uint8)
        gt_vis[:, :, 1] = gt_h_bin * 255  # Green for horizontal
        gt_vis[:, :, 2] = gt_v_bin * 255  # Red for vertical
        panels.append(gt_vis)

        # Panel 3: Pred mask (green=horizontal, red=vertical)
        pred_vis = np.zeros((height, width, 3), dtype=np.uint8)
        pred_vis[:, :, 1] = pred_h * 255  # Green for horizontal
        pred_vis[:, :, 2] = pred_v * 255  # Red for vertical
        panels.append(pred_vis)

        # Panel 4: Diff visualization
        # Green = TP (both GT and pred), Red = FN (GT only), Blue = FP (pred only)
        gt_combined = np.maximum(gt_h_bin, gt_v_bin)
        pred_combined = np.maximum(pred_h, pred_v)

        diff_vis = np.zeros((height, width, 3), dtype=np.uint8)
        tp_mask = (gt_combined > 0) & (pred_combined > 0)
        fn_mask = (gt_combined > 0) & (pred_combined == 0)
        fp_mask = (gt_combined == 0) & (pred_combined > 0)

        diff_vis[tp_mask] = [0, 255, 0]   # Green = TP
        diff_vis[fn_mask] = [0, 0, 255]   # Red = FN (missed)
        diff_vis[fp_mask] = [255, 0, 0]   # Blue = FP (false alarm)
        panels.append(diff_vis)

        # Concatenate panels horizontally
        combined = np.hstack(panels)

        # Add text annotation
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Rank {i+1}: {name} | IoU={m['iou']:.3f} Dice={m['dice']:.3f} | TP={m['tp']} FP={m['fp']} FN={m['fn']}"
        cv2.putText(combined, text, (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, text, (10, 25), font, 0.6, (0, 0, 0), 1)

        # Add panel labels
        labels = ["Original", "GT (G=H, R=V)", "Pred (G=H, R=V)", "Diff (G=TP, R=FN, B=FP)"]
        for j, label in enumerate(labels):
            x = j * width + 10
            cv2.putText(combined, label, (x, height - 10), font, 0.5, (255, 255, 255), 2)
            cv2.putText(combined, label, (x, height - 10), font, 0.5, (0, 0, 0), 1)

        # Save
        out_path = os.path.join(output_dir, f"worst_{i+1:02d}_{name}.png")
        cv2.imwrite(out_path, combined)

    print(f"Saved {min(n, len(sorted_files))} visualizations.")


def main():
    parser = argparse.ArgumentParser(description="Mask-based line segmentation evaluation")
    parser.add_argument("--gt_dir", required=True, help="Ground truth directory (with masks/ subdirectory)")
    parser.add_argument("--pred_dir", required=True, help="Predictions directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (0-1)")
    parser.add_argument("--mask_suffix_h", default="_mask_h.png", help="Horizontal mask suffix")
    parser.add_argument("--mask_suffix_v", default="_mask_v.png", help="Vertical mask suffix")
    parser.add_argument("--sweep_threshold", action="store_true", help="Sweep thresholds to find optimal")
    parser.add_argument("--save_json", type=str, help="Save results to JSON file")
    # Worst case analysis
    parser.add_argument("--show_worst", type=int, default=0, help="Show N worst cases (0=disabled)")
    parser.add_argument("--save_worst", type=str, help="Save worst case visualizations to directory")
    parser.add_argument("--sort_by", default="iou", choices=["iou", "dice", "precision", "recall"],
                        help="Metric to sort worst cases by")
    parser.add_argument("--sort_channel", default="all", choices=["all", "h", "v"],
                        help="Channel to use for sorting (all, h=horizontal, v=vertical)")
    args = parser.parse_args()

    best_thresh = args.threshold  # Default, may be updated by sweep
    if args.sweep_threshold:
        best_thresh, all_results = sweep_thresholds(
            args.gt_dir,
            args.pred_dir,
            mask_suffix_h=args.mask_suffix_h,
            mask_suffix_v=args.mask_suffix_v,
        )
        print("\n" + "=" * 60)
        print("THRESHOLD SWEEP RESULTS")
        print("=" * 60)
        for thresh in sorted(all_results.keys()):
            r = all_results[thresh]["all"]
            print(f"  {thresh:.2f}: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}, F1={r['f1']:.4f}")
        print(f"\nBest threshold: {best_thresh:.2f}")

        # Print detailed results for best threshold
        results, per_file = evaluate_masks(
            args.gt_dir,
            args.pred_dir,
            threshold=best_thresh,
            mask_suffix_h=args.mask_suffix_h,
            mask_suffix_v=args.mask_suffix_v,
        )
        if results:
            print_results(results, len(per_file))
    else:
        results, per_file = evaluate_masks(
            args.gt_dir,
            args.pred_dir,
            threshold=args.threshold,
            mask_suffix_h=args.mask_suffix_h,
            mask_suffix_v=args.mask_suffix_v,
        )
        if results:
            print_results(results, len(per_file))

    if args.save_json and results:
        import json
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    # Worst case analysis
    if per_file and (args.show_worst > 0 or args.save_worst):
        n_worst = args.show_worst if args.show_worst > 0 else 10

        if args.show_worst > 0:
            print_worst_cases(per_file, n=n_worst, sort_by=args.sort_by, channel=args.sort_channel)

        if args.save_worst:
            save_worst_case_visualizations(
                per_file,
                args.gt_dir,
                args.pred_dir,
                args.save_worst,
                n=n_worst,
                sort_by=args.sort_by,
                channel=args.sort_channel,
                threshold=best_thresh,
                mask_suffix_h=args.mask_suffix_h,
                mask_suffix_v=args.mask_suffix_v,
            )


if __name__ == "__main__":
    main()
