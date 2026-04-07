#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate line mask predictions against ground truth.

Computes per-image metrics (Dice, IoU, Precision, Recall), identifies weak
images, and generates a statistical report.

Usage with pre-computed predictions:
    python evaluate_masks.py \
        --pred_dir results/pred_masks \
        --gt_dir data/test/masks \
        --output_dir results/analysis

Usage with on-the-fly inference:
    python evaluate_masks.py \
        --images_dir data/test/images \
        --gt_dir data/test/masks \
        --trained_model eval/model.pth \
        --output_dir results/analysis \
        --threshold_h 0.3 --threshold_v 0.2
"""

import argparse
import csv
import json
import os
import shutil
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

import imgproc
from model import TraceModel


# ── Model loading & inference ──────────────────────────────────────────────

def load_model(model_path, output_ch=2, device="cuda"):
    net = TraceModel(output_ch=output_ch)
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[".".join(k.split(".")[1:])] = v
        state_dict = new_state_dict
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    return net


@torch.no_grad()
def infer_image(net, image, input_size, device):
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = imgproc.normalizeMeanVariance(resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    out = net(x)
    if isinstance(out, tuple):
        out = out[0]
    return out[0].cpu().numpy()  # (H, W, output_ch)


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(gt_bin, pred_bin):
    """Compute pixel-level metrics for a single binary mask pair."""
    tp = int(np.sum(gt_bin & pred_bin))
    fp = int(np.sum(~gt_bin & pred_bin))
    fn = int(np.sum(gt_bin & ~pred_bin))
    tn = int(np.sum(~gt_bin & ~pred_bin))

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0
    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denom if dice_denom > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1,
        "gt_pixels": int(np.sum(gt_bin)),
        "pred_pixels": int(np.sum(pred_bin)),
    }


def evaluate_single_image(gt_h, gt_v, pred_h_bin, pred_v_bin):
    """Compute per-channel and combined metrics for one image."""
    gt_h_bin = gt_h > 127
    gt_v_bin = gt_v > 127

    metrics_h = compute_metrics(gt_h_bin, pred_h_bin)
    metrics_v = compute_metrics(gt_v_bin, pred_v_bin)

    gt_combined = gt_h_bin | gt_v_bin
    pred_combined = pred_h_bin | pred_v_bin
    metrics_all = compute_metrics(gt_combined, pred_combined)

    return {"h": metrics_h, "v": metrics_v, "all": metrics_all}


# ── Reporting ──────────────────────────────────────────────────────────────

def aggregate_metrics(all_results):
    """Compute dataset-level aggregate metrics from per-image results."""
    agg = {}
    for channel in ["h", "v", "all"]:
        tp = sum(r[channel]["tp"] for r in all_results.values())
        fp = sum(r[channel]["fp"] for r in all_results.values())
        fn = sum(r[channel]["fn"] for r in all_results.values())

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        dice_denom = 2 * tp + fp + fn
        dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Also compute mean of per-image metrics
        per_image_dice = [r[channel]["dice"] for r in all_results.values()]
        per_image_iou = [r[channel]["iou"] for r in all_results.values()]

        agg[channel] = {
            "micro_dice": dice,
            "micro_iou": iou,
            "micro_precision": precision,
            "micro_recall": recall,
            "macro_dice": float(np.mean(per_image_dice)),
            "macro_iou": float(np.mean(per_image_iou)),
            "dice_std": float(np.std(per_image_dice)),
            "iou_std": float(np.std(per_image_iou)),
            "total_tp": tp,
            "total_fp": fp,
            "total_fn": fn,
        }
    return agg


def identify_weak_images(all_results, metric="dice", channel="all", bottom_pct=0.1):
    """Identify weakest images by the given metric."""
    scores = [(name, r[channel][metric]) for name, r in all_results.items()]
    scores.sort(key=lambda x: x[1])

    n_weak = max(1, int(len(scores) * bottom_pct))
    weak = scores[:n_weak]
    return weak, scores


def analyze_failure_modes(all_results):
    """Categorize failures into modes: false positives vs false negatives."""
    modes = {
        "high_fp": [],  # model predicts lines where there are none
        "high_fn": [],  # model misses existing lines
        "balanced_poor": [],  # both FP and FN are high
    }

    for name, r in all_results.items():
        m = r["all"]
        if m["gt_pixels"] == 0 and m["pred_pixels"] == 0:
            continue

        total = m["gt_pixels"] + m["pred_pixels"]
        if total == 0:
            continue

        fp_rate = m["fp"] / max(1, m["pred_pixels"])
        fn_rate = m["fn"] / max(1, m["gt_pixels"])

        if m["dice"] < 0.7:
            if fp_rate > 0.5 and fn_rate <= 0.3:
                modes["high_fp"].append((name, m["dice"], fp_rate, fn_rate))
            elif fn_rate > 0.5 and fp_rate <= 0.3:
                modes["high_fn"].append((name, m["dice"], fp_rate, fn_rate))
            else:
                modes["balanced_poor"].append((name, m["dice"], fp_rate, fn_rate))

    for key in modes:
        modes[key].sort(key=lambda x: x[1])

    return modes


def save_comparison_images(weak_images, images_dir, gt_dir, pred_dir, output_dir, max_images=50):
    """Save side-by-side comparison images for the weakest predictions."""
    vis_dir = os.path.join(output_dir, "weak_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (name, score) in enumerate(weak_images[:max_images]):
        # Try to load original image
        img_path = None
        if images_dir:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(images_dir, name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        # Load GT masks
        gt_h_path = os.path.join(gt_dir, f"{name}_mask_h.png")
        gt_v_path = os.path.join(gt_dir, f"{name}_mask_v.png")
        if not os.path.exists(gt_h_path) or not os.path.exists(gt_v_path):
            continue
        gt_h = cv2.imread(gt_h_path, cv2.IMREAD_GRAYSCALE)
        gt_v = cv2.imread(gt_v_path, cv2.IMREAD_GRAYSCALE)

        # Load pred masks (combined or separate)
        combined_path = os.path.join(pred_dir, f"{name}_mask.png")
        if os.path.exists(combined_path):
            combined_img = cv2.imread(combined_path, cv2.IMREAD_COLOR)
            if combined_img is None:
                continue
            pred_h = combined_img[:, :, 1]  # Green channel
            pred_v = combined_img[:, :, 2]  # Red channel
        else:
            pred_h_path = os.path.join(pred_dir, f"{name}_mask_h.png")
            pred_v_path = os.path.join(pred_dir, f"{name}_mask_v.png")
            if not os.path.exists(pred_h_path) or not os.path.exists(pred_v_path):
                continue
            pred_h = cv2.imread(pred_h_path, cv2.IMREAD_GRAYSCALE)
            pred_v = cv2.imread(pred_v_path, cv2.IMREAD_GRAYSCALE)

        # Resize all to same size
        h, w = gt_h.shape[:2]
        pred_h = cv2.resize(pred_h, (w, h), interpolation=cv2.INTER_NEAREST)
        pred_v = cv2.resize(pred_v, (w, h), interpolation=cv2.INTER_NEAREST)

        # Build comparison: GT (green) vs Pred (red), overlap (yellow)
        gt_combined = np.maximum(gt_h, gt_v)
        pred_combined = np.maximum(pred_h, pred_v)

        vis = np.zeros((h, w, 3), dtype=np.uint8)
        # Green = GT only (false negative)
        gt_only = (gt_combined > 127) & (pred_combined <= 127)
        vis[gt_only] = [0, 255, 0]
        # Red = Pred only (false positive)
        pred_only = (pred_combined > 127) & (gt_combined <= 127)
        vis[pred_only] = [0, 0, 255]
        # Yellow = overlap (true positive)
        overlap = (gt_combined > 127) & (pred_combined > 127)
        vis[overlap] = [0, 255, 255]

        # If we have the original image, put it side by side
        if img_path:
            orig = cv2.imread(img_path)
            if orig is not None:
                orig = cv2.resize(orig, (w, h))
                vis = np.hstack([orig, vis])

        out_path = os.path.join(vis_dir, f"{i:04d}_dice{score:.3f}_{name}.png")
        cv2.imwrite(out_path, vis)


def write_report(all_results, agg, weak_images, failure_modes, output_dir):
    """Write CSV and JSON reports."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-image CSV
    csv_path = os.path.join(output_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "dice_all", "iou_all", "precision_all", "recall_all",
            "dice_h", "iou_h", "precision_h", "recall_h",
            "dice_v", "iou_v", "precision_v", "recall_v",
            "gt_pixels_h", "gt_pixels_v", "pred_pixels_h", "pred_pixels_v",
            "fp_all", "fn_all",
        ])
        for name in sorted(all_results.keys()):
            r = all_results[name]
            writer.writerow([
                name,
                f"{r['all']['dice']:.4f}", f"{r['all']['iou']:.4f}",
                f"{r['all']['precision']:.4f}", f"{r['all']['recall']:.4f}",
                f"{r['h']['dice']:.4f}", f"{r['h']['iou']:.4f}",
                f"{r['h']['precision']:.4f}", f"{r['h']['recall']:.4f}",
                f"{r['v']['dice']:.4f}", f"{r['v']['iou']:.4f}",
                f"{r['v']['precision']:.4f}", f"{r['v']['recall']:.4f}",
                r["h"]["gt_pixels"], r["v"]["gt_pixels"],
                r["h"]["pred_pixels"], r["v"]["pred_pixels"],
                r["all"]["fp"], r["all"]["fn"],
            ])

    # Summary JSON
    summary = {
        "total_images": len(all_results),
        "aggregate_metrics": agg,
        "weak_images_count": len(weak_images),
        "weak_images": [{"name": n, "dice": float(s)} for n, s in weak_images],
        "failure_modes": {
            k: [{"name": n, "dice": float(d), "fp_rate": float(fp), "fn_rate": float(fn)}
                for n, d, fp, fn in v]
            for k, v in failure_modes.items()
        },
        "failure_mode_counts": {k: len(v) for k, v in failure_modes.items()},
    }

    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def print_summary(agg, weak_images, failure_modes, total):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total images evaluated: {total}")

    for ch_name, ch_label in [("all", "Combined"), ("h", "Horizontal"), ("v", "Vertical")]:
        m = agg[ch_name]
        print(f"\n  {ch_label}:")
        print(f"    Micro Dice:      {m['micro_dice']:.4f}")
        print(f"    Micro IoU:       {m['micro_iou']:.4f}")
        print(f"    Micro Precision: {m['micro_precision']:.4f}")
        print(f"    Micro Recall:    {m['micro_recall']:.4f}")
        print(f"    Macro Dice:      {m['macro_dice']:.4f} (std: {m['dice_std']:.4f})")
        print(f"    Macro IoU:       {m['macro_iou']:.4f} (std: {m['iou_std']:.4f})")

    print(f"\n{'─' * 70}")
    print(f"WEAK IMAGES (bottom 10%, sorted by dice):")
    print(f"{'─' * 70}")
    for name, score in weak_images[:20]:
        print(f"  {score:.4f}  {name}")
    if len(weak_images) > 20:
        print(f"  ... and {len(weak_images) - 20} more (see CSV for full list)")

    print(f"\n{'─' * 70}")
    print(f"FAILURE MODE BREAKDOWN (images with dice < 0.7):")
    print(f"{'─' * 70}")
    print(f"  High false positives (hallucinated lines): {len(failure_modes['high_fp'])}")
    print(f"  High false negatives (missed lines):       {len(failure_modes['high_fn'])}")
    print(f"  Both FP and FN high:                       {len(failure_modes['balanced_poor'])}")

    # Distribution buckets
    all_dice = [r for _, r in weak_images]
    # weak_images contains all images sorted, use the full scores list
    print(f"\n{'─' * 70}")
    print(f"DICE SCORE DISTRIBUTION:")
    print(f"{'─' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate line mask predictions and identify weak images")

    # Input sources (choose one)
    parser.add_argument("--pred_dir", default=None, help="Directory with pre-computed predicted masks")
    parser.add_argument("--images_dir", default=None, help="Directory with original images (for on-the-fly inference)")
    parser.add_argument("--trained_model", default=None, help="Model weights (required for on-the-fly inference)")

    # GT masks
    parser.add_argument("--gt_dir", required=True, help="Directory with ground truth masks")

    # Output
    parser.add_argument("--output_dir", default="results/analysis", help="Output directory for reports")

    # Inference params (only used with --images_dir)
    parser.add_argument("--input_size", type=int, default=1280, help="Inference input size")
    parser.add_argument("--output_ch", type=int, default=2, help="Number of output channels")
    parser.add_argument("--threshold_h", type=float, default=0.3, help="Threshold for horizontal mask")
    parser.add_argument("--threshold_v", type=float, default=0.2, help="Threshold for vertical mask")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    # Analysis params
    parser.add_argument("--num_samples", type=int, default=0, help="Randomly sample N images to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--bottom_pct", type=float, default=0.1, help="Bottom percentage to flag as weak (0.1 = 10%%)")
    parser.add_argument("--save_vis", action="store_true", help="Save comparison visualizations for weak images")
    parser.add_argument("--max_vis", type=int, default=50, help="Max weak images to visualize")
    args = parser.parse_args()

    if args.pred_dir is None and args.images_dir is None:
        parser.error("Provide either --pred_dir (pre-computed) or --images_dir + --trained_model (on-the-fly)")
    if args.images_dir and not args.trained_model:
        parser.error("--trained_model is required when using --images_dir")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU")

    # Determine which images to evaluate (from GT mask names)
    gt_files = os.listdir(args.gt_dir)
    base_names = set()
    for f in gt_files:
        if f.endswith("_mask_h.png"):
            base = f[: -len("_mask_h.png")]
            # Check that both h and v exist
            if f"{base}_mask_v.png" in gt_files:
                base_names.add(base)
    base_names = sorted(base_names)

    if not base_names:
        print(f"No valid GT mask pairs found in {args.gt_dir}")
        return

    if args.num_samples > 0 and args.num_samples < len(base_names):
        import random
        random.seed(args.seed)
        base_names = sorted(random.sample(base_names, args.num_samples))
        print(f"Sampled {len(base_names)} images from {len(gt_files) // 2} available (seed={args.seed})")
    else:
        print(f"Found {len(base_names)} images with GT masks")

    # Save per-sample debug folder when sampling a subset
    save_samples = args.num_samples > 0 and args.images_dir
    samples_dir = os.path.join(args.output_dir, "samples") if save_samples else None

    # Load model if doing on-the-fly inference
    net = None
    pred_dir_effective = args.pred_dir
    if args.images_dir:
        net = load_model(args.trained_model, output_ch=args.output_ch, device=args.device)
        pred_dir_effective = os.path.join(args.output_dir, "pred_masks")
        os.makedirs(pred_dir_effective, exist_ok=True)

    # Evaluate each image
    all_results = {}
    gt_file_set = set(gt_files)

    for base in tqdm(base_names, desc="Evaluating"):
        gt_h = cv2.imread(os.path.join(args.gt_dir, f"{base}_mask_h.png"), cv2.IMREAD_GRAYSCALE)
        gt_v = cv2.imread(os.path.join(args.gt_dir, f"{base}_mask_v.png"), cv2.IMREAD_GRAYSCALE)
        if gt_h is None or gt_v is None:
            continue

        if net is not None:
            # On-the-fly inference
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(args.images_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            image = imgproc.loadImage(img_path)
            orig_h, orig_w = image.shape[:2]
            heatmap = infer_image(net, image, args.input_size, args.device)
            h_map = cv2.resize(heatmap[:, :, 0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            v_map = cv2.resize(heatmap[:, :, 1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            pred_h_bin = (h_map >= args.threshold_h).astype(np.uint8) * 255
            pred_v_bin = (v_map >= args.threshold_v).astype(np.uint8) * 255

            # Save combined prediction mask (Green=h, Red=v in BGR)
            pred_combined = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            pred_combined[:, :, 1] = pred_h_bin
            pred_combined[:, :, 2] = pred_v_bin
            cv2.imwrite(os.path.join(pred_dir_effective, f"{base}_mask.png"), pred_combined)

            # Save per-sample debug files (flat in samples/ folder)
            if save_samples:
                os.makedirs(samples_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(samples_dir, f"{base}_input.png"))
                gt_vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                gt_vis[:, :, 1] = gt_h  # Green = horizontal
                gt_vis[:, :, 2] = gt_v  # Red = vertical
                cv2.imwrite(os.path.join(samples_dir, f"{base}_gt.png"), gt_vis)
                cv2.imwrite(os.path.join(samples_dir, f"{base}_pred.png"), pred_combined)
                # Diff: Green=TP, Red=FN, Blue=FP
                gt_any = (gt_h > 127) | (gt_v > 127)
                pred_any = (pred_h_bin > 127) | (pred_v_bin > 127)
                diff = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                diff[gt_any & pred_any] = [0, 255, 0]      # TP = green
                diff[gt_any & ~pred_any] = [0, 0, 255]      # FN = red
                diff[~gt_any & pred_any] = [255, 0, 0]      # FP = blue
                cv2.imwrite(os.path.join(samples_dir, f"{base}_diff.png"), diff)
        else:
            # Load pre-computed predictions (combined or separate)
            combined_path = os.path.join(pred_dir_effective, f"{base}_mask.png")
            if os.path.exists(combined_path):
                combined_img = cv2.imread(combined_path, cv2.IMREAD_COLOR)
                if combined_img is None:
                    continue
                if combined_img.shape[:2] != gt_h.shape:
                    combined_img = cv2.resize(combined_img, (gt_h.shape[1], gt_h.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred_h_bin = combined_img[:, :, 1]  # Green channel = horizontal
                pred_v_bin = combined_img[:, :, 2]  # Red channel = vertical
            else:
                pred_h_path = os.path.join(pred_dir_effective, f"{base}_mask_h.png")
                pred_v_path = os.path.join(pred_dir_effective, f"{base}_mask_v.png")
                if not os.path.exists(pred_h_path) or not os.path.exists(pred_v_path):
                    continue
                pred_h_bin = cv2.imread(pred_h_path, cv2.IMREAD_GRAYSCALE)
                pred_v_bin = cv2.imread(pred_v_path, cv2.IMREAD_GRAYSCALE)
                if pred_h_bin is None or pred_v_bin is None:
                    continue
                if pred_h_bin.shape != gt_h.shape:
                    pred_h_bin = cv2.resize(pred_h_bin, (gt_h.shape[1], gt_h.shape[0]), interpolation=cv2.INTER_NEAREST)
                    pred_v_bin = cv2.resize(pred_v_bin, (gt_v.shape[1], gt_v.shape[0]), interpolation=cv2.INTER_NEAREST)

        metrics = evaluate_single_image(gt_h, gt_v, pred_h_bin > 127, pred_v_bin > 127)
        all_results[base] = metrics

    if not all_results:
        print("No images evaluated successfully")
        return

    # Aggregate
    agg = aggregate_metrics(all_results)

    # Identify weak images
    weak_images, all_scores = identify_weak_images(
        all_results, metric="dice", channel="all", bottom_pct=args.bottom_pct
    )

    # Failure mode analysis
    failure_modes = analyze_failure_modes(all_results)

    # Print dice distribution
    dice_scores = [s for _, s in all_scores]
    buckets = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
    print(f"\n{'─' * 70}")
    print("DICE SCORE DISTRIBUTION:")
    print(f"{'─' * 70}")
    for lo, hi in buckets:
        count = sum(1 for d in dice_scores if lo <= d < hi)
        bar = "#" * (count * 50 // max(1, len(dice_scores)))
        label = f"[{lo:.2f}-{hi:.2f})" if hi < 1.01 else f"[{lo:.2f}-1.00]"
        print(f"  {label}: {count:5d} {bar}")

    # Save reports
    csv_path, json_path = write_report(all_results, agg, weak_images, failure_modes, args.output_dir)

    # Print summary
    print_summary(agg, weak_images, failure_modes, len(all_results))

    # Save visualizations
    if args.save_vis and pred_dir_effective:
        print(f"\nSaving comparison visualizations for {min(len(weak_images), args.max_vis)} weak images...")
        save_comparison_images(
            weak_images, args.images_dir, args.gt_dir, pred_dir_effective,
            args.output_dir, max_images=args.max_vis,
        )

    print(f"\nReports saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    if save_samples:
        print(f"  Samples:  {samples_dir}")
    if args.save_vis:
        print(f"  Visualizations: {os.path.join(args.output_dir, 'weak_visualizations')}")


if __name__ == "__main__":
    main()
