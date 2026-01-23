#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Line Mask Evaluation Script

Run inference on validation set and compute mask metrics.
Outputs JSON with dice score for integration with training loop.

Usage:
    python eval_line_mask.py \
        --trained_model eval/ckpt_1000.pth \
        --data_dir data/line_mask_dataset \
        --input_size 1280

Output format (to stdout):
    {"dice": 0.85, "iou": 0.74, "precision": 0.87, "recall": 0.83}
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

from model import TraceModel


def load_model(model_path, output_ch=2, device="cuda"):
    """Load trained model."""
    net = TraceModel(output_ch=output_ch)
    state_dict = torch.load(model_path, map_location=device)

    # Handle DataParallel state dict
    if list(state_dict.keys())[0].startswith("module"):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    return net


def preprocess_image(img, input_size):
    """Preprocess image for inference."""
    h, w = img.shape[:2]
    # Resize maintaining aspect ratio
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Pad to square
    canvas = np.ones((input_size, input_size, 3), dtype=np.uint8) * 255
    canvas[:new_h, :new_w] = img_resized

    # Normalize
    img_norm = canvas.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5  # Normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)

    return img_tensor, (h, w), scale


@torch.no_grad()
def run_inference(net, img_tensor, device):
    """Run inference and return prediction."""
    img_tensor = img_tensor.to(device)
    out = net(img_tensor)
    if isinstance(out, tuple):
        out = out[0]
    return out.cpu().numpy()[0]  # Remove batch dim


def compute_metrics(gt_h, gt_v, pred_h, pred_v, threshold=0.5):
    """Compute pixel-level metrics."""
    # Binarize
    gt_h_bin = gt_h > 127
    gt_v_bin = gt_v > 127
    pred_h_bin = pred_h >= threshold
    pred_v_bin = pred_v >= threshold

    results = {}

    # Per-channel metrics
    for name, gt, pred in [("h", gt_h_bin, pred_h_bin), ("v", gt_v_bin, pred_v_bin)]:
        tp = np.sum(gt & pred)
        fp = np.sum(~gt & pred)
        fn = np.sum(gt & ~pred)

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        dice_denom = 2 * tp + fp + fn
        dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results[name] = {"iou": iou, "dice": dice, "precision": precision, "recall": recall,
                         "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    # Combined metrics
    gt_combined = gt_h_bin | gt_v_bin
    pred_combined = pred_h_bin | pred_v_bin

    tp = np.sum(gt_combined & pred_combined)
    fp = np.sum(~gt_combined & pred_combined)
    fn = np.sum(gt_combined & ~pred_combined)

    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results["all"] = {"iou": iou, "dice": dice, "precision": precision, "recall": recall,
                      "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    return results


def evaluate_dataset(net, data_dir, phase, input_size, device, scale_down=2, threshold=0.5):
    """Evaluate on a dataset split."""
    images_dir = os.path.join(data_dir, phase, "images")
    masks_dir = os.path.join(data_dir, phase, "masks")

    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        print(f"Warning: {phase} split not found", file=sys.stderr)
        return None

    # Accumulators
    acc = {
        "h": {"tp": 0, "fp": 0, "fn": 0},
        "v": {"tp": 0, "fp": 0, "fn": 0},
        "all": {"tp": 0, "fp": 0, "fn": 0},
    }

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(images_dir, img_name)
        mask_h_path = os.path.join(masks_dir, base + "_mask_h.png")
        mask_v_path = os.path.join(masks_dir, base + "_mask_v.png")

        if not os.path.exists(mask_h_path) or not os.path.exists(mask_v_path):
            continue

        # Load image and masks
        img = cv2.imread(img_path)
        gt_h = cv2.imread(mask_h_path, cv2.IMREAD_GRAYSCALE)
        gt_v = cv2.imread(mask_v_path, cv2.IMREAD_GRAYSCALE)

        if img is None or gt_h is None or gt_v is None:
            continue

        # Run inference
        img_tensor, orig_size, scale = preprocess_image(img, input_size)
        pred = run_inference(net, img_tensor, device)

        # Get prediction at output resolution
        out_size = input_size // scale_down
        pred_h = pred[0, :out_size, :out_size]  # Channel 0 = horizontal
        pred_v = pred[1, :out_size, :out_size]  # Channel 1 = vertical

        # Resize predictions to match GT
        pred_h = cv2.resize(pred_h, (gt_h.shape[1], gt_h.shape[0]))
        pred_v = cv2.resize(pred_v, (gt_v.shape[1], gt_v.shape[0]))

        # Compute metrics
        metrics = compute_metrics(gt_h, gt_v, pred_h, pred_v, threshold=threshold)

        for key in acc:
            acc[key]["tp"] += metrics[key]["tp"]
            acc[key]["fp"] += metrics[key]["fp"]
            acc[key]["fn"] += metrics[key]["fn"]

    # Compute final metrics
    results = {}
    for key in acc:
        tp = acc[key]["tp"]
        fp = acc[key]["fp"]
        fn = acc[key]["fn"]

        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0
        dice_denom = 2 * tp + fp + fn
        dice = (2 * tp) / dice_denom if dice_denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results[key] = {"iou": iou, "dice": dice, "precision": precision, "recall": recall}

    return results


def main():
    parser = argparse.ArgumentParser(description="Line mask evaluation")
    parser.add_argument("--trained_model", required=True, help="Path to trained model")
    parser.add_argument("--data_dir", required=True, help="Data directory with train/val/test splits")
    parser.add_argument("--phase", default="val", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--input_size", type=int, default=1280, help="Input size for inference")
    parser.add_argument("--output_ch", type=int, default=2, help="Number of output channels")
    parser.add_argument("--scale_down", type=int, default=2, help="Output scale down factor")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU", file=sys.stderr)

    # Load model
    net = load_model(args.trained_model, output_ch=args.output_ch, device=args.device)

    # Evaluate
    results = evaluate_dataset(
        net, args.data_dir, args.phase, args.input_size,
        args.device, args.scale_down, args.threshold
    )

    if results is None:
        print('{"error": "evaluation failed"}')
        sys.exit(1)

    # Output JSON with dice score (compatible with train.py parsing)
    output = {
        "dice": results["all"]["dice"],
        "iou": results["all"]["iou"],
        "precision": results["all"]["precision"],
        "recall": results["all"]["recall"],
        "dice_h": results["h"]["dice"],
        "dice_v": results["v"]["dice"],
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
