#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for line mask model.

Runs inference on a folder of images and saves predicted mask images.

Usage:
    python infer_masks.py \
        --input data/test/images \
        --trained_model eval/model.pth \
        --output_dir results/pred_masks \
        --input_size 1280 \
        --threshold_h 0.3 \
        --threshold_v 0.2
"""

import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
from tqdm import tqdm

import imgproc
from model import TraceModel


def load_model(model_path, output_ch=2, device="cuda"):
    net = TraceModel(output_ch=output_ch)
    state_dict = torch.load(model_path, map_location=device)

    # Handle DataParallel state dict
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[".".join(k.split(".")[1:])] = v
        state_dict = new_state_dict

    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    return net


def preprocess_image(image, input_size):
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = imgproc.normalizeMeanVariance(resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return resized, x


@torch.no_grad()
def infer_image(net, image, input_size, device):
    resized, x = preprocess_image(image, input_size)
    x = x.to(device)
    out = net(x)
    if isinstance(out, tuple):
        out = out[0]
    heatmap = out[0].cpu().numpy()  # (H, W, output_ch)
    return heatmap


def main():
    parser = argparse.ArgumentParser(description="Line mask inference - save predicted masks")
    parser.add_argument("--input", required=True, help="Directory of input images")
    parser.add_argument("--trained_model", required=True, help="Path to model weights")
    parser.add_argument("--output_dir", default="results/pred_masks", help="Output directory")
    parser.add_argument("--input_size", type=int, default=1280, help="Inference input size")
    parser.add_argument("--output_ch", type=int, default=2, help="Number of output channels")
    parser.add_argument("--threshold_h", type=float, default=0.3, help="Threshold for horizontal mask")
    parser.add_argument("--threshold_v", type=float, default=0.2, help="Threshold for vertical mask")
    parser.add_argument("--save_raw", action="store_true", help="Save raw heatmaps (float, 0-255 grayscale)")
    parser.add_argument("--save_separate", action="store_true", help="Also save separate _mask_h/_mask_v files")
    parser.add_argument("--num_samples", type=int, default=0, help="Randomly sample N images to infer (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU")

    # Collect image files
    exts = {".png", ".jpg", ".jpeg", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not image_files:
        print(f"No images found in {args.input}")
        return

    if args.num_samples > 0 and args.num_samples < len(image_files):
        import random
        random.seed(args.seed)
        image_files = sorted(random.sample(image_files, args.num_samples))
        print(f"Sampled {len(image_files)} images (seed={args.seed})")

    os.makedirs(args.output_dir, exist_ok=True)
    net = load_model(args.trained_model, output_ch=args.output_ch, device=args.device)

    print(f"Running inference on {len(image_files)} images...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(args.input, img_name)
        image = imgproc.loadImage(img_path)
        orig_h, orig_w = image.shape[:2]

        heatmap = infer_image(net, image, args.input_size, args.device)
        h_map = heatmap[:, :, 0]
        v_map = heatmap[:, :, 1]

        # Resize to original image size
        h_map = cv2.resize(h_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        v_map = cv2.resize(v_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        base = os.path.splitext(img_name)[0]

        if args.save_raw:
            # Save raw heatmaps as grayscale images (0-255)
            cv2.imwrite(
                os.path.join(args.output_dir, f"{base}_pred_h.png"),
                np.clip(h_map * 255, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(args.output_dir, f"{base}_pred_v.png"),
                np.clip(v_map * 255, 0, 255).astype(np.uint8),
            )

        # Save binary masks
        h_bin = (h_map >= args.threshold_h).astype(np.uint8) * 255
        v_bin = (v_map >= args.threshold_v).astype(np.uint8) * 255

        # Combined color mask: Green=horizontal, Red=vertical, Yellow=overlap (BGR)
        combined = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        combined[:, :, 1] = h_bin  # Green channel = horizontal
        combined[:, :, 2] = v_bin  # Red channel = vertical
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_mask.png"), combined)

        if args.save_separate:
            cv2.imwrite(os.path.join(args.output_dir, f"{base}_mask_h.png"), h_bin)
            cv2.imwrite(os.path.join(args.output_dir, f"{base}_mask_v.png"), v_bin)

    print(f"Saved {len(image_files)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
