# -*- coding: utf-8 -*-
import argparse
import json
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch

import file_utils
from tqdm import tqdm
import imgproc
from model import TraceModel


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_model(args):
    net = TraceModel(output_ch=args.output_ch)
    if args.cuda and torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        state = torch.load(args.trained_model)
        state = copyStateDict(state)
        net.module.load_state_dict(state)
    else:
        state = torch.load(args.trained_model, map_location="cpu")
        state = copyStateDict(state)
        net.load_state_dict(state)
    net.eval()
    return net


def preprocess_image(image, input_size):
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = imgproc.normalizeMeanVariance(resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return resized, x


def straighten_horizontal(line):
    x1, y1, x2, y2 = line
    y = int(round((y1 + y2) / 2.0))
    if x1 > x2:
        x1, x2 = x2, x1
    return x1, y, x2, y


def straighten_vertical(line):
    x1, y1, x2, y2 = line
    x = int(round((x1 + x2) / 2.0))
    if y1 > y2:
        y1, y2 = y2, y1
    return x, y1, x, y2


def extract_lines(bin_img, hough_threshold, min_line_length, max_line_gap):
    lines = cv2.HoughLinesP(
        bin_img,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [tuple(line[0]) for line in lines]


def clamp_point(pt, width, height):
    x = max(0, min(width - 1, int(pt[0])))
    y = max(0, min(height - 1, int(pt[1])))
    return x, y


def scale_lines(lines, scale_x, scale_y, width, height):
    scaled = []
    for x1, y1, x2, y2 in lines:
        p1 = clamp_point((round(x1 * scale_x), round(y1 * scale_y)), width, height)
        p2 = clamp_point((round(x2 * scale_x), round(y2 * scale_y)), width, height)
        scaled.append((p1, p2))
    return scaled


def infer_image(net, image, args):
    resized, x = preprocess_image(image, args.input_size)
    if args.cuda and torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y = y[0]
    heatmap = y[0].cpu().numpy()

    resized_h, resized_w = resized.shape[:2]
    h_map_resized = cv2.resize(heatmap[:, :, 0], (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    v_map_resized = cv2.resize(heatmap[:, :, 1], (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    orig_h, orig_w = image.shape[:2]
    h_map = cv2.resize(h_map_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    v_map = cv2.resize(v_map_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    h_bin = (h_map_resized >= args.threshold_h).astype(np.uint8) * 255
    v_bin = (v_map_resized >= args.threshold_v).astype(np.uint8) * 255

    raw_h = extract_lines(h_bin, args.hough_threshold, args.min_line_length, args.max_line_gap)
    raw_v = extract_lines(v_bin, args.hough_threshold, args.min_line_length, args.max_line_gap)

    if args.straighten:
        raw_h = [straighten_horizontal(line) for line in raw_h]
        raw_v = [straighten_vertical(line) for line in raw_v]

    return resized, h_map, v_map, raw_h, raw_v


def main():
    parser = argparse.ArgumentParser(description="Line segmentation inference")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--trained_model", required=True, help="Path to model weights")
    parser.add_argument("--output_dir", default="results/predictions", help="Output directory for predictions")
    parser.add_argument("--input_size", type=int, default=1280, help="Resize size used for inference")
    parser.add_argument("--output_ch", type=int, default=2, help="Number of output channels")
    parser.add_argument("--threshold_h", type=float, default=0.3, help="Threshold for horizontal heatmap")
    parser.add_argument("--threshold_v", type=float, default=0.3, help="Threshold for vertical heatmap")
    parser.add_argument("--hough_threshold", type=int, default=60, help="HoughLinesP threshold")
    parser.add_argument("--min_line_length", type=int, default=30, help="HoughLinesP minimum line length")
    parser.add_argument("--max_line_gap", type=int, default=10, help="HoughLinesP maximum line gap")
    parser.add_argument("--straighten", default=True, type=lambda v: v.lower() in ("1", "true", "yes", "y"))
    parser.add_argument("--save_heatmap", action="store_true", help="Save heatmap debug images")
    parser.add_argument("--use_compare", action="store_true", help="save final mask along with the input image to compare")
    parser.add_argument("--cuda", default=True, type=lambda v: v.lower() in ("1", "true", "yes", "y"))
    args = parser.parse_args()

    if os.path.isdir(args.input):
        image_list = file_utils.get_image_list(args.input)
    else:
        image_list = [args.input]

    if not image_list:
        raise ValueError("No input images found.")

    os.makedirs(args.output_dir, exist_ok=True)
    net = load_model(args)

    for image_path in tqdm(image_list):
        image = imgproc.loadImage(image_path)
        orig_h, orig_w = image.shape[:2]
        resized, h_map, v_map, raw_h, raw_v = infer_image(net, image, args)

        scale_x = orig_w / float(resized.shape[1])
        scale_y = orig_h / float(resized.shape[0])
        h_lines = scale_lines(raw_h, scale_x, scale_y, orig_w, orig_h)
        v_lines = scale_lines(raw_v, scale_x, scale_y, orig_w, orig_h)

        lines = []
        for start, end in h_lines:
            lines.append({"type": "horizontal", "points": [list(start), list(end)]})
        for start, end in v_lines:
            lines.append({"type": "vertical", "points": [list(start), list(end)]})

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.output_dir, f"{base}.json")
        if not args.save_heatmap:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"filename": os.path.basename(image_path), "lines": lines}, f, ensure_ascii=False)

        if args.save_heatmap:
            heat_h = np.clip(h_map * 255, 0, 255).astype(np.uint8)
            heat_v = np.clip(v_map * 255, 0, 255).astype(np.uint8)
            combined = np.maximum(heat_h, heat_v)
            if not args.use_compare:
                cv2.imwrite(os.path.join(args.output_dir, f"{base}_heatmap_h.png"), heat_h)
                cv2.imwrite(os.path.join(args.output_dir, f"{base}_heatmap_v.png"), heat_v)
                cv2.imwrite(os.path.join(args.output_dir, f"{base}_heatmap_combined.png"), combined)
            else:
                resized_combined = cv2.resize(combined, (image.shape[:2][::-1]))
                if len(resized_combined.shape) == 2:
                    resized_combined = cv2.cvtColor(resized_combined, cv2.COLOR_GRAY2BGR)
                compared_image = np.hstack((image, resized_combined ))

                cv2.imwrite(os.path.join(args.output_dir, f"{base}_combined.png"), compared_image)
    print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
