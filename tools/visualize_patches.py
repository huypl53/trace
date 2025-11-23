#!/usr/bin/env python3
"""Visualize patch line annotations."""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def render_patch(json_data, image_path, output_path, thickness=2):
    """Overlay line annotations onto the patch image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for line in json_data.get("lines", []):
        visible = line.get("visible", False)
        color = (0, 200, 0) if visible else (200, 50, 50)
        x1 = line["x1"]
        y1 = line["y1"]
        x2 = line["x2"]
        y2 = line["y2"]
        draw.line((x1, y1, x2, y2), fill=color, width=thickness)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render line annotations for every patch.")
    parser.add_argument("patch_dir", help="Directory containing PATCH_DATA (JSON + PNG pairs)")
    parser.add_argument("output_dir", help="Directory to save rendered images")
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness in pixels (default: 2)")
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(patch_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {patch_dir}")

    for json_file in json_files:
        with open(json_file, "r") as f:
            json_data = json.load(f)

        image_name = json_data.get("image")
        if not image_name:
            print(f"Skipping {json_file.name}: missing image reference")
            continue
        image_path = patch_dir / image_name
        if not image_path.exists():
            print(f"Skipping {json_file.name}: image {image_name} not found")
            continue

        output_path = output_dir / image_name
        render_patch(json_data, image_path, output_path, args.thickness)
        print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
