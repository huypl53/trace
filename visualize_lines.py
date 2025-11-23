#!/usr/bin/env python3
"""Visualize extracted lines on images for verification."""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def draw_lines_on_image(image_path, lines, show_invisible=True):
    """
    Draw lines on image for visualization.

    Args:
        image_path: Path to the image
        lines: List of line dictionaries
        show_invisible: Whether to draw invisible lines (in different color)

    Returns:
        Image with lines drawn
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Draw invisible lines first (if enabled) so visible lines are on top
    if show_invisible:
        for line in lines:
            if not line['visible']:
                p1 = (int(line['x1']), int(line['y1']))
                p2 = (int(line['x2']), int(line['y2']))
                # Draw invisible lines in blue with dashed effect
                cv2.line(img, p1, p2, color=(255, 128, 0), thickness=1, lineType=cv2.LINE_AA)

    # Draw visible lines
    for line in lines:
        if line['visible']:
            p1 = (int(line['x1']), int(line['y1']))
            p2 = (int(line['x2']), int(line['y2']))
            # Draw visible lines in red
            cv2.line(img, p1, p2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    return img


def create_comparison_view(image, lines, title="Line Visualization"):
    """Create a side-by-side comparison with statistics."""
    h, w = image.shape[:2]

    # Count lines
    total_lines = len(lines)
    visible_lines = sum(1 for l in lines if l['visible'])
    invisible_lines = total_lines - visible_lines
    horizontal_visible = sum(1 for l in lines if l['visible'] and l['direction'] == 'horizontal')
    vertical_visible = sum(1 for l in lines if l['visible'] and l['direction'] == 'vertical')

    # Create stats text
    stats = [
        f"{title}",
        f"Total lines: {total_lines}",
        f"Visible: {visible_lines} ({visible_lines/total_lines*100:.1f}%)",
        f"  - Horizontal: {horizontal_visible}",
        f"  - Vertical: {vertical_visible}",
        f"Invisible: {invisible_lines} ({invisible_lines/total_lines*100:.1f}%)",
        "",
        "Legend:",
        "Red = Visible",
        "Blue = Invisible",
    ]

    # Add text overlay
    overlay = image.copy()
    y_offset = 30
    for i, text in enumerate(stats):
        cv2.putText(overlay, text, (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


def visualize_dataset(data_dir, output_dir, show_invisible=True):
    """
    Visualize all line data in a directory.

    Args:
        data_dir: Directory containing JSON and image pairs
        output_dir: Output directory for visualizations
        show_invisible: Whether to show invisible lines
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(data_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} files to visualize")

    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")

        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_name = data.get('image')
        if not image_name:
            print(f"  Warning: No image field in {json_file.name}, skipping...")
            continue

        image_path = data_path / image_name
        if not image_path.exists():
            print(f"  Warning: Image {image_name} not found, skipping...")
            continue

        lines = data['lines']

        # Draw lines
        img_with_lines = draw_lines_on_image(image_path, lines, show_invisible)

        # Create comparison view with stats
        img_final = create_comparison_view(img_with_lines, lines, json_file.stem)

        # Save
        output_file = output_path / f"{json_file.stem}_visualization.png"
        cv2.imwrite(str(output_file), img_final)

        visible_count = sum(1 for l in lines if l['visible'])
        print(f"  Saved visualization to {output_file.name}")
        print(f"  Lines: {len(lines)} total, {visible_count} visible ({visible_count/len(lines)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize extracted lines on images")
    parser.add_argument("data_dir", help="Directory containing JSON and image pairs (e.g., LINE_DATA)")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory (default: visualizations)")
    parser.add_argument("--no-invisible", action="store_true", help="Don't show invisible lines")
    parser.add_argument("--pattern", help="Only process files matching pattern (e.g., 'page_23*')")

    args = parser.parse_args()

    # If pattern specified, create temporary filtered view
    if args.pattern:
        data_path = Path(args.data_dir)
        json_files = list(data_path.glob(f"{args.pattern}.json"))
        if not json_files:
            print(f"No files matching pattern '{args.pattern}' found in {args.data_dir}")
            return 1
        print(f"Processing {len(json_files)} files matching '{args.pattern}'")

    visualize_dataset(args.data_dir, args.output_dir, show_invisible=not args.no_invisible)

    print(f"\n✓ Visualizations saved to {args.output_dir}/")
    print(f"  Open them to verify the line extraction results!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
