#!/usr/bin/env python3
"""Crop tables from JSON/image pairs with random padding."""

import argparse
import json
import os
import random
from pathlib import Path
from PIL import Image
import copy


def get_table_bounds(table):
    """Calculate the bounding box of a table based on its position and size."""
    x = table['x']
    y = table['y']
    width = table['width']
    height = table['height']
    return x, y, x + width, y + height


def crop_table(input_dir, output_dir, pad_min, pad_max):
    """
    Crop tables from JSON/image pairs with random padding.

    Args:
        input_dir: Directory containing JSON and image pairs
        output_dir: Output directory for cropped tables
        pad_min: Minimum padding in pixels
        pad_max: Maximum padding in pixels
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_path.glob("*.json"))

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        # Read JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Find corresponding image
        image_name = json_file.stem
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_file = None
        for ext in image_extensions:
            candidate = input_path / f"{image_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            print(f"Warning: No image found for {json_file.name}, skipping...")
            continue

        # Load image
        image = Image.open(image_file)
        img_width, img_height = image.size

        # Find all tables
        tables = [item for item in data['items'] if item['type'] == 'table']

        if not tables:
            print(f"Warning: No tables found in {json_file.name}, skipping...")
            continue

        # Process each table
        for table_idx, table in enumerate(tables):
            # Get table bounds
            x1, y1, x2, y2 = get_table_bounds(table)

            # Generate random padding
            pad_top = random.randint(pad_min, pad_max)
            pad_bottom = random.randint(pad_min, pad_max)
            pad_left = random.randint(pad_min, pad_max)
            pad_right = random.randint(pad_min, pad_max)

            # Calculate crop bounds with padding
            crop_x1 = max(0, x1 - pad_left)
            crop_y1 = max(0, y1 - pad_top)
            crop_x2 = min(img_width, x2 + pad_right)
            crop_y2 = min(img_height, y2 + pad_bottom)

            # Crop image
            cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Create new JSON with adjusted coordinates
            new_data = copy.deepcopy(data)
            new_data['items'] = [table]

            # Adjust table coordinates
            table_copy = new_data['items'][0]
            table_copy['x'] = x1 - crop_x1
            table_copy['y'] = y1 - crop_y1

            # Generate output filenames
            if len(tables) > 1:
                output_name = f"{image_name}_table_{table_idx}"
            else:
                output_name = image_name

            # Save cropped image
            output_image_path = output_path / f"{output_name}{image_file.suffix}"
            cropped_image.save(output_image_path)

            # Save adjusted JSON
            output_json_path = output_path / f"{output_name}.json"
            with open(output_json_path, 'w') as f:
                json.dump(new_data, f, indent=2)

            print(f"  Saved table {table_idx} to {output_name}")


def main():
    parser = argparse.ArgumentParser(description="Crop tables from JSON/image pairs")
    parser.add_argument("input_dir", help="Input directory containing JSON and image pairs")
    parser.add_argument("--output-dir", default="CROP_DATA", help="Output directory (default: CROP_DATA)")
    parser.add_argument("--pad-min", type=int, default=5, help="Minimum padding in pixels (default: 5)")
    parser.add_argument("--pad-max", type=int, default=20, help="Maximum padding in pixels (default: 20)")

    args = parser.parse_args()

    crop_table(args.input_dir, args.output_dir, args.pad_min, args.pad_max)
    print("Done!")


if __name__ == "__main__":
    main()
