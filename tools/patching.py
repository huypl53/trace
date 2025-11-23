#!/usr/bin/env python3
"""Create random patches from line data with line coordinate updates."""

import argparse
import json
import random
from pathlib import Path
from PIL import Image
import sys

# Add parent directory to path to import types
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.types import Line


def clip_line_to_patch(line, patch_x, patch_y, patch_w, patch_h):
    """
    Clip a line to patch boundaries and shift to patch coordinates.

    Args:
        line: Line object
        patch_x, patch_y: Top-left corner of the patch
        patch_w, patch_h: Width and height of the patch

    Returns:
        Updated Line object or None if line is outside patch
    """
    # Calculate patch boundaries
    x_min, y_min = patch_x, patch_y
    x_max, y_max = patch_x + patch_w, patch_y + patch_h

    if line.direction == 'horizontal':
        # Horizontal line: y1 == y2
        y = line.y1

        # Check if line is outside patch vertically
        if y < y_min or y > y_max:
            return None

        # Clip x coordinates
        x1 = max(x_min, min(x_max, line.x1))
        x2 = max(x_min, min(x_max, line.x2))

        # Check if line has length after clipping
        if abs(x2 - x1) < 1e-6:
            return None

        # Shift to patch coordinates
        return Line(
            x1=x1 - patch_x,
            y1=y - patch_y,
            x2=x2 - patch_x,
            y2=y - patch_y,
            direction='horizontal',
            visible=line.visible
        )

    else:  # vertical
        # Vertical line: x1 == x2
        x = line.x1

        # Check if line is outside patch horizontally
        if x < x_min or x > x_max:
            return None

        # Clip y coordinates
        y1 = max(y_min, min(y_max, line.y1))
        y2 = max(y_min, min(y_max, line.y2))

        # Check if line has length after clipping
        if abs(y2 - y1) < 1e-6:
            return None

        # Shift to patch coordinates
        return Line(
            x1=x - patch_x,
            y1=y1 - patch_y,
            x2=x - patch_x,
            y2=y2 - patch_y,
            direction='vertical',
            visible=line.visible
        )


def create_patches(image_path, lines_data, num_patches, min_size, max_size, output_dir, base_name):
    """
    Create random patches from an image and its lines.

    Args:
        image_path: Path to the image file
        lines_data: List of Line objects
        num_patches: Number of patches to create
        min_size: Minimum patch size
        max_size: Maximum patch size
        output_dir: Output directory
        base_name: Base name for output files
    """
    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size

    patches_created = 0

    # Clamp patch sizes to image dimensions
    min_patch_w = min(min_size, img_width)
    min_patch_h = min(min_size, img_height)
    max_patch_w = min(max_size, img_width)
    max_patch_h = min(max_size, img_height)

    if min_patch_w <= 0 or min_patch_h <= 0:
        print(f"Warning: Image {image_path.name} too small for patch creation, skipping...")
        return 0

    # If requested min > image size we'll end up with identical min/max; warn so users know
    if min_size > img_width or min_size > img_height:
        print(f"Warning: Image {image_path.name} smaller than min patch size; using image size instead.")

    for i in range(num_patches * 10):  # Try more times to ensure we get enough valid patches
        if patches_created >= num_patches:
            break

        # Random patch size
        patch_w = random.randint(min_patch_w, max_patch_w)
        patch_h = random.randint(min_patch_h, max_patch_h)

        # Random start location
        max_x = img_width - patch_w
        max_y = img_height - patch_h

        if max_x < 0 or max_y < 0:
            print(f"Warning: Image too small for patch size {patch_w}x{patch_h}, skipping...")
            continue

        patch_x = random.randint(0, max_x)
        patch_y = random.randint(0, max_y)

        # Crop image
        cropped = image.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))

        # Update lines
        updated_lines = []
        for line in lines_data:
            clipped_line = clip_line_to_patch(line, patch_x, patch_y, patch_w, patch_h)
            if clipped_line:
                updated_lines.append(clipped_line)

        # Skip patches with no lines
        if not updated_lines:
            continue

        # Save patch
        patch_name = f"{base_name}_patch_{patches_created}"
        output_image_path = output_dir / f"{patch_name}.png"
        cropped.save(output_image_path)

        # Save lines
        output_json = {
            'image': f"{patch_name}.png",
            'lines': [line.to_dict() for line in updated_lines]
        }
        output_json_path = output_dir / f"{patch_name}.json"
        with open(output_json_path, 'w') as f:
            json.dump(output_json, f, indent=2)

        patches_created += 1

    return patches_created


def process_directory(input_dir, output_dir, num_patches, min_size, max_size):
    """
    Process all files in the input directory and create patches.

    Args:
        input_dir: Input directory (LINE_DATA)
        output_dir: Output directory (PATCH_DATA)
        num_patches: Number of patches per image
        min_size: Minimum patch size
        max_size: Maximum patch size
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_path.glob("*.json"))

    total_patches = 0

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        # Read JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Get image path
        image_name = data.get('image')
        if not image_name:
            print(f"Warning: No image field in {json_file.name}, skipping...")
            continue

        image_path = input_path / image_name
        if not image_path.exists():
            print(f"Warning: Image {image_name} not found, skipping...")
            continue

        # Parse lines
        lines_data = [Line(**line_dict) for line_dict in data['lines']]

        # Create patches
        base_name = json_file.stem
        patches_created = create_patches(
            image_path, lines_data, num_patches, min_size, max_size, output_path, base_name
        )

        total_patches += patches_created
        print(f"  Created {patches_created} patches")

    print(f"Total patches created: {total_patches}")


def main():
    parser = argparse.ArgumentParser(description="Create random patches from line data")
    parser.add_argument("input_dir", help="Input directory (LINE_DATA)")
    parser.add_argument("--output-dir", default="PATCH_DATA", help="Output directory (default: PATCH_DATA)")
    parser.add_argument("-n", "--num-patches", type=int, default=10, help="Number of patches per image (default: 10)")
    parser.add_argument("--min-size", type=int, default=256, help="Minimum patch size (default: 256)")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum patch size (default: 512)")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.num_patches, args.min_size, args.max_size)
    print("Done!")


if __name__ == "__main__":
    main()
