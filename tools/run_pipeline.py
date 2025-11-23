#!/usr/bin/env python3
"""
Run the complete table line extraction pipeline end-to-end.

This script automates all steps from raw JSON/image pairs to training-ready patches.
"""

import argparse
import json
import sys
import shutil
from pathlib import Path
import time


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num, total_steps, description):
    """Print step header."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 70)


def run_step1_crop(input_dir, output_dir, pad_min, pad_max, clean):
    """Run Step 1: Crop tables."""
    from crop_json_tables import crop_table

    output_path = Path(output_dir)

    # Clean output directory if requested
    if clean and output_path.exists():
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_path)

    # Run cropping
    start_time = time.time()
    crop_table(input_dir, output_dir, pad_min, pad_max)
    elapsed = time.time() - start_time

    # Count outputs
    json_count = len(list(output_path.glob("*.json")))
    image_count = len(list(output_path.glob("*.png")))

    print(f"✓ Completed in {elapsed:.2f}s")
    print(f"✓ Created {json_count} tables")

    return json_count


def run_step2_extract(input_dir, output_dir, clean):
    """Run Step 2: Extract lines."""
    from extract_labels import process_directory

    output_path = Path(output_dir)

    # Clean output directory if requested
    if clean and output_path.exists():
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_path)

    # Run extraction
    start_time = time.time()
    process_directory(input_dir, output_dir)
    elapsed = time.time() - start_time

    # Count outputs and lines
    json_files = list(output_path.glob("*.json"))
    total_lines = 0
    visible_lines = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        lines = data.get('lines', [])
        total_lines += len(lines)
        visible_lines += sum(1 for line in lines if line.get('visible', False))

    print(f"✓ Completed in {elapsed:.2f}s")
    print(f"✓ Extracted {total_lines} total lines ({visible_lines} visible)")

    return len(json_files), total_lines, visible_lines


def run_step3_patch(input_dir, output_dir, num_patches, min_size, max_size, clean):
    """Run Step 3: Create patches."""
    from patching import process_directory

    output_path = Path(output_dir)

    # Clean output directory if requested
    if clean and output_path.exists():
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_path)

    # Run patching
    start_time = time.time()
    process_directory(input_dir, output_dir, num_patches, min_size, max_size)
    elapsed = time.time() - start_time

    # Count outputs
    patch_count = len(list(output_path.glob("*.json")))

    print(f"✓ Completed in {elapsed:.2f}s")
    print(f"✓ Created {patch_count} patches")

    return patch_count


def verify_output(output_dir):
    """Verify the final output is ready for training."""
    output_path = Path(output_dir)

    if not output_path.exists():
        return False, "Output directory does not exist"

    json_files = list(output_path.glob("*.json"))
    if not json_files:
        return False, "No JSON files found"

    # Check first file
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    if 'image' not in data:
        return False, "JSON missing 'image' field"

    if 'lines' not in data:
        return False, "JSON missing 'lines' field"

    image_file = output_path / data['image']
    if not image_file.exists():
        return False, f"Image file not found: {data['image']}"

    return True, f"Ready for training with {len(json_files)} samples"


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete table line extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python run_pipeline.py samples

  # Custom output directories
  python run_pipeline.py samples --crop-dir MY_CROP --line-dir MY_LINES --patch-dir MY_PATCHES

  # More patches with larger size range
  python run_pipeline.py samples -n 20 --min-size 384 --max-size 768

  # Skip cleaning intermediate directories
  python run_pipeline.py samples --no-clean

  # Only run specific steps
  python run_pipeline.py samples --steps 1,2  # Only crop and extract
  python run_pipeline.py samples --steps 3    # Only create patches (requires previous steps)
        """
    )

    # Input/Output
    parser.add_argument("input_dir", help="Input directory containing JSON and image pairs")
    parser.add_argument("--output-dir", help="Base directory to store all pipeline outputs (CROP_DATA/LINE_DATA/PATCH_DATA inside)")
    parser.add_argument("--crop-dir", default="CROP_DATA", help="Output directory for cropped tables (default: CROP_DATA)")
    parser.add_argument("--line-dir", default="LINE_DATA", help="Output directory for line data (default: LINE_DATA)")
    parser.add_argument("--patch-dir", default="PATCH_DATA", help="Output directory for patches (default: PATCH_DATA)")

    # Step 1 options
    parser.add_argument("--pad-min", type=int, default=5, help="Minimum padding in pixels (default: 5)")
    parser.add_argument("--pad-max", type=int, default=20, help="Maximum padding in pixels (default: 20)")

    # Step 3 options
    parser.add_argument("-n", "--num-patches", type=int, default=10, help="Number of patches per image (default: 10)")
    parser.add_argument("--min-size", type=int, default=256, help="Minimum patch size (default: 256)")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum patch size (default: 512)")

    # Control options
    parser.add_argument("--steps", help="Comma-separated list of steps to run (1,2,3). Default: all steps")
    parser.add_argument("--no-clean", action="store_true", help="Don't clean output directories before running")
    parser.add_argument("--verify-only", action="store_true", help="Only verify the final output without running pipeline")

    args = parser.parse_args()

    # Normalize output directories if a base output directory was provided
    if args.output_dir:
        base_output = Path(args.output_dir)
        base_output.mkdir(parents=True, exist_ok=True)

        if not Path(args.crop_dir).is_absolute():
            args.crop_dir = str(base_output / args.crop_dir)
        if not Path(args.line_dir).is_absolute():
            args.line_dir = str(base_output / args.line_dir)
        if not Path(args.patch_dir).is_absolute():
            args.patch_dir = str(base_output / args.patch_dir)

    # Determine which steps to run
    if args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
    else:
        steps_to_run = {1, 2, 3}

    # Verify only mode
    if args.verify_only:
        print_banner("Verifying Pipeline Output")
        success, message = verify_output(args.patch_dir)
        if success:
            print(f"✓ {message}")
            return 0
        else:
            print(f"✗ Verification failed: {message}")
            return 1

    # Start pipeline
    print_banner("Table Line Extraction Pipeline")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directories:")
    print(f"  - Cropped tables: {args.crop_dir}")
    print(f"  - Line data: {args.line_dir}")
    print(f"  - Patches: {args.patch_dir}")

    # Verify input directory exists
    if not Path(args.input_dir).exists():
        print(f"\n✗ Error: Input directory '{args.input_dir}' does not exist")
        return 1

    # Check if input has data
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"\n✗ Error: No JSON files found in '{args.input_dir}'")
        return 1

    print(f"\nFound {len(json_files)} JSON files in input directory")

    clean = not args.no_clean
    total_steps = len(steps_to_run)
    current_step = 0

    stats = {}
    overall_start = time.time()

    try:
        # Step 1: Crop tables
        if 1 in steps_to_run:
            current_step += 1
            print_step(current_step, total_steps, "Crop Tables")
            table_count = run_step1_crop(
                args.input_dir,
                args.crop_dir,
                args.pad_min,
                args.pad_max,
                clean
            )
            stats['tables'] = table_count

        # Step 2: Extract lines
        if 2 in steps_to_run:
            current_step += 1
            print_step(current_step, total_steps, "Extract Lines")

            # Check if input exists
            if not Path(args.crop_dir).exists():
                print(f"✗ Error: {args.crop_dir} does not exist. Run step 1 first.")
                return 1

            file_count, total_lines, visible_lines = run_step2_extract(
                args.crop_dir,
                args.line_dir,
                clean
            )
            stats['total_lines'] = total_lines
            stats['visible_lines'] = visible_lines

        # Step 3: Create patches
        if 3 in steps_to_run:
            current_step += 1
            print_step(current_step, total_steps, "Create Patches")

            # Check if input exists
            if not Path(args.line_dir).exists():
                print(f"✗ Error: {args.line_dir} does not exist. Run step 2 first.")
                return 1

            patch_count = run_step3_patch(
                args.line_dir,
                args.patch_dir,
                args.num_patches,
                args.min_size,
                args.max_size,
                clean
            )
            stats['patches'] = patch_count

        overall_elapsed = time.time() - overall_start

        # Summary
        print_banner("Pipeline Complete!")
        print(f"Total time: {overall_elapsed:.2f}s\n")
        print("Statistics:")
        if 'tables' in stats:
            print(f"  - Tables extracted: {stats['tables']}")
        if 'total_lines' in stats:
            print(f"  - Lines extracted: {stats['total_lines']} ({stats['visible_lines']} visible)")
        if 'patches' in stats:
            print(f"  - Training patches: {stats['patches']}")

        # Verify output
        if 3 in steps_to_run:
            print("\nVerifying output...")
            success, message = verify_output(args.patch_dir)
            if success:
                print(f"✓ {message}")
                print(f"\nNext steps:")
                print(f"  1. Use LineDataset to load data from '{args.patch_dir}'")
                print(f"  2. Apply LineAugmentation for training")
                print(f"  3. Train your model!")
            else:
                print(f"✗ Verification failed: {message}")
                return 1

        return 0

    except Exception as e:
        print(f"\n✗ Pipeline failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
