#!/usr/bin/env python3
"""Test the full table line extraction pipeline."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualization")


def test_step1_crop():
    """Test Step 1: Crop tables from JSON files."""
    print("\n=== Testing Step 1: Crop Tables ===")
    from tools.crop_json_tables import crop_table

    input_dir = "samples"
    output_dir = "CROP_DATA"

    # Clean output directory if exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    # Run cropping
    crop_table(input_dir, output_dir, pad_min=5, pad_max=20)

    # Verify output
    output_path = Path(output_dir)
    json_files = list(output_path.glob("*.json"))
    image_files = list(output_path.glob("*.png"))

    print(f"✓ Created {len(json_files)} JSON files")
    print(f"✓ Created {len(image_files)} image files")

    assert len(json_files) > 0, "No JSON files created!"
    assert len(image_files) > 0, "No image files created!"

    return output_dir


def test_step2_types():
    """Test Step 2: Line type validation."""
    print("\n=== Testing Step 2: Line Type ===")
    from tools.types import Line

    # Test valid horizontal line
    h_line = Line(x1=10, y1=20, x2=100, y2=20, direction='horizontal', visible=True)
    print(f"✓ Created horizontal line: {h_line}")

    # Test valid vertical line
    v_line = Line(x1=50, y1=10, x2=50, y2=100, direction='vertical', visible=False)
    print(f"✓ Created vertical line: {v_line}")

    # Test invalid horizontal line (should raise error)
    try:
        invalid_h = Line(x1=10, y1=20, x2=100, y2=30, direction='horizontal', visible=True)
        assert False, "Should have raised validation error!"
    except ValueError as e:
        print(f"✓ Correctly rejected invalid horizontal line: {e}")

    # Test invalid vertical line (should raise error)
    try:
        invalid_v = Line(x1=10, y1=20, x2=20, y2=100, direction='vertical', visible=True)
        assert False, "Should have raised validation error!"
    except ValueError as e:
        print(f"✓ Correctly rejected invalid vertical line: {e}")


def test_step3_extract():
    """Test Step 3: Extract lines from tables."""
    print("\n=== Testing Step 3: Extract Lines ===")
    from tools.extract_labels import process_directory

    input_dir = "CROP_DATA"
    output_dir = "LINE_DATA"

    # Clean output directory if exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    # Run extraction
    process_directory(input_dir, output_dir)

    # Verify output
    output_path = Path(output_dir)
    json_files = list(output_path.glob("*.json"))
    image_files = list(output_path.glob("*.png"))

    print(f"✓ Created {len(json_files)} JSON files with line data")
    print(f"✓ Copied {len(image_files)} image files")

    assert len(json_files) > 0, "No line JSON files created!"

    # Check line data
    import json
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    print(f"✓ Sample has {len(data['lines'])} lines")

    return output_dir


def test_step4_patching():
    """Test Step 4: Create patches."""
    print("\n=== Testing Step 4: Create Patches ===")
    from tools.patching import process_directory

    input_dir = "LINE_DATA"
    output_dir = "PATCH_DATA"

    # Clean output directory if exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    # Run patching
    process_directory(input_dir, output_dir, num_patches=5, min_size=256, max_size=512)

    # Verify output
    output_path = Path(output_dir)
    json_files = list(output_path.glob("*.json"))
    image_files = list(output_path.glob("*.png"))

    print(f"✓ Created {len(json_files)} patch JSON files")
    print(f"✓ Created {len(image_files)} patch images")

    assert len(json_files) > 0, "No patch JSON files created!"

    return output_dir


def test_step5_loader():
    """Test Step 5: Dataset loader."""
    print("\n=== Testing Step 5: Dataset Loader ===")
    from loader import LineDataset

    data_dir = "PATCH_DATA"

    # Test without augmentation
    dataset = LineDataset(data_dir, scale_down=2, visible_only=False, transform=None)
    print(f"✓ Created dataset with {len(dataset)} samples")

    # Test loading a sample
    img, gt, weight = dataset[0]
    print(f"✓ Loaded sample 0:")
    print(f"  - Image shape: {img.shape}")
    print(f"  - GT shape: {gt.shape}")
    print(f"  - Weight shape: {weight.shape}")

    assert img.shape[0] == 3, "Image should have 3 channels (RGB)"
    assert gt.shape[2] == 5, "GT should have 5 channels (corner, h, v, ih, iv)"

    # Test visible_only mode
    dataset_visible = LineDataset(data_dir, scale_down=2, visible_only=True, transform=None)
    img_v, gt_v, weight_v = dataset_visible[0]
    print(f"✓ Loaded sample 0 (visible only):")
    print(f"  - GT shape: {gt_v.shape}")
    assert gt_v.shape[2] == 3, "GT (visible only) should have 3 channels (corner, h, v)"

    return dataset


def test_step6_augmentation():
    """Test Step 6: Augmentation."""
    print("\n=== Testing Step 6: Augmentation ===")
    from loader import LineDataset
    from augmentations import LineAugmentation

    data_dir = "PATCH_DATA"

    # Create augmentation
    aug = LineAugmentation(size=512, enable_photometric=True, enable_crop=True)
    print("✓ Created LineAugmentation")

    # Test with dataset
    dataset = LineDataset(data_dir, scale_down=2, visible_only=False, transform=aug)
    img, gt, weight = dataset[0]

    print(f"✓ Applied augmentation to sample 0:")
    print(f"  - Image shape: {img.shape}")
    print(f"  - GT shape: {gt.shape}")

    # Image should be resized to 512x512
    assert img.shape[1] == 512 and img.shape[2] == 512, "Image should be 512x512 after augmentation"

    return dataset


def visualize_sample(dataset, idx=0, output_path="test_visualization.png"):
    """Visualize a sample from the dataset."""
    if not HAS_MATPLOTLIB:
        print(f"\n=== Skipping Visualization (matplotlib not available) ===")
        return

    print(f"\n=== Visualizing Sample {idx} ===")

    img, gt, weight = dataset[idx]

    # Convert tensors to numpy
    img_np = img.permute(1, 2, 0).numpy()
    gt_np = gt.numpy()

    # Denormalize image (reverse normalizeMeanVariance)
    # The normalization typically does: (img - mean) / std
    # We'll just clip to [0, 1] for visualization
    img_np = np.clip(img_np, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')

    # Channel 0: Corners (should be zeros for line data)
    axes[0, 1].imshow(gt_np[:, :, 0], cmap='hot')
    axes[0, 1].set_title("GT: Corners")
    axes[0, 1].axis('off')

    # Channel 1: Visible horizontal lines
    axes[0, 2].imshow(gt_np[:, :, 1], cmap='hot')
    axes[0, 2].set_title("GT: Horizontal (visible)")
    axes[0, 2].axis('off')

    # Channel 2: Visible vertical lines
    axes[1, 0].imshow(gt_np[:, :, 2], cmap='hot')
    axes[1, 0].set_title("GT: Vertical (visible)")
    axes[1, 0].axis('off')

    # Channel 3: Invisible horizontal lines (if exists)
    if gt_np.shape[2] > 3:
        axes[1, 1].imshow(gt_np[:, :, 3], cmap='hot')
        axes[1, 1].set_title("GT: Horizontal (invisible)")
        axes[1, 1].axis('off')

    # Channel 4: Invisible vertical lines (if exists)
    if gt_np.shape[2] > 4:
        axes[1, 2].imshow(gt_np[:, :, 4], cmap='hot')
        axes[1, 2].set_title("GT: Vertical (invisible)")
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Table Line Extraction Pipeline")
    print("=" * 60)

    try:
        # Step 1: Crop tables
        test_step1_crop()

        # Step 2: Test Line type
        test_step2_types()

        # Step 3: Extract lines
        test_step3_extract()

        # Step 4: Create patches
        test_step4_patching()

        # Step 5: Test loader
        dataset = test_step5_loader()

        # Step 6: Test augmentation
        dataset_aug = test_step6_augmentation()

        # Visualize results
        visualize_sample(dataset, idx=0, output_path="test_no_aug.png")
        visualize_sample(dataset_aug, idx=0, output_path="test_with_aug.png")

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - CROP_DATA/     : Cropped tables")
        print("  - LINE_DATA/     : Extracted lines")
        print("  - PATCH_DATA/    : Random patches")
        print("  - test_no_aug.png: Visualization without augmentation")
        print("  - test_with_aug.png: Visualization with augmentation")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
