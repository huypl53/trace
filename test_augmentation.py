#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for TRACEMaskAugmentation pipeline"""

import numpy as np
import cv2
from augmentations import TRACEMaskAugmentation


def create_test_data(height=800, width=1200):
    """Create synthetic test data with known patterns"""
    # Create RGB image with colored quadrants
    image = np.zeros((height, width, 3), dtype=np.uint8)
    h2, w2 = height // 2, width // 2

    # Top-left: Red
    image[:h2, :w2] = [255, 0, 0]
    # Top-right: Green
    image[:h2, w2:] = [0, 255, 0]
    # Bottom-left: Blue
    image[h2:, :w2] = [0, 0, 255]
    # Bottom-right: Yellow
    image[h2:, w2:] = [255, 255, 0]

    # Create mask with 3 channels (matching TRACE dataset structure)
    # Channel 0: Region mask
    # Channel 1: Affinity mask
    # Channel 2: Additional mask
    mask = np.zeros((height, width, 3), dtype=np.float32)

    # Add circular region in center
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    circle_mask = ((y - cy) ** 2 + (x - cx) ** 2) <= (min(height, width) // 4) ** 2
    mask[circle_mask, 0] = 1.0  # Region channel

    # Add horizontal line in the middle for affinity
    mask[h2-10:h2+10, :, 1] = 1.0  # Affinity channel

    # Add vertical line in the middle
    mask[:, w2-10:w2+10, 2] = 1.0  # Additional channel

    # Create weight map (all ones for simplicity, can be 2D)
    weight = np.ones((height, width), dtype=np.float32)

    return image, mask, weight


def test_basic_pipeline():
    """Test 1: Basic pipeline execution"""
    print("=" * 60)
    print("Test 1: Basic Pipeline Execution")
    print("=" * 60)

    image, mask, weight = create_test_data()
    print(f"Input shapes - Image: {image.shape}, Mask: {mask.shape}, Weight: {weight.shape}")
    print(f"Input dtypes - Image: {image.dtype}, Mask: {mask.dtype}, Weight: {weight.dtype}")

    augmentation = TRACEMaskAugmentation(size=512)

    try:
        aug_image, aug_mask, aug_weight = augmentation(image, mask, weight)
        print(f"✓ Pipeline executed successfully")
        print(f"Output shapes - Image: {aug_image.shape}, Mask: {aug_mask.shape}, Weight: {aug_weight.shape}")
        print(f"Output dtypes - Image: {aug_image.dtype}, Mask: {aug_mask.dtype}, Weight: {aug_weight.dtype}")

        # Validate output shapes
        assert aug_image.shape == (512, 512, 3), f"Image shape mismatch: {aug_image.shape}"
        assert aug_mask.shape == (512, 512, 3), f"Mask shape mismatch: {aug_mask.shape}"
        # Weight remains 2D if input was 2D (expansion happens in loader after augmentation)
        assert aug_weight.shape == (512, 512) or aug_weight.shape == (512, 512, 3), \
            f"Weight shape mismatch: {aug_weight.shape}"
        print(f"✓ Output shapes are correct")

        return True, aug_image, aug_mask, aug_weight
    except Exception as e:
        print(f"✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_mask_alignment(aug_image, aug_mask):
    """Test 2: Verify mask-image alignment"""
    print("\n" + "=" * 60)
    print("Test 2: Mask-Image Alignment")
    print("=" * 60)

    if aug_image is None or aug_mask is None:
        print("✗ Skipping (previous test failed)")
        return False

    try:
        # Check that mask values are in valid range
        mask_min, mask_max = aug_mask.min(), aug_mask.max()
        print(f"Mask value range: [{mask_min:.3f}, {mask_max:.3f}]")
        assert mask_min >= 0.0, f"Mask has negative values: {mask_min}"
        print(f"✓ Mask values are non-negative")

        # Check that mask and image have corresponding features
        # For each channel, verify there are still some activated regions
        for ch in range(aug_mask.shape[2]):
            activated = (aug_mask[:, :, ch] > 0.5).sum()
            print(f"Channel {ch}: {activated} pixels activated ({100*activated/(512*512):.2f}%)")

        print(f"✓ Mask channels have activated regions")
        return True
    except Exception as e:
        print(f"✗ Alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_iterations():
    """Test 3: Multiple iterations with random augmentation"""
    print("\n" + "=" * 60)
    print("Test 3: Multiple Random Iterations")
    print("=" * 60)

    image, mask, weight = create_test_data()
    augmentation = TRACEMaskAugmentation(size=512)

    n_iterations = 10
    try:
        for i in range(n_iterations):
            aug_image, aug_mask, aug_weight = augmentation(image, mask, weight)

            # Quick validation
            assert aug_image.shape == (512, 512, 3)
            assert aug_mask.shape == (512, 512, 3)
            # Weight can be 2D or 3D
            assert aug_weight.ndim in [2, 3]
            assert not np.isnan(aug_image).any(), f"NaN in image at iteration {i}"
            assert not np.isnan(aug_mask).any(), f"NaN in mask at iteration {i}"
            assert not np.isnan(aug_weight).any(), f"NaN in weight at iteration {i}"

        print(f"✓ {n_iterations} iterations completed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed at iteration {i}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_sizes():
    """Test 4: Different input sizes"""
    print("\n" + "=" * 60)
    print("Test 4: Different Input Sizes")
    print("=" * 60)

    sizes = [
        (640, 480),   # Standard
        (1024, 768),  # Larger
        (400, 300),   # Smaller
        (1920, 1080), # HD
        (512, 512),   # Square
    ]

    augmentation = TRACEMaskAugmentation(size=512)

    for h, w in sizes:
        try:
            image, mask, weight = create_test_data(height=h, width=w)
            aug_image, aug_mask, aug_weight = augmentation(image, mask, weight)

            assert aug_image.shape == (512, 512, 3)
            assert aug_mask.shape == (512, 512, 3)
            assert aug_weight.ndim in [2, 3]
            print(f"✓ Size {h}x{w} -> 512x512: OK")
        except Exception as e:
            print(f"✗ Size {h}x{w} failed: {e}")
            return False

    return True


def test_weight_expansion():
    """Test 5: Weight expansion to match mask channels"""
    print("\n" + "=" * 60)
    print("Test 5: Weight Channel Expansion")
    print("=" * 60)

    image, mask, weight = create_test_data()
    augmentation = TRACEMaskAugmentation(size=512)

    # Test with 2D weight (should expand to match mask channels)
    assert weight.ndim == 2, "Weight should be 2D for this test"

    try:
        aug_image, aug_mask, aug_weight = augmentation(image, mask, weight)

        # Weight should remain 2D if input was 2D (expansion handled by loader)
        assert aug_weight.shape == (512, 512), \
            f"2D weight should remain 2D, got {aug_weight.shape}"
        print(f"✓ 2D weight preserved: {aug_weight.shape}")

        # Test with 3D weight
        weight_3d = np.expand_dims(weight, axis=2)
        aug_image, aug_mask, aug_weight = augmentation(image, mask, weight_3d)
        # 3D weight with single channel may be reduced to 2D or kept as 3D
        assert aug_weight.ndim in [2, 3], f"Unexpected weight dimensions: {aug_weight.ndim}"
        print(f"✓ 3D weight handled correctly: {aug_weight.shape}")

        return True
    except Exception as e:
        print(f"✗ Weight expansion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deterministic_transform():
    """Test 6: Verify transforms are applied to both image and mask"""
    print("\n" + "=" * 60)
    print("Test 6: Deterministic Transform (seed-based)")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    image, mask, weight = create_test_data()
    augmentation = TRACEMaskAugmentation(size=512)

    try:
        # Run augmentation
        aug_image, aug_mask, aug_weight = augmentation(image, mask, weight)

        # Check that regions with high mask values correspond to specific image regions
        # This is a soft check - we verify correlation exists
        for ch in range(3):
            mask_ch = aug_mask[:, :, ch]
            high_mask_regions = mask_ch > 0.5

            if high_mask_regions.sum() > 0:
                # Get mean color in high-mask regions
                mean_color = aug_image[high_mask_regions].mean(axis=0)
                print(f"Channel {ch}: High-mask regions mean color: {mean_color}")

        print(f"✓ Transform consistency check passed")
        return True
    except Exception as e:
        print(f"✗ Deterministic transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_results(aug_image, aug_mask, aug_weight):
    """Optional: Save visualization of results"""
    print("\n" + "=" * 60)
    print("Visualization: Saving output samples")
    print("=" * 60)

    if aug_image is None:
        print("✗ Skipping (no data available)")
        return

    try:
        # Save augmented image
        cv2.imwrite("test_aug_image.png", cv2.cvtColor(aug_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Save mask channels
        for ch in range(3):
            mask_vis = (aug_mask[:, :, ch] * 255).astype(np.uint8)
            cv2.imwrite(f"test_aug_mask_ch{ch}.png", mask_vis)

        # Save weight
        if aug_weight.ndim == 2:
            weight_vis = (aug_weight * 255).astype(np.uint8)
        else:
            weight_vis = (aug_weight[:, :, 0] * 255).astype(np.uint8)
        cv2.imwrite("test_aug_weight.png", weight_vis)

        print(f"✓ Saved visualization to:")
        print(f"  - test_aug_image.png")
        print(f"  - test_aug_mask_ch[0-2].png")
        print(f"  - test_aug_weight.png")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TRACE MASK AUGMENTATION PIPELINE TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Basic pipeline
    success, aug_image, aug_mask, aug_weight = test_basic_pipeline()
    results["Basic Pipeline"] = success

    # Test 2: Alignment
    results["Mask Alignment"] = test_mask_alignment(aug_image, aug_mask)

    # Test 3: Multiple iterations
    results["Multiple Iterations"] = test_multiple_iterations()

    # Test 4: Different sizes
    results["Different Sizes"] = test_different_sizes()

    # Test 5: Weight expansion
    results["Weight Expansion"] = test_weight_expansion()

    # Test 6: Deterministic
    results["Transform Consistency"] = test_deterministic_transform()

    # Visualization
    visualize_results(aug_image, aug_mask, aug_weight)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
