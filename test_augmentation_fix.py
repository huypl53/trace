#!/usr/bin/env python3
"""Test the augmentation fix for small images."""

import numpy as np
from augmentations import LineAugmentation
from tools.types import Line


def test_small_image_augmentation():
    """Test that augmentation handles small images without crashing."""

    # Create a small image (303x400) - smaller than target_size // 2
    small_img = np.random.randint(0, 255, (303, 400, 3), dtype=np.uint8)

    # Create some sample lines
    lines = [
        Line(x1=10, y1=50, x2=390, y2=50, direction='horizontal', visible=True),
        Line(x1=100, y1=10, x2=100, y2=290, direction='vertical', visible=True),
        Line(x1=200, y1=10, x2=200, y2=290, direction='vertical', visible=False),
    ]

    # Create augmentation with size that could cause issues
    # target_size=512, so target_size//2 = 256
    # But image height is 303, which is > 256, so should work
    aug = LineAugmentation(size=512, enable_photometric=False, enable_crop=True)

    print("Testing with image size:", small_img.shape)
    print(f"Augmentation target size: 512 (min crop: 256)")

    # Try augmentation multiple times to test different random crops
    success_count = 0
    skip_count = 0

    for i in range(10):
        try:
            img_aug, lines_aug = aug(small_img.copy(), lines.copy())
            success_count += 1
            print(f"  Run {i+1}: Success - output shape {img_aug.shape}, {len(lines_aug)} lines")
        except ValueError as e:
            print(f"  Run {i+1}: ERROR - {e}")
            raise

    print(f"\n✓ All {success_count} runs succeeded!")

    # Test with even smaller image that should skip augmentation
    very_small_img = np.random.randint(0, 255, (200, 250, 3), dtype=np.uint8)
    lines_small = [
        Line(x1=10, y1=100, x2=240, y2=100, direction='horizontal', visible=True),
    ]

    print(f"\nTesting with very small image: {very_small_img.shape}")
    print(f"Image is smaller than target_size//2 (256), should skip crop augmentation")

    for i in range(5):
        img_aug, lines_aug = aug(very_small_img.copy(), lines_small.copy())
        # After augmentation, should be resized to 512x512
        assert img_aug.shape == (512, 512, 3), f"Expected (512, 512, 3), got {img_aug.shape}"
        print(f"  Run {i+1}: Handled correctly - final shape {img_aug.shape}")

    print(f"\n✓ Small image handling works correctly!")


def test_edge_cases():
    """Test various edge case image sizes."""

    test_cases = [
        (100, 100, "Very small square"),
        (600, 300, "Wide and short"),
        (300, 600, "Tall and narrow"),
        (256, 256, "Exactly half target size"),
        (255, 255, "Just below half target size"),
        (512, 512, "Exactly target size"),
        (1024, 1024, "Larger than target size"),
    ]

    aug = LineAugmentation(size=512, enable_photometric=False, enable_crop=True)

    print("\nTesting edge cases:")
    for h, w, description in test_cases:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        lines = [
            Line(x1=10, y1=h//2, x2=w-10, y2=h//2, direction='horizontal', visible=True),
        ]

        try:
            img_aug, lines_aug = aug(img, lines)
            assert img_aug.shape == (512, 512, 3)
            print(f"  ✓ {description} ({h}x{w}): Success")
        except Exception as e:
            print(f"  ✗ {description} ({h}x{w}): FAILED - {e}")
            raise


if __name__ == "__main__":
    print("=" * 70)
    print("Testing LineAugmentation with Small Images")
    print("=" * 70)

    test_small_image_augmentation()
    test_edge_cases()

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
