#!/usr/bin/env python3
"""Diagnose GT quality - check if heatmaps are reasonable."""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path


def analyze_gt_sample(dataset, idx=0):
    """Analyze a single GT sample."""
    print(f"\n{'='*70}")
    print(f"Analyzing sample {idx}")
    print(f"{'='*70}")

    # Load sample
    img, gt, weight = dataset[idx]

    # Convert to numpy
    img_np = img.permute(1, 2, 0).numpy()
    gt_np = gt.numpy()
    weight_np = weight.numpy()

    print(f"\nShapes:")
    print(f"  Image: {img.shape}")
    print(f"  GT: {gt.shape}")
    print(f"  Weight: {weight.shape}")

    # Analyze each channel
    print(f"\nGT Channels Analysis:")
    channel_names = [
        "0: Corners (keypoints)",
        "1: Visible Horizontal Lines",
        "2: Visible Vertical Lines",
        "3: Invisible Horizontal Lines",
        "4: Invisible Vertical Lines"
    ]

    num_channels = gt_np.shape[2] if len(gt_np.shape) == 3 else gt_np.shape[0]

    for i in range(min(num_channels, 5)):
        if len(gt_np.shape) == 3:
            channel = gt_np[..., i]
        else:
            channel = gt_np[i]

        nonzero = np.count_nonzero(channel)
        total = channel.size
        max_val = channel.max()
        mean_val = channel.mean()

        print(f"\n  {channel_names[i] if i < len(channel_names) else f'Channel {i}'}:")
        print(f"    Non-zero pixels: {nonzero:,} / {total:,} ({nonzero/total*100:.2f}%)")
        print(f"    Max value: {max_val:.4f}")
        print(f"    Mean value: {mean_val:.4f}")

        if nonzero == 0:
            print(f"    ⚠️  WARNING: Channel is completely empty!")

    # Check if GT is too sparse
    total_nonzero = np.count_nonzero(gt_np)
    total_pixels = gt_np.size
    density = total_nonzero / total_pixels * 100

    print(f"\n{'='*70}")
    print(f"Overall GT Density: {density:.4f}%")

    if density < 0.1:
        print("⚠️  WARNING: GT is very sparse (< 0.1% non-zero pixels)")
        print("   This might make training difficult!")
    elif density < 1.0:
        print("✓ GT density is reasonable for line detection")
    else:
        print("✓ GT density is good")

    return gt_np, img_np


def compare_with_original_loader(patch_data_path):
    """Compare with original TRACE dataset if available."""
    print(f"\n{'='*70}")
    print(f"Comparing with Original XML-based Data")
    print(f"{'='*70}")

    try:
        from loader import TRACE_Dataset
        from parsers.xml_parser import ParserTRACE

        # Try to load original dataset
        print("\nAttempting to load original TRACE dataset...")
        print("(This will fail if you don't have the original data)")

        # This is just to show the comparison - may not work without original data

    except Exception as e:
        print(f"\n⚠️  Could not load original dataset: {e}")
        print("   This is expected if you only have the new line-based data.")


def visualize_gt_channels(gt_np, img_np, output_path="gt_diagnosis.png"):
    """Create visualization of GT channels."""
    print(f"\n{'='*70}")
    print(f"Creating GT Visualization")
    print(f"{'='*70}")

    num_channels = gt_np.shape[2] if len(gt_np.shape) == 3 else gt_np.shape[0]

    # Create figure with all channels
    rows = 2
    cols = 3

    fig_img = np.ones((gt_np.shape[0] * rows + 50, gt_np.shape[1] * cols, 3), dtype=np.uint8) * 255

    # Denormalize image for display
    img_display = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # Channel 0: Original image
    y_offset = 0
    x_offset = 0
    fig_img[y_offset:y_offset+img_display.shape[0], x_offset:x_offset+img_display.shape[1]] = img_display
    cv2.putText(fig_img, "Input Image", (x_offset+10, y_offset+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Channels 1-5: GT heatmaps
    channel_names = [
        "Corners",
        "Visible H-Lines",
        "Visible V-Lines",
        "Invisible H-Lines",
        "Invisible V-Lines"
    ]

    positions = [
        (0, 1), (0, 2),  # Row 0, cols 1-2
        (1, 0), (1, 1), (1, 2)  # Row 1, cols 0-2
    ]

    for i in range(min(num_channels, 5)):
        if i >= len(positions):
            break

        row, col = positions[i]
        y_offset = row * gt_np.shape[0] + 50
        x_offset = col * gt_np.shape[1]

        if len(gt_np.shape) == 3:
            channel = gt_np[..., i]
        else:
            channel = gt_np[i]

        # Normalize channel for visualization
        if channel.max() > 0:
            channel_vis = (channel / channel.max() * 255).astype(np.uint8)
        else:
            channel_vis = np.zeros_like(channel, dtype=np.uint8)

        # Apply colormap
        channel_colored = cv2.applyColorMap(channel_vis, cv2.COLORMAP_HOT)

        fig_img[y_offset:y_offset+channel.shape[0], x_offset:x_offset+channel.shape[1]] = channel_colored

        # Add label
        label = channel_names[i] if i < len(channel_names) else f"Ch {i}"
        cv2.putText(fig_img, label, (x_offset+10, y_offset+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add stats
        nonzero = np.count_nonzero(channel)
        stats_text = f"{nonzero} px ({nonzero/channel.size*100:.2f}%)"
        cv2.putText(fig_img, stats_text, (x_offset+10, y_offset+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save
    cv2.imwrite(output_path, fig_img)
    print(f"✓ Saved visualization to: {output_path}")

    return fig_img


def check_multiple_samples(dataset, num_samples=10):
    """Check statistics across multiple samples."""
    print(f"\n{'='*70}")
    print(f"Analyzing {num_samples} samples")
    print(f"{'='*70}")

    densities = []
    visible_h_counts = []
    visible_v_counts = []
    invisible_h_counts = []
    invisible_v_counts = []

    for i in range(min(num_samples, len(dataset))):
        img, gt, weight = dataset[i]
        gt_np = gt.numpy()

        # Calculate density
        total_nonzero = np.count_nonzero(gt_np)
        total_pixels = gt_np.size
        density = total_nonzero / total_pixels * 100
        densities.append(density)

        # Count pixels in each channel
        if gt_np.shape[2] >= 5:
            visible_h_counts.append(np.count_nonzero(gt_np[..., 1]))
            visible_v_counts.append(np.count_nonzero(gt_np[..., 2]))
            invisible_h_counts.append(np.count_nonzero(gt_np[..., 3]))
            invisible_v_counts.append(np.count_nonzero(gt_np[..., 4]))

    print(f"\nGT Density Statistics:")
    print(f"  Mean: {np.mean(densities):.4f}%")
    print(f"  Min: {np.min(densities):.4f}%")
    print(f"  Max: {np.max(densities):.4f}%")

    if visible_h_counts:
        print(f"\nAverage Non-zero Pixels per Channel:")
        print(f"  Visible Horizontal: {np.mean(visible_h_counts):.0f} pixels")
        print(f"  Visible Vertical: {np.mean(visible_v_counts):.0f} pixels")
        print(f"  Invisible Horizontal: {np.mean(invisible_h_counts):.0f} pixels")
        print(f"  Invisible Vertical: {np.mean(invisible_v_counts):.0f} pixels")

    # Recommendations
    print(f"\n{'='*70}")
    print("Recommendations:")
    print(f"{'='*70}")

    avg_density = np.mean(densities)

    if avg_density < 0.1:
        print("⚠️  GT is very sparse - this will make training hard!")
        print("   Suggestions:")
        print("   1. Check if line visibility detection is correct")
        print("   2. Use thicker lines in GT (increase thickness in LineGTTransform)")
        print("   3. Verify that borderWidth=None is being treated as visible")
    elif avg_density < 0.5:
        print("⚠️  GT is somewhat sparse")
        print("   Consider increasing line thickness if loss stays high")
    else:
        print("✓ GT density looks reasonable")

    avg_visible = np.mean(visible_h_counts) + np.mean(visible_v_counts) if visible_h_counts else 0
    avg_invisible = np.mean(invisible_h_counts) + np.mean(invisible_v_counts) if invisible_h_counts else 0

    if avg_visible > 0 and avg_invisible > avg_visible * 3:
        print("⚠️  Much more invisible lines than visible!")
        print("   This might confuse the model. Check visibility detection.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose GT quality for line detection")
    parser.add_argument("data_dir", help="Path to PATCH_DATA directory")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to analyze (default: 0)")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to analyze (default: 10)")
    parser.add_argument("--output", default="gt_diagnosis.png", help="Output visualization path")

    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"GT Quality Diagnosis Tool")
    print(f"{'='*70}")

    # Load dataset
    from loader import LineDataset

    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = LineDataset(
        data_dir=args.data_dir,
        scale_down=2,
        visible_only=False,
        transform=None  # No augmentation for diagnosis
    )

    print(f"✓ Loaded {len(dataset)} samples")

    # Analyze single sample in detail
    gt_np, img_np = analyze_gt_sample(dataset, args.sample_idx)

    # Visualize
    visualize_gt_channels(gt_np, img_np, args.output)

    # Check multiple samples
    check_multiple_samples(dataset, args.num_samples)

    print(f"\n{'='*70}")
    print("Diagnosis Complete!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"1. Check the visualization: {args.output}")
    print(f"2. Verify the LINE_DATA visualizations to ensure lines are extracted correctly")
    print(f"3. If GT is too sparse, consider:")
    print(f"   - Increasing line thickness in LineGTTransform")
    print(f"   - Checking border visibility detection logic")


if __name__ == "__main__":
    main()
