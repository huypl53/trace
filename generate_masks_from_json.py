# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import numpy as np
import tqdm

import imgproc
from loader import GTTransform
from parsers.json_parser import ParserTRACEJSON


def visualize_mask(gt_image, output_path, orig_width=None, orig_height=None, scale_down=2):
    """Save mask visualizations as binary images.
    
    Args:
        gt_image: Mask image (height, width, channels)
        output_path: Directory to save visualizations
        orig_width: Original image width (for scaling up visualization)
        orig_height: Original image height (for scaling up visualization)
        scale_down: Scale down factor used for mask generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Scale up mask to original image size for visualization if provided
    if orig_width is not None and orig_height is not None:
        mask_height, mask_width = gt_image.shape[:2]
        if mask_width != orig_width or mask_height != orig_height:
            # Resize each channel
            gt_image_scaled = np.zeros((orig_height, orig_width, gt_image.shape[2]), dtype=gt_image.dtype)
            for ch in range(gt_image.shape[2]):
                gt_image_scaled[:, :, ch] = cv2.resize(
                    gt_image[:, :, ch],
                    (orig_width, orig_height),
                    interpolation=cv2.INTER_LINEAR
                )
            gt_image = gt_image_scaled
    
    # Extract individual channels
    corner_heatmap = gt_image[:, :, 0]
    hor_visible = gt_image[:, :, 1]
    ver_visible = gt_image[:, :, 2]
    hor_invisible = gt_image[:, :, 3]
    ver_invisible = gt_image[:, :, 4]
    
    # Convert to binary (0-255) images
    def to_binary(channel):
        binary = (channel > 0).astype(np.uint8) * 255
        return binary
    
    # Save individual channels
    cv2.imwrite(os.path.join(output_path, "corner_heatmap.png"), to_binary(corner_heatmap))
    cv2.imwrite(os.path.join(output_path, "horizontal_visible.png"), to_binary(hor_visible))
    cv2.imwrite(os.path.join(output_path, "vertical_visible.png"), to_binary(ver_visible))
    cv2.imwrite(os.path.join(output_path, "horizontal_invisible.png"), to_binary(hor_invisible))
    cv2.imwrite(os.path.join(output_path, "vertical_invisible.png"), to_binary(ver_invisible))
    
    # Create combined visualization (RGB overlay)
    # Red: visible horizontal, Green: visible vertical, Blue: invisible lines
    combined = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
    combined[:, :, 0] = to_binary(hor_visible)  # Red channel for visible horizontal
    combined[:, :, 1] = to_binary(ver_visible)  # Green channel for visible vertical
    combined[:, :, 2] = to_binary(hor_invisible + ver_invisible)  # Blue channel for invisible
    
    # Add corner heatmap as white overlay
    corner_binary = to_binary(corner_heatmap)
    combined = np.maximum(combined, np.stack([corner_binary, corner_binary, corner_binary], axis=-1))
    
    cv2.imwrite(os.path.join(output_path, "combined.png"), combined)


def generate_masks(root_path, dataset, phase, scale_down=2, output_dir=None):
    """Generate mask images from JSON files offline."""
    parser = ParserTRACEJSON(root_path, dataset, phase)
    
    # Determine base folder for saving masks (same logic as parser)
    base_folder = os.path.join(root_path, dataset)
    phase_folder = os.path.join(base_folder, phase)
    if os.path.exists(phase_folder):
        base_folder = phase_folder
    
    # Set up output directory for visualizations
    if output_dir is None:
        output_dir = os.path.join(base_folder, "mask_visualizations")
    
    print(f"Generating masks for {len(parser.gt)} images...")
    print(f"Visualizations will be saved to: {output_dir}")
    
    for gt_entry in tqdm.tqdm(parser.gt):
        img_file = gt_entry["file_name"]
        basename, ext = os.path.splitext(os.path.basename(img_file))
        mask_file = os.path.join(base_folder, f"{basename}.npy")
        
        try:
            # Always load image to get dimensions (needed for visualization scaling)
            img = imgproc.loadImage(img_file)
            height, width, _ = img.shape
            
            # Load mask if it exists, otherwise generate it
            if os.path.exists(mask_file):
                mask_data = np.load(mask_file, allow_pickle=True).item()
                gt_image = mask_data["mask"]
            else:
                # Extract quads and lines
                quads = gt_entry["quads"]
                lines = gt_entry["lines"]
                
                if not quads:
                    print(f"Warning: No cells found for {img_file}")
                    continue
                
                # Convert to numpy array and normalize coordinates using original image dimensions
                # (Same as loader.py: normalize by original image size, then GTTransform will denormalize using mask dimensions)
                quads = np.array(quads, dtype=np.float32).reshape(-1, 8)
                num_pt = int(quads.shape[1] / 2)
                quads[:, : 2 * num_pt] /= [width, height] * num_pt
                
                # Prepare data for GTTransform
                gt_gathered = []
                for attr, quad in zip(lines, quads):
                    gt_gathered.append({"quad": quad, "line": attr})
                
                # Generate mask at scaled-down size (same as loader.py: width/scale_down, height/scale_down)
                mask_width = width / scale_down
                mask_height = height / scale_down
                gt_image, gt_weight = GTTransform(gt_gathered, mask_width, mask_height)
                
                # Save mask and weight
                np.save(mask_file, {"mask": gt_image, "weight": gt_weight})
            
            # Always save visualizations (even if mask already existed)
            # Scale up to original image size for easier comparison
            vis_output_path = os.path.join(output_dir, basename)
            visualize_mask(gt_image, vis_output_path, width, height, scale_down)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print("Mask generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mask images from JSON files")
    parser.add_argument("--data_path", type=str, required=True, help="Root path to datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--phase", type=str, default="train", help="Phase: train or test")
    parser.add_argument("--scale_down", type=int, default=2, help="Scale down factor for masks")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for mask visualizations (default: <dataset>/mask_visualizations)")
    args = parser.parse_args()
    
    generate_masks(args.data_path, args.dataset, args.phase, args.scale_down, args.output_dir)

