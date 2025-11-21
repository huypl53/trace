# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import numpy as np

import imgproc
from loader import GTTransform
from parsers.json_parser import ParserTRACEJSON
from parsers.xml_parser import ParserTRACE


def visualize_mask_channels(gt_image, output_path, prefix=""):
    """Save individual mask channels as binary images."""
    os.makedirs(output_path, exist_ok=True)
    
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
    cv2.imwrite(os.path.join(output_path, f"{prefix}corner_heatmap.png"), to_binary(corner_heatmap))
    cv2.imwrite(os.path.join(output_path, f"{prefix}horizontal_visible.png"), to_binary(hor_visible))
    cv2.imwrite(os.path.join(output_path, f"{prefix}vertical_visible.png"), to_binary(ver_visible))
    cv2.imwrite(os.path.join(output_path, f"{prefix}horizontal_invisible.png"), to_binary(hor_invisible))
    cv2.imwrite(os.path.join(output_path, f"{prefix}vertical_invisible.png"), to_binary(ver_invisible))
    
    # Create combined visualization (all channels overlaid)
    combined = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
    combined[:, :, 0] = to_binary(hor_visible)  # Red for horizontal visible
    combined[:, :, 1] = to_binary(ver_visible)  # Green for vertical visible
    combined[:, :, 2] = to_binary(corner_heatmap)  # Blue for corners
    cv2.imwrite(os.path.join(output_path, f"{prefix}combined.png"), combined)
    
    return combined


def compare_parsers(root_path, dataset, phase, scale_down=2, num_samples=5, output_dir="parser_comparison"):
    """Compare ground truth masks from XML and JSON parsers."""
    
    # Initialize parsers
    xml_parser = ParserTRACE(root_path, dataset, phase)
    json_parser = ParserTRACEJSON(root_path, dataset, phase)
    
    print(f"XML parser found {xml_parser.lenFiles()} files")
    print(f"JSON parser found {json_parser.lenFiles()} files")
    
    # Find common files
    xml_files = {os.path.basename(entry["file_name"]) for entry in xml_parser.gt}
    json_files = {os.path.basename(entry["file_name"]) for entry in json_parser.gt}
    common_files = xml_files & json_files
    
    print(f"Found {len(common_files)} files with both XML and JSON")
    
    if len(common_files) == 0:
        print("No common files found. Cannot compare.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare up to num_samples files
    samples_to_compare = min(num_samples, len(common_files))
    sample_files = list(common_files)[:samples_to_compare]
    
    for basename in sample_files:
        print(f"\nComparing: {basename}")
        
        # Find entries in both parsers
        xml_entry = None
        json_entry = None
        
        for entry in xml_parser.gt:
            if os.path.basename(entry["file_name"]) == basename:
                xml_entry = entry
                break
        
        for entry in json_parser.gt:
            if os.path.basename(entry["file_name"]) == basename:
                json_entry = entry
                break
        
        if xml_entry is None or json_entry is None:
            print(f"  Warning: Could not find entry for {basename}")
            continue
        
        # Load image to get dimensions
        img_file = xml_entry["file_name"]
        img = imgproc.loadImage(img_file)
        height, width, _ = img.shape
        
        print(f"  Image size: {width}x{height}")
        print(f"  XML quads: {len(xml_entry['quads'])}")
        print(f"  JSON quads: {len(json_entry['quads'])}")
        
        # Prepare data for GTTransform (same normalization as loader.py)
        def prepare_gt(quads, lines, img_width, img_height):
            # Normalize quads by original image size
            quads = np.array(quads, dtype=np.float32).reshape(-1, 8)
            num_pt = int(quads.shape[1] / 2)
            quads[:, : 2 * num_pt] /= [img_width, img_height] * num_pt
            
            # Prepare data for GTTransform
            gt_gathered = []
            for attr, quad in zip(lines, quads):
                gt_gathered.append({"quad": quad, "line": attr})
            
            return gt_gathered
        
        xml_gt = prepare_gt(xml_entry["quads"], xml_entry["lines"], width, height)
        json_gt = prepare_gt(json_entry["quads"], json_entry["lines"], width, height)
        
        # Generate masks using GTTransform
        mask_width = width / scale_down
        mask_height = height / scale_down
        
        xml_mask, xml_weight = GTTransform(xml_gt, mask_width, mask_height)
        json_mask, json_weight = GTTransform(json_gt, mask_width, mask_height)
        
        print(f"  XML mask shape: {xml_mask.shape}")
        print(f"  JSON mask shape: {json_mask.shape}")
        
        # Create comparison directory for this file
        file_output_dir = os.path.join(output_dir, os.path.splitext(basename)[0])
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Save original image
        cv2.imwrite(os.path.join(file_output_dir, "original_image.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Visualize XML mask
        xml_combined = visualize_mask_channels(xml_mask, file_output_dir, prefix="xml_")
        
        # Visualize JSON mask
        json_combined = visualize_mask_channels(json_mask, file_output_dir, prefix="json_")
        
        # Create side-by-side comparison
        # Scale up masks to original image size for comparison
        xml_combined_scaled = cv2.resize(xml_combined, (width, height), interpolation=cv2.INTER_NEAREST)
        json_combined_scaled = cv2.resize(json_combined, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Create side-by-side image
        side_by_side = np.hstack([xml_combined_scaled, json_combined_scaled])
        
        # Add labels
        label_height = 50
        side_by_side_with_labels = np.ones((height + label_height, width * 2, 3), dtype=np.uint8) * 255
        side_by_side_with_labels[:height, :width] = xml_combined_scaled
        side_by_side_with_labels[:height, width:] = json_combined_scaled
        
        # Add text labels
        cv2.putText(side_by_side_with_labels, "XML Parser", (10, height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(side_by_side_with_labels, "JSON Parser", (width + 10, height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imwrite(os.path.join(file_output_dir, "comparison_side_by_side.png"), side_by_side_with_labels)
        
        print(f"  Saved comparison to: {file_output_dir}")
    
    print(f"\nComparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare XML and JSON parser outputs")
    parser.add_argument("--data_path", type=str, required=True, help="Root path to datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--phase", type=str, default="train", help="Phase: train or test")
    parser.add_argument("--scale_down", type=int, default=2, help="Scale down factor for masks")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to compare")
    parser.add_argument("--output_dir", type=str, default="parser_comparison", help="Output directory for comparisons")
    args = parser.parse_args()
    
    compare_parsers(args.data_path, args.dataset, args.phase, args.scale_down, args.num_samples, args.output_dir)

