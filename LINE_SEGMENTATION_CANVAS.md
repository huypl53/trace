# Line Segmentation Training With Canvas JSON

This guide explains how to prepare data and train a line-segmentation model from the canvas JSON format.

## 1) Input Data

Place canvas JSON files and their corresponding images **in the same folder**:

```
data/raw_canvas/
  canvas_001.json     # Contains "image_path": "page1.png"
  page1.png           # Original color image
  canvas_002.json
  page2.png
  ...
```

The script crops regions from the **original color images** using the table's explicit `width`/`height` properties.

## 1b) Optional: Augment Canvas JSON (Layout/Style/Text)

Use the augmentation script to create new canvas JSONs by:
- Changing row/column sizes
- Randomly merging cells
- Mutating cell text (Japanese + digits + a few ASCII chars by default)
- Jittering colors
- Removing borders (outer/internal) for negative examples
- Adding background contrast without borders
- Creating dense text and narrow columns

### Basic Usage

```bash
uv run python -m scripts.augment_canvas_data \
    --input_dir data/raw_canvas \
    --output_dir data/raw_canvas_aug \
    --num_aug 3 \
    --fill_empty_text
```

### Advanced Usage (with negative example augmentations)

To help the model avoid false positives (detecting borders where none exist):

```bash
uv run python -m scripts.augment_canvas_data \
    --input_dir data/raw_canvas \
    --output_dir data/raw_canvas_aug \
    --num_aug 3 \
    --fill_empty_text \
    --remove_outer_borders_prob 0.3 \
    --remove_internal_borders_prob 0.2 \
    --bg_contrast_prob 0.2 \
    --dense_text_prob 0.15 \
    --narrow_col_prob 0.1
```

### Augmentation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num_aug` | 1 | Number of augmented versions per input |
| `--fill_empty_text` | False | Fill empty cells with random text |
| `--show_all_borders` | False | Force all borders visible (width=1) |
| `--remove_outer_borders_prob` | 0.0 | Probability to remove outer table borders |
| `--remove_internal_borders_prob` | 0.0 | Probability to remove internal borders |
| `--bg_contrast_prob` | 0.0 | Probability to add contrasting backgrounds without borders |
| `--dense_text_prob` | 0.0 | Probability to add dense text filling cells |
| `--narrow_col_prob` | 0.0 | Probability to make columns narrow (15-25px) |

### Negative Example Augmentations

These augmentations help train the model to avoid common false positives:

| Augmentation | Addresses |
|--------------|-----------|
| `remove_outer_borders` | Model detecting invisible table boundaries |
| `remove_internal_borders` | Model hallucinating borders between cells |
| `bg_contrast` | Background color boundaries mistaken for borders |
| `dense_text` | Large text patterns confused with borders |
| `narrow_col` | Vertical text columns detected as vertical borders |

This script only modifies JSON. If you rely on real images, re-render or use the fallback renderer in the next step.
Point `--input_dir` in the next step to the augmented folder if you want to use the new JSONs.

## 1c) Generate Synthetic Table Images from Canvas JSON

Use the `draw_table_from_json.py` script to render table images directly from canvas JSON (without needing original images). This is useful for:
- Generating training data from augmented JSON
- Creating synthetic datasets with precise border control
- Debugging border visibility rules

### Basic Usage

```bash
uv run python -m scripts.draw_table_from_json \
    --input_dir data/raw_canvas \
    --output_dir data/synthetic_tables \
    --recursive
```

### Output Structure (default: with masks and train/val/test split)

```
data/synthetic_tables/
├── train/
│   ├── images/
│   │   ├── canvas_001_table.png
│   │   └── ...
│   └── masks/
│       ├── canvas_001_table_mask_h.png
│       ├── canvas_001_table_mask_v.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | (required) | Directory containing canvas JSON files |
| `--output_dir` | (required) | Output directory for generated images |
| `--recursive` | False | Process subdirectories recursively |
| `--no_masks` | False | Disable mask generation (masks enabled by default) |
| `--split` | 0.8 0.1 0.1 | Train/val/test split ratios |
| `--no_split` | False | Don't split, output all to single directory |
| `--seed` | 42 | Random seed for reproducible splits |
| `--num_workers` | CPU count | Number of parallel workers |

### Examples

```bash
# Generate with custom split
uv run python -m scripts.draw_table_from_json \
    --input_dir data/augmented_canvas \
    --output_dir data/synthetic_dataset \
    --recursive \
    --split 0.7 0.15 0.15

# Generate without split (all in one directory)
uv run python -m scripts.draw_table_from_json \
    --input_dir data/raw_canvas \
    --output_dir data/all_tables \
    --recursive \
    --no_split

# Generate images only (no masks)
uv run python -m scripts.draw_table_from_json \
    --input_dir data/raw_canvas \
    --output_dir data/images_only \
    --recursive \
    --no_masks
```

### Key Features

- **Japanese text support**: Automatically uses CJK fonts for proper rendering
- **Border visibility**: Only draws borders when `border*Width > 0`
- **Dashed borders**: Supports solid and dashed border styles
- **Merged cells**: Handles rowspan/colspan correctly
- **Multiprocessing**: Fast parallel generation

## 2) Convert Canvas JSON -> Line Dataset

Use the `prepare_line_dataset.py` script to:
- Crop table regions from **original color images** (using table's width/height)
- Extract line annotations (horizontal/vertical) from canvas JSON
- Split data into train/val/test sets
- Generate corresponding line JSON annotations

### Basic Usage (with train/val/test split)

```bash
uv run python -m scripts.prepare_line_dataset \
    --input_dir data/raw_canvas \
    --output_dir data/line_dataset \
    --padding 5 \
    --output_mode json \
    --split 0.8 0.1 0.1 \
    --seed 42 
    # --show_all_borders True
```

### Mask Output (for RGB + mask training)

```bash
uv run python -m scripts.prepare_line_dataset \
    --input_dir data/raw_canvas \
    --output_dir data/line_mask_dataset \
    --padding 5 \
    --output_mode mask \
    --split 0.8 0.1 0.1 \
    --seed 42
```

### Without Split (single output directory)

```bash
uv run python scripts/prepare_line_dataset.py \
    --input_dir data/raw_canvas \
    --output_dir data/line_dataset \
    --padding 10 \
    --no_split
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | (required) | Directory containing canvas JSON and images |
| `--output_dir` | (required) | Output directory for processed dataset |
| `--padding` | 5 | Padding around table when cropping (pixels) |
| `--split` | 0.8 0.1 0.1 | Train/val/test split ratios |
| `--seed` | 42 | Random seed for reproducible splits |
| `--no_split` | False | Don't split, output all to single 'all' directory |
| `--show_all_borders` | True | Draw all borders regardless of border width visibility |
| `--output_mode` | json | Output format: `json` (image+json) or `mask` (images/masks) |

### Output Structure

```
data/line_dataset/
├── train/
│   ├── canvas_001.png           # Cropped color image (from original)
│   ├── canvas_001.json          # Line annotations
│   ├── canvas_002_table0.png    # First table (if multi-table canvas)
│   ├── canvas_002_table0.json
│   ├── canvas_002_table1.png    # Second table from same canvas
│   └── canvas_002_table1.json
├── val/
│   └── ...
└── test/
    └── ...
```

### Output Structure (mask mode)

```
data/line_mask_dataset/
├── train/
│   ├── images/
│   │   ├── canvas_001.png
│   │   └── ...
│   └── masks/
│       ├── canvas_001_mask_h.png
│       ├── canvas_001_mask_v.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Tile Mask Dataset into Near-Square Patches

If your data already exists as RGB images + masks, you can tile them offline into
near-square patches (sliding window). Only the tiled outputs are saved.

```bash
uv run python -m scripts.tile_line_mask_dataset \
    --input_dir data/line_mask_dataset \
    --output_dir data/line_mask_dataset_tiled \
    --tile_size 1280 \
    --tile_stride 1024 \
    --max_aspect_ratio 1.2
```

Notes:
- Expects `images/` and `masks/` folders under each split (or directly under `--input_dir`).
- Mask filenames must match `*_mask_h.png` and `*_mask_v.png`.
- If no split folders are present, tiled outputs are written directly under `--output_dir`.

**Training Pipeline:**
- **INPUT**: Cropped color image (from original document image)
- **GROUND TRUTH**: Line JSON annotations (converted to heatmaps during training)

### Output JSON Format

```json
{
  "filename": "canvas_001.png",
  "lines": [
    {"type": "horizontal", "points": [[10, 50], [500, 50]], "thickness": 2},
    {"type": "vertical", "points": [[100, 10], [100, 400]], "thickness": 1}
  ]
}
```

### Notes

- Each table in a canvas becomes a separate image+annotation pair
- Hidden cells are ignored
- Merged cells use `rowspan`/`colspan` to expand border extents
- Border visibility rule: if `border*Width` is missing or > 0, the line is drawn

## 3) (Optional) Visualize Line Masks

Inspect generated lines by rendering a mask directly from a canvas JSON:

```bash
uv run python draw_canvas_line_mask.py --input samples/canvas-table-demo.json --out_dir samples
```

Process all canvas JSONs in a directory (optionally recursive):

```bash
uv run python draw_canvas_line_mask.py --input data/line_dataset/train --out_dir samples/masks
```

```bash
uv run python draw_canvas_line_mask.py --input data/line_dataset/train --out_dir samples/masks --recursive
```

Outputs:
- `*_mask_h.png` (horizontal lines)
- `*_mask_v.png` (vertical lines)
- `*_mask_preview.png` (green = horizontal, red = vertical)

## 3b) Full Data Preparation Pipeline (One Command)

Run the entire data preparation pipeline with a single script:

```bash
./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas
```

This runs all steps in sequence:
1. **Augment** canvas JSON (layout/style/text mutations + negative examples)
2. **Generate** synthetic table images from augmented JSON
3. **Prepare** line dataset with masks from original images
4. **Tile** mask dataset into near-square patches (smart tiling)

### Pipeline Options

```bash
./scripts/prepare_line_data_pipeline.sh \
    --input_dir data/raw_canvas \
    --output_base data/line_seg_pipeline \
    --num_aug 5 \
    --tile_size 480 \
    --tile_stride 360
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | (required) | Input directory with canvas JSON files |
| `--output_base` | `data/line_seg_pipeline` | Base output directory |
| `--num_aug` | 3 | Number of augmentations per input |
| `--seed` | 42 | Random seed |
| `--recursive` | False | Process subdirectories recursively |
| `--split` | `0.8 0.1 0.1` | Train/val/test split ratios |
| `--tile_size` | 480 | Tile size in pixels |
| `--tile_stride` | 360 | Tile stride in pixels |

### Augmentation Probabilities

| Option | Default | Description |
|--------|---------|-------------|
| `--remove_outer_borders_prob` | 0.3 | Remove outer table borders |
| `--remove_internal_borders_prob` | 0.2 | Remove internal borders |
| `--bg_contrast_prob` | 0.2 | Add contrasting backgrounds |
| `--dense_text_prob` | 0.15 | Add dense text filling |
| `--narrow_col_prob` | 0.1 | Make columns narrow |

### Smart Tiling Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min_line_pixels` | 100 | Skip tiles with fewer line pixels |
| `--no_content_bbox` | False | Don't restrict tiling to content bbox |
| `--bbox_padding` | 50 | Padding around content bbox |

### Skip Steps

```bash
# Skip augmentation (use existing augmented data)
./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas --skip_augment

# Skip synthetic generation
./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas --skip_synthetic

# Skip tiling
./scripts/prepare_line_data_pipeline.sh --input_dir data/raw_canvas --skip_tile
```

### Output Structure

```
data/line_seg_pipeline/
├── augmented_json/           # Step 1: Augmented canvas JSONs
├── synthetic_images/         # Step 2: Rendered table images + masks
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   └── test/
├── line_mask_dataset/        # Step 3: Dataset from original images
└── synthetic_images_tiled/   # Step 4: Tiled patches
    ├── train/
    ├── val/
    └── test/
```

## 4) Train the Line Segmentation Model

Use the line segmentation config and select the line task:

```bash
uv run python train.py \
    --task line \
    --config_file configs/train_line.json \
    --output_ch 2 \
    --batch_size 4 \
    --resume trace_wtw.pth
```

Common overrides:
- `--line_thickness` to control line width in the heatmaps
- `--use_gaussian` to smooth heatmaps
- `--train_size` to set the input resize dimension

## 4b) Train with RGB + Mask Images

If you already have line masks, you can train directly from RGB images and two
mask files per image (`*_mask_h.png` and `*_mask_v.png`). The loader normalizes
mask values to `[0, 1]` using each mask's max value (safe for varying pixel
values).

Expected layout:

```
data/line_mask_dataset/
├── train/
│   ├── images/
│   │   ├── sample_001.png
│   │   └── ...
│   └── masks/
│       ├── sample_001_mask_h.png
│       ├── sample_001_mask_v.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Training command:

```bash
uv run python train.py \
    --task line \
    --line_data_mode mask \
    --config_file configs/train_line_mask.json \
    --output_ch 2
```

### Evaluate During Training (subprocess approach)

Run the separate evaluation script to check model performance:

```bash
# Evaluate on validation set
uv run python eval_line_mask.py \
    --trained_model eval/ckpt_1000.pth \
    --data_dir data/line_mask_dataset \
    --phase val \
    --input_size 1280

# Output: {"dice": 0.85, "iou": 0.74, "precision": 0.87, "recall": 0.83, ...}
```

## 5) Run Inference (Generate Predictions)

Run inference on a folder of images and write line JSON predictions:

```bash
uv run python line_infer.py \
    --input data/line_dataset/test \
    --trained_model eval/ckpt_100000.pth \
    --output_dir results/predictions \
    --input_size 1280 \
    --save_heatmap \
    --resize_map \
    --use_compare \
    --threshold_h 0.3 \
    --threshold_v 0.3
```

Notes:
- `--input_size` should match the `train_size` used during training.
- Predictions are saved as `results/predictions/<image_basename>.json`.
- Add `--save_heatmap` to dump debug heatmaps next to predictions (`*_heatmap_h.png`, `*_heatmap_v.png`, and `*_heatmap_combined.png`).

## 5b) Run Inference (Save Predicted Masks)

Run inference on a folder of images and save predicted binary masks (instead of JSON lines):

```bash
uv run python infer_masks.py \
    --input data/line_mask_dataset/test/images \
    --trained_model eval/model.pth \
    --output_dir results/pred_masks \
    --input_size 1280 \
    --threshold_h 0.3 \
    --threshold_v 0.2
```

Add `--save_raw` to also save raw heatmap images (grayscale 0-255) alongside binary masks.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Directory of input images |
| `--trained_model` | (required) | Path to model weights |
| `--output_dir` | `results/pred_masks` | Output directory |
| `--input_size` | 1280 | Inference input size (should match `train_size`) |
| `--output_ch` | 2 | Number of output channels |
| `--threshold_h` | 0.3 | Binarization threshold for horizontal mask |
| `--threshold_v` | 0.2 | Binarization threshold for vertical mask |
| `--save_raw` | False | Also save raw heatmaps as grayscale images |
| `--num_samples` | 0 | Randomly sample N images (0 = all) |
| `--seed` | 42 | Random seed for sampling |
| `--device` | `cuda` | Device (`cuda`/`cpu`) |

### Output Structure

```
results/pred_masks/
├── sample_001_mask_h.png     # Binary horizontal mask
├── sample_001_mask_v.png     # Binary vertical mask
├── sample_001_pred_h.png     # Raw heatmap (only with --save_raw)
├── sample_001_pred_v.png     # Raw heatmap (only with --save_raw)
└── ...
```

## 5c) Evaluate Predictions & Identify Weak Images

Compute per-image metrics, identify weak predictions, and generate a statistical report.

### Mode A: With pre-computed predictions (from `infer_masks.py`)

```bash
uv run python evaluate_masks.py \
    --pred_dir results/pred_masks \
    --gt_dir data/line_mask_dataset/test/masks \
    --output_dir results/analysis \
    --save_vis
```

### Mode B: On-the-fly inference (all-in-one)

```bash
uv run python evaluate_masks.py \
    --images_dir data/line_mask_dataset/test/images \
    --gt_dir data/line_mask_dataset/test/masks \
    --trained_model eval/model.pth \
    --output_dir results/analysis \
    --threshold_h 0.3 --threshold_v 0.2 \
    --save_vis
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pred_dir` | None | Directory with pre-computed predicted masks |
| `--images_dir` | None | Directory with original images (on-the-fly inference) |
| `--trained_model` | None | Model weights (required with `--images_dir`) |
| `--gt_dir` | (required) | Directory with ground truth masks |
| `--output_dir` | `results/analysis` | Output directory for reports |
| `--input_size` | 1280 | Inference input size |
| `--output_ch` | 2 | Number of output channels |
| `--threshold_h` | 0.3 | Threshold for horizontal mask |
| `--threshold_v` | 0.2 | Threshold for vertical mask |
| `--device` | `cuda` | Device (`cuda`/`cpu`) |
| `--num_samples` | 0 | Randomly sample N images to evaluate (0 = all) |
| `--seed` | 42 | Random seed for sampling |
| `--bottom_pct` | 0.1 | Bottom percentage flagged as weak (0.1 = 10%) |
| `--save_vis` | False | Save comparison visualizations for weak images |
| `--max_vis` | 50 | Max weak images to visualize |

### Output Structure

```
results/analysis/
├── per_image_metrics.csv         # Per-image Dice, IoU, Precision, Recall (h/v/combined)
├── evaluation_report.json        # Aggregate metrics, weak image list, failure modes
├── pred_masks/                   # Saved predictions (only with --images_dir)
└── weak_visualizations/          # Side-by-side comparisons (only with --save_vis)
    ├── 0000_dice0.123_sample_001.png
    ├── 0001_dice0.234_sample_002.png
    └── ...
```

### Visualization Legend

Each weak image visualization shows:
- **Green** = False negative (GT lines missed by the model)
- **Red** = False positive (model hallucinated lines)
- **Yellow** = True positive (correct overlap)

If `--images_dir` is provided, the original image is shown side-by-side.

### Failure Mode Categories

Images with dice < 0.7 are categorized into:

| Mode | Description |
|------|-------------|
| `high_fp` | Model hallucinating lines where none exist |
| `high_fn` | Model missing real lines |
| `balanced_poor` | Both false positives and false negatives are high |

### Example Output

```
======================================================================
EVALUATION SUMMARY
======================================================================
Total images evaluated: 500

  Combined:
    Micro Dice:      0.8395
    Micro IoU:       0.7234
    Micro Precision: 0.8512
    Micro Recall:    0.8281
    Macro Dice:      0.8150 (std: 0.1234)
    Macro IoU:       0.7023 (std: 0.1456)

  Horizontal:
    Micro Dice:      0.8543
    ...

  Vertical:
    Micro Dice:      0.8229
    ...

──────────────────────────────────────────────────────────────────────
WEAK IMAGES (bottom 10%, sorted by dice):
──────────────────────────────────────────────────────────────────────
  0.1234  sample_042
  0.2345  sample_118
  ...

──────────────────────────────────────────────────────────────────────
FAILURE MODE BREAKDOWN (images with dice < 0.7):
──────────────────────────────────────────────────────────────────────
  High false positives (hallucinated lines): 12
  High false negatives (missed lines):       8
  Both FP and FN high:                       5

──────────────────────────────────────────────────────────────────────
DICE SCORE DISTRIBUTION:
──────────────────────────────────────────────────────────────────────
  [0.00-0.30):     3
  [0.30-0.50):     7 #
  [0.50-0.70):    15 ###
  [0.70-0.80):    42 ########
  [0.80-0.90):   156 ###############################
  [0.90-0.95):   187 #####################################
  [0.95-1.00]:    90 ##################
```

## 6) Evaluate Line Segmentation

Evaluate predictions against ground truth JSONs:

```bash
uv run python evaluation/line_eval.py \
    --gt_path data/line_dataset/test \
    --pred_path results/predictions \
    --threshold 10 \
    --save_json results/metrics.json
```

### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gt_path` | (required) | Ground truth directory |
| `--pred_path` | (required) | Predictions directory |
| `--threshold` | 10 | Line matching threshold in pixels |
| `--line_thickness` | 2 | Line thickness for pixel metrics |
| `--save_json` | None | Save results to JSON file |

## 6b) Evaluate Mask-based Predictions

If your model outputs prediction masks directly (not JSON), use the mask evaluation script:

```bash
uv run python evaluation/mask_eval.py \
    --gt_dir data/line_mask_dataset/test \
    --pred_dir results/predictions \
    --threshold 0.5 \
    --save_json results/mask_metrics.json
```

### Finding Optimal Threshold

Sweep thresholds to find the best binarization threshold:

```bash
uv run python evaluation/mask_eval.py \
    --gt_dir data/line_mask_dataset/test \
    --pred_dir results/predictions \
    --sweep_threshold
```

### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gt_dir` | (required) | Ground truth directory (with masks/ subdirectory) |
| `--pred_dir` | (required) | Predictions directory |
| `--threshold` | 0.5 | Binarization threshold for predictions (0-1) |
| `--sweep_threshold` | False | Sweep thresholds 0.1-0.9 to find optimal |
| `--mask_suffix_h` | `_mask_h.png` | Horizontal mask suffix |
| `--mask_suffix_v` | `_mask_v.png` | Vertical mask suffix |
| `--save_json` | None | Save results to JSON file |

### Metrics Computed

| Metric | Formula | Description |
|--------|---------|-------------|
| **IoU** | TP / (TP + FP + FN) | Intersection over Union (Jaccard Index) |
| **Dice** | 2·TP / (2·TP + FP + FN) | Dice coefficient (same as F1) |
| **Precision** | TP / (TP + FP) | What fraction of predictions are correct |
| **Recall** | TP / (TP + FN) | What fraction of GT is detected |
| **F1** | 2·P·R / (P + R) | Harmonic mean of precision and recall |

### Worst Case Analysis

Find and visualize the worst performing samples to debug model issues:

```bash
# Print 10 worst cases sorted by IoU
uv run python evaluation/mask_eval.py \
    --gt_dir data/line_mask_dataset/test \
    --pred_dir results/predictions \
    --threshold 0.5 \
    --show_worst 10

# Save visualizations of worst cases
uv run python evaluation/mask_eval.py \
    --gt_dir data/line_mask_dataset/test \
    --pred_dir results/predictions \
    --threshold 0.5 \
    --show_worst 20 \
    --save_worst results/worst_cases \
    --sort_by iou \
    --sort_channel all
```

### Worst Case Options

| Option | Default | Description |
|--------|---------|-------------|
| `--show_worst` | 0 | Print N worst cases (0=disabled) |
| `--save_worst` | None | Directory to save worst case visualizations |
| `--sort_by` | iou | Metric to sort by (iou, dice, precision, recall) |
| `--sort_channel` | all | Channel for sorting (all, h=horizontal, v=vertical) |

### Visualization Output

Each worst case visualization shows 4 panels side-by-side:
1. **Original**: The input image
2. **GT**: Ground truth mask (green=horizontal, red=vertical)
3. **Pred**: Predicted mask (green=horizontal, red=vertical)
4. **Diff**: Error visualization (green=TP, red=FN missed, blue=FP false alarm)

### Example Output

```
============================================================
MASK-BASED LINE SEGMENTATION EVALUATION
============================================================
Evaluated: 50 image pairs

COMBINED:
  IoU:       0.7234
  Dice:      0.8395
  Precision: 0.8512
  Recall:    0.8281
  F1:        0.8395
  (TP=125,432, FP=21,876, FN=25,891)

HORIZONTAL:
  IoU:       0.7456
  Dice:      0.8543
  ...

VERTICAL:
  IoU:       0.6987
  Dice:      0.8229
  ...
```

## 7) Compare GT vs Pred Line Labels (Visual Debug)

Render overlays that show GT vs predicted lines, plus a TP/FP/FN diff layer.
Pred coordinates are scaled from a fixed `--pred_size` (default: 1280) to the
original image size before drawing.

```bash
uv run python scripts/compare_line_labels.py \
    --gt_dir data/line_dataset/test \
    --pred_dir results/predictions \
    --image_dir data/line_dataset/test \
    --out_dir results/compare \
    --pred_size 1280
```

Notes:
- If predictions are already in original image coordinates, set `--pred_size 0`.
- If images are missing, use `--image_size W,H` to draw on a blank canvas.
- Add `--recursive` to search nested folders (output preserves subfolders).
### Metrics Computed

**Line-level metrics** (based on endpoint matching):
- `line_all`: All lines combined
- `line_h`: Horizontal lines only
- `line_v`: Vertical lines only

**Pixel-level metrics** (based on rendered masks):
- `pixel_all`: All line pixels combined
- `pixel_h`: Horizontal line pixels
- `pixel_v`: Vertical line pixels

Each reports: Precision, Recall, F1, TP, FP, FN

### Example Output

```
============================================================
LINE SEGMENTATION EVALUATION RESULTS
============================================================
Evaluated: 50 files (missing: 0)

LINE_ALL:
  Precision: 0.9234
  Recall:    0.8912
  F1:        0.9070
  (TP=412, FP=34, FN=50)

PIXEL_ALL:
  Precision: 0.8876
  Recall:    0.9123
  F1:        0.8998
  (TP=125432, FP=15876, FN=12045)
```
