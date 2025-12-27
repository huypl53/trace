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

## 2) Convert Canvas JSON -> Line Dataset

Use the `prepare_line_dataset.py` script to:
- Crop table regions from **original color images** (using table's width/height)
- Extract line annotations (horizontal/vertical) from canvas JSON
- Split data into train/val/test sets
- Generate corresponding line JSON annotations

### Basic Usage (with train/val/test split)

```bash
uv run python scripts/prepare_line_dataset.py \
    --input_dir data/raw_canvas \
    --output_dir data/line_dataset \
    --padding 5 \
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

## 4) Train the Line Segmentation Model

Use the line segmentation config and select the line task:

```bash
uv run python train.py \
    --task line \
    --config_file configs/train_line.json \
    --output_ch 2 \
    --resume trace_wtw.pth
```

Common overrides:
- `--line_thickness` to control line width in the heatmaps
- `--use_gaussian` to smooth heatmaps
- `--train_size` to set the input resize dimension

## 5) Run Inference (Generate Predictions)

Run inference on a folder of images and write line JSON predictions:

```bash
uv run python line_infer.py \
    --input data/line_dataset/test \
    --trained_model eval/ckpt_100000.pth \
    --output_dir results/predictions \
    --input_size 1280 \
    --threshold_h 0.3 \
    --threshold_v 0.3
```

Notes:
- `--input_size` should match the `train_size` used during training.
- Predictions are saved as `results/predictions/<image_basename>.json`.
- Add `--save_heatmap` to dump debug heatmaps next to predictions (`*_heatmap_h.png`, `*_heatmap_v.png`, and `*_heatmap_combined.png`).

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
