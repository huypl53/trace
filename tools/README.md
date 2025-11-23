# Table Line Extraction Tools

This directory contains tools for extracting and processing table grid lines from JSON table definitions for training line detection models.

## Quick Start

**Run the entire pipeline with one command:**

```bash
# From the project root directory
uv run python tools/run_pipeline.py samples

# Save everything under a single directory
uv run python tools/run_pipeline.py samples --output-dir runs/exp1

# Or with custom options
uv run python tools/run_pipeline.py samples -n 20 --min-size 384 --max-size 768
```

This will automatically:
1. Crop tables from your JSON/image pairs
2. Extract grid lines from cell borders
3. Create random training patches
4. Verify the output is ready for training

See [run_pipeline.py](#run_pipelinepy---end-to-end-automation) for all options.

## Overview

These tools form a pipeline that converts annotated table data into training samples for line detection:

```
Input: JSON tables with cell data
  ↓ crop_json_tables.py
  ↓ extract_labels.py
  ↓ patching.py
Output: Training patches with line annotations
```

## Tools

### 0. run_pipeline.py - End-to-End Automation

**NEW!** Automated script that runs the entire pipeline with one command.

**Usage:**
```bash
# Run full pipeline with defaults
uv run python tools/run_pipeline.py samples

# Custom output directories
uv run python tools/run_pipeline.py samples \
  --crop-dir MY_CROP \
  --line-dir MY_LINES \
  --patch-dir MY_PATCHES

# Store all outputs under one folder
uv run python tools/run_pipeline.py samples --output-dir runs/exp1

# More patches with larger size range
uv run python tools/run_pipeline.py samples -n 20 --min-size 384 --max-size 768

# Only run specific steps
uv run python tools/run_pipeline.py samples --steps 1,2  # Only crop and extract
uv run python tools/run_pipeline.py samples --steps 3    # Only patches

# Verify existing output
uv run python tools/run_pipeline.py PATCH_DATA --verify-only
```

**All Options:**
- `input_dir` (required): Input directory with JSON/image pairs
- `--output-dir`: Base directory to store all outputs (CROP_DATA/LINE_DATA/PATCH_DATA inside)
- `--crop-dir`: Output for cropped tables (default: CROP_DATA)
- `--line-dir`: Output for line data (default: LINE_DATA)
- `--patch-dir`: Output for patches (default: PATCH_DATA)
- `--pad-min`, `--pad-max`: Padding range for cropping (default: 5-20)
- `-n, --num-patches`: Patches per image (default: 10)
- `--min-size`, `--max-size`: Patch size range (default: 256-512)
- `--steps`: Run specific steps (e.g., "1,2" or "3")
- `--no-clean`: Don't clean output directories before running
- `--verify-only`: Just verify output without running

**Example Output:**
```
======================================================================
  Table Line Extraction Pipeline
======================================================================
Input directory: samples
Output directories:
  - Cropped tables: runs/exp1/CROP_DATA
  - Line data: runs/exp1/LINE_DATA
  - Patches: runs/exp1/PATCH_DATA

Found 2 JSON files in input directory

[Step 1/3] Crop Tables
----------------------------------------------------------------------
✓ Completed in 0.11s
✓ Created 4 tables

[Step 2/3] Extract Lines
----------------------------------------------------------------------
✓ Completed in 0.01s
✓ Extracted 911 total lines (366 visible)

[Step 3/3] Create Patches
----------------------------------------------------------------------
✓ Completed in 0.13s
✓ Created 20 patches

======================================================================
  Pipeline Complete!
======================================================================
Total time: 0.36s

Statistics:
  - Tables extracted: 4
  - Lines extracted: 911 (366 visible)
  - Training patches: 20

Next steps:
  1. Use LineDataset to load data from 'PATCH_DATA'
  2. Apply LineAugmentation for training
  3. Train your model!
```

---

### 1. types.py - Line Type Definition

Pydantic model for representing table grid lines with validation.

**Line Model:**
```python
from tools.types import Line

# Create a horizontal line
h_line = Line(
    x1=10.0, y1=20.0,
    x2=100.0, y2=20.0,
    direction='horizontal',
    visible=True
)

# Create a vertical line
v_line = Line(
    x1=50.0, y1=10.0,
    x2=50.0, y2=100.0,
    direction='vertical',
    visible=False
)
```

**Validation Rules:**
- Horizontal lines: `y1` must equal `y2` (within 1e-6 tolerance)
- Vertical lines: `x1` must equal `x2` (within 1e-6 tolerance)
- Direction must be either `'horizontal'` or `'vertical'`
- Visible is a boolean indicating if the line should be rendered

**Methods:**
- `to_dict()`: Convert to dictionary for JSON serialization

---

### 2. crop_json_tables.py - Extract Tables with Padding

Crops individual tables from full-page JSON/image pairs with random padding.

**Usage:**
```bash
# Basic usage
uv run tools/crop_json_tables.py <input_dir> [options]

# Example
uv run tools/crop_json_tables.py samples --output-dir CROP_DATA --pad-min 5 --pad-max 20
```

**Arguments:**
- `input_dir` (required): Directory containing JSON and image pairs

**Options:**
- `--output-dir`: Output directory (default: `CROP_DATA`)
- `--pad-min`: Minimum padding in pixels (default: `5`)
- `--pad-max`: Maximum padding in pixels (default: `20`)

**What it does:**
1. Finds all JSON files in the input directory
2. Locates corresponding images (.png, .jpg, .jpeg)
3. Extracts each table from the JSON `items` array (type: "table")
4. Adds random padding around each table
5. Crops the image to the padded table region
6. Adjusts table coordinates in the JSON to match the crop
7. Saves cropped image and updated JSON to output directory

**Input Format:**
```json
{
  "items": [
    {
      "type": "table",
      "x": 100,
      "y": 200,
      "width": 500,
      "height": 300,
      "properties": { ... }
    }
  ]
}
```

**Output:**
- If one table: `<original_name>.json` and `<original_name>.png`
- If multiple tables: `<original_name>_table_0.json`, `<original_name>_table_1.json`, etc.

**Example Output:**
```
samples/page_10.json (1 table) → CROP_DATA/page_10.json, page_10.png
samples/page_23.json (3 tables) → CROP_DATA/page_23_table_0.json, page_23_table_0.png
                                          page_23_table_1.json, page_23_table_1.png
                                          page_23_table_2.json, page_23_table_2.png
```

---

### 3. extract_labels.py - Convert Cell Borders to Lines

Extracts grid lines from table cell data, creating deduplicated Line objects.

**Usage:**
```bash
# Basic usage
uv run tools/extract_labels.py <input_dir> [options]

# Example
uv run tools/extract_labels.py CROP_DATA --output-dir LINE_DATA
```

**Arguments:**
- `input_dir` (required): Directory with cropped table JSON/image pairs (from step 1)

**Options:**
- `--output-dir`: Output directory (default: `LINE_DATA`)

**What it does:**
1. Reads table JSON files from input directory
2. Processes table cell data to extract grid lines
3. Handles merged cells (colspan/rowspan) correctly
4. Skips hidden cells
5. Creates deduplicated grid lines (not individual cell borders)
6. Marks lines as visible/invisible based on border width
7. Saves Line objects as JSON alongside images

**Cell Data Processing:**

The tool reads from `properties.cellData` which contains cell-level styling:

```json
{
  "properties": {
    "rows": 10,
    "columns": 5,
    "columnWidths": {"0": 80, "1": 100, ...},
    "rowHeights": {"0": 20, "1": 20, ...},
    "cellData": {
      "0-0": {
        "cellStyle": {
          "borderTopWidth": 1,
          "borderBottomWidth": 1,
          "borderLeftWidth": 1,
          "borderRightWidth": 0
        }
      }
    },
    "mergedCells": {
      "0-0": {"colspan": 2, "rowspan": 1}
    },
    "hiddenCells": {
      "0-1": true
    }
  }
}
```

**Line Visibility:**
- `borderWidth > 0` or `borderWidth == None` → visible
- `borderWidth == 0` → invisible (explicitly hidden)
- `None` is treated as visible (default table border style)

**Line Deduplication and Visibility Rules:**

Adjacent cells sharing a border produce a **single line**, not two overlapping lines.

**Important:** A shared line is visible **only if ALL adjacent cells mark it as visible**. If ANY cell marks the border as invisible (`borderWidth == 0`), the entire shared line becomes invisible. This ensures proper representation of intentionally hidden borders.

**Output Format:**
```json
{
  "image": "page_10.png",
  "lines": [
    {
      "x1": 10.0, "y1": 20.0,
      "x2": 100.0, "y2": 20.0,
      "direction": "horizontal",
      "visible": true
    },
    {
      "x1": 50.0, "y1": 10.0,
      "x2": 50.0, "y2": 100.0,
      "direction": "vertical",
      "visible": false
    }
  ]
}
```

**Example Output:**
From 1 table with 23 rows × 14 columns:
- Extracted: 410 total lines
- Visible: 216 lines
- Invisible: 194 lines

---

### 4. patching.py - Create Random Patches

Creates random crops from line data with coordinate transformations.

**Usage:**
```bash
# Basic usage
uv run tools/patching.py <input_dir> [options]

# Example
uv run tools/patching.py LINE_DATA --output-dir PATCH_DATA -n 10 --min-size 256 --max-size 512
```

**Arguments:**
- `input_dir` (required): Directory with line JSON/image pairs (from step 2)

**Options:**
- `--output-dir`: Output directory (default: `PATCH_DATA`)
- `-n, --num-patches`: Number of patches per image (default: `10`)
- `--min-size`: Minimum patch size in pixels (default: `256`)
- `--max-size`: Maximum patch size in pixels (default: `512`)

**What it does:**
1. Reads line JSON files from input directory
2. For each image, creates N random patches
3. Random patch size between min-size and max-size
4. Random patch location within image bounds
5. Crops the image
6. Updates line coordinates:
   - **Shift**: Translate lines to patch-local coordinates
   - **Truncate**: Clip lines at patch boundaries
   - **Remove**: Discard lines completely outside patch
7. Saves patch images and updated line JSON

**Line Coordinate Transformation:**

For a patch at `(patch_x, patch_y)` with size `(patch_w, patch_h)`:

**Horizontal Line:**
```
Original: (x1, y, x2, y)
If y outside [patch_y, patch_y + patch_h] → Remove
Else:
  x1_new = clip(x1, patch_x, patch_x + patch_w) - patch_x
  x2_new = clip(x2, patch_x, patch_x + patch_w) - patch_x
  y_new = y - patch_y
```

**Vertical Line:**
```
Original: (x, y1, x, y2)
If x outside [patch_x, patch_x + patch_w] → Remove
Else:
  y1_new = clip(y1, patch_y, patch_y + patch_h) - patch_y
  y2_new = clip(y2, patch_y, patch_y + patch_h) - patch_y
  x_new = x - patch_x
```

**Output:**
For each input image, creates multiple patches:
```
LINE_DATA/page_10.json → PATCH_DATA/page_10_patch_0.json, page_10_patch_0.png
                                    page_10_patch_1.json, page_10_patch_1.png
                                    ...
                                    page_10_patch_9.json, page_10_patch_9.png
```

**Example:**
From 4 tables with 10 patches each:
- Input: 4 images with line annotations
- Output: 40 patches (256-512px each) with updated line coordinates

---

## Complete Pipeline Example

Process sample data through all steps:

```bash
# Step 1: Crop tables (samples → CROP_DATA)
uv run tools/crop_json_tables.py samples \
  --output-dir CROP_DATA \
  --pad-min 5 \
  --pad-max 20

# Step 2: Extract lines (CROP_DATA → LINE_DATA)
uv run tools/extract_labels.py CROP_DATA \
  --output-dir LINE_DATA

# Step 3: Create patches (LINE_DATA → PATCH_DATA)
uv run tools/patching.py LINE_DATA \
  --output-dir PATCH_DATA \
  -n 10 \
  --min-size 256 \
  --max-size 512
```

**Result:**
- `CROP_DATA/`: Individual tables with padding
- `LINE_DATA/`: Tables with extracted line annotations
- `PATCH_DATA/`: Training patches ready for use

---

## Integration with Training Pipeline

After running the tools, use the data with the `LineDataset` loader:

```python
from loader import LineDataset
from augmentations import LineAugmentation

# Create augmentation
aug = LineAugmentation(size=512)

# Load dataset
dataset = LineDataset(
    data_dir="PATCH_DATA",
    scale_down=2,
    visible_only=False,  # Include invisible lines
    transform=aug
)

# Use with PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for images, gt_heatmaps, weights in loader:
    # Train your model
    pass
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────┐
│ samples/                                        │
│ ├── page_10.json (full page with tables)       │
│ └── page_10.png                                 │
└─────────────────────────────────────────────────┘
                      ↓
           [crop_json_tables.py]
                      ↓
┌─────────────────────────────────────────────────┐
│ CROP_DATA/                                      │
│ ├── page_10.json (single table + padding)      │
│ └── page_10.png (cropped)                      │
└─────────────────────────────────────────────────┘
                      ↓
            [extract_labels.py]
                      ↓
┌─────────────────────────────────────────────────┐
│ LINE_DATA/                                      │
│ ├── page_10.json (image + Line objects)        │
│ └── page_10.png (same as CROP_DATA)            │
└─────────────────────────────────────────────────┘
                      ↓
              [patching.py]
                      ↓
┌─────────────────────────────────────────────────┐
│ PATCH_DATA/                                     │
│ ├── page_10_patch_0.json (patch + lines)       │
│ ├── page_10_patch_0.png                        │
│ ├── page_10_patch_1.json                       │
│ ├── page_10_patch_1.png                        │
│ └── ...                                         │
└─────────────────────────────────────────────────┘
                      ↓
              [LineDataset]
                      ↓
            Training-ready data
```

---

## Technical Details

### Coordinate System
- Origin: Top-left corner (0, 0)
- Units: Pixels
- X-axis: Increases to the right
- Y-axis: Increases downward

### Line Representation
Lines are represented by two endpoints:
- Horizontal: `(x1, y, x2, y)` where `y1 == y2 == y`
- Vertical: `(x, y1, x, y2)` where `x1 == x2 == x`

### Merged Cells
Cells can span multiple rows/columns:
```json
"mergedCells": {
  "0-0": {"colspan": 3, "rowspan": 2}
}
```
Cell at row 0, column 0 spans 3 columns and 2 rows.

### Hidden Cells
Hidden cells (usually due to merges) are skipped:
```json
"hiddenCells": {
  "0-1": true,  // Cell at row 0, col 1 is hidden
  "0-2": true
}
```

### Border Width Detection
```python
visible = border_width > 0
invisible = border_width == 0
```

---

## Troubleshooting

**No tables found:**
- Check that JSON has `items` array with `type: "table"`

**Image not found:**
- Ensure image has same base name as JSON
- Supported: .png, .jpg, .jpeg

**No lines extracted:**
- Verify `cellData` exists in table properties
- Check that cells have `cellStyle` with border widths

**Patches have no lines:**
- Increase `--num-patches` to try more locations
- Increase patch size with `--max-size`

**Validation errors:**
- Lines must be perfectly horizontal or vertical
- Check for floating-point precision issues

**Augmentation errors (ValueError: empty range for randrange):**
- This occurred with images smaller than half the target augmentation size
- **Fixed:** The `LineRandomResizeCrop` augmentation now automatically handles small images
- Images smaller than the minimum crop size will skip random cropping and only be resized
- No action needed - the fix is already applied

---

## Visualization

Visualize extracted lines on images to verify the extraction results.

**Script:** `visualize_lines.py` (located in project root)

**Usage:**
```bash
# Visualize all data
uv run python visualize_lines.py LINE_DATA

# Visualize specific files
uv run python visualize_lines.py LINE_DATA --pattern "page_23*"

# Hide invisible lines
uv run python visualize_lines.py LINE_DATA --no-invisible

# Custom output directory
uv run python visualize_lines.py LINE_DATA --output-dir my_visualizations
```

**Output:**
- Creates overlay images with lines drawn on the original tables
- **Red lines** = Visible borders
- **Blue lines** = Invisible borders
- Includes statistics overlay showing line counts and percentages

**Example:**
```bash
uv run python visualize_lines.py LINE_DATA
# Outputs to visualizations/ folder
# Files: <original_name>_visualization.png
```

This is very useful for:
- Verifying border visibility detection is correct
- Checking if line extraction matches the visual table structure
- Debugging issues with specific tables

---

## Testing

Run the complete test suite:

```bash
cd /mnt/Code/code/AI/segmentation/trace/offline-data
uv run test_pipeline.py
```

This will:
1. Run all pipeline steps
2. Validate Line type checking
3. Verify data formats
4. Test dataset loading
5. Test augmentation
6. Generate visualizations (if matplotlib available)

---

## Dependencies

- Python 3.7+
- Pillow (PIL)
- pydantic
- NumPy (for testing)
- OpenCV (for dataset)
- PyTorch (for dataset)

---

## Output Statistics (Sample Data)

From the included test samples:

| Step | Input | Output | Details |
|------|-------|--------|---------|
| 1. Crop | 2 pages | 4 tables | 1 table + 3 tables |
| 2. Extract | 4 tables | 911 lines total | 366 visible, 545 invisible |
| 3. Patch | 4 tables | 40 patches | 10 per table |

Example from largest table:
- Size: 23 rows × 14 columns
- Lines: 410 total (216 visible, 194 invisible)
- Grid structure: ~18 horizontal + ~15 vertical lines (deduplicated)

---

## License

Part of the TRACE offline data processing pipeline.

## Contact

For issues or questions about this pipeline, please refer to the main project documentation.
