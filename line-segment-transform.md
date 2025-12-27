# Technical Spec: Adapting TRACE Model for Line Segmentation

**Author**: AI Analysis
**Date**: 2025-12-26
**Status**: Ready for Development

---

## Executive Summary

The TRACE table reconstruction model can be adapted for **ruled/grid line segmentation** with minimal architectural changes. The model is a generic heatmap segmentation network - all table-specific logic is in the data preprocessing layer.

**Key Finding**: The model reads XML only to generate binary/heatmap masks. The neural network itself is domain-agnostic.

---

## How the Current Pipeline Works

### 1. XML Parsing (`parsers/xml_parser.py`)
- Reads cell coordinates (`<Coords points="x1,y1 x2,y2 x3,y3 x4,y4"/>`)
- Reads line attributes (`<Lines top="0" bottom="1" left="0" right="0"/>`)
- Returns: `quads` (cell corners) and `lines` (boolean flags for each edge)

### 2. Mask Generation (`loader.py:GTTransform`)
Converts parsed data into **5-channel heatmap masks**:

| Channel | Content | How Generated |
|---------|---------|---------------|
| 0 | Corner points | Gaussian kernel (sigma=40) placed at each corner |
| 1 | Horizontal lines (present) | `cv2.line()` drawn at top/bottom edges where `lines[0]=1` or `lines[1]=1` |
| 2 | Vertical lines (present) | `cv2.line()` drawn at left/right edges where `lines[2]=1` or `lines[3]=1` |
| 3 | Horizontal lines (absent) | `cv2.line()` where `lines[0]=0` or `lines[1]=0` |
| 4 | Vertical lines (absent) | `cv2.line()` where `lines[2]=0` or `lines[3]=0` |

### 3. Model Architecture (`model.py`)
- **Encoder**: ResNet50 or ViT backbone
- **Decoder**: U-Net style upsampling with skip connections
- **Output**: `output_ch=5` channels (configurable)
- The model is **architecture-agnostic** to what those channels represent

### 4. Loss Function (`loss.py`)
- Per-channel Huber loss with OHEM (Online Hard Example Mining)
- Treats each channel independently

---

## Why Line Segmentation Is Feasible

The architecture is essentially a **generic multi-class heatmap segmentation network**. The 5 channels for table reconstruction are arbitrary - you can change them to any line segmentation targets.

**What makes it adaptable:**
1. The model doesn't "know" it's detecting tables - it just learns to produce heatmaps matching ground truth
2. `output_ch` is a simple parameter to change
3. The data pipeline is modular (parser + mask generator are separate)
4. Loss function works for any number of channels

---

## What Needs to Change for Line Segmentation

### Option A: Minimal Changes (Same XML Format)
If your line annotations can fit the current XML schema (lines defined by start/end points as cell edges):
- Modify `GTTransform()` to generate line heatmaps instead of corner+edge masks
- Change `output_ch` to match your line classes (e.g., 1 for binary, N for N line types)

### Option B: New Annotation Format
If you need different annotation (e.g., polylines, line segments with class labels):
1. Create new parser (e.g., `parsers/line_parser.py`)
2. Create new mask generator in `loader.py`
3. Update `TRACE_Dataset` to use new parser

---

## Key Files to Modify

| File | Current Role | Changes Needed |
|------|-------------|----------------|
| `parsers/xml_parser.py` | Parse table cells | Create new line parser or modify |
| `loader.py:GTTransform` | Generate 5-ch masks | Generate line segmentation masks |
| `model.py` | Model definition | Change `output_ch` parameter |
| `train.py` | Training script | Minimal (just config changes) |

---

## Summary

Your intuition is correct:
- **Current flow**: XML → parse coords → draw masks → train model
- **For lines**: XML/new format → parse line coords → draw line masks → train model

The model is a general-purpose segmentation network that learns from whatever masks you provide. The table-specific logic is entirely in the data preprocessing, not the model architecture.

---

# Implementation Plan: Ruled/Grid Line Segmentation

## Recommended Annotation Format

Since ruled/grid lines are straight horizontal and vertical lines, I recommend a **simple JSON format**:

```json
{
  "filename": "document.png",
  "lines": [
    {"type": "horizontal", "points": [[10, 50], [500, 50]]},
    {"type": "vertical", "points": [[100, 10], [100, 400]]},
    {"type": "horizontal", "points": [[10, 100], [500, 100]]}
  ]
}
```

This is simpler than XML and sufficient for straight lines. Each line is defined by start and end points.

## Output Channels

For ruled/grid lines, I recommend **2 or 3 channels**:

| Channel | Purpose |
|---------|---------|
| 0 | Horizontal lines heatmap |
| 1 | Vertical lines heatmap |
| 2 (optional) | Line endpoints/intersections |

The current model uses 5 channels (corners + 4 edge types). For pure line detection, 2 channels (H + V) may be sufficient.

## Implementation Steps

### Step 1: Create Line Parser (`parsers/line_parser.py`)
```python
class ParserLine:
    def parseGT(self):
        # Read JSON annotation
        # Return: {"h_lines": [...], "v_lines": [...]}
```

### Step 2: Create Line Mask Generator (`loader.py`)
```python
def LineGTTransform(target, width, height):
    heatmap_h = np.zeros((height, width), dtype=np.float32)
    heatmap_v = np.zeros((height, width), dtype=np.float32)

    for line in target["h_lines"]:
        cv2.line(heatmap_h, line[0], line[1], color=1, thickness=3)
    for line in target["v_lines"]:
        cv2.line(heatmap_v, line[0], line[1], color=1, thickness=3)

    # Optional: Apply Gaussian blur for softer targets
    heatmap_h = cv2.GaussianBlur(heatmap_h, (5, 5), 0)
    heatmap_v = cv2.GaussianBlur(heatmap_v, (5, 5), 0)

    return np.stack([heatmap_h, heatmap_v], axis=-1)
```

### Step 3: Create Line Dataset (`loader.py`)
```python
class Line_Dataset(data.Dataset):
    # Similar to TRACE_Dataset but uses ParserLine and LineGTTransform
```

### Step 4: Update Model Output Channels
In `model.py`, change `output_ch=5` to `output_ch=2` (or 3 if including endpoints).

### Step 5: Create Training Config
```json
{
  "name": "line_segmentation",
  "train_size": 1280,
  "output_ch": 2,
  "thresholds": [0.3, 0.2]
}
```

## Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| Create | `parsers/line_parser.py` | JSON line annotation parser |
| Modify | `loader.py` | Add `LineGTTransform` and `Line_Dataset` |
| Modify | `model.py` | Change `output_ch` default or make configurable |
| Create | `configs/train_line.json` | Training configuration |
| Modify | `train.py` | Add option to select dataset type |

## Alternative: Reuse Current Format

If you prefer to reuse the existing XML format, you could represent lines as degenerate cells (cells with zero area but defined edges). However, the JSON approach is cleaner for pure line detection.

---

# Detailed Implementation Guide

## Task 1: Create Line Annotation Parser

**File**: `parsers/line_parser.py`

```python
# -*- coding: utf-8 -*-
import json
import os
from file_utils import get_files


class ParserLine:
    """Parser for JSON-based line annotations."""

    def __init__(self, rootpath, dataset, phase="train"):
        self.rootpath = rootpath
        self.dataset = dataset
        self.phase = phase

        # Get image and annotation files
        data_path = os.path.join(rootpath, dataset, phase)
        self.img_files = get_files(data_path, image=True)
        self.gt_files = get_files(data_path, extension=".json")
        self.file_idx = 0

    def lenFiles(self):
        return len(self.img_files)

    def parseGT(self):
        """Parse a single annotation file and return line data."""
        if self.file_idx >= len(self.img_files):
            self.file_idx = 0  # Reset for next epoch

        img_file = self.img_files[self.file_idx]
        gt_file = self.gt_files[self.file_idx]
        self.file_idx += 1

        with open(gt_file, "r") as f:
            data = json.load(f)

        h_lines = []  # Horizontal lines
        v_lines = []  # Vertical lines

        for line in data.get("lines", []):
            points = line["points"]
            start = tuple(points[0])
            end = tuple(points[1])

            if line["type"] == "horizontal":
                h_lines.append((start, end))
            elif line["type"] == "vertical":
                v_lines.append((start, end))

        return img_file, {"h_lines": h_lines, "v_lines": v_lines}
```

---

## Task 2: Create Line Mask Generator

**File**: `loader.py` (add new function)

```python
def LineGTTransform(target, width, height, line_thickness=3, use_gaussian=True):
    """
    Generate heatmap masks for horizontal and vertical lines.

    Args:
        target: Dict with 'h_lines' and 'v_lines' lists
        width: Output mask width
        height: Output mask height
        line_thickness: Pixel thickness of drawn lines (default: 3)
        use_gaussian: Apply Gaussian blur for softer targets (default: True)

    Returns:
        heatmap: np.ndarray of shape (height, width, 2)
                 Channel 0 = horizontal lines, Channel 1 = vertical lines
        weight_mask: np.ndarray of shape (height, width)
    """
    height = int(height)
    width = int(width)

    heatmap_h = np.zeros((height, width), dtype=np.float32)
    heatmap_v = np.zeros((height, width), dtype=np.float32)
    weight_mask = np.ones((height, width), dtype=np.float32)

    # Draw horizontal lines
    for start, end in target.get("h_lines", []):
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.line(heatmap_h, start, end, color=1.0, thickness=line_thickness)

    # Draw vertical lines
    for start, end in target.get("v_lines", []):
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.line(heatmap_v, start, end, color=1.0, thickness=line_thickness)

    # Optional: Apply Gaussian blur for smoother gradients
    if use_gaussian:
        heatmap_h = cv2.GaussianBlur(heatmap_h, (5, 5), 0)
        heatmap_v = cv2.GaussianBlur(heatmap_v, (5, 5), 0)
        # Re-normalize to [0, 1]
        if heatmap_h.max() > 0:
            heatmap_h /= heatmap_h.max()
        if heatmap_v.max() > 0:
            heatmap_v /= heatmap_v.max()

    # Stack into 2-channel output
    heatmap = np.stack([heatmap_h, heatmap_v], axis=-1)

    return heatmap, weight_mask
```

---

## Task 3: Create Line Dataset Class

**File**: `loader.py` (add new class)

```python
class Line_Dataset(data.Dataset):
    """Dataset for line segmentation training."""

    def __init__(self, datasets, rootpath, scale_down=2, phase="train",
                 transform=None, line_thickness=3):
        self.rootpath = rootpath
        self.transform = transform
        self.scale_down = scale_down
        self.line_thickness = line_thickness

        # Initialize parsers
        self.parsers = []
        for dataset in datasets.split(","):
            parser = ParserLine(rootpath, dataset, phase)
            self.parsers.append({
                "name": dataset,
                "num": parser.lenFiles(),
                "parser": parser
            })

        self.total_files = sum(p["num"] for p in self.parsers)

    def __len__(self):
        return self.total_files

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        # Select parser (assuming single dataset for simplicity)
        parser = self.parsers[0]["parser"]

        # Load image and annotations
        img_file, gt = parser.parseGT()
        img = imgproc.loadImage(img_file)
        height, width, _ = img.shape

        # Normalize line coordinates to [0, 1] range
        h_lines_norm = []
        for start, end in gt["h_lines"]:
            h_lines_norm.append((
                (start[0] / width, start[1] / height),
                (end[0] / width, end[1] / height)
            ))

        v_lines_norm = []
        for start, end in gt["v_lines"]:
            v_lines_norm.append((
                (start[0] / width, start[1] / height),
                (end[0] / width, end[1] / height)
            ))

        gt_norm = {"h_lines": h_lines_norm, "v_lines": v_lines_norm}

        # Apply augmentation if provided
        if self.transform is not None:
            img, gt_norm = self.transform(img, gt_norm)
            width = height = self.transform.size

        # Generate heatmap at reduced resolution
        out_w = int(width / self.scale_down)
        out_h = int(height / self.scale_down)

        # Denormalize to output size
        h_lines_out = [
            ((int(s[0] * out_w), int(s[1] * out_h)),
             (int(e[0] * out_w), int(e[1] * out_h)))
            for s, e in gt_norm["h_lines"]
        ]
        v_lines_out = [
            ((int(s[0] * out_w), int(s[1] * out_h)),
             (int(e[0] * out_w), int(e[1] * out_h)))
            for s, e in gt_norm["v_lines"]
        ]

        gt_out = {"h_lines": h_lines_out, "v_lines": v_lines_out}
        gt_image, gt_weight = LineGTTransform(
            gt_out, out_w, out_h,
            line_thickness=self.line_thickness
        )

        # Normalize image
        img = imgproc.normalizeMeanVariance(img)

        # Convert to tensors
        img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(gt_image.astype(np.float32))

        # Expand weight to match gt channels
        _, _, gt_ch = gt_image.shape
        gt_weight = np.array([gt_weight] * gt_ch).transpose(1, 2, 0)
        weight_tensor = torch.from_numpy(gt_weight.astype(np.float32))

        return img_tensor, gt_tensor, weight_tensor
```

---

## Task 4: Update Model Output Channels

**File**: `model.py`

Find the `TraceModel` class and change the `output_ch` parameter:

```python
# BEFORE (line ~20)
class TraceModel(nn.Module):
    def __init__(self, output_ch=5, pretrained=True, freeze=False, basenet="resnet50"):

# AFTER
class TraceModel(nn.Module):
    def __init__(self, output_ch=2, pretrained=True, freeze=False, basenet="resnet50"):
```

Or better, make it configurable via command-line:

```python
# In train.py, add argument
parser.add_argument('--output_ch', default=2, type=int, help='Number of output channels')

# Then use:
model = TraceModel(output_ch=args.output_ch, ...)
```

---

## Task 5: Create Training Configuration

**File**: `configs/train_line.json`

```json
{
    "name": "line_segmentation",
    "train_size": 1280,
    "canvas_size": 1280,
    "output_ch": 2,
    "thresholds": [0.3, 0.2],
    "line_thickness": 3,
    "use_gaussian": true
}
```

---

## Task 6: Update Training Script

**File**: `train.py`

Add dataset selection logic:

```python
# Add argument
parser.add_argument('--task', default='table', type=str,
                    choices=['table', 'line'], help='Task type')

# In main():
if args.task == 'line':
    from parsers.line_parser import ParserLine
    dataset = Line_Dataset(
        args.train_sets,
        rootpath=args.data_path,
        phase="train",
        scale_down=scale_down,
        transform=transform,
        line_thickness=config.get("line_thickness", 3)
    )
else:
    dataset = TRACE_Dataset(...)  # existing code
```

---

## Task 7: Prepare Training Data

### Annotation Format

Each image should have a corresponding `.json` file:

```
data/
  line_dataset/
    train/
      image_001.png
      image_001.json
      image_002.png
      image_002.json
      ...
```

### JSON Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["filename", "lines"],
    "properties": {
        "filename": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "points"],
                "properties": {
                    "type": {"enum": ["horizontal", "vertical"]},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "minItems": 2,
                        "maxItems": 2
                    }
                }
            }
        }
    }
}
```

### Example Annotation

```json
{
    "filename": "document_001.png",
    "lines": [
        {"type": "horizontal", "points": [[50, 100], [750, 100]]},
        {"type": "horizontal", "points": [[50, 200], [750, 200]]},
        {"type": "vertical", "points": [[100, 50], [100, 500]]},
        {"type": "vertical", "points": [[400, 50], [400, 500]]}
    ]
}
```

---

## Training Command

```bash
python train.py \
    --task line \
    --config configs/train_line.json \
    --train_sets line_dataset \
    --data_path /path/to/data \
    --output_ch 2 \
    --batch_size 16 \
    --lr 3e-4 \
    --max_iter 50000
```

---

## Post-Processing for Inference

After getting the 2-channel heatmap output, apply line detection:

```python
def extract_lines_from_heatmap(heatmap, threshold=0.3):
    """
    Extract line coordinates from predicted heatmaps.

    Args:
        heatmap: np.ndarray of shape (H, W, 2)
        threshold: Confidence threshold

    Returns:
        h_lines: List of horizontal line segments
        v_lines: List of vertical line segments
    """
    h_map = heatmap[:, :, 0]
    v_map = heatmap[:, :, 1]

    # Threshold
    h_binary = (h_map > threshold).astype(np.uint8)
    v_binary = (v_map > threshold).astype(np.uint8)

    # Use HoughLinesP for line detection
    h_lines = cv2.HoughLinesP(h_binary, 1, np.pi/180, 50,
                               minLineLength=30, maxLineGap=10)
    v_lines = cv2.HoughLinesP(v_binary, 1, np.pi/180, 50,
                               minLineLength=30, maxLineGap=10)

    return h_lines, v_lines
```

---

## Summary: Files to Create/Modify

| Priority | Action | File | Description |
|----------|--------|------|-------------|
| 1 | Create | `parsers/line_parser.py` | JSON annotation parser |
| 2 | Modify | `loader.py` | Add `LineGTTransform()` and `Line_Dataset` class |
| 3 | Modify | `model.py` | Change default `output_ch` from 5 to 2 |
| 4 | Create | `configs/train_line.json` | Training configuration |
| 5 | Modify | `train.py` | Add `--task` argument and dataset selection |
| 6 | Create | Training data | Images + JSON annotations |

---

## Estimated Effort

| Task | Complexity | Notes |
|------|------------|-------|
| Line parser | Low | ~50 lines of code |
| Mask generator | Low | ~30 lines of code |
| Dataset class | Medium | ~80 lines, adapt from TRACE_Dataset |
| Model change | Trivial | 1-line change |
| Training script | Low | ~10 lines added |
| Data preparation | Variable | Depends on annotation tool used |

---

## Questions for Dev Team

1. **Annotation tool**: What tool will be used to annotate line data? (LabelImg, CVAT, custom tool?)
2. **Line types**: Are there other line types beyond horizontal/vertical? (diagonal, curved?)
3. **Inference requirements**: Is real-time inference needed? What hardware target?
4. **Dataset size**: How many training images are available/planned?
