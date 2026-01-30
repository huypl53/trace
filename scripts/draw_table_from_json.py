#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Draw canvas items from JSON files.

This script extracts items (table, label, shape) from canvas JSON and renders
them as images, with precise handling of cell borders (visible only when width > 0).

Supports:
- Tables with solid and dashed border styles
- Labels with text rendering (including Japanese, underline, alignment)
- Shapes (rectangle) with borders and backgrounds
- Merged cells (rowspan/colspan)
- Recursive directory processing
- Line mask generation (horizontal and vertical)
"""

import argparse
import json
import multiprocessing as mp
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Japanese font paths to try (in order of preference)
JAPANESE_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
    "C:/Windows/Fonts/msgothic.ttc",  # Windows
]

JAPANESE_BOLD_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
]

_font_cache = {}


def parse_hex_color(value):
    """Parse hex color string to RGB tuple."""
    if not isinstance(value, str):
        return (0, 0, 0)
    value = value.strip()
    if value == "transparent":
        return None
    if not value.startswith("#"):
        return (0, 0, 0)
    try:
        if len(value) == 7:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
            return (r, g, b)
        elif len(value) == 4:
            r = int(value[1], 16) * 17
            g = int(value[2], 16) * 17
            b = int(value[3], 16) * 17
            return (r, g, b)
    except ValueError:
        pass
    return (0, 0, 0)


def build_sizes(size_map, count, total):
    """Build list of row heights or column widths from map."""
    if count <= 0:
        return []
    size_map = size_map or {}
    if total is None:
        total = sum(float(v) for v in size_map.values()) if size_map else 0.0
    default = (total / count) if total else 0.0
    sizes = []
    for i in range(count):
        key = str(i)
        sizes.append(float(size_map.get(key, default)))
    return sizes


def parse_cell_key(key):
    """Parse cell key 'row-col' to (row, col) tuple."""
    parts = key.split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def draw_dashed_line(draw, start, end, color, width=1, dash_length=4, gap_length=2):
    """Draw a dashed line from start to end."""
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5

    if length == 0:
        return

    ux = dx / length
    uy = dy / length

    pos = 0
    drawing = True

    while pos < length:
        if drawing:
            seg_end = min(pos + dash_length, length)
            sx = x1 + ux * pos
            sy = y1 + uy * pos
            ex = x1 + ux * seg_end
            ey = y1 + uy * seg_end
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos = seg_end + gap_length
        else:
            pos += gap_length
        drawing = not drawing


def draw_styled_line(draw, start, end, color, style, width=1):
    """Draw a line with the specified style (solid or dashed)."""
    if width <= 0:
        return
    if style == "dashed":
        draw_dashed_line(draw, start, end, color, width)
    else:
        draw.line([start, end], fill=color, width=width)


def get_border_width(cell_style, side, default_borders):
    """Get border width for a side, considering defaults.

    Border is visible if:
    - Cell has explicit width > 0, OR
    - Cell has no explicit setting and default_borders.all is True

    Border is hidden if:
    - Cell has explicit width = 0, OR
    - Cell has explicit style = 'none', OR
    - No explicit setting and no defaults
    """
    width_key = f"border{side}Width"
    style_key = f"border{side}Style"

    # Check if cell has explicit border settings
    if width_key in cell_style:
        width = cell_style[width_key]
        # Explicit width = 0 means hidden
        if width == 0 or width == 0.0:
            return 0
        return width

    # Check if style is explicitly 'none'
    if cell_style.get(style_key) == "none":
        return 0

    # Use default if available
    if default_borders and default_borders.get("all"):
        return default_borders.get("width", 1)

    return 0


def collect_cell_borders(x1, y1, x2, y2, cell_style, default_borders=None):
    """Collect border lines from cell style.

    Returns list of line dicts with type, points, thickness.
    Borders are only collected if their width > 0.

    Args:
        default_borders: Table-level cellBorders settings (with 'all', 'width', 'color')
    """
    lines = []
    default_color = default_borders.get("color", "#000000") if default_borders else "#000000"

    # Top border (horizontal)
    top_width = get_border_width(cell_style, "Top", default_borders)
    if top_width and top_width > 0:
        lines.append({
            "type": "horizontal",
            "points": [[int(x1), int(y1)], [int(x2), int(y1)]],
            "thickness": int(top_width),
            "style": cell_style.get("borderTopStyle", "solid"),
            "color": cell_style.get("borderTopColor", default_color),
        })

    # Bottom border (horizontal)
    bottom_width = get_border_width(cell_style, "Bottom", default_borders)
    if bottom_width and bottom_width > 0:
        lines.append({
            "type": "horizontal",
            "points": [[int(x1), int(y2)], [int(x2), int(y2)]],
            "thickness": int(bottom_width),
            "style": cell_style.get("borderBottomStyle", "solid"),
            "color": cell_style.get("borderBottomColor", default_color),
        })

    # Left border (vertical)
    left_width = get_border_width(cell_style, "Left", default_borders)
    if left_width and left_width > 0:
        lines.append({
            "type": "vertical",
            "points": [[int(x1), int(y1)], [int(x1), int(y2)]],
            "thickness": int(left_width),
            "style": cell_style.get("borderLeftStyle", "solid"),
            "color": cell_style.get("borderLeftColor", default_color),
        })

    # Right border (vertical)
    right_width = get_border_width(cell_style, "Right", default_borders)
    if right_width and right_width > 0:
        lines.append({
            "type": "vertical",
            "points": [[int(x2), int(y1)], [int(x2), int(y2)]],
            "thickness": int(right_width),
            "style": cell_style.get("borderRightStyle", "solid"),
            "color": cell_style.get("borderRightColor", default_color),
        })

    return lines


def draw_cell_borders(draw, x1, y1, x2, y2, cell_style, default_borders=None):
    """Draw cell borders based on cell style properties.

    Borders are only drawn if their width > 0.

    Args:
        default_borders: Table-level cellBorders settings (with 'all', 'width', 'color')
    """
    default_color = default_borders.get("color", "#000000") if default_borders else "#000000"

    # Top border
    top_width = get_border_width(cell_style, "Top", default_borders)
    if top_width and top_width > 0:
        top_color = parse_hex_color(cell_style.get("borderTopColor", default_color))
        top_style = cell_style.get("borderTopStyle", "solid")
        if top_color:
            draw_styled_line(draw, (x1, y1), (x2, y1), top_color, top_style, int(top_width))

    # Bottom border
    bottom_width = get_border_width(cell_style, "Bottom", default_borders)
    if bottom_width and bottom_width > 0:
        bottom_color = parse_hex_color(cell_style.get("borderBottomColor", default_color))
        bottom_style = cell_style.get("borderBottomStyle", "solid")
        if bottom_color:
            draw_styled_line(draw, (x1, y2), (x2, y2), bottom_color, bottom_style, int(bottom_width))

    # Left border
    left_width = get_border_width(cell_style, "Left", default_borders)
    if left_width and left_width > 0:
        left_color = parse_hex_color(cell_style.get("borderLeftColor", default_color))
        left_style = cell_style.get("borderLeftStyle", "solid")
        if left_color:
            draw_styled_line(draw, (x1, y1), (x1, y2), left_color, left_style, int(left_width))

    # Right border
    right_width = get_border_width(cell_style, "Right", default_borders)
    if right_width and right_width > 0:
        right_color = parse_hex_color(cell_style.get("borderRightColor", default_color))
        right_style = cell_style.get("borderRightStyle", "solid")
        if right_color:
            draw_styled_line(draw, (x2, y1), (x2, y2), right_color, right_style, int(right_width))


def get_font(font_size, font_weight="normal"):
    """Try to load a Japanese-capable font, fallback to default if not available."""
    cache_key = (font_size, font_weight)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font_paths = JAPANESE_BOLD_FONT_PATHS if font_weight == "bold" else JAPANESE_FONT_PATHS

    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            _font_cache[cache_key] = font
            return font
        except (IOError, OSError):
            continue

    # Fallback to default
    try:
        font = ImageFont.load_default()
        _font_cache[cache_key] = font
        return font
    except Exception:
        _font_cache[cache_key] = None
        return None


def draw_cell_text(draw, x1, y1, x2, y2, cell_data, cell_style, default_font_size):
    """Draw text content in a cell."""
    value = cell_data.get("value", "")
    if not value:
        return

    text = str(value)

    # Get style properties
    font_size = cell_style.get("fontSize") or cell_data.get("properties", {}).get("fontSize") or default_font_size or 11
    font_size = int(font_size)
    font_weight = cell_style.get("fontWeight", "normal")
    text_color = parse_hex_color(cell_style.get("color", "#000000"))
    text_align = cell_style.get("textAlign", "left")

    # Padding (handle None values)
    padding_left = cell_style.get("paddingLeft") or 2
    padding_right = cell_style.get("paddingRight") or 2
    padding_top = cell_style.get("paddingTop") or 2
    padding_bottom = cell_style.get("paddingBottom") or 2

    # Calculate text area
    text_x1 = x1 + padding_left
    text_y1 = y1 + padding_top
    text_x2 = x2 - padding_right
    text_y2 = y2 - padding_bottom

    text_width = text_x2 - text_x1
    text_height = text_y2 - text_y1

    if text_width <= 0 or text_height <= 0:
        return

    font = get_font(font_size, font_weight)

    # Get text bounding box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * font_size * 0.6, font_size

    # Calculate position based on alignment
    if text_align == "center":
        tx = text_x1 + (text_width - tw) / 2
    elif text_align == "right":
        tx = text_x2 - tw
    else:  # left
        tx = text_x1

    # Vertical center
    ty = text_y1 + (text_height - th) / 2

    if text_color:
        draw.text((tx, ty), text, fill=text_color, font=font)


def draw_label(draw, label_item, offset_x=0, offset_y=0):
    """Draw a label item.

    Args:
        draw: PIL ImageDraw object
        label_item: The label item from JSON
        offset_x: X offset for positioning
        offset_y: Y offset for positioning
    """
    props = label_item.get("properties", {})
    text = props.get("text", "")
    if not text:
        return

    x = label_item.get("x", 0) + offset_x
    y = label_item.get("y", 0) + offset_y
    width = label_item.get("width", 0)
    height = label_item.get("height", 0)

    # Get style properties
    font_size = int(props.get("fontSize", 11))
    font_weight = props.get("fontWeight", "normal")
    text_color = parse_hex_color(props.get("color", "#000000"))
    text_align = props.get("textAlign", "left")
    text_decoration = props.get("textDecoration", "none")
    bg_color = props.get("backgroundColor", "transparent")

    # Draw background if not transparent
    if bg_color and bg_color != "transparent":
        bg_rgb = parse_hex_color(bg_color)
        if bg_rgb:
            draw.rectangle([x, y, x + width, y + height], fill=bg_rgb)

    # Get font
    font = get_font(font_size, font_weight)

    # Get text bounding box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * font_size * 0.6, font_size

    # Calculate position based on alignment
    if text_align == "center":
        tx = x + (width - tw) / 2
    elif text_align == "right":
        tx = x + width - tw
    else:  # left
        tx = x

    # Vertical center
    ty = y + (height - th) / 2

    # Draw text
    if text_color:
        draw.text((tx, ty), text, fill=text_color, font=font)

        # Draw underline if specified
        if text_decoration == "underline":
            underline_y = ty + th + 1
            draw.line([(tx, underline_y), (tx + tw, underline_y)], fill=text_color, width=1)


def draw_shape(draw, shape_item, offset_x=0, offset_y=0):
    """Draw a shape item.

    Args:
        draw: PIL ImageDraw object
        shape_item: The shape item from JSON
        offset_x: X offset for positioning
        offset_y: Y offset for positioning

    Returns:
        List of line dicts for mask generation
    """
    props = shape_item.get("properties", {})
    shape_type = props.get("shapeType", "rectangle")

    x = shape_item.get("x", 0) + offset_x
    y = shape_item.get("y", 0) + offset_y
    width = shape_item.get("width", 0)
    height = shape_item.get("height", 0)

    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)

    lines = []

    if shape_type == "rectangle":
        # Draw background
        bg_color = props.get("backgroundColor", "transparent")
        if bg_color and bg_color != "transparent":
            bg_rgb = parse_hex_color(bg_color)
            if bg_rgb:
                draw.rectangle([x1, y1, x2, y2], fill=bg_rgb)

        # Draw border
        border_width = props.get("borderWidth", 0)
        if border_width and border_width > 0:
            border_color = parse_hex_color(props.get("borderColor", "#000000"))
            border_style = props.get("borderStyle", "solid")
            border_width = int(border_width)

            if border_color:
                # Draw all four sides
                draw_styled_line(draw, (x1, y1), (x2, y1), border_color, border_style, border_width)  # top
                draw_styled_line(draw, (x1, y2), (x2, y2), border_color, border_style, border_width)  # bottom
                draw_styled_line(draw, (x1, y1), (x1, y2), border_color, border_style, border_width)  # left
                draw_styled_line(draw, (x2, y1), (x2, y2), border_color, border_style, border_width)  # right

                # Collect lines for mask
                lines.append({
                    "type": "horizontal",
                    "points": [[x1, y1], [x2, y1]],
                    "thickness": border_width,
                })
                lines.append({
                    "type": "horizontal",
                    "points": [[x1, y2], [x2, y2]],
                    "thickness": border_width,
                })
                lines.append({
                    "type": "vertical",
                    "points": [[x1, y1], [x1, y2]],
                    "thickness": border_width,
                })
                lines.append({
                    "type": "vertical",
                    "points": [[x2, y1], [x2, y2]],
                    "thickness": border_width,
                })

    return lines


def draw_line_masks(lines, width, height):
    """Draw horizontal and vertical line masks.

    Returns:
        mask_h: Horizontal lines mask (white lines on black)
        mask_v: Vertical lines mask (white lines on black)
    """
    mask_h = np.zeros((height, width), dtype=np.uint8)
    mask_v = np.zeros((height, width), dtype=np.uint8)

    for line in lines:
        line_type = line.get("type")
        points = line.get("points", [])
        if len(points) != 2:
            continue

        start = tuple(map(int, points[0]))
        end = tuple(map(int, points[1]))
        thickness = line.get("thickness", 1)

        try:
            thickness = int(round(float(thickness)))
        except (TypeError, ValueError):
            thickness = 1
        thickness = max(1, thickness)

        if line_type == "horizontal":
            cv2.line(mask_h, start, end, color=255, thickness=thickness)
        elif line_type == "vertical":
            cv2.line(mask_v, start, end, color=255, thickness=thickness)

    return mask_h, mask_v


def draw_table_on_canvas(draw, table_item, offset_x=0, offset_y=0, collect_lines=True):
    """Draw a table item on an existing canvas.

    Args:
        draw: PIL ImageDraw object
        table_item: The table item from JSON
        offset_x: X offset for positioning
        offset_y: Y offset for positioning
        collect_lines: Whether to collect lines for mask generation

    Returns:
        List of line dicts for mask generation
    """
    props = table_item.get("properties", {})
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))

    if rows <= 0 or cols <= 0:
        return []

    # Get table position and dimensions
    table_x = table_item.get("x", 0) + offset_x
    table_y = table_item.get("y", 0) + offset_y
    table_width = props.get("width") or table_item.get("width")
    table_height = props.get("height") or table_item.get("height")

    # Build row heights and column widths
    row_heights = build_sizes(props.get("rowHeights", {}), rows, table_height)
    col_widths = build_sizes(props.get("columnWidths", {}), cols, table_width)

    # Get cell data and merge info
    cell_data = props.get("cellData", {}) or {}
    merged_cells = props.get("mergedCells", {}) or {}
    hidden_cells = props.get("hiddenCells", {}) or {}
    default_font_size = props.get("fontSize", 11)
    cell_borders = props.get("cellBorders", {}) or {}

    # Calculate cell positions (relative to table position)
    row_positions = [table_y]
    for h in row_heights:
        row_positions.append(row_positions[-1] + h)

    col_positions = [table_x]
    for w in col_widths:
        col_positions.append(col_positions[-1] + w)

    # Collect all lines for mask generation
    all_lines = []

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"

            # Skip hidden cells (part of merged cell)
            if hidden_cells.get(key):
                continue

            # Get cell bounds
            x1 = col_positions[c]
            y1 = row_positions[r]

            # Check for merged cell
            merge_info = merged_cells.get(key, {})
            rowspan = int(merge_info.get("rowspan", 1))
            colspan = int(merge_info.get("colspan", 1))

            end_row = min(r + rowspan, rows)
            end_col = min(c + colspan, cols)

            x2 = col_positions[end_col]
            y2 = row_positions[end_row]

            # Get cell style
            cell = cell_data.get(key, {})
            cell_style = cell.get("cellStyle", {}) or {}

            # Draw cell background if specified
            cell_bg = cell_style.get("backgroundColor")
            if cell_bg and cell_bg != "transparent":
                bg_rgb = parse_hex_color(cell_bg)
                if bg_rgb:
                    draw.rectangle([x1, y1, x2, y2], fill=bg_rgb)

            # Draw text
            draw_cell_text(draw, x1, y1, x2, y2, cell, cell_style, default_font_size)

            # Draw borders (using table-level cellBorders as default)
            draw_cell_borders(draw, x1, y1, x2, y2, cell_style, cell_borders)

            # Collect lines for mask
            if collect_lines:
                cell_lines = collect_cell_borders(x1, y1, x2, y2, cell_style, cell_borders)
                all_lines.extend(cell_lines)

    return all_lines


def draw_canvas(canvas_data, output_path, generate_masks=True, mask_dir=None, padding=10):
    """Draw all items from canvas data to an image file.

    Args:
        canvas_data: The canvas JSON data
        output_path: Path to save the main image
        generate_masks: If True, also generate horizontal and vertical line masks
        mask_dir: Directory to save masks (if None, saves next to image)
        padding: Padding around the canvas

    Returns:
        True if successful, False otherwise
    """
    # Get canvas dimensions
    canvas_width = int(canvas_data.get("canvasWidth", 800))
    canvas_height = int(canvas_data.get("canvasHeight", 600))

    # Add padding
    img_width = canvas_width + padding * 2
    img_height = canvas_height + padding * 2

    # Create image with canvas background
    bg_color_str = canvas_data.get("canvasBackground", "#FFFFFF")
    bg_color = parse_hex_color(bg_color_str) or (255, 255, 255)
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Get all items
    items = canvas_data.get("items", [])
    if not items:
        return False

    # Collect all lines for mask generation
    all_lines = []

    # Draw items in order (tables first for background, then shapes, then labels)
    # Sort by type to ensure proper layering
    tables = [item for item in items if item.get("type") == "table"]
    shapes = [item for item in items if item.get("type") == "shape"]
    labels = [item for item in items if item.get("type") == "label"]

    # Draw tables
    for table in tables:
        table_lines = draw_table_on_canvas(draw, table, offset_x=padding, offset_y=padding, collect_lines=generate_masks)
        all_lines.extend(table_lines)

    # Draw shapes
    for shape in shapes:
        shape_lines = draw_shape(draw, shape, offset_x=padding, offset_y=padding)
        all_lines.extend(shape_lines)

    # Draw labels (on top)
    for label in labels:
        draw_label(draw, label, offset_x=padding, offset_y=padding)

    # Save main image
    img.save(output_path)

    # Generate and save masks if requested
    if generate_masks:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        if mask_dir:
            mask_h_path = os.path.join(mask_dir, f"{base_name}_mask_h.png")
            mask_v_path = os.path.join(mask_dir, f"{base_name}_mask_v.png")
        else:
            base_path = os.path.splitext(output_path)[0]
            mask_h_path = f"{base_path}_mask_h.png"
            mask_v_path = f"{base_path}_mask_v.png"

        mask_h, mask_v = draw_line_masks(all_lines, img_width, img_height)
        cv2.imwrite(mask_h_path, mask_h)
        cv2.imwrite(mask_v_path, mask_v)

    return True


def draw_table(table_item, output_path, generate_masks=True, mask_dir=None):
    """Draw a single table item to an image file (legacy function).

    Args:
        table_item: The table item from JSON
        output_path: Path to save the main image
        generate_masks: If True, also generate horizontal and vertical line masks
        mask_dir: Directory to save masks (if None, saves next to image)

    Returns:
        True if successful, False otherwise
    """
    props = table_item.get("properties", {})
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))

    if rows <= 0 or cols <= 0:
        return False

    # Get table dimensions
    table_width = props.get("width") or table_item.get("width")
    table_height = props.get("height") or table_item.get("height")

    # Build row heights and column widths
    row_heights = build_sizes(props.get("rowHeights", {}), rows, table_height)
    col_widths = build_sizes(props.get("columnWidths", {}), cols, table_width)

    # Calculate actual dimensions
    total_width = int(sum(col_widths)) if col_widths else int(table_width or 100)
    total_height = int(sum(row_heights)) if row_heights else int(table_height or 100)

    # Add padding
    padding = 10
    img_width = total_width + padding * 2
    img_height = total_height + padding * 2

    # Create image
    bg_color = parse_hex_color(props.get("backgroundColor", "#ffffff")) or (255, 255, 255)
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Create a temporary table item at origin for drawing
    temp_table = dict(table_item)
    temp_table["x"] = 0
    temp_table["y"] = 0

    # Draw the table
    all_lines = draw_table_on_canvas(draw, temp_table, offset_x=padding, offset_y=padding, collect_lines=generate_masks)

    # Save main image
    img.save(output_path)

    # Generate and save masks if requested
    if generate_masks:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        if mask_dir:
            mask_h_path = os.path.join(mask_dir, f"{base_name}_mask_h.png")
            mask_v_path = os.path.join(mask_dir, f"{base_name}_mask_v.png")
        else:
            base_path = os.path.splitext(output_path)[0]
            mask_h_path = f"{base_path}_mask_h.png"
            mask_v_path = f"{base_path}_mask_v.png"

        mask_h, mask_v = draw_line_masks(all_lines, img_width, img_height)
        cv2.imwrite(mask_h_path, mask_h)
        cv2.imwrite(mask_v_path, mask_v)

    return True


def list_json_files(input_path, recursive=False):
    """List all JSON files in the input path."""
    if os.path.isfile(input_path):
        return [input_path]
    json_files = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for name in files:
                if name.endswith(".json"):
                    json_files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_path):
            if name.endswith(".json"):
                json_files.append(os.path.join(input_path, name))
    return sorted(json_files)


def process_json_file(json_path, output_dir, generate_masks=True, image_dir=None, mask_dir=None, mode="canvas"):
    """Process a single JSON file and draw items.

    Args:
        json_path: Path to the JSON file
        output_dir: Base output directory (fallback if image_dir not specified)
        generate_masks: Whether to generate mask images
        image_dir: Directory to save images (if None, uses output_dir)
        mask_dir: Directory to save masks (if None, saves next to images)
        mode: "canvas" to draw full canvas with all items, "tables" to draw tables only

    Returns:
        Number of images processed
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {json_path}: {e}")
        return 0

    # Check for items
    items = data.get("items", [])
    if not items:
        return 0

    # Determine base name
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # Determine output directories
    img_out_dir = image_dir or output_dir
    os.makedirs(img_out_dir, exist_ok=True)
    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)

    count = 0

    if mode == "canvas":
        # Draw full canvas with all items (table, label, shape)
        out_name = f"{base_name}.png"
        out_path = os.path.join(img_out_dir, out_name)

        if draw_canvas(data, out_path, generate_masks=generate_masks, mask_dir=mask_dir):
            count = 1
    else:
        # Legacy mode: draw tables only
        tables = [item for item in items if item.get("type") == "table"]
        if not tables:
            return 0

        for i, table in enumerate(tables):
            suffix = f"_table{i + 1}" if len(tables) > 1 else "_table"
            out_name = f"{base_name}{suffix}.png"
            out_path = os.path.join(img_out_dir, out_name)

            if draw_table(table, out_path, generate_masks=generate_masks, mask_dir=mask_dir):
                count += 1

    return count


def _process_wrapper(args_tuple):
    """Wrapper for multiprocessing - unpacks arguments."""
    json_path, output_dir, generate_masks, image_dir, mask_dir, mode = args_tuple
    return process_json_file(json_path, output_dir, generate_masks, image_dir, mask_dir, mode)


def split_files(file_list, ratios, seed=42):
    """Split file list into train/val/test sets."""
    rng = random.Random(seed)
    files = list(file_list)
    rng.shuffle(files)
    n = len(files)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Draw canvas images from JSON files (tables, labels, shapes)"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory or JSON file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for images"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Disable mask generation (masks are generated by default)"
    )
    parser.add_argument(
        "--mode",
        choices=["canvas", "tables"],
        default="canvas",
        help="Drawing mode: 'canvas' draws full canvas with all items, 'tables' draws tables only (default: canvas)"
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)"
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Don't split data, process all to single directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPU cores)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input path not found: {args.input_dir}")

    # Validate split ratios
    if not args.no_split and abs(sum(args.split) - 1.0) > 0.01:
        print(f"Error: Split ratios must sum to 1.0, got {sum(args.split)}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = list_json_files(args.input_dir, args.recursive)

    if not json_files:
        print("No JSON files found.")
        return

    generate_masks = not args.no_masks
    num_workers = args.num_workers or mp.cpu_count()

    print(f"Found {len(json_files)} JSON files")
    print(f"Mode: {args.mode}, workers: {num_workers}, masks={'enabled' if generate_masks else 'disabled'}")

    # Split files or process all together
    if args.no_split:
        splits = {"all": json_files}
    else:
        splits = split_files(json_files, args.split, args.seed)
        print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    total_tables = 0

    for split_name, split_files_list in splits.items():
        if not split_files_list:
            continue

        # Setup output directories
        split_dir = os.path.join(args.output_dir, split_name)
        if generate_masks:
            image_dir = os.path.join(split_dir, "images")
            mask_dir = os.path.join(split_dir, "masks")
        else:
            image_dir = split_dir
            mask_dir = None

        os.makedirs(image_dir, exist_ok=True)
        if mask_dir:
            os.makedirs(mask_dir, exist_ok=True)

        # Prepare work items
        work_items = [
            (json_path, args.output_dir, generate_masks, image_dir, mask_dir, args.mode)
            for json_path in split_files_list
        ]

        # Process in parallel
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_wrapper, work_items),
                total=len(split_files_list),
                desc=f"Processing {split_name}"
            ))
            split_count = sum(results)
            total_tables += split_count

        print(f"  {split_name}: {split_count} images")

    print(f"\nDone. Generated {total_tables} images from {len(json_files)} JSON files.")
    if generate_masks:
        print(f"Also generated {total_tables * 2} mask images (_mask_h.png, _mask_v.png).")


if __name__ == "__main__":
    main()
