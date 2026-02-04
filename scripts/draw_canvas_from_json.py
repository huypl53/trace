#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Draw canvas items from JSON files.

This script extracts items (table, label, shape) from canvas JSON and renders
them as images, with precise handling of cell borders (visible only when width > 0).

Supports:
- Tables with solid and dashed border styles
- Labels with text rendering (including Japanese, alignment)
- Shapes (rectangle) with borders and backgrounds
- Merged cells (rowspan/colspan)
- Recursive directory processing
- Line mask generation (horizontal and vertical)
- Combined mask generation (both horizontal and vertical in one image)

Usage Examples:
    # Basic usage with train/val/test split
    python draw_canvas_from_json.py \\
        --input_dir /path/to/json \\
        --output_dir /path/to/output

    # No split, keep original directory structure
    python draw_canvas_from_json.py \\
        --input_dir /path/to/json \\
        --output_dir /path/to/output \\
        --no_split --keep_structure

    # Generate combined masks (h+v in one image)
    python draw_canvas_from_json.py \\
        --input_dir /path/to/json \\
        --output_dir /path/to/output \\
        --combined_mask_dir /path/to/combined_masks

    # Scale output to 2x size
    python draw_canvas_from_json.py \\
        --input_dir /path/to/json \\
        --output_dir /path/to/output \\
        --scale 2.0

    # Full example with all options
    python draw_canvas_from_json.py \\
        --input_dir /path/to/json \\
        --output_dir /path/to/output \\
        --no_split --keep_structure \\
        --combined_mask_dir /path/to/combined_masks \\
        --mode canvas \\
        --num_workers 8 \\
        --scale 1.5
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
LOCAL_FONT_DIR = os.path.join(REPO_DIR, "fonts")

# Japanese font paths to try (in order of preference)
JAPANESE_FONT_PATHS = [
    os.path.join(LOCAL_FONT_DIR, "NotoSansCJK-Regular.ttc"),
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
    os.path.join(LOCAL_FONT_DIR, "NotoSansCJK-Bold.ttc"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
]

_font_cache = {}


def scale_value(value, scale):
    """Scale a numeric value by the scale factor.

    Args:
        value: The value to scale (can be int, float, or None)
        scale: The scale factor

    Returns:
        Scaled value as int, or 0 if value is None
    """
    if value is None:
        return 0
    return int(round(float(value) * scale))


def scale_font_size(font_size, scale):
    """Scale a font size by the scale factor.

    Args:
        font_size: The font size to scale
        scale: The scale factor

    Returns:
        Scaled font size as int (minimum 1)
    """
    if font_size is None:
        return 11  # default font size
    scaled = int(round(float(font_size) * scale))
    return max(1, scaled)


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


def draw_dashed_line(draw, start, end, color, width=1, dash_length=4, gap_length=2, scale=1.0):
    """Draw a dashed line from start to end.

    Args:
        draw: PIL ImageDraw object
        start: Start point (x, y)
        end: End point (x, y)
        color: Line color
        width: Line width
        dash_length: Length of dash segments
        gap_length: Length of gaps between dashes
        scale: Scale factor for dash/gap lengths
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2) ** 0.5

    if length == 0:
        return

    ux = dx / length
    uy = dy / length

    # Scale dash and gap lengths
    scaled_dash = dash_length * scale
    scaled_gap = gap_length * scale

    pos = 0
    drawing = True

    while pos < length:
        if drawing:
            seg_end = min(pos + scaled_dash, length)
            sx = x1 + ux * pos
            sy = y1 + uy * pos
            ex = x1 + ux * seg_end
            ey = y1 + uy * seg_end
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos = seg_end + scaled_gap
        else:
            pos += scaled_gap
        drawing = not drawing


def draw_styled_line(draw, start, end, color, style, width=1, scale=1.0):
    """Draw a line with the specified style (solid or dashed).

    Args:
        draw: PIL ImageDraw object
        start: Start point (x, y)
        end: End point (x, y)
        color: Line color
        style: Line style ('solid' or 'dashed')
        width: Line width
        scale: Scale factor for dashed line patterns
    """
    if width <= 0:
        return
    if style == "dashed":
        draw_dashed_line(draw, start, end, color, width, scale=scale)
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


def is_white_or_near_white(color):
    """Check if a color is white or very close to white.

    Args:
        color: RGB tuple (r, g, b) or None

    Returns:
        True if the color is white or near-white (brightness > 250)
    """
    if color is None:
        return True
    r, g, b = color
    # Consider colors very close to white (brightness > 250) as effectively white
    # This helps handle slight variations in white color definitions
    brightness = (r + g + b) / 3
    return brightness > 250


def collect_cell_borders(x1, y1, x2, y2, cell_style, default_borders=None):
    """Collect border lines from cell style.

    Returns list of line dicts with type, points, thickness.
    Borders are only collected if their width > 0 and color is not white/near-white.

    Args:
        default_borders: Table-level cellBorders settings (with 'all', 'width', 'color')
    """
    lines = []
    default_color = (
        default_borders.get("color", "#000000") if default_borders else "#000000"
    )

    # Top border (horizontal)
    top_width = get_border_width(cell_style, "Top", default_borders)
    if top_width and top_width > 0:
        top_color_str = cell_style.get("borderTopColor", default_color)
        top_color = parse_hex_color(top_color_str)
        # Skip white/near-white borders
        if not is_white_or_near_white(top_color):
            lines.append(
                {
                    "type": "horizontal",
                    "points": [[int(x1), int(y1)], [int(x2), int(y1)]],
                    "thickness": int(top_width),
                    "style": cell_style.get("borderTopStyle", "solid"),
                    "color": top_color_str,
                }
            )

    # Bottom border (horizontal)
    bottom_width = get_border_width(cell_style, "Bottom", default_borders)
    if bottom_width and bottom_width > 0:
        bottom_color_str = cell_style.get("borderBottomColor", default_color)
        bottom_color = parse_hex_color(bottom_color_str)
        # Skip white/near-white borders
        if not is_white_or_near_white(bottom_color):
            lines.append(
                {
                    "type": "horizontal",
                    "points": [[int(x1), int(y2)], [int(x2), int(y2)]],
                    "thickness": int(bottom_width),
                    "style": cell_style.get("borderBottomStyle", "solid"),
                    "color": bottom_color_str,
                }
            )

    # Left border (vertical)
    left_width = get_border_width(cell_style, "Left", default_borders)
    if left_width and left_width > 0:
        left_color_str = cell_style.get("borderLeftColor", default_color)
        left_color = parse_hex_color(left_color_str)
        # Skip white/near-white borders
        if not is_white_or_near_white(left_color):
            lines.append(
                {
                    "type": "vertical",
                    "points": [[int(x1), int(y1)], [int(x1), int(y2)]],
                    "thickness": int(left_width),
                    "style": cell_style.get("borderLeftStyle", "solid"),
                    "color": left_color_str,
                }
            )

    # Right border (vertical)
    right_width = get_border_width(cell_style, "Right", default_borders)
    if right_width and right_width > 0:
        right_color_str = cell_style.get("borderRightColor", default_color)
        right_color = parse_hex_color(right_color_str)
        # Skip white/near-white borders
        if not is_white_or_near_white(right_color):
            lines.append(
                {
                    "type": "vertical",
                    "points": [[int(x2), int(y1)], [int(x2), int(y2)]],
                    "thickness": int(right_width),
                    "style": cell_style.get("borderRightStyle", "solid"),
                    "color": right_color_str,
                }
            )

    return lines


def draw_cell_borders(draw, x1, y1, x2, y2, cell_style, default_borders=None, scale=1.0):
    """Draw cell borders based on cell style properties.

    Borders are only drawn if their width > 0 and color is not white/near-white.
    White borders are skipped because they blend with the background and can
    deteriorate segmentation model performance.

    Args:
        draw: PIL ImageDraw object
        x1, y1, x2, y2: Cell coordinates (already scaled)
        cell_style: Cell style properties
        default_borders: Table-level cellBorders settings (with 'all', 'width', 'color')
        scale: Scale factor for border widths and dash patterns
    """
    default_color = (
        default_borders.get("color", "#000000") if default_borders else "#000000"
    )

    # Top border
    top_width = get_border_width(cell_style, "Top", default_borders)
    if top_width and top_width > 0:
        top_color = parse_hex_color(cell_style.get("borderTopColor", default_color))
        top_style = cell_style.get("borderTopStyle", "solid")
        scaled_width = scale_value(top_width, scale)
        # Skip white/near-white borders
        if top_color and scaled_width > 0 and not is_white_or_near_white(top_color):
            draw_styled_line(
                draw, (x1, y1), (x2, y1), top_color, top_style, scaled_width, scale=scale
            )

    # Bottom border
    bottom_width = get_border_width(cell_style, "Bottom", default_borders)
    if bottom_width and bottom_width > 0:
        bottom_color = parse_hex_color(
            cell_style.get("borderBottomColor", default_color)
        )
        bottom_style = cell_style.get("borderBottomStyle", "solid")
        scaled_width = scale_value(bottom_width, scale)
        # Skip white/near-white borders
        if bottom_color and scaled_width > 0 and not is_white_or_near_white(bottom_color):
            draw_styled_line(
                draw, (x1, y2), (x2, y2), bottom_color, bottom_style, scaled_width, scale=scale
            )

    # Left border
    left_width = get_border_width(cell_style, "Left", default_borders)
    if left_width and left_width > 0:
        left_color = parse_hex_color(cell_style.get("borderLeftColor", default_color))
        left_style = cell_style.get("borderLeftStyle", "solid")
        scaled_width = scale_value(left_width, scale)
        # Skip white/near-white borders
        if left_color and scaled_width > 0 and not is_white_or_near_white(left_color):
            draw_styled_line(
                draw, (x1, y1), (x1, y2), left_color, left_style, scaled_width, scale=scale
            )

    # Right border
    right_width = get_border_width(cell_style, "Right", default_borders)
    if right_width and right_width > 0:
        right_color = parse_hex_color(cell_style.get("borderRightColor", default_color))
        right_style = cell_style.get("borderRightStyle", "solid")
        scaled_width = scale_value(right_width, scale)
        # Skip white/near-white borders
        if right_color and scaled_width > 0 and not is_white_or_near_white(right_color):
            draw_styled_line(
                draw, (x2, y1), (x2, y2), right_color, right_style, scaled_width, scale=scale
            )


def get_font(font_size, font_weight="normal"):
    """Try to load a Japanese-capable font, fallback to default if not available."""
    cache_key = (font_size, font_weight)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font_paths = (
        JAPANESE_BOLD_FONT_PATHS if font_weight == "bold" else JAPANESE_FONT_PATHS
    )

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


def draw_cell_text(draw, x1, y1, x2, y2, cell_data, cell_style, default_font_size, scale=1.0, image=None):
    """Draw text content in a cell with clipping to cell boundaries.

    Args:
        draw: PIL ImageDraw object
        x1, y1, x2, y2: Cell coordinates (already scaled)
        cell_data: Cell data dict
        cell_style: Cell style properties
        default_font_size: Default font size (unscaled)
        scale: Scale factor
        image: PIL Image object (required for clipping support when text overflows)
    """
    value = cell_data.get("value", "")
    if not value:
        return

    text = str(value)

    # Get style properties (font size needs scaling)
    font_size = (
        cell_style.get("fontSize")
        or cell_data.get("properties", {}).get("fontSize")
        or default_font_size
        or 11
    )
    font_size = scale_font_size(font_size, scale)
    font_weight = cell_style.get("fontWeight", "normal")
    text_color = parse_hex_color(cell_style.get("color", "#000000"))
    text_align = cell_style.get("textAlign", "left")

    # Padding (handle None values) - scale padding
    padding_left = scale_value(cell_style.get("paddingLeft") or 2, scale)
    padding_right = scale_value(cell_style.get("paddingRight") or 2, scale)
    padding_top = scale_value(cell_style.get("paddingTop") or 2, scale)
    padding_bottom = scale_value(cell_style.get("paddingBottom") or 2, scale)

    # Calculate text area (the clipping region)
    text_x1 = int(x1 + padding_left)
    text_y1 = int(y1 + padding_top)
    text_x2 = int(x2 - padding_right)
    text_y2 = int(y2 - padding_bottom)

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
        # Get the offset from textbbox (text may not start at 0,0)
        text_offset_x = bbox[0]
        text_offset_y = bbox[1]
    except Exception:
        tw, th = len(text) * font_size * 0.6, font_size
        text_offset_x = 0
        text_offset_y = 0

    # Calculate position based on alignment (relative to text area)
    if text_align == "center":
        rel_tx = (text_width - tw) / 2
    elif text_align == "right":
        rel_tx = text_width - tw
    else:  # left
        rel_tx = 0

    # Vertical center (relative to text area)
    rel_ty = (text_height - th) / 2

    if not text_color:
        return

    # Check if text exceeds cell boundaries and needs clipping
    text_exceeds_bounds = tw > text_width or th > text_height

    if text_exceeds_bounds and image is not None:
        # Use clipping approach: draw text on a temporary image, then paste clipped region
        # Create a temporary RGBA image for the text (with transparency)
        temp_width = max(1, int(tw) + 2)
        temp_height = max(1, int(th) + 2)
        temp_img = Image.new("RGBA", (temp_width, temp_height), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        # Draw text at origin (adjusted for text offset)
        temp_draw.text((-text_offset_x, -text_offset_y), text, fill=text_color + (255,), font=font)

        # Calculate the source region to crop from temp image
        # This handles the case where text position is negative (e.g., right-aligned overflow)
        src_x1 = max(0, int(-rel_tx))
        src_y1 = max(0, int(-rel_ty))
        src_x2 = min(temp_img.width, int(-rel_tx + text_width))
        src_y2 = min(temp_img.height, int(-rel_ty + text_height))

        if src_x2 > src_x1 and src_y2 > src_y1:
            # Crop the visible portion of the text
            cropped = temp_img.crop((src_x1, src_y1, src_x2, src_y2))

            # Calculate destination position
            dst_x = text_x1 + max(0, int(rel_tx))
            dst_y = text_y1 + max(0, int(rel_ty))

            # Paste onto the main image using alpha channel as mask
            image.paste(cropped, (dst_x, dst_y), cropped)
    else:
        # Text fits within bounds, draw directly (no clipping needed)
        tx = text_x1 + rel_tx
        ty = text_y1 + rel_ty
        draw.text((tx, ty), text, fill=text_color, font=font)


def draw_label_on_canvas(draw, label_item, offset_x=0, offset_y=0, scale=1.0):
    """Draw a label item on a canvas.

    Args:
        draw: PIL ImageDraw object
        label_item: The label item from JSON
        offset_x: X offset for positioning (already scaled)
        offset_y: Y offset for positioning (already scaled)
        scale: Scale factor
    """
    props = label_item.get("properties", {})
    text = props.get("text", "")
    if not text:
        return

    # Scale position and dimensions
    x = scale_value(label_item.get("x", 0), scale) + offset_x
    y = scale_value(label_item.get("y", 0), scale) + offset_y
    width = scale_value(label_item.get("width", 0), scale)
    height = scale_value(label_item.get("height", 0), scale)

    # Get style properties (scale font size)
    font_size = scale_font_size(props.get("fontSize", 11), scale)
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

        # # Draw underline if specified (scale underline width)
        # if text_decoration == "underline":
        #     underline_y = ty + th + scale_value(1, scale)
        #     underline_width = max(1, scale_value(1, scale))
        #     draw.line(
        #         [(tx, underline_y), (tx + tw, underline_y)], fill=text_color, width=underline_width
        #     )


def draw_label(draw, label_item, offset_x=0, offset_y=0):
    """Draw a label item (legacy function without scale).

    Args:
        draw: PIL ImageDraw object
        label_item: The label item from JSON
        offset_x: X offset for positioning
        offset_y: Y offset for positioning
    """
    draw_label_on_canvas(draw, label_item, offset_x, offset_y, scale=1.0)


def draw_shape_on_canvas(draw, shape_item, offset_x=0, offset_y=0, scale=1.0):
    """Draw a shape item on a canvas.

    Args:
        draw: PIL ImageDraw object
        shape_item: The shape item from JSON
        offset_x: X offset for positioning (already scaled)
        offset_y: Y offset for positioning (already scaled)
        scale: Scale factor

    Returns:
        List of line dicts for mask generation
    """
    props = shape_item.get("properties", {})
    shape_type = props.get("shapeType", "rectangle")

    # Scale position and dimensions
    x = scale_value(shape_item.get("x", 0), scale) + offset_x
    y = scale_value(shape_item.get("y", 0), scale) + offset_y
    width = scale_value(shape_item.get("width", 0), scale)
    height = scale_value(shape_item.get("height", 0), scale)

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

        # Draw border (scale border width)
        border_width = props.get("borderWidth", 0)
        if border_width and border_width > 0:
            border_color = parse_hex_color(props.get("borderColor", "#000000"))
            border_style = props.get("borderStyle", "solid")
            scaled_border_width = scale_value(border_width, scale)

            if border_color and scaled_border_width > 0:
                # Draw all four sides
                draw_styled_line(
                    draw, (x1, y1), (x2, y1), border_color, border_style, scaled_border_width, scale=scale
                )  # top
                draw_styled_line(
                    draw, (x1, y2), (x2, y2), border_color, border_style, scaled_border_width, scale=scale
                )  # bottom
                draw_styled_line(
                    draw, (x1, y1), (x1, y2), border_color, border_style, scaled_border_width, scale=scale
                )  # left
                draw_styled_line(
                    draw, (x2, y1), (x2, y2), border_color, border_style, scaled_border_width, scale=scale
                )  # right

                # Collect lines for mask (with scaled thickness)
                lines.append(
                    {
                        "type": "horizontal",
                        "points": [[x1, y1], [x2, y1]],
                        "thickness": scaled_border_width,
                    }
                )
                lines.append(
                    {
                        "type": "horizontal",
                        "points": [[x1, y2], [x2, y2]],
                        "thickness": scaled_border_width,
                    }
                )
                lines.append(
                    {
                        "type": "vertical",
                        "points": [[x1, y1], [x1, y2]],
                        "thickness": scaled_border_width,
                    }
                )
                lines.append(
                    {
                        "type": "vertical",
                        "points": [[x2, y1], [x2, y2]],
                        "thickness": scaled_border_width,
                    }
                )

    return lines


def draw_shape(draw, shape_item, offset_x=0, offset_y=0):
    """Draw a shape item (legacy function without scale).

    Args:
        draw: PIL ImageDraw object
        shape_item: The shape item from JSON
        offset_x: X offset for positioning
        offset_y: Y offset for positioning

    Returns:
        List of line dicts for mask generation
    """
    return draw_shape_on_canvas(draw, shape_item, offset_x, offset_y, scale=1.0)


def draw_line_masks(lines, width, height, combined=False, scale=1.0):
    """Draw horizontal and vertical line masks.

    Args:
        lines: List of line dicts with type, points, thickness (already scaled)
        width: Image width (already scaled)
        height: Image height (already scaled)
        combined: If True, also generate a combined mask with both line types
        scale: Scale factor (for reference, lines should already be scaled)

    Returns:
        mask_h: Horizontal lines mask (white lines on black)
        mask_v: Vertical lines mask (white lines on black)
        mask_combined: Combined mask (if combined=True), otherwise None
    """
    mask_h = np.zeros((height, width), dtype=np.uint8)
    mask_v = np.zeros((height, width), dtype=np.uint8)
    mask_combined = np.zeros((height, width), dtype=np.uint8) if combined else None

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
            if combined:
                cv2.line(mask_combined, start, end, color=255, thickness=thickness)
        elif line_type == "vertical":
            cv2.line(mask_v, start, end, color=255, thickness=thickness)
            if combined:
                cv2.line(mask_combined, start, end, color=255, thickness=thickness)

    return mask_h, mask_v, mask_combined


def draw_table_on_canvas(draw, table_item, offset_x=0, offset_y=0, collect_lines=True, scale=1.0, image=None):
    """Draw a table item on an existing canvas.

    Args:
        draw: PIL ImageDraw object
        table_item: The table item from JSON
        offset_x: X offset for positioning (already scaled)
        offset_y: Y offset for positioning (already scaled)
        collect_lines: Whether to collect lines for mask generation
        scale: Scale factor
        image: PIL Image object (required for text clipping support)

    Returns:
        List of line dicts for mask generation
    """
    props = table_item.get("properties", {})
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))

    # Get header row settings
    header_rows = int(props.get("headerRows", 0))
    header_bg_color = props.get("headerBackgroundColor")

    if rows <= 0 or cols <= 0:
        return []

    # Get table position and dimensions (scale them)
    table_x = scale_value(table_item.get("x", 0), scale) + offset_x
    table_y = scale_value(table_item.get("y", 0), scale) + offset_y
    table_width = props.get("width") or table_item.get("width")
    table_height = props.get("height") or table_item.get("height")

    # Build row heights and column widths (then scale them)
    row_heights = build_sizes(props.get("rowHeights", {}), rows, table_height)
    col_widths = build_sizes(props.get("columnWidths", {}), cols, table_width)

    # Scale row heights and column widths
    row_heights = [h * scale for h in row_heights]
    col_widths = [w * scale for w in col_widths]

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

    # Pre-compute cell bounds for all visible cells
    cell_bounds = {}
    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"
            if hidden_cells.get(key):
                continue

            x1 = int(col_positions[c])
            y1 = int(row_positions[r])

            merge_info = merged_cells.get(key, {})
            rowspan = int(merge_info.get("rowspan", 1))
            colspan = int(merge_info.get("colspan", 1))

            end_row = min(r + rowspan, rows)
            end_col = min(c + colspan, cols)

            x2 = int(col_positions[end_col])
            y2 = int(row_positions[end_row])

            cell_bounds[key] = (x1, y1, x2, y2)

    # Pass 1: Draw all cell backgrounds
    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"
            if key not in cell_bounds:
                continue

            x1, y1, x2, y2 = cell_bounds[key]
            cell = cell_data.get(key, {})
            cell_style = cell.get("cellStyle", {}) or {}

            cell_bg = cell_style.get("backgroundColor")
            if not cell_bg or cell_bg == "transparent":
                if r < header_rows and header_bg_color:
                    cell_bg = header_bg_color
            if cell_bg and cell_bg != "transparent":
                bg_rgb = parse_hex_color(cell_bg)
                if bg_rgb:
                    draw.rectangle([x1, y1, x2, y2], fill=bg_rgb)

    # Pass 2: Draw all cell text
    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"
            if key not in cell_bounds:
                continue

            x1, y1, x2, y2 = cell_bounds[key]
            cell = cell_data.get(key, {})
            cell_style = cell.get("cellStyle", {}) or {}

            draw_cell_text(draw, x1, y1, x2, y2, cell, cell_style, default_font_size, scale=scale, image=image)

    # Pass 3: Draw all cell borders (on top of backgrounds and text)
    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"
            if key not in cell_bounds:
                continue

            x1, y1, x2, y2 = cell_bounds[key]
            cell = cell_data.get(key, {})
            cell_style = cell.get("cellStyle", {}) or {}

            draw_cell_borders(draw, x1, y1, x2, y2, cell_style, cell_borders, scale=scale)

            if collect_lines:
                cell_lines = collect_cell_borders_scaled(
                    x1, y1, x2, y2, cell_style, cell_borders, scale=scale
                )
                all_lines.extend(cell_lines)

    return all_lines


def collect_cell_borders_scaled(x1, y1, x2, y2, cell_style, default_borders=None, scale=1.0):
    """Collect border lines from cell style with scaled thickness.

    Returns list of line dicts with type, points, thickness (scaled).
    Borders are only collected if their width > 0 and color is not white/near-white.

    Args:
        x1, y1, x2, y2: Cell coordinates (already scaled)
        cell_style: Cell style properties
        default_borders: Table-level cellBorders settings (with 'all', 'width', 'color')
        scale: Scale factor for thickness
    """
    lines = []
    default_color = (
        default_borders.get("color", "#000000") if default_borders else "#000000"
    )

    # Top border (horizontal)
    top_width = get_border_width(cell_style, "Top", default_borders)
    if top_width and top_width > 0:
        top_color_str = cell_style.get("borderTopColor", default_color)
        top_color = parse_hex_color(top_color_str)
        scaled_width = scale_value(top_width, scale)
        # Skip white/near-white borders
        if scaled_width > 0 and not is_white_or_near_white(top_color):
            lines.append(
                {
                    "type": "horizontal",
                    "points": [[int(x1), int(y1)], [int(x2), int(y1)]],
                    "thickness": scaled_width,
                    "style": cell_style.get("borderTopStyle", "solid"),
                    "color": top_color_str,
                }
            )

    # Bottom border (horizontal)
    bottom_width = get_border_width(cell_style, "Bottom", default_borders)
    if bottom_width and bottom_width > 0:
        bottom_color_str = cell_style.get("borderBottomColor", default_color)
        bottom_color = parse_hex_color(bottom_color_str)
        scaled_width = scale_value(bottom_width, scale)
        # Skip white/near-white borders
        if scaled_width > 0 and not is_white_or_near_white(bottom_color):
            lines.append(
                {
                    "type": "horizontal",
                    "points": [[int(x1), int(y2)], [int(x2), int(y2)]],
                    "thickness": scaled_width,
                    "style": cell_style.get("borderBottomStyle", "solid"),
                    "color": bottom_color_str,
                }
            )

    # Left border (vertical)
    left_width = get_border_width(cell_style, "Left", default_borders)
    if left_width and left_width > 0:
        left_color_str = cell_style.get("borderLeftColor", default_color)
        left_color = parse_hex_color(left_color_str)
        scaled_width = scale_value(left_width, scale)
        # Skip white/near-white borders
        if scaled_width > 0 and not is_white_or_near_white(left_color):
            lines.append(
                {
                    "type": "vertical",
                    "points": [[int(x1), int(y1)], [int(x1), int(y2)]],
                    "thickness": scaled_width,
                    "style": cell_style.get("borderLeftStyle", "solid"),
                    "color": left_color_str,
                }
            )

    # Right border (vertical)
    right_width = get_border_width(cell_style, "Right", default_borders)
    if right_width and right_width > 0:
        right_color_str = cell_style.get("borderRightColor", default_color)
        right_color = parse_hex_color(right_color_str)
        scaled_width = scale_value(right_width, scale)
        # Skip white/near-white borders
        if scaled_width > 0 and not is_white_or_near_white(right_color):
            lines.append(
                {
                    "type": "vertical",
                    "points": [[int(x2), int(y1)], [int(x2), int(y2)]],
                    "thickness": scaled_width,
                    "style": cell_style.get("borderRightStyle", "solid"),
                    "color": right_color_str,
                }
            )

    return lines


def draw_canvas(
    canvas_data,
    output_path,
    generate_masks=True,
    mask_dir=None,
    padding=10,
    combined_mask_dir=None,
    scale=1.0,
):
    """Draw all items from canvas data to an image file.

    Args:
        canvas_data: The canvas JSON data
        output_path: Path to save the main image
        generate_masks: If True, also generate horizontal and vertical line masks
        mask_dir: Directory to save masks (if None, saves next to image)
        padding: Padding around the canvas (will be scaled)
        combined_mask_dir: Directory to save combined masks (both H and V lines)
        scale: Scale factor for the entire output (default 1.0)

    Returns:
        True if successful, False otherwise
    """
    # Get canvas dimensions and scale them
    canvas_width = scale_value(canvas_data.get("canvasWidth", 800), scale)
    canvas_height = scale_value(canvas_data.get("canvasHeight", 600), scale)

    # Scale padding
    scaled_padding = scale_value(padding, scale)

    # Add scaled padding
    img_width = canvas_width + scaled_padding * 2
    img_height = canvas_height + scaled_padding * 2

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

    # Draw tables (pass scaled padding as offset, scale factor for internal scaling, image for clipping)
    for table in tables:
        table_lines = draw_table_on_canvas(
            draw,
            table,
            offset_x=scaled_padding,
            offset_y=scaled_padding,
            collect_lines=generate_masks,
            scale=scale,
            image=img,
        )
        all_lines.extend(table_lines)

    # Draw shapes (with scale)
    for shape in shapes:
        shape_lines = draw_shape_on_canvas(
            draw, shape, offset_x=scaled_padding, offset_y=scaled_padding, scale=scale
        )
        all_lines.extend(shape_lines)

    # Draw labels (on top, with scale)
    for label in labels:
        draw_label_on_canvas(
            draw, label, offset_x=scaled_padding, offset_y=scaled_padding, scale=scale
        )

    # Save main image
    img.save(output_path)

    # Generate and save masks if requested
    if generate_masks or combined_mask_dir:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        generate_combined = combined_mask_dir is not None
        mask_h, mask_v, mask_combined = draw_line_masks(
            all_lines, img_width, img_height, combined=generate_combined, scale=scale
        )

        if generate_masks:
            if mask_dir:
                mask_h_path = os.path.join(mask_dir, f"{base_name}_mask_h.png")
                mask_v_path = os.path.join(mask_dir, f"{base_name}_mask_v.png")
            else:
                base_path = os.path.splitext(output_path)[0]
                mask_h_path = f"{base_path}_mask_h.png"
                mask_v_path = f"{base_path}_mask_v.png"

            cv2.imwrite(mask_h_path, mask_h)
            cv2.imwrite(mask_v_path, mask_v)

        if combined_mask_dir and mask_combined is not None:
            mask_combined_path = os.path.join(
                combined_mask_dir, f"{base_name}_mask_combined.png"
            )
            cv2.imwrite(mask_combined_path, mask_combined)

    return True


def draw_table(table_item, output_path, generate_masks=True, mask_dir=None, combined_mask_dir=None, scale=1.0):
    """Draw a single table item to an image file (legacy function).

    Args:
        table_item: The table item from JSON
        output_path: Path to save the main image
        generate_masks: If True, also generate horizontal and vertical line masks
        mask_dir: Directory to save masks (if None, saves next to image)
        combined_mask_dir: Directory to save combined masks (both H and V lines)
        scale: Scale factor for the entire output (default 1.0)

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

    # Calculate actual dimensions and scale them
    total_width = int(sum(col_widths)) if col_widths else int(table_width or 100)
    total_height = int(sum(row_heights)) if row_heights else int(table_height or 100)

    # Scale dimensions
    total_width = scale_value(total_width, scale)
    total_height = scale_value(total_height, scale)

    # Add scaled padding
    padding = 10
    scaled_padding = scale_value(padding, scale)
    img_width = total_width + scaled_padding * 2
    img_height = total_height + scaled_padding * 2

    # Create image
    bg_color = parse_hex_color(props.get("backgroundColor", "#ffffff")) or (
        255,
        255,
        255,
    )
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Create a temporary table item at origin for drawing
    temp_table = dict(table_item)
    temp_table["x"] = 0
    temp_table["y"] = 0

    # Draw the table with scale and image for text clipping
    all_lines = draw_table_on_canvas(
        draw,
        temp_table,
        offset_x=scaled_padding,
        offset_y=scaled_padding,
        collect_lines=generate_masks,
        scale=scale,
        image=img,
    )

    # Save main image
    img.save(output_path)

    # Generate and save masks if requested
    if generate_masks or combined_mask_dir:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        generate_combined = combined_mask_dir is not None
        mask_h, mask_v, mask_combined = draw_line_masks(
            all_lines, img_width, img_height, combined=generate_combined, scale=scale
        )

        if generate_masks:
            if mask_dir:
                mask_h_path = os.path.join(mask_dir, f"{base_name}_mask_h.png")
                mask_v_path = os.path.join(mask_dir, f"{base_name}_mask_v.png")
            else:
                base_path = os.path.splitext(output_path)[0]
                mask_h_path = f"{base_path}_mask_h.png"
                mask_v_path = f"{base_path}_mask_v.png"

            cv2.imwrite(mask_h_path, mask_h)
            cv2.imwrite(mask_v_path, mask_v)

        if combined_mask_dir and mask_combined is not None:
            mask_combined_path = os.path.join(
                combined_mask_dir, f"{base_name}_mask_combined.png"
            )
            cv2.imwrite(mask_combined_path, mask_combined)

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


def process_json_file(
    json_path,
    output_dir,
    generate_masks=True,
    image_dir=None,
    mask_dir=None,
    mode="canvas",
    combined_mask_dir=None,
    scale=1.0,
):
    """Process a single JSON file and draw items.

    Args:
        json_path: Path to the JSON file
        output_dir: Base output directory (fallback if image_dir not specified)
        generate_masks: Whether to generate mask images
        image_dir: Directory to save images (if None, uses output_dir)
        mask_dir: Directory to save masks (if None, saves next to images)
        mode: "canvas" to draw full canvas with all items, "tables" to draw tables only
        combined_mask_dir: Directory to save combined masks (both H and V lines)
        scale: Scale factor for the entire output (default 1.0)

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

        if draw_canvas(
            data,
            out_path,
            generate_masks=generate_masks,
            mask_dir=mask_dir,
            combined_mask_dir=combined_mask_dir,
            scale=scale,
        ):
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

            if draw_table(
                table,
                out_path,
                generate_masks=generate_masks,
                mask_dir=mask_dir,
                combined_mask_dir=combined_mask_dir,
                scale=scale,
            ):
                count += 1

    return count


def _process_wrapper(args_tuple):
    """Wrapper for multiprocessing - unpacks arguments."""
    (
        json_path,
        output_dir,
        generate_masks,
        image_dir,
        mask_dir,
        mode,
        combined_mask_dir,
        scale,
    ) = args_tuple
    return process_json_file(
        json_path,
        output_dir,
        generate_masks,
        image_dir,
        mask_dir,
        mode,
        combined_mask_dir,
        scale,
    )


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
        "--input_dir", required=True, help="Input directory or JSON file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for images"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Process subdirectories recursively",
    )
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Disable mask generation (masks are generated by default)",
    )
    parser.add_argument(
        "--combined_mask_dir",
        type=str,
        default=None,
        help="Directory to save combined masks (both horizontal and vertical lines in one image)",
    )
    parser.add_argument(
        "--mode",
        choices=["canvas", "tables"],
        default="canvas",
        help="Drawing mode: 'canvas' draws full canvas with all items, 'tables' draws tables only (default: canvas)",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Don't split data, process all to single directory",
    )
    parser.add_argument(
        "--keep_structure",
        action="store_true",
        help="Preserve input directory structure in output (use with --no_split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for the entire output (default: 1.0). Scales canvas size, positions, dimensions, font sizes, and line thicknesses.",
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
    print(
        f"Mode: {args.mode}, workers: {num_workers}, masks={'enabled' if generate_masks else 'disabled'}, scale={args.scale}"
    )

    # Split files or process all together
    if args.no_split:
        splits = {"all": json_files}
    else:
        splits = split_files(json_files, args.split, args.seed)
        print(
            f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    total_tables = 0

    # Handle --keep_structure mode (preserves input directory structure)
    if args.keep_structure and args.no_split:
        # Create base output directories
        base_image_dir = os.path.join(args.output_dir, "images")
        base_mask_dir = (
            os.path.join(args.output_dir, "masks") if generate_masks else None
        )
        os.makedirs(base_image_dir, exist_ok=True)
        if base_mask_dir:
            os.makedirs(base_mask_dir, exist_ok=True)

        # Prepare work items with relative path structure preserved
        work_items = []
        input_dir_abs = os.path.abspath(args.input_dir)
        for json_path in json_files:
            # Compute relative path from input_dir
            json_abs = os.path.abspath(json_path)
            rel_path = os.path.relpath(os.path.dirname(json_abs), input_dir_abs)
            if rel_path == ".":
                rel_path = ""

            # Create output subdirectories
            if rel_path:
                image_dir = os.path.join(base_image_dir, rel_path)
                mask_dir = (
                    os.path.join(base_mask_dir, rel_path) if base_mask_dir else None
                )
            else:
                image_dir = base_image_dir
                mask_dir = base_mask_dir

            os.makedirs(image_dir, exist_ok=True)
            if mask_dir:
                os.makedirs(mask_dir, exist_ok=True)

            # Setup combined mask directory if specified
            combined_mask_dir = None
            if args.combined_mask_dir:
                if rel_path:
                    combined_mask_dir = os.path.join(args.combined_mask_dir, rel_path)
                else:
                    combined_mask_dir = args.combined_mask_dir
                os.makedirs(combined_mask_dir, exist_ok=True)

            work_items.append(
                (
                    json_path,
                    args.output_dir,
                    generate_masks,
                    image_dir,
                    mask_dir,
                    args.mode,
                    combined_mask_dir,
                    args.scale,
                )
            )

        # Process in parallel
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_wrapper, work_items),
                    total=len(json_files),
                    desc="Processing (keep_structure)",
                )
            )
            total_tables = sum(results)

        print(f"  Total: {total_tables} images")
    else:
        # Original behavior: flat directory structure per split
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

            # Setup combined mask directory if specified
            combined_mask_dir = None
            if args.combined_mask_dir:
                combined_mask_dir = os.path.join(args.combined_mask_dir, split_name)
                os.makedirs(combined_mask_dir, exist_ok=True)

            # Prepare work items
            work_items = [
                (
                    json_path,
                    args.output_dir,
                    generate_masks,
                    image_dir,
                    mask_dir,
                    args.mode,
                    combined_mask_dir,
                    args.scale,
                )
                for json_path in split_files_list
            ]

            # Process in parallel
            with mp.Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_process_wrapper, work_items),
                        total=len(split_files_list),
                        desc=f"Processing {split_name}",
                    )
                )
                split_count = sum(results)
                total_tables += split_count

            print(f"  {split_name}: {split_count} images")

    print(f"\nDone. Generated {total_tables} images from {len(json_files)} JSON files.")
    if generate_masks:
        print(
            f"Also generated {total_tables * 2} mask images (_mask_h.png, _mask_v.png)."
        )
    if args.combined_mask_dir:
        print(
            f"Also generated {total_tables} combined mask images (_mask_combined.png)."
        )


if __name__ == "__main__":
    main()
