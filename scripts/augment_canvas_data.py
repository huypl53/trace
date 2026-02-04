#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Augment canvas JSON data by modifying layout, styles, and positions.

Supported Item Types:
- Tables: row/column sizes, merges, cell text, borders, colors
- Text: content, colors, font size
- Images: position and size (via canvas scaling)

Augmentations include:
- Row/column size changes (table)
- Random merges (rowspan/colspan)
- Cell/text mutations
- Color jitter
- Canvas scaling and position jitter
- Border removal (for negative examples)

Usage Examples:

    # Basic table augmentation
    uv run python -m scripts.augment_canvas_data \\
        --input_dir data/raw_canvas \\
        --output_dir data/augmented \\
        --num_aug 5

    # With canvas size and position augmentation
    uv run python -m scripts.augment_canvas_data \\
        --input_dir data/raw_canvas \\
        --output_dir data/augmented \\
        --num_aug 5 \\
        --canvas_scale_prob 0.5 \\
        --canvas_scale_min 0.7 \\
        --canvas_scale_max 1.3 \\
        --jitter_prob 0.8 \\
        --max_jitter 100

    # With font size scaling for text items
    uv run python -m scripts.augment_canvas_data \\
        --input_dir data/raw_canvas \\
        --output_dir data/augmented \\
        --num_aug 3 \\
        --font_scale_prob 0.5 \\
        --font_scale_min 0.7 \\
        --font_scale_max 1.3

    # Negative examples (remove borders, add contrast)
    uv run python -m scripts.augment_canvas_data \\
        --input_dir data/raw_canvas \\
        --output_dir data/augmented \\
        --num_aug 5 \\
        --remove_outer_borders_prob 0.3 \\
        --remove_internal_borders_prob 0.2 \\
        --bg_contrast_prob 0.2

    # Full augmentation (all options)
    uv run python -m scripts.augment_canvas_data \
        --input_dir data/raw_canvas \
        --output_dir data/augmented \
        --prefix hard_case \
        --num_aug 5 \
        --recursive \
        --size_prob 0.8 \
        --row_scale_min 0.7 \
        --row_scale_max 1.3 \
        --col_scale_min 0.7 \
        --col_scale_max 1.3 \
        --merge_prob 0.2 \
        --text_prob 0.3 \
        --color_prob 0.1 \
        --canvas_scale_prob 0.5 \
        --jitter_prob 0.8 \
        --max_jitter 100 \
        --font_scale_prob 0.5 \
        --remove_outer_borders_prob 0.2 \
        --bg_contrast_prob 0.15 \
        --add_labels_prob 0.8 \
        --add_labels_prob 0.8 \
        --label_above_prob 0.7 \
        --label_below_prob 0.7 \
        --label_min_padding 5 \
        --reduce_cols_prob 0.8 \
        --reduce_rows_prob 0.8 \
        --min_cell_area 2 \
        --max_row_remove_ratio 0.7 \
        --max_col_remove_ratio 0.5 \


"""

import argparse
import copy
import json
import math
import os
import random
import string

from tqdm import tqdm

DIGITS = string.digits
ASCII_POOL = string.ascii_letters
JAPANESE_RANGES = [
    (0x3041, 0x3096),  # Hiragana
    (0x30A1, 0x30FF),  # Katakana
    (0x4E00, 0x9FAF),  # CJK Unified Ideographs (subset)
]


def parse_hex_color(value):
    if not isinstance(value, str):
        return None
    if len(value) != 7 or not value.startswith("#"):
        return None
    try:
        r = int(value[1:3], 16)
        g = int(value[3:5], 16)
        b = int(value[5:7], 16)
    except ValueError:
        return None
    return r, g, b


def format_hex_color(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def jitter_color(value, rng, jitter):
    rgb = parse_hex_color(value)
    if rgb is None:
        return value
    r = max(0, min(255, rgb[0] + rng.randint(-jitter, jitter)))
    g = max(0, min(255, rgb[1] + rng.randint(-jitter, jitter)))
    b = max(0, min(255, rgb[2] + rng.randint(-jitter, jitter)))
    return format_hex_color((r, g, b))


def normalize_weights(weights):
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return [1.0 / len(weights) for _ in weights]
    return [max(0.0, w) / total for w in weights]


def random_japanese_char(rng):
    start, end = rng.choice(JAPANESE_RANGES)
    return chr(rng.randint(start, end))


def random_text(
    rng, min_len=1, max_len=8, jp_ratio=0.7, digit_ratio=0.2, ascii_ratio=0.1
):
    length = rng.randint(min_len, max_len)
    jp_ratio, digit_ratio, ascii_ratio = normalize_weights(
        [jp_ratio, digit_ratio, ascii_ratio]
    )
    chars = []
    for _ in range(length):
        roll = rng.random()
        if roll < jp_ratio:
            chars.append(random_japanese_char(rng))
        elif roll < jp_ratio + digit_ratio:
            chars.append(rng.choice(DIGITS))
        else:
            chars.append(rng.choice(ASCII_POOL))
    return "".join(chars)


def mutate_text(
    value,
    rng,
    replace_prob,
    append_prob,
    truncate_prob,
    char_prob,
    fill_empty,
    max_len,
    jp_ratio,
    digit_ratio,
    ascii_ratio,
):
    text = "" if value is None else str(value)
    if not text:
        if not fill_empty:
            return text
        return random_text(rng, 1, max_len, jp_ratio, digit_ratio, ascii_ratio)

    if rng.random() < replace_prob:
        return random_text(rng, 1, max_len, jp_ratio, digit_ratio, ascii_ratio)

    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isdigit() and rng.random() < char_prob:
            chars[i] = rng.choice(DIGITS)
        elif ch.isascii() and ch.isalpha() and rng.random() < char_prob:
            chars[i] = rng.choice(ASCII_POOL)
        elif not ch.isascii() and rng.random() < char_prob:
            chars[i] = random_japanese_char(rng)
    text = "".join(chars)

    if rng.random() < append_prob:
        text += random_text(rng, 1, 3, jp_ratio, digit_ratio, ascii_ratio)
    if rng.random() < truncate_prob and len(text) > 1:
        text = text[: rng.randint(1, len(text))]

    if len(text) > max_len:
        text = text[:max_len]
    return text


def build_sizes(size_map, count, total):
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


def rescale_sizes(sizes, target_total):
    if not sizes or target_total is None:
        return sizes
    total = sum(sizes)
    if total <= 0:
        return sizes
    scale = float(target_total) / total
    scaled = [max(1, int(round(size * scale))) for size in sizes]
    diff = int(round(target_total)) - sum(scaled)
    if scaled:
        scaled[-1] = max(1, scaled[-1] + diff)
    return scaled


def mutate_sizes(sizes, rng, scale_min, scale_max):
    return [max(1.0, size * rng.uniform(scale_min, scale_max)) for size in sizes]


def parse_cell_key(key):
    parts = key.split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def build_occupancy(rows, cols, merged_cells, hidden_cells):
    occ = [[False for _ in range(cols)] for _ in range(rows)]
    for key, info in merged_cells.items():
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        rowspan = int(info.get("rowspan", 1))
        colspan = int(info.get("colspan", 1))
        for rr in range(r, min(rows, r + rowspan)):
            for cc in range(c, min(cols, c + colspan)):
                occ[rr][cc] = True
    for key, hidden in hidden_cells.items():
        if not hidden:
            continue
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        if 0 <= r < rows and 0 <= c < cols:
            occ[r][c] = True
    return occ


def add_random_merges(
    rows,
    cols,
    merged_cells,
    hidden_cells,
    rng,
    merge_prob,
    max_merges,
    max_rowspan,
    max_colspan,
):
    occ = build_occupancy(rows, cols, merged_cells, hidden_cells)
    merges_added = 0
    attempts = rows * cols * 3 if rows and cols else 0

    for _ in range(attempts):
        if merges_added >= max_merges:
            break
        if rng.random() > merge_prob:
            continue
        r = rng.randrange(rows)
        c = rng.randrange(cols)
        if occ[r][c]:
            continue
        rowspan = rng.randint(1, max_rowspan)
        colspan = rng.randint(1, max_colspan)
        if rowspan == 1 and colspan == 1:
            continue
        if r + rowspan > rows or c + colspan > cols:
            continue
        valid = True
        for rr in range(r, r + rowspan):
            for cc in range(c, c + colspan):
                if occ[rr][cc]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        merged_cells[f"{r}-{c}"] = {"rowspan": rowspan, "colspan": colspan}
        occ[r][c] = True
        for rr in range(r, r + rowspan):
            for cc in range(c, c + colspan):
                if rr == r and cc == c:
                    continue
                hidden_cells[f"{rr}-{cc}"] = True
                occ[rr][cc] = True
        merges_added += 1


def maybe_jitter_colors(target, rng, prob, jitter):
    if not isinstance(target, dict):
        return
    for key, value in list(target.items()):
        if key == "color" or key.endswith("Color"):
            if rng.random() < prob:
                target[key] = jitter_color(value, rng, jitter)


def remove_outer_borders(rows, cols, cell_data, hidden_cells, rng, prob):
    """Remove outer table borders (top row top, bottom row bottom, etc.)

    This helps train models to not assume table boundaries are always visible.
    """
    if rng.random() > prob:
        return

    for c in range(cols):
        # Remove top border of first row
        key = f"0-{c}"
        if not hidden_cells.get(key):
            cell = cell_data.setdefault(key, {})
            style = cell.setdefault("cellStyle", {})
            style["borderTopWidth"] = 0

        # Remove bottom border of last row
        key = f"{rows-1}-{c}"
        if not hidden_cells.get(key):
            cell = cell_data.setdefault(key, {})
            style = cell.setdefault("cellStyle", {})
            style["borderBottomWidth"] = 0

    for r in range(rows):
        # Remove left border of first column
        key = f"{r}-0"
        if not hidden_cells.get(key):
            cell = cell_data.setdefault(key, {})
            style = cell.setdefault("cellStyle", {})
            style["borderLeftWidth"] = 0

        # Remove right border of last column
        key = f"{r}-{cols-1}"
        if not hidden_cells.get(key):
            cell = cell_data.setdefault(key, {})
            style = cell.setdefault("cellStyle", {})
            style["borderRightWidth"] = 0


def remove_random_borders(rows, cols, cell_data, hidden_cells, rng, prob):
    """Randomly remove internal borders between cells.

    This creates cases where adjacent cells have no border between them.
    """
    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"
            if hidden_cells.get(key):
                continue
            cell = cell_data.setdefault(key, {})
            style = cell.setdefault("cellStyle", {})

            # Randomly remove each border
            for border in ["Top", "Bottom", "Left", "Right"]:
                if rng.random() < prob:
                    style[f"border{border}Width"] = 0


# Default background colors for contrast augmentation
DEFAULT_BG_COLORS = [
    "#ffffff",  # white
    "#f0f0f0",  # light gray
    "#e8f7f0",  # light green
    "#f0f0e8",  # light yellow
    "#e8e8f7",  # light blue
    "#f7e8e8",  # light red
    "#f5f5dc",  # beige
    "#e0ffff",  # light cyan
]


def add_bg_contrast_without_border(rows, cols, cell_data, hidden_cells, rng, prob, colors=None):
    """Add contrasting backgrounds to adjacent cells without shared borders.

    This helps train models to not confuse color boundaries with borders.
    """
    if rng.random() > prob:
        return

    if colors is None:
        colors = DEFAULT_BG_COLORS

    if len(colors) < 2:
        return

    # Pick random adjacent cell pairs (horizontal)
    num_pairs = max(1, rows * cols // 4)
    for _ in range(num_pairs):
        if cols < 2:
            break
        r = rng.randrange(rows)
        c = rng.randrange(cols - 1)
        key1, key2 = f"{r}-{c}", f"{r}-{c+1}"

        if hidden_cells.get(key1) or hidden_cells.get(key2):
            continue

        # Set different backgrounds
        cell1 = cell_data.setdefault(key1, {})
        cell2 = cell_data.setdefault(key2, {})
        style1 = cell1.setdefault("cellStyle", {})
        style2 = cell2.setdefault("cellStyle", {})

        bg1, bg2 = rng.sample(colors, 2)
        style1["backgroundColor"] = bg1
        style2["backgroundColor"] = bg2

        # Remove shared border
        style1["borderRightWidth"] = 0
        style2["borderLeftWidth"] = 0

    # Pick random adjacent cell pairs (vertical)
    for _ in range(num_pairs):
        if rows < 2:
            break
        r = rng.randrange(rows - 1)
        c = rng.randrange(cols)
        key1, key2 = f"{r}-{c}", f"{r+1}-{c}"

        if hidden_cells.get(key1) or hidden_cells.get(key2):
            continue

        # Set different backgrounds
        cell1 = cell_data.setdefault(key1, {})
        cell2 = cell_data.setdefault(key2, {})
        style1 = cell1.setdefault("cellStyle", {})
        style2 = cell2.setdefault("cellStyle", {})

        bg1, bg2 = rng.sample(colors, 2)
        style1["backgroundColor"] = bg1
        style2["backgroundColor"] = bg2

        # Remove shared border
        style1["borderBottomWidth"] = 0
        style2["borderTopWidth"] = 0


def add_dense_text(rows, cols, cell_data, row_heights, col_widths, hidden_cells, rng, prob):
    """Add dense text that fills most of the cell.

    This helps train models to not confuse text patterns with borders.
    """
    for r in range(rows):
        for c in range(cols):
            if rng.random() > prob:
                continue

            key = f"{r}-{c}"
            if hidden_cells.get(key):
                continue

            cell_h = row_heights.get(str(r), 20)
            cell_w = col_widths.get(str(c), 100)

            # Calculate how many chars to fill the cell
            font_size = max(8, min(int(cell_h) - 2, 14))
            char_width = font_size * 0.6
            chars_fit = max(1, int(cell_w / char_width))

            cell = cell_data.setdefault(key, {})
            cell["value"] = random_text(rng, max(1, chars_fit - 1), chars_fit)
            style = cell.setdefault("cellStyle", {})
            style["fontSize"] = font_size
            style["paddingLeft"] = 1
            style["paddingRight"] = 1
            style["paddingTop"] = 1
            style["paddingBottom"] = 1


def add_narrow_columns(cols, col_widths, rng, prob, min_width=15, max_width=25):
    """Make some columns very narrow (mimics vertical text columns).

    This helps train models to not confuse narrow text columns with vertical lines.
    """
    for c in range(cols):
        if rng.random() < prob:
            col_widths[str(c)] = rng.randint(min_width, max_width)


def reduce_rows(props, rng, min_cell_area, max_remove_ratio):
    """Remove random rows from the table.

    Returns the new row count after removal.
    """
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 1))

    # Calculate min_rows based on min_cell_area constraint: rows * cols >= min_cell_area
    # min_rows = ceil(min_cell_area / cols)
    min_rows = max(1, -(-min_cell_area // cols))  # Ceiling division

    if rows <= min_rows:
        return rows

    max_remove = max(0, int(rows * max_remove_ratio))
    max_remove = min(max_remove, rows - min_rows)
    if max_remove <= 0:
        return rows

    num_remove = rng.randint(1, max_remove)

    # Select rows to remove (avoid first row if it's a header)
    removable = list(range(1, rows)) if rows > 1 else []
    if len(removable) < num_remove:
        return rows

    rows_to_remove = set(rng.sample(removable, num_remove))
    new_rows = rows - num_remove

    # Rebuild row heights
    row_heights = props.get("rowHeights", {})
    new_row_heights = {}
    new_idx = 0
    for r in range(rows):
        if r not in rows_to_remove:
            if str(r) in row_heights:
                new_row_heights[str(new_idx)] = row_heights[str(r)]
            new_idx += 1
    props["rowHeights"] = new_row_heights

    # Update height to match new rowHeights sum
    if new_row_heights:
        new_height = sum(float(v) for v in new_row_heights.values())
        props["height"] = int(round(new_height))
        # Also update item height if accessible (it's passed through props reference)

    # Rebuild cell data, merged cells, hidden cells
    cols = int(props.get("columns", 0))
    cell_data = props.get("cellData", {}) or {}
    merged_cells = props.get("mergedCells", {}) or {}
    hidden_cells = props.get("hiddenCells", {}) or {}

    # Create row mapping (old -> new)
    row_map = {}
    new_idx = 0
    for r in range(rows):
        if r not in rows_to_remove:
            row_map[r] = new_idx
            new_idx += 1

    # Rebuild cell data - only keep cells in rows that weren't removed
    new_cell_data = {}
    for key, cell in cell_data.items():
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        if r in row_map:
            new_key = f"{row_map[r]}-{c}"
            new_cell_data[new_key] = cell
    props["cellData"] = new_cell_data

    # Clean up merged cells - remove any that are affected by row removal
    # A merged cell is affected if:
    # 1. Its anchor row is removed
    # 2. Any row in its span is removed
    new_merged = {}
    for key, info in merged_cells.items():
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        # Skip if anchor row is removed
        if r not in row_map:
            continue
        rowspan = int(info.get("rowspan", 1))
        colspan = int(info.get("colspan", 1))
        # Check if any row in the span is removed
        span_rows = set(range(r, r + rowspan))
        if span_rows & rows_to_remove:
            # Affected by row removal - skip this merge
            # The cells will be regular cells now
            continue
        # Safe to keep the merge
        new_merged[f"{row_map[r]}-{c}"] = info
    props["mergedCells"] = new_merged

    # Clean up hidden cells - only keep those in rows that weren't removed
    new_hidden = {}
    for key, hidden in hidden_cells.items():
        if not hidden:
            continue
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        if r in row_map:
            new_hidden[f"{row_map[r]}-{c}"] = True
    props["hiddenCells"] = new_hidden

    props["rows"] = new_rows
    return new_rows


def reduce_columns(props, rng, min_cell_area, max_remove_ratio):
    """Remove random columns from the table.

    Returns the new column count after removal.
    """
    rows = int(props.get("rows", 1))
    cols = int(props.get("columns", 0))

    # Calculate min_cols based on min_cell_area constraint: rows * cols >= min_cell_area
    # min_cols = ceil(min_cell_area / rows)
    min_cols = max(1, -(-min_cell_area // rows))  # Ceiling division

    if cols <= min_cols:
        return cols

    max_remove = max(0, int(cols * max_remove_ratio))
    max_remove = min(max_remove, cols - min_cols)
    if max_remove <= 0:
        return cols

    num_remove = rng.randint(1, max_remove)

    # Select columns to remove (avoid first column often used for labels)
    removable = list(range(1, cols)) if cols > 1 else []
    if len(removable) < num_remove:
        return cols

    cols_to_remove = set(rng.sample(removable, num_remove))
    new_cols = cols - num_remove

    # Rebuild column widths
    col_widths = props.get("columnWidths", {})
    new_col_widths = {}
    new_idx = 0
    for c in range(cols):
        if c not in cols_to_remove:
            if str(c) in col_widths:
                new_col_widths[str(new_idx)] = col_widths[str(c)]
            new_idx += 1
    props["columnWidths"] = new_col_widths

    # Update width to match new columnWidths sum
    if new_col_widths:
        new_width = sum(float(v) for v in new_col_widths.values())
        props["width"] = int(round(new_width))

    # Create column mapping (old -> new)
    rows = int(props.get("rows", 0))
    col_map = {}
    new_idx = 0
    for c in range(cols):
        if c not in cols_to_remove:
            col_map[c] = new_idx
            new_idx += 1

    # Rebuild cell data - only keep cells in columns that weren't removed
    cell_data = props.get("cellData", {}) or {}
    new_cell_data = {}
    for key, cell in cell_data.items():
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        if c in col_map:
            new_key = f"{r}-{col_map[c]}"
            new_cell_data[new_key] = cell
    props["cellData"] = new_cell_data

    # Clean up merged cells - remove any that are affected by column removal
    # A merged cell is affected if:
    # 1. Its anchor column is removed
    # 2. Any column in its span is removed
    merged_cells = props.get("mergedCells", {}) or {}
    new_merged = {}
    for key, info in merged_cells.items():
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        # Skip if anchor column is removed
        if c not in col_map:
            continue
        colspan = int(info.get("colspan", 1))
        rowspan = int(info.get("rowspan", 1))
        # Check if any column in the span is removed
        span_cols = set(range(c, c + colspan))
        if span_cols & cols_to_remove:
            # Affected by column removal - skip this merge
            # The cells will be regular cells now
            continue
        # Safe to keep the merge
        new_merged[f"{r}-{col_map[c]}"] = info
    props["mergedCells"] = new_merged

    # Clean up hidden cells - only keep those in columns that weren't removed
    hidden_cells = props.get("hiddenCells", {}) or {}
    new_hidden = {}
    for key, hidden in hidden_cells.items():
        if not hidden:
            continue
        parsed = parse_cell_key(key)
        if parsed is None:
            continue
        r, c = parsed
        if c in col_map:
            new_hidden[f"{r}-{col_map[c]}"] = True
    props["hiddenCells"] = new_hidden

    props["columns"] = new_cols
    return new_cols


def scale_table_size(item, props, rng, scale_min, scale_max):
    """Scale the entire table by a random factor.

    This changes both the table dimensions and all row/column sizes proportionally.
    """
    scale = rng.uniform(scale_min, scale_max)

    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))

    # Scale row heights
    row_heights = props.get("rowHeights", {})
    for key in row_heights:
        row_heights[key] = max(1, int(round(float(row_heights[key]) * scale)))
    props["rowHeights"] = row_heights

    # Scale column widths
    col_widths = props.get("columnWidths", {})
    for key in col_widths:
        col_widths[key] = max(1, int(round(float(col_widths[key]) * scale)))
    props["columnWidths"] = col_widths

    # Update table dimensions from actual sums (not by scaling independently)
    # This avoids rounding mismatches between sum of parts and total
    if row_heights:
        new_height = sum(float(v) for v in row_heights.values())
        props["height"] = int(round(new_height))
        item["height"] = float(props["height"])
    elif "height" in props:
        props["height"] = int(round(float(props["height"]) * scale))
        item["height"] = float(props["height"])
    elif "height" in item:
        item["height"] = float(item["height"]) * scale

    if col_widths:
        new_width = sum(float(v) for v in col_widths.values())
        props["width"] = int(round(new_width))
        item["width"] = float(props["width"])
    elif "width" in props:
        props["width"] = int(round(float(props["width"]) * scale))
        item["width"] = float(props["width"])
    elif "width" in item:
        item["width"] = float(item["width"]) * scale

    # Scale cell font sizes and padding
    cell_data = props.get("cellData", {}) or {}
    for cell in cell_data.values():
        cell_style = cell.get("cellStyle", {})
        if "fontSize" in cell_style:
            cell_style["fontSize"] = max(6, int(round(float(cell_style["fontSize"]) * scale)))
        for pad_key in ["paddingLeft", "paddingRight", "paddingTop", "paddingBottom"]:
            if pad_key in cell_style and cell_style[pad_key] is not None:
                cell_style[pad_key] = max(0, int(round(float(cell_style[pad_key]) * scale)))
    props["cellData"] = cell_data

    return scale


def augment_table(item, rng, args):
    if item.get("type") != "table":
        return
    props = item.get("properties", {})
    had_row_heights = bool(props.get("rowHeights"))
    had_col_widths = bool(props.get("columnWidths"))
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))
    if had_row_heights:
        rows = len(props.get("rowHeights", {}))
        props["rows"] = rows
    if had_col_widths:
        cols = len(props.get("columnWidths", {}))
        props["columns"] = cols
    if rows <= 0 or cols <= 0:
        if not props.get("rowHeights"):
            props.pop("rows", None)
        if not props.get("columnWidths"):
            props.pop("columns", None)
        item["properties"] = props
        return

    # Reduce rows/columns first (before other augmentations)
    min_cell_area = getattr(args, "min_cell_area", 2)

    if getattr(args, "reduce_rows_prob", 0) > 0 and rng.random() < args.reduce_rows_prob:
        rows = reduce_rows(
            props, rng,
            min_cell_area,
            getattr(args, "max_row_remove_ratio", 0.5)
        )

    if getattr(args, "reduce_cols_prob", 0) > 0 and rng.random() < args.reduce_cols_prob:
        cols = reduce_columns(
            props, rng,
            min_cell_area,
            getattr(args, "max_col_remove_ratio", 0.5)
        )

    # CRITICAL: Re-read rows/cols from props after reduction
    # The reduce_* functions update props["rows"] and props["columns"] directly,
    # but we need to sync our local variables with these updated values.
    rows = int(props.get("rows", rows))
    cols = int(props.get("columns", cols))

    # CRITICAL: Store original table dimensions BEFORE any scaling
    # These are used to calculate defaults if rowHeights/columnWidths are empty
    _orig_table_w = props.get("width", item.get("width", 0))
    _orig_table_h = props.get("height", item.get("height", 0))
    _orig_table_w = float(_orig_table_w) if _orig_table_w else 0.0
    _orig_table_h = float(_orig_table_h) if _orig_table_h else 0.0

    # CRITICAL: After reduction, ensure rowHeights and columnWidths are populated
    # and have entries for ALL rows/columns (not sparse)
    # Also ensure at least 1 row and 1 column
    row_heights = props.get("rowHeights", {})
    col_widths = props.get("columnWidths", {})

    # Determine actual row/column counts from rowHeights/columnWidths if they exist
    # This ensures rows/columns always match the length of rowHeights/columnWidths
    if had_row_heights:
        if row_heights:
            # Use length of rowHeights as source of truth
            rows = len(row_heights)
        # Ensure at least 1 row
        if rows < 1:
            rows = 1
        # Ensure rowHeights has exactly `rows` entries
        if len(row_heights) != rows:
            if row_heights:
                avg_h = sum(float(v) for v in row_heights.values()) / len(row_heights)
            else:
                avg_h = _orig_table_h / rows if _orig_table_h > 0 else 20.0
            new_row_heights = {}
            for i in range(rows):
                key = str(i)
                if key in row_heights:
                    new_row_heights[key] = row_heights[key]
                else:
                    new_row_heights[key] = avg_h
            props["rowHeights"] = new_row_heights
            row_heights = new_row_heights
        props["rows"] = rows

    if had_col_widths:
        if col_widths:
            # Use length of columnWidths as source of truth
            cols = len(col_widths)
        # Ensure at least 1 column
        if cols < 1:
            cols = 1
        # Ensure columnWidths has exactly `cols` entries
        if len(col_widths) != cols:
            if col_widths:
                avg_w = sum(float(v) for v in col_widths.values()) / len(col_widths)
            else:
                avg_w = _orig_table_w / cols if _orig_table_w > 0 else 100.0
            new_col_widths = {}
            for i in range(cols):
                key = str(i)
                if key in col_widths:
                    new_col_widths[key] = col_widths[key]
                else:
                    new_col_widths[key] = avg_w
            props["columnWidths"] = new_col_widths
            col_widths = new_col_widths
        props["columns"] = cols

    if not had_row_heights and not props.get("rowHeights"):
        props.pop("rowHeights", None)
    if not had_col_widths and not props.get("columnWidths"):
        props.pop("columnWidths", None)

    # Scale entire table size
    if getattr(args, "table_scale_prob", 0) > 0 and rng.random() < args.table_scale_prob:
        scale_table_size(
            item, props, rng,
            getattr(args, "table_scale_min", 0.5),
            getattr(args, "table_scale_max", 1.5)
        )

    if rng.random() < args.size_prob:
        orig_prop_w = props.get("width")
        orig_prop_h = props.get("height")
        orig_item_w = item.get("width")
        orig_item_h = item.get("height")

        target_w = orig_prop_w or orig_item_w
        target_h = orig_prop_h or orig_item_h

        row_sizes = build_sizes(props.get("rowHeights", {}), rows, target_h)
        col_sizes = build_sizes(props.get("columnWidths", {}), cols, target_w)

        row_sizes = mutate_sizes(row_sizes, rng, args.row_scale_min, args.row_scale_max)
        col_sizes = mutate_sizes(col_sizes, rng, args.col_scale_min, args.col_scale_max)

        if args.preserve_table_size and target_h:
            row_sizes = rescale_sizes(row_sizes, target_h)
        if args.preserve_table_size and target_w:
            col_sizes = rescale_sizes(col_sizes, target_w)

        row_sizes = [int(max(1, round(size))) for size in row_sizes]
        col_sizes = [int(max(1, round(size))) for size in col_sizes]

        props["rowHeights"] = {str(i): row_sizes[i] for i in range(rows)}
        props["columnWidths"] = {str(i): col_sizes[i] for i in range(cols)}

        new_w = sum(col_sizes)
        new_h = sum(row_sizes)
        props["width"] = int(round(new_w))
        props["height"] = int(round(new_h))

        # Always update item size to match new table dimensions
        # Calculate any offset between item and prop dimensions (e.g., padding/borders)
        delta_w = 0
        delta_h = 0
        if orig_item_w is not None and orig_prop_w is not None:
            delta_w = float(orig_item_w) - float(orig_prop_w)
        if orig_item_h is not None and orig_prop_h is not None:
            delta_h = float(orig_item_h) - float(orig_prop_h)

        item["width"] = float(new_w + delta_w)
        item["height"] = float(new_h + delta_h)

    merged_cells = props.get("mergedCells", {}) or {}
    hidden_cells = props.get("hiddenCells", {}) or {}
    if args.reset_merges:
        merged_cells = {}
        hidden_cells = {}
    if args.merge_prob > 0 and args.max_merges_per_table > 0:
        add_random_merges(
            rows,
            cols,
            merged_cells,
            hidden_cells,
            rng,
            args.merge_prob,
            args.max_merges_per_table,
            args.max_rowspan,
            args.max_colspan,
        )
    props["mergedCells"] = merged_cells
    props["hiddenCells"] = hidden_cells

    if args.text_prob > 0:
        cell_data = props.get("cellData", {}) or {}
        for key, cell in cell_data.items():
            if rng.random() < args.text_prob:
                cell["value"] = mutate_text(
                    cell.get("value"),
                    rng,
                    args.text_replace_prob,
                    args.text_append_prob,
                    args.text_truncate_prob,
                    args.text_char_prob,
                    args.fill_empty_text,
                    args.text_max_len,
                    args.text_jp_ratio,
                    args.text_digit_ratio,
                    args.text_ascii_ratio,
                )
        props["cellData"] = cell_data

    if args.color_prob > 0:
        maybe_jitter_colors(props, rng, args.color_prob, args.color_jitter)
        maybe_jitter_colors(
            props.get("cellBorders", {}), rng, args.color_prob, args.color_jitter
        )

        cell_data = props.get("cellData", {}) or {}
        for cell in cell_data.values():
            if "cellStyle" in cell:
                maybe_jitter_colors(
                    cell["cellStyle"], rng, args.color_prob, args.color_jitter
                )
            if "style" in cell:
                maybe_jitter_colors(
                    cell["style"], rng, args.color_prob, args.color_jitter
                )
        props["cellData"] = cell_data

    if args.show_all_borders:
        cell_data = props.get("cellData", {}) or {}
        hidden_cells = props.get("hiddenCells", {}) or {}
        for r in range(rows):
            for c in range(cols):
                key = f"{r}-{c}"
                if hidden_cells.get(key):
                    continue
                cell = cell_data.get(key, {}) or {}
                cell_style = cell.get("cellStyle", {}) or {}
                cell_style["borderTopWidth"] = 1
                cell_style["borderBottomWidth"] = 1
                cell_style["borderLeftWidth"] = 1
                cell_style["borderRightWidth"] = 1
                cell["cellStyle"] = cell_style
                cell_data[key] = cell
        props["cellData"] = cell_data

    # New augmentations for negative examples (model confusion cases)
    cell_data = props.get("cellData", {}) or {}
    hidden_cells = props.get("hiddenCells", {}) or {}
    row_heights = props.get("rowHeights", {})
    col_widths = props.get("columnWidths", {})

    # Remove outer table borders
    if getattr(args, "remove_outer_borders_prob", 0) > 0:
        remove_outer_borders(rows, cols, cell_data, hidden_cells, rng, args.remove_outer_borders_prob)

    # Remove random internal borders
    if getattr(args, "remove_internal_borders_prob", 0) > 0:
        remove_random_borders(rows, cols, cell_data, hidden_cells, rng, args.remove_internal_borders_prob)

    # Add background contrast without borders
    if getattr(args, "bg_contrast_prob", 0) > 0:
        add_bg_contrast_without_border(rows, cols, cell_data, hidden_cells, rng, args.bg_contrast_prob)

    # Add dense text in cells
    if getattr(args, "dense_text_prob", 0) > 0:
        add_dense_text(rows, cols, cell_data, row_heights, col_widths, hidden_cells, rng, args.dense_text_prob)

    # Add narrow columns
    if getattr(args, "narrow_col_prob", 0) > 0:
        add_narrow_columns(cols, col_widths, rng, args.narrow_col_prob)
        props["columnWidths"] = col_widths

    props["cellData"] = cell_data

    # ALWAYS sync table dimensions to match actual row/column sizes
    # This ensures consistency even when size_prob wasn't triggered
    # or when row/col reduction happened
    row_heights_map = props.get("rowHeights", {}) or {}
    col_widths_map = props.get("columnWidths", {}) or {}
    has_row_heights = bool(row_heights_map)
    has_col_widths = bool(col_widths_map)

    # Get original table dimensions as reference (from props or item)
    # Use the _orig values captured earlier (before scaling) for proper defaults
    orig_table_w = _orig_table_w
    orig_table_h = _orig_table_h

    # If no explicit row/column sizes, don't force rows/columns into output
    if not has_row_heights and not has_col_widths:
        props.pop("rows", None)
        props.pop("columns", None)
        if "width" in props:
            item["width"] = float(props["width"])
        if "height" in props:
            item["height"] = float(props["height"])
        item["properties"] = props
        return

    if has_row_heights:
        rows = len(row_heights_map)
        # Ensure at least 1 row - if empty, create a default entry
        if rows < 1:
            rows = 1
            default_h = orig_table_h if orig_table_h > 0 else 20.0
            row_heights_map = {"0": default_h}
            props["rowHeights"] = row_heights_map
        props["rows"] = rows
    else:
        props.pop("rows", None)

    if has_col_widths:
        cols = len(col_widths_map)
        # Ensure at least 1 column - if empty, create a default entry
        if cols < 1:
            cols = 1
            default_w = orig_table_w if orig_table_w > 0 else 100.0
            col_widths_map = {"0": default_w}
            props["columnWidths"] = col_widths_map
        props["columns"] = cols
    else:
        props.pop("columns", None)

    if has_row_heights:
        final_h = sum(float(v) for v in row_heights_map.values())
        if final_h <= 0:
            fallback_h = props.get("height", orig_table_h) or item.get("height", 0)
            final_h = float(fallback_h) if fallback_h else 20.0
        final_h = int(round(final_h))
        props["height"] = final_h
        item["height"] = float(final_h)
    elif "height" in props:
        item["height"] = float(props["height"])

    if has_col_widths:
        final_w = sum(float(v) for v in col_widths_map.values())
        if final_w <= 0:
            fallback_w = props.get("width", orig_table_w) or item.get("width", 0)
            final_w = float(fallback_w) if fallback_w else 100.0
        final_w = int(round(final_w))
        props["width"] = final_w
        item["width"] = float(final_w)
    elif "width" in props:
        item["width"] = float(props["width"])

    item["properties"] = props


def get_item_bbox(item):
    """Get bounding box (x, y, width, height) for an item."""
    x = float(item.get("x", 0))
    y = float(item.get("y", 0))
    w = float(item.get("width", 0))
    h = float(item.get("height", 0))

    # For tables, also check properties for size
    if item.get("type") == "table":
        props = item.get("properties", {})
        if "width" in props:
            w = max(w, float(props["width"]))
        if "height" in props:
            h = max(h, float(props["height"]))

    return x, y, w, h


def bboxes_overlap(bbox1, bbox2, margin=0):
    """Check if two bounding boxes overlap.

    Args:
        bbox1: (x, y, w, h) tuple
        bbox2: (x, y, w, h) tuple
        margin: extra margin to add around boxes
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Add margin
    x1 -= margin
    y1 -= margin
    w1 += 2 * margin
    h1 += 2 * margin

    # Check overlap
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    return True


def find_valid_translation(item, all_items, orig_bbox, rng, max_offset, canvas_width, canvas_height, margin=5, min_edge_padding=5):
    """Find a valid translation offset that doesn't cause overlap.

    Strategy: Try random offsets within the allowed range, check for overlaps.
    The table can move into the space freed by size reduction.

    Args:
        item: The item to translate
        all_items: All items in the canvas
        orig_bbox: Original bounding box before any augmentation (x, y, w, h)
        rng: Random number generator
        max_offset: Maximum offset in pixels (or ratio of original size)
        canvas_width: Canvas width limit
        canvas_height: Canvas height limit
        margin: Minimum margin between items
        min_edge_padding: Minimum padding from canvas edges

    Returns:
        (dx, dy) translation offset, or (0, 0) if no valid position found
    """
    curr_bbox = get_item_bbox(item)
    curr_x, curr_y, curr_w, curr_h = curr_bbox
    orig_x, orig_y, orig_w, orig_h = orig_bbox

    # Calculate how much space was freed by size reduction
    freed_w = max(0, orig_w - curr_w)
    freed_h = max(0, orig_h - curr_h)

    # Max offset is the freed space plus some additional movement
    max_dx = freed_w + max_offset
    max_dy = freed_h + max_offset

    # Also allow negative movement (but limited)
    min_dx = -max_offset
    min_dy = -max_offset

    # Get bboxes of all other items
    other_bboxes = []
    for other in all_items:
        if other is item:
            continue
        other_bboxes.append(get_item_bbox(other))

    # Try random positions
    max_attempts = 50
    for _ in range(max_attempts):
        dx = rng.uniform(min_dx, max_dx)
        dy = rng.uniform(min_dy, max_dy)

        new_x = curr_x + dx
        new_y = curr_y + dy

        # Check canvas bounds with min edge padding
        if new_x < min_edge_padding or new_y < min_edge_padding:
            continue
        if canvas_width and new_x + curr_w > canvas_width - min_edge_padding:
            continue
        if canvas_height and new_y + curr_h > canvas_height - min_edge_padding:
            continue

        # Check overlap with other items
        new_bbox = (new_x, new_y, curr_w, curr_h)
        overlap = False
        for other_bbox in other_bboxes:
            if bboxes_overlap(new_bbox, other_bbox, margin):
                overlap = True
                break

        if not overlap:
            return dx, dy

    # No valid position found, return no movement
    return 0, 0


def jitter_item_positions(data, rng, jitter_prob, max_jitter, min_padding=5):
    """Randomly jitter all item positions without causing overlaps.

    Args:
        data: Canvas data with items
        rng: Random number generator
        jitter_prob: Probability to jitter each item
        max_jitter: Maximum jitter in pixels
        min_padding: Minimum padding from canvas edges
    """
    if jitter_prob <= 0:
        return

    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)
    items = data.get("items", [])

    # Collect current bboxes of all items
    current_bboxes = [get_item_bbox(item) for item in items]

    for idx, item in enumerate(items):
        if rng.random() >= jitter_prob:
            continue

        orig_bbox = current_bboxes[idx]
        orig_x, orig_y, orig_w, orig_h = orig_bbox

        # Try random positions until we find a valid one
        max_attempts = 20
        for _ in range(max_attempts):
            dx = rng.uniform(-max_jitter, max_jitter)
            dy = rng.uniform(-max_jitter, max_jitter)
            new_x = orig_x + dx
            new_y = orig_y + dy

            # Check canvas bounds with padding
            if new_x < min_padding or new_y < min_padding:
                continue
            if canvas_width > 0 and new_x + orig_w > canvas_width - min_padding:
                continue
            if canvas_height > 0 and new_y + orig_h > canvas_height - min_padding:
                continue

            # Check overlap with other items
            new_bbox = (new_x, new_y, orig_w, orig_h)
            overlap = False
            for other_idx, other_bbox in enumerate(current_bboxes):
                if other_idx == idx:
                    continue
                if bboxes_overlap(new_bbox, other_bbox, margin=0):
                    overlap = True
                    break

            if not overlap:
                # Valid position found
                item["x"] = new_x
                item["y"] = new_y
                current_bboxes[idx] = new_bbox
                break


def randomize_canvas_size(data, rng, scale_prob, scale_min, scale_max):
    """Randomly scale canvas size and adjust item positions proportionally.

    Args:
        data: Canvas data with items
        rng: Random number generator
        scale_prob: Probability to scale canvas
        scale_min: Minimum scale factor
        scale_max: Maximum scale factor
    """
    if scale_prob <= 0 or rng.random() > scale_prob:
        return

    orig_width = data.get("canvasWidth", 0)
    orig_height = data.get("canvasHeight", 0)
    if orig_width <= 0 or orig_height <= 0:
        return

    scale = rng.uniform(scale_min, scale_max)
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))

    # Scale all item positions and sizes
    for item in data.get("items", []):
        for key in ["x", "y", "width", "height"]:
            if key in item:
                item[key] = float(item[key]) * scale

        # Also scale table properties
        if item.get("type") == "table":
            props = item.get("properties", {})
            for key in ["width", "height"]:
                if key in props:
                    props[key] = float(props[key]) * scale

            # Scale row heights and column widths
            if "rowHeights" in props:
                for k, v in props["rowHeights"].items():
                    props["rowHeights"][k] = float(v) * scale
            if "columnWidths" in props:
                for k, v in props["columnWidths"].items():
                    props["columnWidths"][k] = float(v) * scale

            item["properties"] = props

    data["canvasWidth"] = new_width
    data["canvasHeight"] = new_height


def translate_tables(data, orig_bboxes, rng, args):
    """Translate tables to new positions without overlapping other items.

    Args:
        data: Canvas data with items
        orig_bboxes: Dict mapping item id to original bbox before augmentation
        rng: Random number generator
        args: Command line arguments
    """
    translate_prob = getattr(args, "translate_prob", 0)
    if translate_prob <= 0:
        return

    max_offset = getattr(args, "translate_max_offset", 50)
    margin = getattr(args, "translate_margin", 5)
    min_edge_padding = 5  # Minimum padding from canvas edges

    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)

    items = data.get("items", [])

    # Track current bboxes of all items (updates after each move)
    current_bboxes = {}
    for item in items:
        item_id = item.get("id")
        if item_id:
            current_bboxes[item_id] = get_item_bbox(item)

    for item in items:
        if item.get("type") != "table":
            continue

        if rng.random() > translate_prob:
            continue

        item_id = item.get("id")
        if not item_id:
            continue

        orig_bbox = orig_bboxes.get(item_id)
        if orig_bbox is None:
            # Use current bbox if original not available
            orig_bbox = current_bboxes.get(item_id)
        if orig_bbox is None:
            continue

        # Build list of current bboxes for other items
        other_bboxes = []
        for other in items:
            if other is item:
                continue
            other_id = other.get("id")
            if other_id and other_id in current_bboxes:
                other_bboxes.append(current_bboxes[other_id])

        # Find valid translation using current positions
        curr_bbox = current_bboxes.get(item_id, get_item_bbox(item))
        curr_x, curr_y, curr_w, curr_h = curr_bbox
        orig_x, orig_y, orig_w, orig_h = orig_bbox

        # Calculate how much space was freed by size reduction
        freed_w = max(0, orig_w - curr_w)
        freed_h = max(0, orig_h - curr_h)

        max_dx = freed_w + max_offset
        max_dy = freed_h + max_offset
        min_dx = -max_offset
        min_dy = -max_offset

        # Try random positions
        max_attempts = 50
        for _ in range(max_attempts):
            dx = rng.uniform(min_dx, max_dx)
            dy = rng.uniform(min_dy, max_dy)
            new_x = curr_x + dx
            new_y = curr_y + dy

            # Check canvas bounds with min edge padding
            if new_x < min_edge_padding or new_y < min_edge_padding:
                continue
            if canvas_width and new_x + curr_w > canvas_width - min_edge_padding:
                continue
            if canvas_height and new_y + curr_h > canvas_height - min_edge_padding:
                continue

            # Check overlap with other items (using current positions)
            new_bbox = (new_x, new_y, curr_w, curr_h)
            overlap = False
            for other_bbox in other_bboxes:
                if bboxes_overlap(new_bbox, other_bbox, margin):
                    overlap = True
                    break

            if not overlap:
                # Valid position found - move the item and update tracking
                item["x"] = new_x
                item["y"] = new_y
                current_bboxes[item_id] = new_bbox
                break


def augment_item(item, rng, args):
    """Augment a single canvas item (table, text, image, etc.)."""
    item_type = item.get("type")

    if item_type == "table":
        augment_table(item, rng, args)
    elif item_type == "text":
        augment_text_item(item, rng, args)
    elif item_type == "image":
        augment_image_item(item, rng, args)
    # Add more item types as needed


# Sample label texts that commonly appear above/below tables
DEFAULT_LABEL_TEXTS = [
    "表1", "表2", "表3", "Table 1", "Table 2", "Table 3",
    "データ", "Data", "集計結果", "Summary", "一覧",
    "List", "明細", "Details", "サマリー", "Overview",
    "統計", "Statistics", "記録", "Records", "報告書",
    "Report", "分析", "Analysis", "推移", "Trends",
    "比較", "Comparison", "内訳", "Breakdown",
]


def create_label_item(text, x, y, width, height, rng=None):
    """Create a label item with default properties.

    Args:
        text: Label text content
        x, y: Position
        width, height: Dimensions
        rng: Random number generator for style variations

    Returns:
        Label item dictionary
    """
    import uuid

    item = {
        "id": str(uuid.uuid4()),
        "type": "label",
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "properties": {
            "color": "#000000",
            "fontSize": 12,
            "fontFamily": "MS Mincho",
            "textAlign": "left",
            "fontWeight": "normal",
            "backgroundColor": "transparent",
            "borderRadius": 0.0,
            "padding": 0.0,
            "letterSpacing": 0.0,
            "kerning": False,
            "boxSizing": "border-box",
            "width": "auto",
            "height": "auto",
            "type": "label",
            "text": text,
        },
    }

    # Add style variations
    if rng:
        # Random font size (10-16)
        item["properties"]["fontSize"] = rng.randint(10, 16)

        # Random bold (30% chance)
        if rng.random() < 0.3:
            item["properties"]["fontWeight"] = "bold"

        # Random alignment (mostly left, sometimes center)
        if rng.random() < 0.2:
            item["properties"]["textAlign"] = "center"

        # Random underline (20% chance)
        if rng.random() < 0.2:
            item["properties"]["textDecoration"] = "underline"

        # Slight color jitter (stay dark for readability)
        if rng.random() < 0.3:
            base_colors = ["#000000", "#1a1a1a", "#333333", "#2c2c2c"]
            item["properties"]["color"] = rng.choice(base_colors)

    return item


def add_surrounding_labels(data, rng, args):
    """Add text labels above/below tables to train model to distinguish borders from text.

    This addresses a common model confusion where surrounding text (captions, labels)
    is mistaken for table borders.

    Args:
        data: Canvas data with items
        rng: Random number generator
        args: Command line arguments
    """
    add_labels_prob = getattr(args, "add_labels_prob", 0)
    if add_labels_prob <= 0:
        return

    label_above_prob = getattr(args, "label_above_prob", 0.5)
    label_below_prob = getattr(args, "label_below_prob", 0.5)
    min_label_padding = getattr(args, "label_min_padding", 8)
    max_label_padding = getattr(args, "label_max_padding", 25)
    label_texts = getattr(args, "label_texts", None) or DEFAULT_LABEL_TEXTS

    items = data.get("items", [])
    new_items = []

    # Track all existing bboxes for overlap checking
    existing_bboxes = []
    for item in items:
        existing_bboxes.append(get_item_bbox(item))

    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)

    for item in items:
        if item.get("type") != "table":
            continue

        if rng.random() > add_labels_prob:
            continue

        table_bbox = get_item_bbox(item)
        table_x, table_y, table_w, table_h = table_bbox

        # Estimate label height based on font size (12-16px font + padding)
        label_height = rng.randint(16, 24)
        label_width = min(table_w, rng.randint(80, 200))

        # Try to add label above table
        if rng.random() < label_above_prob:
            padding = rng.randint(min_label_padding, max_label_padding)
            label_x = table_x + rng.uniform(0, max(0, table_w - label_width))
            label_y = table_y - label_height - padding

            # Check if position is valid (within canvas and no overlap)
            label_bbox = (label_x, label_y, label_width, label_height)
            valid = True

            # Check canvas bounds
            if label_y < 0:
                valid = False
            elif canvas_width > 0 and (label_x + label_width > canvas_width):
                valid = False

            # Check overlap with existing items
            if valid:
                for existing_bbox in existing_bboxes:
                    if bboxes_overlap(label_bbox, existing_bbox, margin=2):
                        valid = False
                        break

            if valid:
                text = rng.choice(label_texts)
                label_item = create_label_item(text, label_x, label_y, label_width, label_height, rng)
                new_items.append(label_item)
                existing_bboxes.append(label_bbox)

        # Try to add label below table
        if rng.random() < label_below_prob:
            padding = rng.randint(min_label_padding, max_label_padding)
            label_x = table_x + rng.uniform(0, max(0, table_w - label_width))
            label_y = table_y + table_h + padding

            # Check if position is valid
            label_bbox = (label_x, label_y, label_width, label_height)
            valid = True

            # Check canvas bounds
            if canvas_height > 0 and (label_y + label_height > canvas_height):
                valid = False
            elif canvas_width > 0 and (label_x + label_width > canvas_width):
                valid = False

            # Check overlap with existing items
            if valid:
                for existing_bbox in existing_bboxes:
                    if bboxes_overlap(label_bbox, existing_bbox, margin=2):
                        valid = False
                        break

            if valid:
                text = rng.choice(label_texts)
                label_item = create_label_item(text, label_x, label_y, label_width, label_height, rng)
                new_items.append(label_item)
                existing_bboxes.append(label_bbox)

    # Add new labels to canvas
    if new_items:
        data["items"].extend(new_items)


def augment_text_item(item, rng, args):
    """Augment text item content and style."""
    if item.get("type") != "text":
        return

    # Mutate text content
    text_prob = getattr(args, "text_prob", 0)
    if text_prob > 0 and rng.random() < text_prob:
        current_text = item.get("value", "")
        item["value"] = mutate_text(
            current_text,
            rng,
            getattr(args, "text_replace_prob", 0.2),
            getattr(args, "text_append_prob", 0.2),
            getattr(args, "text_truncate_prob", 0.1),
            getattr(args, "text_char_prob", 0.3),
            getattr(args, "fill_empty_text", False),
            getattr(args, "text_max_len", 12),
            getattr(args, "text_jp_ratio", 0.7),
            getattr(args, "text_digit_ratio", 0.2),
            getattr(args, "text_ascii_ratio", 0.1),
        )

    # Jitter colors
    color_prob = getattr(args, "color_prob", 0)
    if color_prob > 0:
        color_jitter = getattr(args, "color_jitter", 40)
        for key in ["color", "backgroundColor"]:
            if key in item and rng.random() < color_prob:
                item[key] = jitter_color(item[key], rng, color_jitter)
        if "style" in item:
            maybe_jitter_colors(item["style"], rng, color_prob, color_jitter)

    # Mutate font size
    font_scale_prob = getattr(args, "font_scale_prob", 0)
    if font_scale_prob > 0 and rng.random() < font_scale_prob:
        scale_min = getattr(args, "font_scale_min", 0.8)
        scale_max = getattr(args, "font_scale_max", 1.2)
        if "fontSize" in item:
            item["fontSize"] = max(6, int(round(item["fontSize"] * rng.uniform(scale_min, scale_max))))


def augment_image_item(item, rng, args):
    """Augment image item (position and size only)."""
    # Images are binary data, so we mainly adjust position/size
    # Size changes are handled by canvas scaling and jitter
    pass


def augment_canvas(canvas_data, rng, args):
    data = copy.deepcopy(canvas_data)

    # First, optionally scale the entire canvas
    canvas_scale_prob = getattr(args, "canvas_scale_prob", 0)
    if canvas_scale_prob > 0:
        randomize_canvas_size(
            data,
            rng,
            canvas_scale_prob,
            getattr(args, "canvas_scale_min", 0.8),
            getattr(args, "canvas_scale_max", 1.2),
        )

    # Store original bounding boxes before augmentation (for translation)
    orig_bboxes = {}
    for item in data.get("items", []):
        item_id = item.get("id")
        if item_id:
            orig_bboxes[item_id] = get_item_bbox(item)

    # Apply item-specific augmentations (all item types)
    for item in data.get("items", []):
        augment_item(item, rng, args)

    # Calculate canvas bounds from all items and ensure min padding
    min_edge_padding = 5  # Minimum padding from canvas edges
    max_x = 0.0
    max_y = 0.0
    for item in data.get("items", []):
        bbox = get_item_bbox(item)
        item_x, item_y, item_w, item_h = bbox
        max_x = max(max_x, item_x + item_w)
        max_y = max(max_y, item_y + item_h)

    # Set initial canvas size with padding on all sides
    extra_padding = max(0, int(args.canvas_padding))
    if max_x > 0 or max_y > 0:
        data["canvasWidth"] = int(math.ceil(max_x + extra_padding + min_edge_padding))
        data["canvasHeight"] = int(math.ceil(max_y + extra_padding + min_edge_padding))

    # Ensure all items have minimum padding from canvas edges
    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)
    for item in data.get("items", []):
        bbox = get_item_bbox(item)
        item_x, item_y, item_w, item_h = bbox
        adjusted = False

        # Check left/top edges
        if item_x < min_edge_padding:
            item["x"] = min_edge_padding
            adjusted = True
        if item_y < min_edge_padding:
            item["y"] = min_edge_padding
            adjusted = True

        # Check right/bottom edges
        if canvas_width > 0 and item_x + item_w > canvas_width - min_edge_padding:
            item["x"] = max(min_edge_padding, canvas_width - min_edge_padding - item_w)
            adjusted = True
        if canvas_height > 0 and item_y + item_h > canvas_height - min_edge_padding:
            item["y"] = max(min_edge_padding, canvas_height - min_edge_padding - item_h)
            adjusted = True

    # Apply position jitter to all items (with overlap prevention)
    jitter_prob = getattr(args, "jitter_prob", 0)
    max_jitter = getattr(args, "max_jitter", 50)
    if jitter_prob > 0:
        jitter_item_positions(data, rng, jitter_prob, max_jitter, min_edge_padding)

    # Apply table translation (move tables without overlap)
    translate_tables(data, orig_bboxes, rng, args)

    # Recalculate canvas bounds after position changes (with min padding)
    max_x = 0.0
    max_y = 0.0
    for item in data.get("items", []):
        bbox = get_item_bbox(item)
        item_x, item_y, item_w, item_h = bbox
        max_x = max(max_x, item_x + item_w)
        max_y = max(max_y, item_y + item_h)

    if max_x > 0 or max_y > 0:
        extra_padding = max(0, int(args.canvas_padding))
        data["canvasWidth"] = int(math.ceil(max_x + extra_padding + min_edge_padding))
        data["canvasHeight"] = int(math.ceil(max_y + extra_padding + min_edge_padding))

    # Final validation: check for any overlaps and fix them
    _fix_overlaps(data, min_edge_padding)

    # Add surrounding labels to tables (helps model distinguish borders from text)
    add_surrounding_labels(data, rng, args)

    # Recalculate canvas bounds after adding labels (labels may extend bounds)
    max_x = 0.0
    max_y = 0.0
    for item in data.get("items", []):
        bbox = get_item_bbox(item)
        item_x, item_y, item_w, item_h = bbox
        max_x = max(max_x, item_x + item_w)
        max_y = max(max_y, item_y + item_h)

    if max_x > 0 or max_y > 0:
        extra_padding = max(0, int(args.canvas_padding))
        data["canvasWidth"] = int(math.ceil(max_x + extra_padding + min_edge_padding))
        data["canvasHeight"] = int(math.ceil(max_y + extra_padding + min_edge_padding))

    return data


def _fix_overlaps(data, min_padding=5, max_iterations=50):
    """Detect and fix any overlapping items by separating them.

    Args:
        data: Canvas data with items
        min_padding: Minimum spacing between items
        max_iterations: Maximum iterations to resolve overlaps
    """
    items = data.get("items", [])
    if not items:
        return

    canvas_width = data.get("canvasWidth", 0)
    canvas_height = data.get("canvasHeight", 0)

    for iteration in range(max_iterations):
        overlaps_found = False
        item_bboxes = [get_item_bbox(item) for item in items]

        for i, item_a in enumerate(items):
            bbox_a = item_bboxes[i]
            ax, ay, aw, ah = bbox_a

            for j, item_b in enumerate(items):
                if i >= j:
                    continue

                bbox_b = item_bboxes[j]
                if bboxes_overlap(bbox_a, bbox_b, margin=0):
                    overlaps_found = True
                    bx, by, bw, bh = bbox_b

                    # Calculate centers
                    center_a_x = ax + aw / 2
                    center_a_y = ay + ah / 2
                    center_b_x = bx + bw / 2
                    center_b_y = by + bh / 2

                    # Push items apart along the line connecting their centers
                    dx = center_b_x - center_a_x
                    dy = center_b_y - center_a_y
                    dist = max(0.1, (dx**2 + dy**2)**0.5)  # Avoid division by zero

                    push_dist = 2.0  # Push by 2 pixels per iteration
                    move_x = (dx / dist) * push_dist
                    move_y = (dy / dist) * push_dist

                    # Move item A away from B
                    new_ax = ax - move_x
                    new_ay = ay - move_y
                    new_ax = max(min_padding, new_ax)
                    new_ay = max(min_padding, new_ay)
                    if canvas_width > 0:
                        new_ax = min(new_ax, canvas_width - min_padding - aw)
                    if canvas_height > 0:
                        new_ay = min(new_ay, canvas_height - min_padding - ah)
                    item_a["x"] = new_ax
                    item_a["y"] = new_ay

                    # Move item B away from A
                    new_bx = bx + move_x
                    new_by = by + move_y
                    new_bx = max(min_padding, new_bx)
                    new_by = max(min_padding, new_by)
                    if canvas_width > 0:
                        new_bx = min(new_bx, canvas_width - min_padding - bw)
                    if canvas_height > 0:
                        new_by = min(new_by, canvas_height - min_padding - bh)
                    item_b["x"] = new_bx
                    item_b["y"] = new_by

                    # Update cached bboxes
                    item_bboxes[i] = (new_ax, new_ay, aw, ah)
                    item_bboxes[j] = (new_bx, new_by, bw, bh)

        if not overlaps_found:
            break


def list_json_files(input_path, recursive=False):
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


def safe_basename(path, input_dir):
    rel = os.path.relpath(path, input_dir)
    rel = os.path.splitext(rel)[0]
    return rel.replace(os.sep, "__")


def main():
    parser = argparse.ArgumentParser(description="Augment canvas JSON data")
    parser.add_argument(
        "--input_dir", required=True, help="Input directory or JSON file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for augmented JSONs"
    )
    parser.add_argument(
        "--num_aug", type=int, default=3, help="Number of augmentations per input"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--recursive", action="store_true", help="Process subdirectories"
    )

    parser.add_argument(
        "--size_prob",
        type=float,
        default=0.8,
        help="Probability to change row/col sizes",
    )
    parser.add_argument(
        "--row_scale_min", type=float, default=0.7, help="Row height scale min"
    )
    parser.add_argument(
        "--row_scale_max", type=float, default=1.3, help="Row height scale max"
    )
    parser.add_argument(
        "--col_scale_min", type=float, default=0.7, help="Column width scale min"
    )
    parser.add_argument(
        "--col_scale_max", type=float, default=1.3, help="Column width scale max"
    )
    parser.add_argument(
        "--preserve_table_size",
        action="store_true",
        help="Keep total width/height constant",
    )

    parser.add_argument(
        "--merge_prob", type=float, default=0.2, help="Probability to attempt merges"
    )
    parser.add_argument(
        "--max_merges_per_table", type=int, default=3, help="Max merges per table"
    )
    parser.add_argument(
        "--max_rowspan", type=int, default=5, help="Max rowspan for merges"
    )
    parser.add_argument(
        "--max_colspan", type=int, default=5, help="Max colspan for merges"
    )
    parser.add_argument(
        "--reset_merges",
        action="store_true",
        help="Clear existing merges before augmenting",
    )

    parser.add_argument(
        "--text_prob", type=float, default=0.3, help="Probability to mutate cell text"
    )
    parser.add_argument(
        "--text_replace_prob",
        type=float,
        default=0.2,
        help="Probability to replace text entirely",
    )
    parser.add_argument(
        "--text_append_prob",
        type=float,
        default=0.2,
        help="Probability to append random text",
    )
    parser.add_argument(
        "--text_truncate_prob",
        type=float,
        default=0.1,
        help="Probability to truncate text",
    )
    parser.add_argument(
        "--text_char_prob",
        type=float,
        default=0.3,
        help="Per-char mutation probability",
    )
    parser.add_argument(
        "--text_max_len", type=int, default=12, help="Max text length after mutation"
    )
    parser.add_argument(
        "--fill_empty_text",
        action="store_true",
        help="Fill empty cells with random text",
    )
    parser.add_argument(
        "--text_jp_ratio",
        type=float,
        default=0.7,
        help="Ratio of Japanese chars in text",
    )
    parser.add_argument(
        "--text_digit_ratio", type=float, default=0.2, help="Ratio of digits in text"
    )
    parser.add_argument(
        "--text_ascii_ratio",
        type=float,
        default=0.1,
        help="Ratio of ASCII chars in text",
    )

    parser.add_argument(
        "--color_prob", type=float, default=0.3, help="Probability to jitter colors"
    )
    parser.add_argument(
        "--color_jitter", type=int, default=40, help="Color jitter range (0-255)"
    )
    parser.add_argument(
        "--canvas_padding",
        type=int,
        default=5,
        help="Padding (in pixels) added to canvas size",
    )
    parser.add_argument(
        "--show_all_borders",
        default=False,
        action="store_true",
        help="Set all cell border widths to 1 after augmentation",
    )

    # Negative example augmentations (to reduce model confusion)
    parser.add_argument(
        "--remove_outer_borders_prob",
        type=float,
        default=0.0,
        help="Probability to remove outer table borders (train model not to assume boundaries)",
    )
    parser.add_argument(
        "--remove_internal_borders_prob",
        type=float,
        default=0.0,
        help="Probability to remove internal borders between cells",
    )
    parser.add_argument(
        "--bg_contrast_prob",
        type=float,
        default=0.0,
        help="Probability to add contrasting backgrounds to adjacent cells without borders",
    )
    parser.add_argument(
        "--dense_text_prob",
        type=float,
        default=0.0,
        help="Probability to add dense text that fills most of the cell",
    )
    parser.add_argument(
        "--narrow_col_prob",
        type=float,
        default=0.0,
        help="Probability to make columns very narrow (mimics vertical text columns)",
    )

    # Row/column reduction augmentations
    parser.add_argument(
        "--reduce_rows_prob",
        type=float,
        default=0.0,
        help="Probability to remove random rows from table",
    )
    parser.add_argument(
        "--reduce_cols_prob",
        type=float,
        default=0.0,
        help="Probability to remove random columns from table",
    )
    parser.add_argument(
        "--min_cell_area",
        type=int,
        default=2,
        help="Minimum cell area (rows * cols) to keep when reducing. E.g., 2 means at least 2x1, 1x2, or larger.",
    )
    parser.add_argument(
        "--max_row_remove_ratio",
        type=float,
        default=0.5,
        help="Maximum ratio of rows to remove, e.g. 0.5 means up to 50%%",
    )
    parser.add_argument(
        "--max_col_remove_ratio",
        type=float,
        default=0.5,
        help="Maximum ratio of columns to remove, e.g. 0.5 means up to 50%%",
    )

    # Table scaling augmentation
    parser.add_argument(
        "--table_scale_prob",
        type=float,
        default=0.0,
        help="Probability to scale the entire table size",
    )
    parser.add_argument(
        "--table_scale_min",
        type=float,
        default=0.5,
        help="Minimum scale factor for table, e.g. 0.5 means 50%% of original",
    )
    parser.add_argument(
        "--table_scale_max",
        type=float,
        default=1.5,
        help="Maximum scale factor for table, e.g. 1.5 means 150%% of original",
    )

    # Table translation augmentation
    parser.add_argument(
        "--translate_prob",
        type=float,
        default=0.0,
        help="Probability to translate/move table position",
    )
    parser.add_argument(
        "--translate_max_offset",
        type=float,
        default=50,
        help="Maximum translation offset in pixels beyond freed space",
    )
    parser.add_argument(
        "--translate_margin",
        type=float,
        default=5,
        help="Minimum margin between items after translation",
    )

    # Canvas size and position augmentations
    parser.add_argument(
        "--canvas_scale_prob",
        type=float,
        default=0.0,
        help="Probability to scale the entire canvas (items are repositioned proportionally)",
    )
    parser.add_argument(
        "--canvas_scale_min",
        type=float,
        default=0.8,
        help="Minimum canvas scale factor (e.g., 0.8 = 80%% of original size)",
    )
    parser.add_argument(
        "--canvas_scale_max",
        type=float,
        default=1.2,
        help="Maximum canvas scale factor (e.g., 1.2 = 120%% of original size)",
    )
    parser.add_argument(
        "--jitter_prob",
        type=float,
        default=0.0,
        help="Probability to jitter each item's position",
    )
    parser.add_argument(
        "--max_jitter",
        type=float,
        default=50,
        help="Maximum position jitter in pixels (+/-)",
    )

    # Font size augmentation for text items
    parser.add_argument(
        "--font_scale_prob",
        type=float,
        default=0.0,
        help="Probability to scale font size for text items",
    )
    parser.add_argument(
        "--font_scale_min",
        type=float,
        default=0.8,
        help="Minimum font scale factor",
    )
    parser.add_argument(
        "--font_scale_max",
        type=float,
        default=1.2,
        help="Maximum font scale factor",
    )

    # Surrounding labels augmentation (add labels above/below tables)
    parser.add_argument(
        "--add_labels_prob",
        type=float,
        default=0.0,
        help="Probability to add text labels above/below a table (helps model distinguish borders from surrounding text)",
    )
    parser.add_argument(
        "--label_above_prob",
        type=float,
        default=0.5,
        help="Probability to add label above the table (when add_labels_prob triggers)",
    )
    parser.add_argument(
        "--label_below_prob",
        type=float,
        default=0.5,
        help="Probability to add label below the table (when add_labels_prob triggers)",
    )
    parser.add_argument(
        "--label_min_padding",
        type=int,
        default=8,
        help="Minimum padding (in pixels) between table and added labels",
    )
    parser.add_argument(
        "--label_max_padding",
        type=int,
        default=25,
        help="Maximum padding (in pixels) between table and added labels",
    )
    parser.add_argument(
        "--label_texts",
        type=str,
        nargs="*",
        default=None,
        help="Custom label texts to use (default: predefined Japanese/English labels like '表1', 'Table 1', etc.)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to generated augmented filenames (e.g., 'close_labels_' produces 'close_labels_base_aug1.json')",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    json_files = list_json_files(args.input_dir, args.recursive)
    if not json_files:
        raise ValueError("No JSON files found to augment.")

    for path in tqdm(json_files):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = safe_basename(path, args.input_dir)
        for i in range(args.num_aug):
            aug_rng = random.Random(rng.randint(0, 2**31 - 1))
            items = data.get('items', [])
            valid_items = []
            for item in items:
                try:
                    get_item_bbox(item)
                    valid_items.append(item)
                except Exception:
                    pass
            data['items'] = valid_items
            augmented = augment_canvas(data, aug_rng, args)
            out_name = f"{args.prefix}{base}_aug{i + 1}.json"
            out_path = os.path.join(args.output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
