#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Augment canvas JSON tables by modifying layout and styles.

Augmentations include:
- Random row/column size changes
- Random merges (rowspan/colspan)
- Cell text mutations
- Color jitter (table and cell colors)
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


def random_text(rng, min_len=1, max_len=8, jp_ratio=0.7, digit_ratio=0.2, ascii_ratio=0.1):
    length = rng.randint(min_len, max_len)
    jp_ratio, digit_ratio, ascii_ratio = normalize_weights([jp_ratio, digit_ratio, ascii_ratio])
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


def add_random_merges(rows, cols, merged_cells, hidden_cells, rng, merge_prob, max_merges, max_rowspan, max_colspan):
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


def augment_table(item, rng, args):
    if item.get("type") != "table":
        return
    props = item.get("properties", {})
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))
    if rows <= 0 or cols <= 0:
        return

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

        if orig_item_w is not None:
            delta_w = 0
            if orig_prop_w is not None:
                delta_w = float(orig_item_w) - float(orig_prop_w)
            item["width"] = float(new_w + delta_w)
        if orig_item_h is not None:
            delta_h = 0
            if orig_prop_h is not None:
                delta_h = float(orig_item_h) - float(orig_prop_h)
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
            if rng.random() > args.text_prob:
                continue
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
        maybe_jitter_colors(props.get("cellBorders", {}), rng, args.color_prob, args.color_jitter)

        cell_data = props.get("cellData", {}) or {}
        for cell in cell_data.values():
            if "cellStyle" in cell:
                maybe_jitter_colors(cell["cellStyle"], rng, args.color_prob, args.color_jitter)
            if "style" in cell:
                maybe_jitter_colors(cell["style"], rng, args.color_prob, args.color_jitter)
        props["cellData"] = cell_data

    item["properties"] = props


def augment_canvas(canvas_data, rng, args):
    data = copy.deepcopy(canvas_data)
    for item in data.get("items", []):
        augment_table(item, rng, args)
    max_x = 0.0
    max_y = 0.0
    for item in data.get("items", []):
        if item.get("type") != "table":
            continue
        props = item.get("properties", {})
        rows = int(props.get("rows", 0))
        cols = int(props.get("columns", 0))
        if rows <= 0 or cols <= 0:
            continue
        table_x = float(item.get("x", 0))
        table_y = float(item.get("y", 0))
        total_w = props.get("width")
        if total_w is None:
            total_w = item.get("width")
        total_h = props.get("height")
        if total_h is None:
            total_h = item.get("height")
        total_w = float(total_w) if total_w is not None else None
        total_h = float(total_h) if total_h is not None else None

        row_sizes = build_sizes(props.get("rowHeights", {}), rows, total_h)
        col_sizes = build_sizes(props.get("columnWidths", {}), cols, total_w)
        table_w = sum(col_sizes) if col_sizes else float(total_w or 0.0)
        table_h = sum(row_sizes) if row_sizes else float(total_h or 0.0)

        max_x = max(max_x, table_x + table_w)
        max_y = max(max_y, table_y + table_h)

    if max_x > 0 or max_y > 0:
        padding = max(0, int(args.canvas_padding))
        data["canvasWidth"] = int(math.ceil(max_x + padding))
        data["canvasHeight"] = int(math.ceil(max_y + padding))
    return data


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
    parser.add_argument("--input_dir", required=True, help="Input directory or JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for augmented JSONs")
    parser.add_argument("--num_aug", type=int, default=3, help="Number of augmentations per input")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories")

    parser.add_argument("--size_prob", type=float, default=0.8, help="Probability to change row/col sizes")
    parser.add_argument("--row_scale_min", type=float, default=0.7, help="Row height scale min")
    parser.add_argument("--row_scale_max", type=float, default=1.3, help="Row height scale max")
    parser.add_argument("--col_scale_min", type=float, default=0.7, help="Column width scale min")
    parser.add_argument("--col_scale_max", type=float, default=1.3, help="Column width scale max")
    parser.add_argument("--preserve_table_size", action="store_true", help="Keep total width/height constant")

    parser.add_argument("--merge_prob", type=float, default=0.2, help="Probability to attempt merges")
    parser.add_argument("--max_merges_per_table", type=int, default=3, help="Max merges per table")
    parser.add_argument("--max_rowspan", type=int, default=3, help="Max rowspan for merges")
    parser.add_argument("--max_colspan", type=int, default=3, help="Max colspan for merges")
    parser.add_argument("--reset_merges", action="store_true", help="Clear existing merges before augmenting")

    parser.add_argument("--text_prob", type=float, default=0.3, help="Probability to mutate cell text")
    parser.add_argument("--text_replace_prob", type=float, default=0.2, help="Probability to replace text entirely")
    parser.add_argument("--text_append_prob", type=float, default=0.2, help="Probability to append random text")
    parser.add_argument("--text_truncate_prob", type=float, default=0.1, help="Probability to truncate text")
    parser.add_argument("--text_char_prob", type=float, default=0.3, help="Per-char mutation probability")
    parser.add_argument("--text_max_len", type=int, default=12, help="Max text length after mutation")
    parser.add_argument("--fill_empty_text", action="store_true", help="Fill empty cells with random text")
    parser.add_argument("--text_jp_ratio", type=float, default=0.7, help="Ratio of Japanese chars in text")
    parser.add_argument("--text_digit_ratio", type=float, default=0.2, help="Ratio of digits in text")
    parser.add_argument("--text_ascii_ratio", type=float, default=0.1, help="Ratio of ASCII chars in text")

    parser.add_argument("--color_prob", type=float, default=0.3, help="Probability to jitter colors")
    parser.add_argument("--color_jitter", type=int, default=40, help="Color jitter range (0-255)")
    parser.add_argument(
        "--canvas_padding",
        type=int,
        default=2,
        help="Padding (in pixels) added to canvas size",
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
            augmented = augment_canvas(data, aug_rng, args)
            out_name = f"{base}_aug{i + 1}.json"
            out_path = os.path.join(args.output_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
