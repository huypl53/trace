# -*- coding: utf-8 -*-
import math


def _get_table_dim(props, item, key):
    value = props.get(key)
    if value is None:
        return item.get(key)
    return value


def _build_sizes(size_map, count, total):
    if count <= 0:
        return []
    size_map = size_map or {}
    if total is None:
        total = sum(float(v) for v in size_map.values()) if size_map else 0.0
    default = (total / count) if total else 0.0
    sizes = []
    for i in range(count):
        key = str(i)
        if key in size_map:
            sizes.append(float(size_map[key]))
        else:
            sizes.append(default)
    return sizes


def _border_visible(cell_style, key):
    if key not in cell_style:
        return True
    try:
        return float(cell_style[key]) > 0
    except (TypeError, ValueError):
        return True


def _border_thickness(cell_style, key, default=1):
    if key in cell_style:

        try:
            if cell_style[key] is not None:
                return int(cell_style[key] )
        except (TypeError, ValueError):
            return default
    return default


def _round_point(x, y):
    return int(round(x)), int(round(y))


def extract_table_lines(canvas_data):
    lines_h = []
    lines_v = []
    items = canvas_data.get("items", [])
    for item in items:
        if item.get("type") != "table":
            continue
        props = item.get("properties", {})
        rows = int(props.get("rows", 0))
        cols = int(props.get("columns", 0))
        table_x = float(item.get("x", 0))
        table_y = float(item.get("y", 0))
        table_w = _get_table_dim(props, item, "width")
        table_h = _get_table_dim(props, item, "height")
        table_w = float(table_w) if table_w is not None else None
        table_h = float(table_h) if table_h is not None else None

        row_heights = _build_sizes(props.get("rowHeights", {}), rows, table_h)
        col_widths = _build_sizes(props.get("columnWidths", {}), cols, table_w)
        row_offsets = [0.0]
        for h in row_heights:
            row_offsets.append(row_offsets[-1] + h)
        col_offsets = [0.0]
        for w in col_widths:
            col_offsets.append(col_offsets[-1] + w)

        cell_data = props.get("cellData", {}) or {}
        merged_cells = props.get("mergedCells", {}) or {}
        hidden_cells = props.get("hiddenCells", {}) or {}

        for r in range(rows):
            for c in range(cols):
                key = f"{r}-{c}"
                if hidden_cells.get(key):
                    continue
                merged = merged_cells.get(key, {})
                rowspan = int(merged.get("rowspan", 1))
                colspan = int(merged.get("colspan", 1))

                x0 = table_x + col_offsets[c]
                x1 = table_x + col_offsets[min(c + colspan, len(col_offsets) - 1)]
                y0 = table_y + row_offsets[r]
                y1 = table_y + row_offsets[min(r + rowspan, len(row_offsets) - 1)]

                cell_style = {}
                if key in cell_data:
                    cell_style = cell_data[key].get("cellStyle", {}) or {}

                if _border_visible(cell_style, "borderTopWidth"):
                    thickness = _border_thickness(cell_style, "borderTopWidth")
                    lines_h.append((_round_point(x0, y0), _round_point(x1, y0), thickness))
                if _border_visible(cell_style, "borderBottomWidth"):
                    thickness = _border_thickness(cell_style, "borderBottomWidth")
                    lines_h.append((_round_point(x0, y1), _round_point(x1, y1), thickness))
                if _border_visible(cell_style, "borderLeftWidth"):
                    thickness = _border_thickness(cell_style, "borderLeftWidth")
                    lines_v.append((_round_point(x0, y0), _round_point(x0, y1), thickness))
                if _border_visible(cell_style, "borderRightWidth"):
                    thickness = _border_thickness(cell_style, "borderRightWidth")
                    lines_v.append((_round_point(x1, y0), _round_point(x1, y1), thickness))

    return lines_h, lines_v


def extract_single_table_data(item):
    """Extract table bounds, lines, and cell boxes from a table item."""
    props = item.get("properties", {})
    rows = int(props.get("rows", 0))
    cols = int(props.get("columns", 0))
    table_x = float(item.get("x", 0))
    table_y = float(item.get("y", 0))
    table_w = _get_table_dim(props, item, "width")
    table_h = _get_table_dim(props, item, "height")
    table_w = float(table_w) if table_w is not None else None
    table_h = float(table_h) if table_h is not None else None

    row_heights = _build_sizes(props.get("rowHeights", {}), rows, table_h)
    col_widths = _build_sizes(props.get("columnWidths", {}), cols, table_w)

    row_offsets = [0.0]
    for h in row_heights:
        row_offsets.append(row_offsets[-1] + h)
    col_offsets = [0.0]
    for w in col_widths:
        col_offsets.append(col_offsets[-1] + w)

    actual_w = table_w if table_w is not None else col_offsets[-1]
    actual_h = table_h if table_h is not None else row_offsets[-1]

    bounds = (
        int(table_x),
        int(table_y),
        int(math.ceil(table_x + actual_w)),
        int(math.ceil(table_y + actual_h)),
    )

    cell_data = props.get("cellData", {}) or {}
    merged_cells = props.get("mergedCells", {}) or {}
    hidden_cells = props.get("hiddenCells", {}) or {}

    lines_h = []
    lines_v = []
    cells = []

    for r in range(rows):
        for c in range(cols):
            key = f"{r}-{c}"
            if hidden_cells.get(key):
                continue
            merged = merged_cells.get(key, {})
            rowspan = int(merged.get("rowspan", 1))
            colspan = int(merged.get("colspan", 1))

            x0 = table_x + col_offsets[c]
            x1 = table_x + col_offsets[min(c + colspan, len(col_offsets) - 1)]
            y0 = table_y + row_offsets[r]
            y1 = table_y + row_offsets[min(r + rowspan, len(row_offsets) - 1)]

            cell_style = {}
            if key in cell_data:
                cell_style = cell_data[key].get("cellStyle", {}) or {}

            cells.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "style": cell_style,
                }
            )

            if _border_visible(cell_style, "borderTopWidth"):
                thickness = _border_thickness(cell_style, "borderTopWidth")
                lines_h.append((_round_point(x0, y0), _round_point(x1, y0), thickness))
            if _border_visible(cell_style, "borderBottomWidth"):
                thickness = _border_thickness(cell_style, "borderBottomWidth")
                lines_h.append((_round_point(x0, y1), _round_point(x1, y1), thickness))
            if _border_visible(cell_style, "borderLeftWidth"):
                thickness = _border_thickness(cell_style, "borderLeftWidth")
                lines_v.append((_round_point(x0, y0), _round_point(x0, y1), thickness))
            if _border_visible(cell_style, "borderRightWidth"):
                thickness = _border_thickness(cell_style, "borderRightWidth")
                lines_v.append((_round_point(x1, y0), _round_point(x1, y1), thickness))

    return {"bounds": bounds, "lines_h": lines_h, "lines_v": lines_v, "cells": cells}


def infer_canvas_size(canvas_data):
    canvas_width = canvas_data.get("canvasWidth")
    canvas_height = canvas_data.get("canvasHeight")
    if canvas_width is not None and canvas_height is not None:
        return int(math.ceil(float(canvas_width))), int(math.ceil(float(canvas_height)))
    max_x = 0.0
    max_y = 0.0
    for item in canvas_data.get("items", []):
        if item.get("type") != "table":
            continue
        props = item.get("properties", {})
        table_w = _get_table_dim(props, item, "width")
        table_h = _get_table_dim(props, item, "height")
        if table_w is None or table_h is None:
            continue
        max_x = max(max_x, float(item.get("x", 0)) + float(table_w))
        max_y = max(max_y, float(item.get("y", 0)) + float(table_h))
    return int(math.ceil(max_x)), int(math.ceil(max_y))


def get_table_bounds(canvas_data, padding=0):
    """Get bounding boxes of all tables in canvas.

    Args:
        canvas_data: Parsed canvas JSON
        padding: Optional padding around each table

    Returns:
        List of (x_min, y_min, x_max, y_max) tuples
    """
    bounds = []
    for item in canvas_data.get("items", []):
        if item.get("type") != "table":
            continue
        props = item.get("properties", {})
        x = float(item.get("x", 0))
        y = float(item.get("y", 0))
        w = float(props.get("width", item.get("width", 0)))
        h = float(props.get("height", item.get("height", 0)))
        bounds.append((
            int(x - padding),
            int(y - padding),
            int(math.ceil(x + w + padding)),
            int(math.ceil(y + h + padding))
        ))
    return bounds
