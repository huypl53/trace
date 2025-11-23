#!/usr/bin/env python3
"""Extract line labels from cropped table JSON files."""

import argparse
import json
import os
from pathlib import Path
import sys

# Add parent directory to path to import types
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.types import Line


def calculate_cell_position(row, col, row_heights, col_widths, merged_cells, cell_key):
    """
    Calculate the position and size of a cell considering merges.

    Returns: (x, y, width, height, row_span, col_span)
    """
    # Get merge info if exists
    merge_info = merged_cells.get(cell_key, {'colspan': 1, 'rowspan': 1})
    col_span = merge_info.get('colspan', 1)
    row_span = merge_info.get('rowspan', 1)

    # Calculate position
    x = sum(col_widths.get(str(i), 0) for i in range(col))
    y = sum(row_heights.get(str(i), 0) for i in range(row))

    # Calculate size
    width = sum(col_widths.get(str(col + i), 0) for i in range(col_span))
    height = sum(row_heights.get(str(row + i), 0) for i in range(row_span))

    return x, y, width, height, row_span, col_span


def extract_lines_from_table(table_data, table_x_offset=0, table_y_offset=0):
    """
    Extract grid lines from a table.

    Args:
        table_data: Table object from JSON
        table_x_offset: X offset of the table in the image
        table_y_offset: Y offset of the table in the image

    Returns:
        List of Line objects
    """
    properties = table_data['properties']
    rows = properties['rows']
    columns = properties['columns']
    cell_data = properties.get('cellData', {})
    merged_cells = properties.get('mergedCells', {})
    hidden_cells = properties.get('hiddenCells', {})
    row_heights = properties.get('rowHeights', {})
    col_widths = properties.get('columnWidths', {})

    # Calculate cumulative positions for grid lines
    # Horizontal lines (y positions)
    h_line_y = [table_y_offset]
    for i in range(rows):
        h_line_y.append(h_line_y[-1] + row_heights.get(str(i), 0))

    # Vertical lines (x positions)
    v_line_x = [table_x_offset]
    for i in range(columns):
        v_line_x.append(v_line_x[-1] + col_widths.get(str(i), 0))

    # Track visibility of each grid line
    # horizontal_lines[i] represents the line between row i-1 and row i
    # vertical_lines[j] represents the line between column j-1 and column j
    def mark_line(line_dict, key, visible):
        """Update line visibility - a line is invisible if ANY border marks it as invisible."""
        # Start with True (visible), becomes False if any cell marks it invisible
        line_dict[key] = line_dict.get(key, True) and visible

    horizontal_lines = {}  # {(y, x1, x2): visible}
    vertical_lines = {}  # {(x, y1, y2): visible}

    # Process each cell
    for row in range(rows):
        for col in range(columns):
            cell_key = f"{row}-{col}"

            # Skip hidden cells
            if hidden_cells.get(cell_key):
                continue

            # Get cell style
            cell_info = cell_data.get(cell_key, {})
            cell_style = cell_info.get('cellStyle', {})

            # Calculate cell position and span
            x, y, width, height, row_span, col_span = calculate_cell_position(
                row, col, row_heights, col_widths, merged_cells, cell_key
            )

            # Adjust for table offset
            x += table_x_offset
            y += table_y_offset

            # Extract border widths
            # None = default visible border, 0 = explicitly invisible, 1+ = explicitly visible
            border_top_width = cell_style.get('borderTopWidth')
            if border_top_width is None:
                border_top_width = 1

            border_bottom_width = cell_style.get('borderBottomWidth')
            if border_bottom_width is None:
                border_bottom_width = 1

            border_left_width = cell_style.get('borderLeftWidth')
            if border_left_width is None:
                border_left_width = 1

            border_right_width = cell_style.get('borderRightWidth')
            if border_right_width is None:
                border_right_width = 1

            # Top border (horizontal line at row index)
            top_y = y
            h_key = (top_y, x, x + width)
            mark_line(horizontal_lines, h_key, border_top_width > 0)

            # Bottom border (horizontal line at row + row_span index)
            bottom_y = y + height
            h_key = (bottom_y, x, x + width)
            mark_line(horizontal_lines, h_key, border_bottom_width > 0)

            # Left border (vertical line at col index)
            left_x = x
            v_key = (left_x, y, y + height)
            mark_line(vertical_lines, v_key, border_left_width > 0)

            # Right border (vertical line at col + col_span index)
            right_x = x + width
            v_key = (right_x, y, y + height)
            mark_line(vertical_lines, v_key, border_right_width > 0)

    # Convert to Line objects
    lines = []

    # Horizontal lines
    for (y, x1, x2), visible in horizontal_lines.items():
        line = Line(
            x1=x1,
            y1=y,
            x2=x2,
            y2=y,
            direction='horizontal',
            visible=visible
        )
        lines.append(line)

    # Vertical lines
    for (x, y1, y2), visible in vertical_lines.items():
        line = Line(
            x1=x,
            y1=y1,
            x2=x,
            y2=y2,
            direction='vertical',
            visible=visible
        )
        lines.append(line)

    return lines


def process_directory(input_dir, output_dir):
    """
    Process all JSON files in the input directory.

    Args:
        input_dir: Directory containing cropped table JSON/image pairs
        output_dir: Output directory for line JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_path.glob("*.json"))

    for json_file in json_files:
        print(f"Processing {json_file.name}...")

        # Read JSON
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Check if corresponding image exists
        image_name = json_file.stem
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_file = None
        for ext in image_extensions:
            candidate = input_path / f"{image_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            print(f"Warning: No image found for {json_file.name}, skipping...")
            continue

        # Find table (should only be one after cropping)
        tables = [item for item in data['items'] if item['type'] == 'table']

        if not tables:
            print(f"Warning: No tables found in {json_file.name}, skipping...")
            continue

        if len(tables) > 1:
            print(f"Warning: Multiple tables found in {json_file.name}, using first one...")

        table = tables[0]

        # Extract lines
        lines = extract_lines_from_table(table, table['x'], table['y'])

        # Save lines to JSON
        output_json = {
            'image': image_file.name,
            'lines': [line.to_dict() for line in lines]
        }

        output_json_path = output_path / f"{image_name}.json"
        with open(output_json_path, 'w') as f:
            json.dump(output_json, f, indent=2)

        # Copy image to output directory
        import shutil
        output_image_path = output_path / image_file.name
        shutil.copy(image_file, output_image_path)

        print(f"  Extracted {len(lines)} lines (visible: {sum(1 for l in lines if l.visible)})")


def main():
    parser = argparse.ArgumentParser(description="Extract line labels from cropped tables")
    parser.add_argument("input_dir", help="Input directory (CROP_DATA)")
    parser.add_argument("--output-dir", default="LINE_DATA", help="Output directory (default: LINE_DATA)")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
