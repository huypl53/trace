# -*- coding: utf-8 -*-
import json
import os
import random

import file_utils


class ParserTRACEJSON:
    def __init__(self, root_path, dataset, phase):
        self.gt = []
        base_folder = os.path.join(root_path, dataset)
        phase_folder = os.path.join(base_folder, phase)
        
        # If phase folder doesn't exist, use dataset folder directly (flat structure)
        if os.path.exists(phase_folder):
            base_folder = phase_folder
        # else: use base_folder directly (files are in dataset folder, not in phase subfolder)
        
        image_files, _, _ = file_utils.list_files(base_folder)

        for img_file in image_files:
            basename, ext = os.path.splitext(os.path.basename(img_file))
            gt_file = os.path.join(base_folder, f"{basename}.json")

            if os.path.exists(gt_file):
                quads = []
                lines = []
                table_bounds = []
                
                with open(gt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Iterate through items to find tables
                if "items" in data:
                    for item in data["items"]:
                        if item.get("type") == "table":
                            table_quads, table_lines, bounds = self._parse_table(item)
                            quads.extend(table_quads)
                            lines.extend(table_lines)
                            if bounds is not None:
                                table_bounds.append(bounds)

                if quads:  # Only add if we found at least one cell
                    self.gt.append(
                        {
                            "file_name": img_file,
                            "quads": quads,
                            "lines": lines,
                            "table_bounds": table_bounds,
                        }
                    )

    def _parse_table(self, table_item):
        """Parse a single table item and extract cell quads and border visibility."""
        quads = []
        lines = []
        
        props = table_item.get("properties", {})
        table_x = table_item.get("x", 0)
        table_y = table_item.get("y", 0)
        
        # Get column widths and row heights
        column_widths = props.get("columnWidths", {})
        row_heights = props.get("rowHeights", {})
        cell_data = props.get("cellData", {})
        
        # Convert string keys to integers and sort
        col_indices = sorted([int(k) for k in column_widths.keys()])
        row_indices = sorted([int(k) for k in row_heights.keys()])
        col_idx_to_pos = {idx: i for i, idx in enumerate(col_indices)}
        row_idx_to_pos = {idx: i for i, idx in enumerate(row_indices)}
        
        # Calculate cumulative positions
        col_positions = [0]
        for col_idx in col_indices:
            col_positions.append(col_positions[-1] + column_widths[str(col_idx)])
        
        row_positions = [0]
        for row_idx in row_indices:
            row_positions.append(row_positions[-1] + row_heights[str(row_idx)])
        
        merged_cells = props.get("mergedCells", {}) or {}
        hidden_cells = {key for key, val in (props.get("hiddenCells", {}) or {}).items() if val}

        table_bounds = None
        if col_positions and row_positions:
            table_width = col_positions[-1]
            table_height = row_positions[-1]
            table_bounds = {
                "x1": table_x,
                "y1": table_y,
                "x2": table_x + table_width,
                "y2": table_y + table_height,
            }

        # Process each cell
        for cell_key, cell_info in cell_data.items():
            # Parse row-col from key like "0-0"
            try:
                row_idx, col_idx = map(int, cell_key.split("-"))
            except ValueError:
                continue
            
            # Skip hidden cells
            if cell_key in hidden_cells:
                continue

            # Skip if indices are out of range
            if col_idx not in col_idx_to_pos or row_idx not in row_idx_to_pos:
                # Column or row index not found in widths/heights
                continue
            col_pos_idx = col_idx_to_pos[col_idx]
            row_pos_idx = row_idx_to_pos[row_idx]

            # Determine span (default 1x1, overridden by mergedCells entry)
            merged_info = merged_cells.get(cell_key, {})
            colspan = int(merged_info.get("colspan") or 1)
            rowspan = int(merged_info.get("rowspan") or 1)
            colspan = max(1, colspan)
            rowspan = max(1, rowspan)
            
            # Calculate cell position relative to table
            x1 = col_positions[col_pos_idx]
            y1 = row_positions[row_pos_idx]
            end_col_pos = min(len(col_positions) - 1, col_pos_idx + colspan)
            end_row_pos = min(len(row_positions) - 1, row_pos_idx + rowspan)
            x2 = col_positions[end_col_pos]
            y2 = row_positions[end_row_pos]
            
            # Convert to absolute coordinates (relative to image)
            abs_x1 = table_x + x1
            abs_y1 = table_y + y1
            abs_x2 = table_x + x2
            abs_y2 = table_y + y2
            
            # Build quad as [x1, y1, x2, y1, x2, y2, x1, y2]
            quad = [abs_x1, abs_y1, abs_x2, abs_y1, abs_x2, abs_y2, abs_x1, abs_y2]
            quads.append(quad)
            
            # Extract border visibility from cellStyle
            cell_style = cell_info.get("cellStyle", {})
            border_top = (cell_style.get("borderTopWidth") or 0) > 0
            border_bottom = (cell_style.get("borderBottomWidth") or 0) > 0
            border_left = (cell_style.get("borderLeftWidth") or 0) > 0
            border_right = (cell_style.get("borderRightWidth") or 0) > 0
            
            # Format: [top, bottom, left, right] matching XML parser
            line = [border_top, border_bottom, border_left, border_right]
            lines.append(line)
        
        return quads, lines, table_bounds

    def getDatasetSize(self):
        return len(self.gt)

    def lenFiles(self):
        return len(self.gt)

    def parseGT(self, index=-1):
        if index == -1:
            gt = self.gt[random.randrange(0, len(self.gt))]
        else:
            gt = self.gt[index]

        return gt["file_name"], gt

