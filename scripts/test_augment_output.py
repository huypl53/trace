#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to validate augmented canvas JSON output.

Checks the following invariants for all tables:
1. rows == len(rowHeights) when rowHeights exists
2. columns == len(columnWidths) when columnWidths exists
3. height == sum(rowHeights) when rowHeights exists
4. width == sum(columnWidths) when columnWidths exists
5. rows >= 1 and columns >= 1 when rowHeights/columnWidths exist
6. item["height"] == props["height"] and item["width"] == props["width"]

Usage:
    uv run python -m scripts.test_augment_output --input_dir <augmented_dir>
    uv run python -m scripts.test_augment_output --input_file <single_json_file>
"""

import argparse
import json
import os
import sys
from pathlib import Path


class ValidationError:
    def __init__(self, file_path, item_id, error_type, message, details=None):
        self.file_path = file_path
        self.item_id = item_id
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def __str__(self):
        details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"[{self.error_type}] {self.file_path} (item {self.item_id}): {self.message} ({details_str})"


def validate_table(file_path, item):
    """Validate a single table item. Returns list of ValidationError."""
    errors = []
    item_id = item.get("id", "unknown")
    props = item.get("properties", {})

    row_heights = props.get("rowHeights", {})
    col_widths = props.get("columnWidths", {})
    rows = props.get("rows")
    cols = props.get("columns")
    prop_height = props.get("height")
    prop_width = props.get("width")
    item_height = item.get("height")
    item_width = item.get("width")

    # Check 1: rows == len(rowHeights)
    if row_heights:
        actual_rows = len(row_heights)
        if rows is not None and rows != actual_rows:
            errors.append(ValidationError(
                file_path, item_id, "ROWS_MISMATCH",
                f"rows ({rows}) != len(rowHeights) ({actual_rows})",
                {"rows": rows, "len_rowHeights": actual_rows}
            ))

        # Check 5a: rows >= 1
        if actual_rows < 1:
            errors.append(ValidationError(
                file_path, item_id, "MIN_ROWS",
                f"len(rowHeights) ({actual_rows}) < 1",
                {"len_rowHeights": actual_rows}
            ))

        # Check 3: height == sum(rowHeights)
        try:
            sum_row_heights = sum(float(v) for v in row_heights.values())
            sum_row_heights_rounded = int(round(sum_row_heights))
            if prop_height is not None:
                prop_height_int = int(round(float(prop_height)))
                if abs(prop_height_int - sum_row_heights_rounded) > 1:  # Allow 1px tolerance for rounding
                    errors.append(ValidationError(
                        file_path, item_id, "HEIGHT_MISMATCH",
                        f"props.height ({prop_height_int}) != sum(rowHeights) ({sum_row_heights_rounded})",
                        {"props_height": prop_height_int, "sum_rowHeights": sum_row_heights_rounded}
                    ))
        except (ValueError, TypeError) as e:
            errors.append(ValidationError(
                file_path, item_id, "INVALID_ROW_HEIGHTS",
                f"Cannot sum rowHeights: {e}",
                {"rowHeights": row_heights}
            ))

    # Check 2: columns == len(columnWidths)
    if col_widths:
        actual_cols = len(col_widths)
        if cols is not None and cols != actual_cols:
            errors.append(ValidationError(
                file_path, item_id, "COLS_MISMATCH",
                f"columns ({cols}) != len(columnWidths) ({actual_cols})",
                {"columns": cols, "len_columnWidths": actual_cols}
            ))

        # Check 5b: columns >= 1
        if actual_cols < 1:
            errors.append(ValidationError(
                file_path, item_id, "MIN_COLS",
                f"len(columnWidths) ({actual_cols}) < 1",
                {"len_columnWidths": actual_cols}
            ))

        # Check 4: width == sum(columnWidths)
        try:
            sum_col_widths = sum(float(v) for v in col_widths.values())
            sum_col_widths_rounded = int(round(sum_col_widths))
            if prop_width is not None:
                prop_width_int = int(round(float(prop_width)))
                if abs(prop_width_int - sum_col_widths_rounded) > 1:  # Allow 1px tolerance for rounding
                    errors.append(ValidationError(
                        file_path, item_id, "WIDTH_MISMATCH",
                        f"props.width ({prop_width_int}) != sum(columnWidths) ({sum_col_widths_rounded})",
                        {"props_width": prop_width_int, "sum_columnWidths": sum_col_widths_rounded}
                    ))
        except (ValueError, TypeError) as e:
            errors.append(ValidationError(
                file_path, item_id, "INVALID_COL_WIDTHS",
                f"Cannot sum columnWidths: {e}",
                {"columnWidths": col_widths}
            ))

    # Check 6: item dimensions match props dimensions
    if prop_height is not None and item_height is not None:
        prop_h = int(round(float(prop_height)))
        item_h = int(round(float(item_height)))
        if abs(prop_h - item_h) > 1:
            errors.append(ValidationError(
                file_path, item_id, "ITEM_HEIGHT_MISMATCH",
                f"item.height ({item_h}) != props.height ({prop_h})",
                {"item_height": item_h, "props_height": prop_h}
            ))

    if prop_width is not None and item_width is not None:
        prop_w = int(round(float(prop_width)))
        item_w = int(round(float(item_width)))
        if abs(prop_w - item_w) > 1:
            errors.append(ValidationError(
                file_path, item_id, "ITEM_WIDTH_MISMATCH",
                f"item.width ({item_w}) != props.width ({prop_w})",
                {"item_width": item_w, "props_width": prop_w}
            ))

    # Check rowHeights keys are contiguous 0..n-1
    if row_heights:
        expected_keys = {str(i) for i in range(len(row_heights))}
        actual_keys = set(row_heights.keys())
        if expected_keys != actual_keys:
            errors.append(ValidationError(
                file_path, item_id, "NON_CONTIGUOUS_ROW_KEYS",
                f"rowHeights keys are not contiguous 0..{len(row_heights)-1}",
                {"expected": sorted(expected_keys), "actual": sorted(actual_keys)}
            ))

    # Check columnWidths keys are contiguous 0..n-1
    if col_widths:
        expected_keys = {str(i) for i in range(len(col_widths))}
        actual_keys = set(col_widths.keys())
        if expected_keys != actual_keys:
            errors.append(ValidationError(
                file_path, item_id, "NON_CONTIGUOUS_COL_KEYS",
                f"columnWidths keys are not contiguous 0..{len(col_widths)-1}",
                {"expected": sorted(expected_keys), "actual": sorted(actual_keys)}
            ))

    return errors


def validate_file(file_path):
    """Validate a single JSON file. Returns list of ValidationError."""
    errors = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [ValidationError(file_path, "N/A", "INVALID_JSON", str(e))]
    except Exception as e:
        return [ValidationError(file_path, "N/A", "READ_ERROR", str(e))]

    items = data.get("items", [])
    for item in items:
        if item.get("type") == "table":
            errors.extend(validate_table(file_path, item))

    return errors


def list_json_files(input_path, recursive=False):
    """List all JSON files in a directory."""
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


def main():
    parser = argparse.ArgumentParser(description="Validate augmented canvas JSON output")
    parser.add_argument("--input_dir", help="Directory containing augmented JSON files")
    parser.add_argument("--input_file", help="Single JSON file to validate")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all errors")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary only")
    args = parser.parse_args()

    if not args.input_dir and not args.input_file:
        parser.error("Must specify --input_dir or --input_file")

    input_path = args.input_file or args.input_dir
    json_files = list_json_files(input_path, args.recursive)

    if not json_files:
        print("No JSON files found")
        sys.exit(1)

    print(f"Validating {len(json_files)} JSON files...")

    all_errors = []
    files_with_errors = set()
    error_counts = {}

    for file_path in json_files:
        errors = validate_file(file_path)
        if errors:
            files_with_errors.add(file_path)
            all_errors.extend(errors)
            for err in errors:
                error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

    # Print results
    if args.verbose and all_errors:
        print("\n=== ERRORS ===")
        for err in all_errors:
            print(err)

    print(f"\n=== SUMMARY ===")
    print(f"Total files checked: {len(json_files)}")
    print(f"Files with errors: {len(files_with_errors)}")
    print(f"Total errors: {len(all_errors)}")

    if error_counts:
        print("\nError breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

    if not args.summary and files_with_errors and not args.verbose:
        print("\nFiles with errors (first 10):")
        for f in list(files_with_errors)[:10]:
            print(f"  {f}")
        if len(files_with_errors) > 10:
            print(f"  ... and {len(files_with_errors) - 10} more")

    if all_errors:
        print("\n❌ VALIDATION FAILED")
        sys.exit(1)
    else:
        print("\n✓ ALL VALIDATIONS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
