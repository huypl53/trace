"""Resize images in a directory tree by a fixed downscale factor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

try:  # Pillow >= 9.1.0
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - fallback for older Pillow
    RESAMPLE = Image.LANCZOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize every image found inside INPUT_DIR and its subdirectories "
            "by a downscale factor of 2. If OUTPUT_DIR is omitted the images "
            "are overwritten in-place."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Root directory to scan for images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to mirror the resized images into",
    )
    parser.add_argument(
        "--scale-down",
        type=float,
        default=2.0,
        help="Downscale factor (default: 2.0)",
    )
    return parser.parse_args()


def _iter_images(root_dir: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            path = Path(dirpath, name)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _resize_image(src: Path, dest: Path, scale_down: float) -> None:
    if scale_down <= 0:
        raise ValueError("scale_down must be positive")

    with Image.open(src) as img:
        new_width = max(1, int(round(img.width / scale_down)))
        new_height = max(1, int(round(img.height / scale_down)))

        if new_width == img.width and new_height == img.height and src == dest:
            return

        resized = img.resize((new_width, new_height), RESAMPLE)
        _ensure_parent(dest)
        resized.save(dest, format=img.format)


def resize_tree(input_dir: Path, output_dir: Path | None, scale_down: float = 2.0) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    image_paths = list(_iter_images(input_dir))
    if not image_paths:
        return 0

    total_processed = 0
    for src in image_paths:
        rel_path = src.relative_to(input_dir)
        dest = src if output_dir is None else output_dir / rel_path
        _resize_image(src, dest, scale_down)
        total_processed += 1
    return total_processed


def main() -> None:
    args = _parse_args()
    total = resize_tree(args.input_dir, args.output_dir, args.scale_down)
    print(f"Resized {total} image(s).")


if __name__ == "__main__":
    main()
