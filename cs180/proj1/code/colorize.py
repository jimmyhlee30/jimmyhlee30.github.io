#!/usr/bin/env python3
"""Colorize Prokudin-Gorskii glass plates using NCC and a coarse-to-fine pyramid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_gray(path: Path) -> np.ndarray:
    """Read an 8- or 16-bit plate and return float32 intensities in [0, 1]."""
    image = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    maximum = image.max()
    return image / maximum if maximum > 0 else image


def split_channels(plate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The scans are stacked vertically in blue, green, red order."""
    height = plate.shape[0] // 3
    blue = plate[:height]
    green = plate[height : 2 * height]
    red = plate[2 * height : 3 * height]
    return blue, green, red


def ncc(reference: np.ndarray, moving: np.ndarray, margin: int) -> float:
    """Normalized cross-correlation on an interior region, avoiding roll edges."""
    if margin * 2 >= min(reference.shape):
        margin = 0
    ref = reference[margin:-margin or None, margin:-margin or None]
    mov = moving[margin:-margin or None, margin:-margin or None]
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    denom = np.linalg.norm(ref) * np.linalg.norm(mov)
    return float(np.sum(ref * mov) / denom) if denom > 1e-12 else -np.inf


def align_at_scale(
    moving: np.ndarray, reference: np.ndarray, center: tuple[int, int], radius: int
) -> tuple[int, int]:
    """Exhaustively search a square window of row/column translations."""
    best_shift = center
    best_score = -np.inf
    margin = max(radius + 2, int(min(reference.shape) * 0.10))
    for row_shift in range(center[0] - radius, center[0] + radius + 1):
        for col_shift in range(center[1] - radius, center[1] + radius + 1):
            shifted = np.roll(moving, (row_shift, col_shift), axis=(0, 1))
            score = ncc(reference, shifted, margin)
            if score > best_score:
                best_score, best_shift = score, (row_shift, col_shift)
    return best_shift


def resize_half(image: np.ndarray) -> np.ndarray:
    """Pillow's antialiased resize provides smoothing before downsampling."""
    height, width = image.shape
    return np.asarray(
        Image.fromarray(image).resize(
            (max(1, width // 2), max(1, height // 2)), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )


def pyramid(image: np.ndarray, smallest_side_limit: int = 350) -> list[np.ndarray]:
    levels = [image]
    while max(levels[-1].shape) > smallest_side_limit:
        levels.append(resize_half(levels[-1]))
    return levels


def align_pyramid(moving: np.ndarray, reference: np.ndarray) -> tuple[int, int]:
    """Estimate a large translation coarsely, then refine it at every scale."""
    moving_levels, reference_levels = pyramid(moving), pyramid(reference)
    shift = (0, 0)
    for level, (mov, ref) in enumerate(zip(reversed(moving_levels), reversed(reference_levels))):
        if level:
            shift = (shift[0] * 2, shift[1] * 2)
            shift = align_at_scale(mov, ref, shift, radius=2)
        else:
            shift = align_at_scale(mov, ref, shift, radius=15)
    return shift


def colorize(path: Path) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    blue, green, red = split_channels(read_gray(path))
    green_shift = align_pyramid(green, blue)
    red_shift = align_pyramid(red, blue)
    aligned_green = np.roll(green, green_shift, axis=(0, 1))
    aligned_red = np.roll(red, red_shift, axis=(0, 1))
    return np.dstack((aligned_red, aligned_green, blue)), green_shift, red_shift


def save_image(image: np.ndarray, path: Path, max_width: int | None = None) -> None:
    output = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
    if max_width and output.width > max_width:
        output.thumbnail((max_width, max_width * 3), Image.Resampling.LANCZOS)
    output.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="one plate image or a directory of plates")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--web-preview", action="store_true", help="resize saved images to 1200 px wide")
    args = parser.parse_args()

    paths = ([args.input] if args.input.is_file() else
             sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}))
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image, green_shift, red_shift = colorize(path)
        save_image(image, args.output / f"{path.stem}.jpg", 1200 if args.web_preview else None)
        print(f"{path.name}: green {green_shift}, red {red_shift}")


if __name__ == "__main__":
    main()
