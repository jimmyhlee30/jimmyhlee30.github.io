#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from colorize import align_at_scale, pyramid, read_gray, split_channels


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/jameslee/Downloads/CS180_fa2026_proj1_data")
MEDIA = ROOT / "media"
FONT = ImageFont.load_default()


def rgb_image(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image


def labeled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(panel)
    draw.text((16, 13), label, fill="#1c1b19", font=FONT)
    image.thumbnail((size[0] - 32, size[1] - 52), Image.Resampling.LANCZOS)
    panel.paste(image, ((size[0] - image.width) // 2, 38 + (size[1] - 42 - image.height) // 2))
    return panel


def alignment_figure() -> None:
    blue, green, red = split_channels(read_gray(DATA / "cathedral.jpg"))
    raw_plate = Image.open(DATA / "cathedral.jpg").convert("L")
    unaligned = np.dstack((red, green, blue))
    aligned = np.asarray(Image.open(MEDIA / "cathedral.jpg").convert("RGB"), dtype=np.uint8) / 255
    panels = [
        labeled_panel(raw_plate.convert("RGB"), "1. Original BGR glass plate", (310, 410)),
        labeled_panel(rgb_image(unaligned, (370, 340)), "2. RGB stack, unaligned", (400, 410)),
        labeled_panel(rgb_image(aligned, (370, 340)), "3. NCC-aligned RGB result", (400, 410)),
    ]
    canvas = Image.new("RGB", (1130, 410), "#f6f4ee")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + 15
    canvas.save(MEDIA / "alignment_pipeline.jpg", quality=92)


def pyramid_figure() -> None:
    blue, green, _ = split_channels(read_gray(DATA / "church.tif"))
    blue_levels, green_levels = pyramid(blue), pyramid(green)
    shift = (0, 0)
    records = []
    for index, (mov, ref) in enumerate(zip(reversed(green_levels), reversed(blue_levels))):
        if index == 0:
            shift = align_at_scale(mov, ref, shift, 15)
            caption = f"Coarsest level: search ±15\nshift {shift}"
        else:
            shift = (shift[0] * 2, shift[1] * 2)
            shift = align_at_scale(mov, ref, shift, 2)
            caption = f"Finer level: double, refine ±2\nshift {shift}"
        records.append((ref, caption))
    picks = [records[0], records[len(records) // 2], records[-1]]
    panels = []
    for ref, caption in picks:
        img = Image.fromarray(np.clip(ref * 255, 0, 255).astype(np.uint8)).convert("RGB")
        panels.append(labeled_panel(img, caption, (360, 330)))
    canvas = Image.new("RGB", (1110, 330), "#f6f4ee")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * 375, 0))
    canvas.save(MEDIA / "pyramid_refinement.jpg", quality=92)


if __name__ == "__main__":
    alignment_figure()
    pyramid_figure()
