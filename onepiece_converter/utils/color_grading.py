"""Arc-based color grading for Part 3 output."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance


ARC_PALETTES: Dict[str, List[Tuple[int, int, int]]] = {
    "adventure": [(246, 206, 81), (51, 129, 214), (245, 245, 245), (221, 154, 82), (49, 84, 138)],
    "dramatic": [(38, 57, 92), (90, 103, 128), (172, 178, 184), (53, 57, 72), (218, 219, 221)],
    "wano": [(194, 38, 36), (231, 184, 66), (34, 28, 39), (162, 111, 65), (95, 33, 40)],
}


def _apply_temperature_shift(image: Image.Image, target_palette: List[Tuple[int, int, int]]) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    target = np.array(target_palette, dtype=np.float32).mean(axis=0)
    cur = arr.reshape(-1, 3).mean(axis=0)
    shift = (target - cur) * 0.18
    arr[..., 0] = np.clip(arr[..., 0] + shift[0], 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] + shift[1], 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] + shift[2], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def apply_arc_color_grading(
    image: Image.Image,
    arc: str = "adventure",
    reference_input: Image.Image | None = None,
) -> Image.Image:
    palette = ARC_PALETTES.get(arc, ARC_PALETTES["adventure"])
    graded = image.convert("RGB")
    graded = ImageEnhance.Color(graded).enhance(1.25)
    graded = ImageEnhance.Contrast(graded).enhance(1.10)
    graded = _apply_temperature_shift(graded, palette)

    if reference_input is not None:
        in_brightness = float(np.array(reference_input.convert("RGB")).mean())
        out_arr = np.array(graded, dtype=np.float32)
        out_brightness = float(out_arr.mean())
        if out_brightness > 0:
            ratio = np.clip(in_brightness / out_brightness, 0.85, 1.15)
            out_arr = np.clip(out_arr * ratio, 0, 255)
            graded = Image.fromarray(out_arr.astype(np.uint8), mode="RGB")
    return graded

