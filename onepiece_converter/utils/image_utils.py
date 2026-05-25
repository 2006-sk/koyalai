"""Image utility helpers for resize, comparison rendering, and metadata."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin


TARGET_SIZE: Tuple[int, int] = (512, 512)


def timestamp_string() -> str:
    """Return a compact timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_image(path: Path) -> Image.Image:
    """Load image from path as RGB."""
    return Image.open(path).convert("RGB")


def resize_with_padding(image: Image.Image, size: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Resize image to fit target size while preserving aspect ratio with padding."""
    return ImageOps.pad(image, size=size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))


def save_side_by_side(
    original: Image.Image,
    lineart: Image.Image,
    output: Image.Image,
    output_dir: Path,
    stem: str,
) -> Path:
    """Save original, lineart, output in one horizontal comparison image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (original.width * 3, original.height))
    canvas.paste(original, (0, 0))
    canvas.paste(lineart.convert("RGB"), (original.width, 0))
    canvas.paste(output, (original.width * 2, 0))
    out_path = output_dir / f"{stem}_comparison.png"
    canvas.save(out_path)
    return out_path


def save_image_with_metadata(image: Image.Image, path: Path, metadata: Dict[str, Any]) -> None:
    """Save image and embed metadata text in PNG tEXt chunks."""
    png_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        png_info.add_text(str(key), str(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, pnginfo=png_info)


def save_metadata_json(metadata: Dict[str, Any], path: Path) -> None:
    """Save metadata to JSON with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB uint8 numpy array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)

