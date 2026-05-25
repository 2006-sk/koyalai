"""Preprocessing utilities for lineart extraction and edge computations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from controlnet_aux import LineartDetector
from PIL import Image

# Some environments have a partial `tensorflow` module that misses Tensor/Variable,
# which can crash einops backend detection. Patch minimal placeholders if needed.
try:
    import tensorflow as tf  # type: ignore

    if not hasattr(tf, "Tensor"):
        tf.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    if not hasattr(tf, "Variable"):
        tf.Variable = type("Variable", (), {})  # type: ignore[attr-defined]
except Exception:
    pass


class LineartPreprocessor:
    """Wrapper around controlnet_aux lineart detector."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        # Local annotator download can be partial if interrupted. In that case,
        # fall back to the hosted annotator repo to avoid hard failure.
        local_ready = (
            model_dir
            and model_dir.exists()
            and (model_dir / "sk_model.pth").exists()
            and (model_dir / "sk_model2.pth").exists()
        )
        if local_ready:
            self.detector = LineartDetector.from_pretrained(model_dir.as_posix())
            return
        self.detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

    def extract_lineart(self, image: Image.Image) -> Image.Image:
        """Extract lineart map from an RGB image."""
        return self.detector(image, detect_resolution=512, image_resolution=512)


def to_gray_array(image: Image.Image) -> np.ndarray:
    """Convert image to grayscale uint8 array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def edge_map(image: Image.Image, low: int = 100, high: int = 200) -> np.ndarray:
    """Compute Canny edge map for metric comparisons."""
    gray = to_gray_array(image)
    return cv2.Canny(gray, low, high)


def edge_density(edge_img: np.ndarray) -> float:
    """Calculate ratio of edge pixels in an edge map."""
    return float(np.count_nonzero(edge_img)) / float(edge_img.size)


def laplacian_variance(image: Image.Image) -> float:
    """Estimate image sharpness using variance of Laplacian."""
    gray = to_gray_array(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

