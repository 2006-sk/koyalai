"""Scene analysis and dynamic prompt construction for Part 3."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


ARC_DESCRIPTORS = {
    "adventure": "adventure mood, heroic energy, ocean breeze palette",
    "dramatic": "dramatic mood, cinematic contrast, intense atmosphere",
    "wano": "wano arc mood, japanese festival styling, ornate ink accents",
}


def dominant_colors_kmeans(image: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
    arr = np.array(image.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _compactness, labels, centers = cv2.kmeans(
        arr,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    ordered = centers[np.argsort(-counts)]
    return [tuple(int(c) for c in row) for row in ordered]


def analyze_scene(image: Image.Image) -> Dict[str, object]:
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    brightness = float(np.mean(0.2126 * r + 0.7152 * g + 0.0722 * b))
    blue_ratio = float(np.mean((b > r + 20) & (b > g + 20)))
    green_ratio = float(np.mean((g > r + 15) & (g > b + 10)))
    gray_ratio = float(np.mean(np.abs(r - g) < 15) * np.mean(np.abs(g - b) < 15))

    if blue_ratio > 0.18 and green_ratio < 0.15:
        scene_type = "water"
    elif green_ratio > 0.20:
        scene_type = "nature"
    elif gray_ratio > 0.30:
        scene_type = "urban"
    elif brightness > 125:
        scene_type = "outdoor"
    else:
        scene_type = "indoor"

    time_of_day = "day" if brightness >= 105 else "night"
    return {
        "dominant_colors": dominant_colors_kmeans(image, k=5),
        "scene_type": scene_type,
        "time_of_day": time_of_day,
        "brightness": brightness,
        "resolution": (w, h),
    }


def build_dynamic_prompt(
    scene_context: Dict[str, object],
    person_count: int,
    arc: str = "adventure",
) -> tuple[str, str]:
    _ = scene_context, person_count, arc
    positive = (
        "large expressive anime eyes, big bright eyes, thick dark eyelashes, "
        "wide open eyes, detailed iris, nami eyes, robin eyes, "
        "female character, feminine facial features, female one piece character, "
        "defined nose, oda style nose, manga nose, feminine but defined facial "
        "features, strong feminine jaw, bold black outlines on face, "
        "cel shaded skin tones, "
        "masterpiece, best quality, official one piece art, eiichiro oda art style, "
        "anime cel shading, bold black ink outlines, clean line art, "
        "thick bold black outlines, heavy ink lines, high contrast anime shading, "
        "vivid saturated colors, cel shaded with hard shadows, "
        "detailed clothing folds, fabric texture, one piece character outfit, "
        "detailed outdoor background, day lighting, atmospheric depth, vivid environment, "
        "adventure mood, heroic energy, ocean breeze palette, "
        "single character focus, hero shot, "
        "8k, sharp focus, vibrant colors, professional illustration, manga panel quality"
    )
    negative = (
        "photorealistic, realistic skin, soft gradients on face, blurry face, "
        "3d render, smooth skin, photograph, western cartoon, chibi, deformed, "
        "ugly, watermark, text, extra limbs, bad anatomy, low quality, "
        "jpeg artifacts, noise, out of frame, small eyes, narrow eyes, "
        "squinting, half closed eyes, no eyelashes, tired eyes, "
        "male face, masculine features, stub nose, triangle nose"
    )
    return positive, negative

