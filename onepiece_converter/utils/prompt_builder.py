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
    base = (
        "one piece anime style, eiichiro oda, shounen manga, "
        "bold black outlines, flat cel shading, bright colors, "
        "thick bold black outlines, heavy ink lines, strong line art, "
        "high contrast anime shading, vivid saturated colors, "
        "cel shaded with hard shadows"
    )
    scene_desc = (
        f"{scene_context['scene_type']} scene, {scene_context['time_of_day']} lighting, "
        "coherent environment, vivid background"
    )
    arc_desc = ARC_DESCRIPTORS.get(arc, ARC_DESCRIPTORS["adventure"])
    face_desc = (
        "female character, feminine facial features, female anime face, "
        "small delicate nose, soft feminine jaw, female one piece character"
    )
    multi_desc = (
        "multiple characters, each with distinct appearance"
        if person_count > 1
        else "single character focus"
    )
    positive = f"{base}, {scene_desc}, {arc_desc}, {face_desc}, {multi_desc}"
    negative = (
        "realistic, photorealistic, 3d, horror, scary, gradient skin, blurry, deformed, "
        "watermark, ugly, extra limbs, bad anatomy, male face, masculine features, "
        "male nose, stub nose, triangle nose, male jaw, masculine"
    )
    return positive, negative

