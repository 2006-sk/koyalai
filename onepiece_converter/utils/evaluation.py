"""Evaluation helpers for Part 3 tests."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def dominant_colors(image: Image.Image, k: int = 8) -> List[Tuple[int, int, int]]:
    arr = np.array(image.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _compact, labels, centers = cv2.kmeans(arr, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    ordered = centers[np.argsort(-counts)]
    return [tuple(int(c) for c in row) for row in ordered]


def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    arr = np.uint8([[list(rgb)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return lab


def delta_e(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(rgb_to_lab(c1) - rgb_to_lab(c2)))


def palette_distance(pal_a: Sequence[Tuple[int, int, int]], pal_b: Sequence[Tuple[int, int, int]]) -> float:
    # Symmetric nearest-neighbor mean distance as lightweight EMD proxy.
    if not pal_a or not pal_b:
        return 1e9
    d1 = [min(delta_e(a, b) for b in pal_b) for a in pal_a]
    d2 = [min(delta_e(b, a) for a in pal_a) for b in pal_b]
    return float((np.mean(d1) + np.mean(d2)) / 2.0)


def orb_good_matches(input_image: Image.Image, output_image: Image.Image) -> int:
    gray_in = cv2.cvtColor(np.array(input_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray_out = cv2.cvtColor(np.array(output_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(gray_in, None)
    kp2, des2 = orb.detectAndCompute(gray_out, None)
    if des1 is None or des2 is None or not kp1 or not kp2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if n is not None and m.distance < 0.75 * n.distance]
    return len(good)


class CLIPScorer:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def similarity(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        img = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((img * txt).sum().item())

    def style_score(self, image: Image.Image) -> float:
        positive = "one piece anime eiichiro oda manga bold outlines"
        negative = "photograph realistic photorealistic"
        return self.similarity(image, positive) - self.similarity(image, negative)

