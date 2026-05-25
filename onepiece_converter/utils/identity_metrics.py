"""Identity similarity metrics using DeepFace embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
from deepface import DeepFace
from PIL import Image


def _embedding_from_image(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    reps = DeepFace.represent(
        img_path=arr,
        model_name="VGG-Face",
        detector_backend="skip",
        enforce_detection=False,
    )
    if not reps:
        raise RuntimeError("No embedding returned by DeepFace.")
    emb = np.array(reps[0]["embedding"], dtype=np.float32)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def identity_improvement(
    input_face: Image.Image,
    part1_face: Image.Image,
    part2_face: Image.Image,
) -> Dict[str, float]:
    input_emb = _embedding_from_image(input_face)
    p1_emb = _embedding_from_image(part1_face)
    p2_emb = _embedding_from_image(part2_face)

    p1_sim = cosine_similarity(input_emb, p1_emb)
    p2_sim = cosine_similarity(input_emb, p2_emb)
    improvement = p2_sim - p1_sim

    return {
        "part1_similarity": p1_sim,
        "part2_similarity": p2_sim,
        "improvement": improvement,
    }

