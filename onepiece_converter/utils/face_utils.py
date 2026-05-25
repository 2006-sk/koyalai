"""Face detection and crop utilities for Part 2 identity conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class FaceDetectionResult:
    face_crop: Optional[Image.Image]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    face_area_ratio: float
    eyes_detected: bool
    laplacian_variance: float


def detect_and_crop_face(image: Image.Image) -> Optional[Image.Image]:
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    padding = int(0.2 * max(w, h))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_array.shape[1], x + w + padding)
    y2 = min(img_array.shape[0], y + h + padding)
    face_crop = image.crop((x1, y1, x2, y2)).resize((256, 256), Image.Resampling.LANCZOS)
    return face_crop


def has_eyes(face_crop: Image.Image) -> bool:
    img_array = np.array(face_crop.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_detector.detectMultiScale(gray, 1.1, 3)
    return len(eyes) >= 1


def is_blurry(image: Image.Image, threshold: float = 80.0) -> bool:
    img_array = np.array(image.convert("L"))
    lap_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    return float(lap_var) < threshold


class FaceExtractor:
    """OpenCV Haar-based face extraction wrapper."""

    def extract_face_crop(self, image: Image.Image, target_size: int = 256) -> FaceDetectionResult:
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) == 0:
            return FaceDetectionResult(None, None, 0.0, False, 0.0)

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        padding = int(0.2 * max(fw, fh))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + fw + padding)
        y2 = min(h, y + fh + padding)
        face_crop = image.crop((x1, y1, x2, y2)).resize((target_size, target_size), Image.Resampling.LANCZOS)

        bbox_area = float(fw * fh)
        full_area = float(max(1, w * h))
        area_ratio = bbox_area / full_area
        eyes_ok = has_eyes(face_crop)
        lap_var = float(cv2.Laplacian(np.array(face_crop.convert("L")), cv2.CV_64F).var())

        return FaceDetectionResult(
            face_crop=face_crop,
            bbox_xyxy=(int(x), int(y), int(x + fw), int(y + fh)),
            face_area_ratio=area_ratio,
            eyes_detected=eyes_ok,
            laplacian_variance=lap_var,
        )


def save_face_crop(face_crop: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    face_crop.save(output_path)

