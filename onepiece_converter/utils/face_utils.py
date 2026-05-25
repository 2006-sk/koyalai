"""Face detection and crop utilities for Part 2 identity conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


LEFT_EYE_IDS = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE_IDS = [362, 263, 387, 386, 385, 384, 398]


@dataclass
class FaceDetectionResult:
    face_crop: Optional[Image.Image]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    face_area_ratio: float
    eyes_detected: bool
    laplacian_variance: float


class FaceExtractor:
    """MediaPipe-based face detection and crop extraction."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_primary_face_bbox(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        result = self.detector.process(rgb)
        if not result.detections:
            return None
        box = result.detections[0].location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, int((box.xmin + box.width) * w))
        y2 = min(h, int((box.ymin + box.height) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        width: int,
        height: int,
        padding_ratio: float = 0.25,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        ex1 = max(0, x1 - pad_x)
        ey1 = max(0, y1 - pad_y)
        ex2 = min(width, x2 + pad_x)
        ey2 = min(height, y2 + pad_y)
        return ex1, ey1, ex2, ey2

    def _eyes_detected(self, face_crop: Image.Image) -> bool:
        rgb = np.array(face_crop.convert("RGB"))
        mesh_result = self.mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            return False
        landmarks = mesh_result.multi_face_landmarks[0].landmark
        return all(0.0 <= landmarks[i].x <= 1.0 and 0.0 <= landmarks[i].y <= 1.0 for i in LEFT_EYE_IDS + RIGHT_EYE_IDS)

    def _laplacian_var(self, image: Image.Image) -> float:
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def extract_face_crop(self, image: Image.Image, target_size: int = 256) -> FaceDetectionResult:
        rgb = np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        bbox = self.detect_primary_face_bbox(image)
        if bbox is None:
            return FaceDetectionResult(
                face_crop=None,
                bbox_xyxy=None,
                face_area_ratio=0.0,
                eyes_detected=False,
                laplacian_variance=0.0,
            )

        ex1, ey1, ex2, ey2 = self._expand_bbox(bbox, width=w, height=h)
        crop = image.crop((ex1, ey1, ex2, ey2)).resize((target_size, target_size), Image.Resampling.LANCZOS)
        bbox_area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        full_area = float(max(1, w * h))
        area_ratio = bbox_area / full_area

        eyes_ok = self._eyes_detected(crop)
        blur_score = self._laplacian_var(crop)

        return FaceDetectionResult(
            face_crop=crop,
            bbox_xyxy=bbox,
            face_area_ratio=area_ratio,
            eyes_detected=eyes_ok,
            laplacian_variance=blur_score,
        )


def save_face_crop(face_crop: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    face_crop.save(output_path)

