"""Spatial person detection and visualization helpers for Part 3."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _face_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_person_map(image: Image.Image) -> Dict[str, object]:
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = _face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    persons: List[Dict[str, object]] = []
    for idx, (x, y, w, h) in enumerate(faces):
        cx = x + w // 2
        cy = y + h // 2
        bw = int(w * 2.0)
        bh = int(h * 2.0)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(arr.shape[1], cx + bw // 2)
        y2 = min(arr.shape[0], cy + bh // 2)
        persons.append(
            {
                "face_bbox": (int(x), int(y), int(x + w), int(y + h)),
                "person_bbox": (int(x1), int(y1), int(x2), int(y2)),
                "face_area": int(w * h),
                "x_center": int(cx),
                "label": f"person_{idx + 1}",
            }
        )

    persons.sort(key=lambda p: p["x_center"])
    if persons:
        primary = max(persons, key=lambda p: p["face_area"])
        for p in persons:
            p["is_primary"] = p is primary
            if p["is_primary"]:
                p["label"] = "primary_person"

    return {
        "person_count": len(persons),
        "persons": persons,
        "primary_person": next((p for p in persons if p.get("is_primary")), None),
    }


def save_person_map_visualization(image: Image.Image, person_map: Dict[str, object], output_path: Path) -> None:
    draw_img = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(draw_img)
    for person in person_map.get("persons", []):
        pbox = tuple(person["person_bbox"])
        fbox = tuple(person["face_bbox"])
        label = str(person["label"])
        color = (255, 140, 0) if person.get("is_primary") else (0, 200, 255)
        drawer.rectangle(pbox, outline=color, width=3)
        drawer.rectangle(fbox, outline=(255, 255, 255), width=2)
        drawer.text((pbox[0] + 4, max(0, pbox[1] - 14)), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_img.save(output_path)

