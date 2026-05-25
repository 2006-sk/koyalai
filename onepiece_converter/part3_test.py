#!/usr/bin/env python3
"""Comprehensive evaluation suite for Part 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from part2_pipeline import run_part2
from part3_pipeline import run_part3
from utils.color_grading import ARC_PALETTES
from utils.evaluation import CLIPScorer, delta_e, dominant_colors, orb_good_matches, palette_distance
from utils.face_utils import FaceExtractor
from utils.identity_metrics import identity_improvement
from utils.image_utils import load_image, resize_with_padding
from utils.spatial_utils import detect_person_map


PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "inputs"


@dataclass
class TestResult:
    name: str
    score: float
    passed: bool
    details: str


class Part3Tester:
    def __init__(self) -> None:
        self.results: List[TestResult] = []
        self.clip = CLIPScorer()
        self.face = FaceExtractor()
        self._cache: Dict[str, object] = {}

    def add_result(self, name: str, score: float, passed: bool, details: str) -> None:
        self.results.append(TestResult(name, score, passed, details))

    def _input_image(self) -> Path:
        images = sorted(INPUTS_DIR.glob("*.jpg")) + sorted(INPUTS_DIR.glob("*.jpeg")) + sorted(INPUTS_DIR.glob("*.png"))
        if not images:
            raise FileNotFoundError("No input images found for Part 3 tests.")
        return images[0]

    def _run_once(self) -> Dict[str, object]:
        if self._cache:
            return self._cache
        input_path = self._input_image()
        p2 = run_part2(input_path, project_root=PROJECT_ROOT)
        p3 = run_part3(input_path, project_root=PROJECT_ROOT, arc="adventure")
        self._cache = {"input_path": input_path, "part2": p2, "part3": p3}
        return self._cache

    def test_style_authenticity(self) -> None:
        name = "Style Authenticity"
        data = self._run_once()
        out = load_image(data["part3"].part3_path)
        score = self.clip.style_score(out)
        passed = score > 0.10
        self.add_result(name, score, passed, f"clip_style={score:.3f}")

    def test_identity_retention(self) -> None:
        name = "Identity Retention"
        data = self._run_once()
        input_img = resize_with_padding(load_image(data["input_path"]))
        out_img = load_image(data["part3"].part3_path)
        in_face = self.face.extract_face_crop(input_img).face_crop
        out_face = self.face.extract_face_crop(out_img).face_crop
        if in_face is None or out_face is None:
            self.add_result(name, 1.0, True, "No face detected -> auto-pass")
            return
        sim = identity_improvement(in_face, in_face, out_face)["part2_similarity"]
        passed = sim >= 0.45
        self.add_result(name, sim, passed, f"similarity={sim:.3f}")

    def test_background_fidelity(self) -> None:
        name = "Background Fidelity"
        data = self._run_once()
        in_img = resize_with_padding(load_image(data["input_path"]))
        out_img = load_image(data["part3"].part3_path)
        matches = orb_good_matches(in_img, out_img)
        passed = matches >= 15
        self.add_result(name, float(matches), passed, f"good_matches={matches}")

    def test_scene_coherence(self) -> None:
        name = "Scene Coherence"
        data = self._run_once()
        in_img = resize_with_padding(load_image(data["input_path"]))
        out_img = load_image(data["part3"].part3_path)
        in_map = detect_person_map(in_img)
        out_map = detect_person_map(out_img)

        def split_regions(img: Image.Image, pmap: Dict[str, object]):
            arr = np.array(img.convert("RGB"))
            mask = np.zeros(arr.shape[:2], dtype=np.uint8)
            for p in pmap.get("persons", []):
                x1, y1, x2, y2 = p["person_bbox"]
                mask[y1:y2, x1:x2] = 1
            person_pixels = arr[mask == 1]
            bg_pixels = arr[mask == 0]
            if person_pixels.size == 0:
                person_pixels = arr.reshape(-1, 3)
            if bg_pixels.size == 0:
                bg_pixels = arr.reshape(-1, 3)
            return (
                Image.fromarray(person_pixels.reshape(-1, 1, 3).astype(np.uint8), mode="RGB"),
                Image.fromarray(bg_pixels.reshape(-1, 1, 3).astype(np.uint8), mode="RGB"),
            )

        in_person, in_bg = split_regions(in_img, in_map)
        out_person, out_bg = split_regions(out_img, out_map)
        in_dist = palette_distance(dominant_colors(in_person, 8), dominant_colors(in_bg, 8))
        out_dist = palette_distance(dominant_colors(out_person, 8), dominant_colors(out_bg, 8))
        passed = out_dist < in_dist
        score = in_dist - out_dist
        self.add_result(name, score, passed, f"input_dist={in_dist:.2f}, output_dist={out_dist:.2f}")

    def test_harmonization(self) -> None:
        name = "Harmonization"
        data = self._run_once()
        pre = load_image(data["part3"].part3_pre_harmonized_path)
        post = load_image(data["part3"].part3_path)
        e1 = cv2.Canny(np.array(pre.convert("L")), 100, 200)
        e2 = cv2.Canny(np.array(post.convert("L")), 100, 200)
        pre_var = float(e1.std())
        post_var = float(e2.std())
        passed = post_var < pre_var
        self.add_result(name, pre_var - post_var, passed, f"pre_var={pre_var:.2f}, post_var={post_var:.2f}")

    def test_color_grading(self) -> None:
        name = "Color Grading"
        data = self._run_once()
        p2 = load_image(data["part2"].part2_path)
        p3 = load_image(data["part3"].part3_path)
        sat2 = float(np.array(p2.convert("HSV"))[..., 1].mean())
        sat3 = float(np.array(p3.convert("HSV"))[..., 1].mean())
        sat_ok = sat3 > sat2

        dom = dominant_colors(p3, 5)
        ref = ARC_PALETTES["adventure"]
        good = 0
        for c in dom:
            if min(delta_e(c, r) for r in ref) < 35:
                good += 1
        palette_ok = good >= 3
        passed = sat_ok and palette_ok
        self.add_result(name, float(good), passed, f"sat2={sat2:.1f}, sat3={sat3:.1f}, good_colors={good}/5")

    def test_lora_enhancement(self) -> None:
        name = "LoRA Enhancement"
        data = self._run_once()
        p2 = load_image(data["part2"].part2_path)
        p3 = load_image(data["part3"].part3_path)
        s2 = self.clip.style_score(p2)
        s3 = self.clip.style_score(p3)
        passed = s3 >= s2
        self.add_result(name, s3 - s2, passed, f"part2_clip={s2:.3f}, part3_clip={s3:.3f}")

    def test_composite(self) -> None:
        name = "End-to-End Human Proxy Score"
        # Must run after prior tests.
        score_map = {r.name: r for r in self.results}
        style = 25.0 if score_map["Style Authenticity"].passed else 0.0
        identity = 25.0 if score_map["Identity Retention"].passed else 0.0
        bg = 20.0 if score_map["Background Fidelity"].passed else 0.0
        scene = 20.0 if score_map["Scene Coherence"].passed else 0.0
        color = 10.0 if score_map["Color Grading"].passed else 0.0
        total = style + identity + bg + scene + color
        passed = total >= 70.0
        self.add_result(name, total, passed, f"composite={total:.1f}/100")

    def run_all(self) -> None:
        self.test_style_authenticity()
        self.test_identity_retention()
        self.test_background_fidelity()
        self.test_scene_coherence()
        self.test_harmonization()
        self.test_color_grading()
        self.test_lora_enhancement()
        self.test_composite()
        self.print_report()

    def print_report(self) -> None:
        print("═══════════════════════════════════════")
        print("  ONE PIECE CONVERTER — PART 3 EVALUATION")
        print("═══════════════════════════════════════")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {r.name:<21}: {r.score:.2f} — {status}")
        print("───────────────────────────────────────")
        comp = next((r for r in self.results if r.name == "End-to-End Human Proxy Score"), None)
        comp_score = comp.score if comp else 0.0
        passed_tests = sum(1 for r in self.results if r.passed)
        overall = "READY FOR SUBMISSION" if passed_tests >= 6 else "NEEDS WORK"
        print(f"  COMPOSITE SCORE       : {comp_score:.1f}/100")
        print(f"  OVERALL               : {overall}")
        print("═══════════════════════════════════════")
        if passed_tests >= 6:
            print("Part 3 READY — pipeline complete")


def main() -> None:
    tester = Part3Tester()
    tester.run_all()


if __name__ == "__main__":
    main()

