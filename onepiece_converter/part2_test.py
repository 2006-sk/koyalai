#!/usr/bin/env python3
"""Automated test suite for Part 2 identity-preserving pipeline."""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

from part1_pipeline import run_part1
from part2_pipeline import run_part2
from utils.face_utils import FaceExtractor
from utils.identity_metrics import identity_improvement
from utils.image_utils import load_image, pil_to_numpy_rgb, resize_with_padding
from utils.preprocessor import edge_density, edge_map, laplacian_variance


PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "inputs"


@dataclass
class TestResult:
    name: str
    passed: bool
    score: str
    details: str


def compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov_ab = float(((a - mu_a) * (b - mu_b)).mean())
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def pearson_corr(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    std_a = arr_a.std()
    std_b = arr_b.std()
    if std_a == 0 or std_b == 0:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def edge_density_grid(edge_img: np.ndarray, grid_size: int = 3) -> List[float]:
    h, w = edge_img.shape
    values: List[float] = []
    cell_h = h // grid_size
    cell_w = w // grid_size
    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * cell_h
            y2 = h if gy == grid_size - 1 else (gy + 1) * cell_h
            x1 = gx * cell_w
            x2 = w if gx == grid_size - 1 else (gx + 1) * cell_w
            values.append(edge_density(edge_img[y1:y2, x1:x2]))
    return values


def max_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10**8:
        return rss / float(1024**3)
    return (rss * 1024.0) / float(1024**3)


class Part2Tester:
    def __init__(self) -> None:
        self.results: List[TestResult] = []
        self.face_extractor = FaceExtractor()

    def add_result(self, name: str, passed: bool, score: str, details: str) -> None:
        self.results.append(TestResult(name=name, passed=passed, score=score, details=details))

    def _images(self) -> List[Path]:
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        images: List[Path] = []
        for p in patterns:
            images.extend(sorted(INPUTS_DIR.glob(p)))
        unique = sorted({img.resolve() for img in images})
        if not unique:
            raise FileNotFoundError("No images found in inputs/ for Part 2 tests.")
        return unique

    def test_face_detection_reliability(self) -> None:
        name = "TEST 1 — Face Detection Reliability"
        images = self._images()
        detected = 0
        notes: List[str] = []
        for img_path in images:
            img = resize_with_padding(load_image(img_path))
            result = self.face_extractor.extract_face_crop(img)
            ok = result.face_crop is not None
            detected += int(ok)
            notes.append(f"{img_path.name}:{'OK' if ok else 'NO'}")

        rate = detected / max(1, len(images))
        passed = rate >= 0.8
        self.add_result(name, passed, f"{rate:.2%}", " | ".join(notes))

    def test_face_crop_quality(self) -> None:
        name = "TEST 2 — Face Crop Quality"
        images = self._images()
        detected_results = []
        notes: List[str] = []
        for img_path in images:
            img = resize_with_padding(load_image(img_path))
            result = self.face_extractor.extract_face_crop(img)
            if result.face_crop is None:
                continue
            area_ok = 0.40 <= result.face_area_ratio <= 0.90
            eyes_ok = result.eyes_detected
            blur_ok = result.laplacian_variance > 80.0
            detected_results.append(area_ok and eyes_ok and blur_ok)
            notes.append(
                f"{img_path.name}: area={result.face_area_ratio:.3f}({'OK' if area_ok else 'NO'}) "
                f"eyes={'OK' if eyes_ok else 'NO'} "
                f"lap={result.laplacian_variance:.2f}({'OK' if blur_ok else 'NO'})"
            )
        if not detected_results:
            self.add_result(name, False, "0%", "No detected faces available for crop quality test.")
            return
        pass_rate = sum(detected_results) / len(detected_results)
        passed = pass_rate >= 0.8
        self.add_result(name, passed, f"{pass_rate:.2%}", " | ".join(notes))

    def test_identity_preservation(self) -> None:
        name = "TEST 3 — Identity Preservation Score"
        image = self._images()[0]
        try:
            p1 = run_part1(image, project_root=PROJECT_ROOT)
            p2 = run_part2(image, project_root=PROJECT_ROOT)
            input_img = resize_with_padding(load_image(image))
            p1_img = load_image(p1.output_path)
            p2_img = load_image(p2.part2_path)

            input_face = self.face_extractor.extract_face_crop(input_img).face_crop
            p1_face = self.face_extractor.extract_face_crop(p1_img).face_crop
            p2_face = self.face_extractor.extract_face_crop(p2_img).face_crop
            if input_face is None or p1_face is None or p2_face is None:
                self.add_result(name, False, "0/2", "Face crop unavailable for one or more images.")
                return

            metrics = identity_improvement(input_face, p1_face, p2_face)
            improved = metrics["improvement"] >= 0.10
            absolute = metrics["part2_similarity"] >= 0.40
            pass_count = sum([improved, absolute])
            passed = pass_count == 2
            details = (
                f"part1_sim={metrics['part1_similarity']:.3f}, "
                f"part2_sim={metrics['part2_similarity']:.3f}, "
                f"improvement={metrics['improvement']:.3f} "
                f"({'OK' if improved else 'NO'}) | abs({'OK' if absolute else 'NO'})"
            )
            print(f"Identity preservation improved by {metrics['improvement'] * 100:.2f}% over baseline")
            self.add_result(name, passed, f"{pass_count}/2", details)
        except Exception as exc:
            self.add_result(name, False, "0/2", f"Identity test failed: {exc}")

    def test_background_consistency(self) -> None:
        name = "TEST 4 — Background Consistency Maintained"
        image = self._images()[0]
        try:
            p2 = run_part2(image, project_root=PROJECT_ROOT)
            input_img = resize_with_padding(load_image(image))
            out_img = load_image(p2.part2_path)

            edge_in = edge_map(input_img)
            edge_out = edge_map(out_img)
            ssim_edges = compute_ssim(edge_in, edge_out)
            check_a = ssim_edges > 0.20
            corr = pearson_corr(edge_density_grid(edge_in), edge_density_grid(edge_out))
            check_b = corr > 0.6
            check_c = out_img.size == (512, 512)
            pass_count = sum([check_a, check_b, check_c])
            passed = pass_count >= 2
            details = (
                f"edge_ssim={ssim_edges:.3f}({'OK' if check_a else 'NO'}), "
                f"grid_corr={corr:.3f}({'OK' if check_b else 'NO'}), "
                f"resolution={out_img.size}({'OK' if check_c else 'NO'})"
            )
            self.add_result(name, passed, f"{pass_count}/3", details)
        except Exception as exc:
            self.add_result(name, False, "0/3", f"Background consistency test failed: {exc}")

    def test_memory_safety(self) -> None:
        name = "TEST 5 — Memory Safety"
        image = self._images()[0]
        rss_values: List[float] = []
        run_times: List[float] = []
        for idx in range(3):
            start = time.perf_counter()
            _ = run_part2(image, project_root=PROJECT_ROOT)
            run_times.append(time.perf_counter() - start)
            rss_values.append(psutil.Process().memory_info().rss / float(1024**2))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        growth = max(rss_values) - min(rss_values) if rss_values else 10**9
        passed = growth < 500.0
        details = (
            f"rss_mb={','.join(f'{v:.1f}' for v in rss_values)} "
            f"growth_mb={growth:.1f} ({'OK' if passed else 'NO'}) "
            f"times_s={','.join(f'{t:.1f}' for t in run_times)}"
        )
        self.add_result(name, passed, "PASS" if passed else "FAIL", details)

    def test_no_face_fallback(self) -> None:
        name = "TEST 6 — No Face Fallback"
        blank_path = INPUTS_DIR / "_part2_no_face_blank.png"
        blank = Image.new("RGB", (512, 512), (255, 255, 255))
        blank.save(blank_path)
        try:
            result = run_part2(blank_path, project_root=PROJECT_ROOT)
            out_img = load_image(result.part2_path)
            mean_pixel = float(pil_to_numpy_rgb(out_img).mean())
            blur_var = laplacian_variance(out_img)
            quality_ok = 30.0 <= mean_pixel <= 225.0 and blur_var > 30.0
            fallback_ok = bool(result.metadata.get("no_face_fallback", False))
            passed = fallback_ok and quality_ok
            details = (
                f"fallback={fallback_ok}, mean={mean_pixel:.2f}, "
                f"lap_var={blur_var:.2f}, quality_ok={quality_ok}"
            )
            self.add_result(name, passed, "PASS" if passed else "FAIL", details)
        except Exception as exc:
            self.add_result(name, False, "FAIL", f"No-face fallback failed: {exc}")
        finally:
            if blank_path.exists():
                blank_path.unlink()

    def run_all(self) -> None:
        self.test_face_detection_reliability()
        self.test_face_crop_quality()
        self.test_identity_preservation()
        self.test_background_consistency()
        self.test_memory_safety()
        self.test_no_face_fallback()
        self.print_report()

    def print_report(self) -> None:
        print("\n" + "=" * 74)
        print("PART 2 TEST REPORT")
        print("=" * 74)
        passed_count = 0
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{result.name}: {status} ({result.score})")
            print(f"  Details: {result.details}")
            if result.passed:
                passed_count += 1
        total = len(self.results)
        print("-" * 74)
        print(f"Overall score: {passed_count}/{total} tests passed")
        if passed_count >= 5:
            print("Part 2 READY — proceed to Part 3")
        else:
            print("Part 2 NOT READY — address failed tests before Part 3")
        print("=" * 74 + "\n")


def main() -> None:
    tester = Part2Tester()
    tester.run_all()


if __name__ == "__main__":
    main()

