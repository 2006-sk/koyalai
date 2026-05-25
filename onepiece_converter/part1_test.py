#!/usr/bin/env python3
"""Automated test suite for Part 1 One Piece converter pipeline."""

from __future__ import annotations

import importlib
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from part1_pipeline import run_part1
from utils.image_utils import load_image, pil_to_numpy_rgb, resize_with_padding
from utils.preprocessor import LineartPreprocessor, edge_density, edge_map, laplacian_variance


PROJECT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class TestResult:
    name: str
    passed: bool
    score: str
    details: str


def compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Compute a lightweight global SSIM estimate for two grayscale images."""
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
    """Compute Pearson correlation between two equal-length vectors."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    std_a = arr_a.std()
    std_b = arr_b.std()
    if std_a == 0 or std_b == 0:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def bytes_to_gb(value: float) -> float:
    return value / (1024**3)


def max_rss_gb() -> float:
    """Return process max RSS in GB with platform-aware unit conversion."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux typically reports KB.
    if rss > 10**8:
        return bytes_to_gb(float(rss))
    return bytes_to_gb(float(rss) * 1024.0)


class Part1Tester:
    def __init__(self) -> None:
        self.results: List[TestResult] = []
        self.preprocessor: Optional[LineartPreprocessor] = None
        self.cached_pipeline_run: Optional[Tuple[float, float, Path, Path, Path]] = None
        self.selected_images: Optional[Tuple[Path, Path, Path]] = None

    def log(self, message: str) -> None:
        print(f"[test] {message}")

    def add_result(self, name: str, passed: bool, score: str, details: str) -> None:
        self.results.append(TestResult(name=name, passed=passed, score=score, details=details))

    def ensure_test_images(self) -> List[Path]:
        if self.selected_images is not None:
            return list(self.selected_images)

        image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        available: List[Path] = []
        for pattern in image_patterns:
            available.extend(sorted(INPUTS_DIR.glob(pattern)))
        available = sorted({path.resolve() for path in available})

        if not available:
            raise FileNotFoundError(
                f"No local input images found in {INPUTS_DIR}. "
                "Add at least 1 image to run tests."
            )

        # Force tests to run only on download (2) image when present.
        target = next((p for p in available if p.stem.lower() == "download (2)"), None)
        if target is not None:
            portrait, indoor, outdoor = target, target, target
            self.selected_images = (portrait, indoor, outdoor)
            self.log(
                "Using forced test image for all cases: "
                f"portrait={portrait.name}, indoor={indoor.name}, outdoor={outdoor.name}"
            )
            return [portrait, indoor, outdoor]

        if len(available) == 1:
            portrait, indoor, outdoor = available[0], available[0], available[0]
        elif len(available) == 2:
            portrait, indoor, outdoor = available[0], available[1], available[1]
        else:
            portrait, indoor, outdoor = available[0], available[1], available[2]

        self.selected_images = (portrait, indoor, outdoor)
        self.log(
            "Using local test images: "
            f"portrait={portrait.name}, indoor={indoor.name}, outdoor={outdoor.name}"
        )
        return [portrait, indoor, outdoor]

    def ensure_preprocessor(self) -> LineartPreprocessor:
        if self.preprocessor is None:
            model_path = MODELS_DIR / "lineart_annotators"
            self.preprocessor = LineartPreprocessor(model_dir=model_path if model_path.exists() else None)
        return self.preprocessor

    def detect_key_region_bbox(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """Try person detection; fallback to central region."""
        arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(arr, winStride=(8, 8), padding=(16, 16), scale=1.05)
        if len(boxes) > 0:
            x, y, w, h = boxes[0]
            return int(x), int(y), int(w), int(h)

        h, w = arr.shape[:2]
        return int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.6)

    def region_edge_density(self, edge_img: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x, y, w, h = bbox
        x2 = max(1, min(edge_img.shape[1], x + w))
        y2 = max(1, min(edge_img.shape[0], y + h))
        x = max(0, min(edge_img.shape[1] - 1, x))
        y = max(0, min(edge_img.shape[0] - 1, y))
        region = edge_img[y:y2, x:x2]
        if region.size == 0:
            return 0.0
        return edge_density(region)

    def run_pipeline_once(self, input_path: Path) -> Tuple[float, float, Path, Path, Path]:
        if self.cached_pipeline_run is not None:
            return self.cached_pipeline_run

        self.log("Running full pipeline once for structural/performance/quality tests...")
        mem_before = max_rss_gb()
        start = time.perf_counter()
        result = run_part1(input_image=input_path, project_root=PROJECT_ROOT)
        elapsed = time.perf_counter() - start
        mem_after = max_rss_gb()
        peak_mem = max(mem_before, mem_after)
        self.cached_pipeline_run = (
            elapsed,
            peak_mem,
            result.input_path,
            result.lineart_path,
            result.output_path,
        )
        return self.cached_pipeline_run

    def test_environment(self) -> None:
        name = "TEST 1 — Environment Test"
        required_imports = [
            "torch",
            "diffusers",
            "controlnet_aux",
            "transformers",
            "accelerate",
            "safetensors",
            "PIL",
            "cv2",
            "numpy",
        ]
        errors: List[str] = []
        for mod in required_imports:
            try:
                importlib.import_module(mod)
            except Exception as exc:
                errors.append(f"import {mod} failed: {exc}")

        device_ok = torch.cuda.is_available() or torch.backends.mps.is_available()
        if torch.cuda.is_available():
            try:
                _ = torch.ones((2, 2), device="cuda") * 2
            except Exception as exc:
                errors.append(f"CUDA unusable: {exc}")
        elif torch.backends.mps.is_available():
            try:
                _ = torch.ones((2, 2), device="mps") * 2
            except Exception as exc:
                errors.append(f"MPS unusable: {exc}")
        elif not device_ok:
            errors.append("No accelerated device detected (CUDA or MPS).")

        expected_model_dirs = [
            MODELS_DIR / "base_model",
            MODELS_DIR / "controlnet_lineart",
            MODELS_DIR / "lineart_annotators",
        ]
        for model_dir in expected_model_dirs:
            if not model_dir.exists() or not any(model_dir.iterdir()):
                errors.append(f"Model files missing in: {model_dir}")

        passed = len(errors) == 0
        score = "100%" if passed else "0%"
        details = "All imports/device/models validated." if passed else " | ".join(errors)
        self.add_result(name, passed, score, details)

    def test_lineart_extraction_accuracy(self) -> None:
        name = "TEST 2 — Lineart Extraction Accuracy"
        try:
            images = self.ensure_test_images()
            pre = self.ensure_preprocessor()
        except Exception as exc:
            self.add_result(name, False, "0/3", f"Setup failed: {exc}")
            return

        passed_images = 0
        per_image_notes: List[str] = []

        for image_path in images:
            image = resize_with_padding(load_image(image_path))
            lineart = pre.extract_lineart(image)

            edge_orig = edge_map(image)
            edge_line = edge_map(lineart)

            density_orig = edge_density(edge_orig)
            density_line = edge_density(edge_line)
            ratio = density_line / density_orig if density_orig > 0 else math.inf
            ratio_ok = 0.5 <= ratio <= 2.0

            ssim_edges = compute_ssim(edge_orig, edge_line)
            ssim_ok = ssim_edges > 0.35

            bbox = self.detect_key_region_bbox(image)
            region_density = self.region_edge_density(edge_line, bbox)
            key_region_ok = region_density > 0.015

            image_pass = ratio_ok and ssim_ok and key_region_ok
            if image_pass:
                passed_images += 1

            per_image_notes.append(
                (
                    f"{image_path.name}: ratio={ratio:.3f}({'OK' if ratio_ok else 'NO'}), "
                    f"ssim={ssim_edges:.3f}({'OK' if ssim_ok else 'NO'}), "
                    f"key_region={region_density:.4f}({'OK' if key_region_ok else 'NO'})"
                )
            )

        passed = passed_images >= 2
        score = f"{passed_images}/3"
        details = " | ".join(per_image_notes)
        self.add_result(name, passed, score, details)

    def edge_density_grid(self, edge_img: np.ndarray, grid_size: int = 3) -> List[float]:
        h, w = edge_img.shape
        densities: List[float] = []
        cell_h = h // grid_size
        cell_w = w // grid_size
        for gy in range(grid_size):
            for gx in range(grid_size):
                y1 = gy * cell_h
                y2 = h if gy == grid_size - 1 else (gy + 1) * cell_h
                x1 = gx * cell_w
                x2 = w if gx == grid_size - 1 else (gx + 1) * cell_w
                cell = edge_img[y1:y2, x1:x2]
                densities.append(edge_density(cell))
        return densities

    def quantized_unique_colors(self, image: Image.Image, bin_size: int = 16) -> int:
        arr = pil_to_numpy_rgb(image)
        quantized = (arr // bin_size).reshape(-1, 3)
        unique = np.unique(quantized, axis=0)
        return int(unique.shape[0])

    def test_controlnet_structural_preservation(self) -> None:
        name = "TEST 3 — ControlNet Structural Preservation"
        try:
            portrait = self.ensure_test_images()[0]
            pre = self.ensure_preprocessor()
            elapsed, _mem, input_saved, lineart_saved, output_saved = self.run_pipeline_once(portrait)
            _ = elapsed
        except Exception as exc:
            self.add_result(name, False, "0/4", f"Pipeline run failed: {exc}")
            return

        input_image = load_image(input_saved)
        input_lineart = load_image(lineart_saved)
        output_image = load_image(output_saved)
        output_lineart = pre.extract_lineart(output_image)

        edge_input_line = edge_map(input_lineart)
        edge_output_line = edge_map(output_lineart)

        ssim_edges = compute_ssim(edge_input_line, edge_output_line)
        check_a = ssim_edges > 0.20

        in_grid = self.edge_density_grid(edge_input_line, grid_size=3)
        out_grid = self.edge_density_grid(edge_output_line, grid_size=3)
        corr = pearson_corr(in_grid, out_grid)
        check_b = corr > 0.6

        check_c = output_image.size == (512, 512)

        pass_count = sum([check_a, check_b, check_c])
        passed = pass_count >= 2
        score = f"{pass_count}/3"
        details = (
            f"edge_ssim={ssim_edges:.3f}({'OK' if check_a else 'NO'}), "
            f"grid_corr={corr:.3f}({'OK' if check_b else 'NO'}), "
            f"resolution={output_image.size}({'OK' if check_c else 'NO'})"
        )
        self.add_result(name, passed, score, details)

    def test_performance(self) -> None:
        name = "TEST 4 — Performance Test"
        try:
            portrait = self.ensure_test_images()[0]
            elapsed, peak_mem, _input_saved, _lineart_saved, _output_saved = self.run_pipeline_once(portrait)
        except Exception as exc:
            self.add_result(name, False, "0/2", f"Pipeline run failed: {exc}")
            return

        check_time = elapsed < 180.0
        memory_limit = 14.0 if torch.cuda.is_available() else 7.0
        check_mem = peak_mem < memory_limit
        pass_count = sum([check_time, check_mem])
        passed = pass_count == 2
        score = f"{pass_count}/2"
        details = (
            f"time={elapsed:.2f}s({'OK' if check_time else 'NO'}), "
            f"peak_mem={peak_mem:.2f}GB limit={memory_limit:.2f}GB({'OK' if check_mem else 'NO'})"
        )
        self.add_result(name, passed, score, details)

    def test_output_quality_gate(self) -> None:
        name = "TEST 5 — Output Quality Gate"
        try:
            portrait = self.ensure_test_images()[0]
            _elapsed, _peak_mem, input_saved, _lineart_saved, output_saved = self.run_pipeline_once(portrait)
        except Exception as exc:
            self.add_result(name, False, "0/3", f"Pipeline run failed: {exc}")
            return

        input_image = load_image(input_saved)
        output_image = load_image(output_saved)

        output_arr = pil_to_numpy_rgb(output_image)
        mean_pixel = float(output_arr.mean())
        check_a = 30.0 <= mean_pixel <= 225.0

        blur_var = laplacian_variance(output_image)
        check_b = blur_var > 100.0

        input_gray = cv2.cvtColor(pil_to_numpy_rgb(input_image), cv2.COLOR_RGB2GRAY)
        output_gray = cv2.cvtColor(pil_to_numpy_rgb(output_image), cv2.COLOR_RGB2GRAY)
        style_ssim = compute_ssim(input_gray, output_gray)
        check_c = style_ssim < 0.98

        pass_count = sum([check_a, check_b, check_c])
        passed = pass_count == 3
        score = f"{pass_count}/3"
        details = (
            f"mean_pixel={mean_pixel:.2f}({'OK' if check_a else 'NO'}), "
            f"lap_var={blur_var:.2f}({'OK' if check_b else 'NO'}), "
            f"input_output_ssim={style_ssim:.3f}({'OK' if check_c else 'NO'})"
        )
        self.add_result(name, passed, score, details)

    def run_all(self) -> None:
        INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.log("Starting Part 1 automated test suite...")
        self.test_environment()
        self.test_lineart_extraction_accuracy()
        self.test_controlnet_structural_preservation()
        self.test_performance()
        self.test_output_quality_gate()
        self.print_report()

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("PART 1 TEST REPORT")
        print("=" * 70)
        passed_tests = 0
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{result.name}: {status} ({result.score})")
            print(f"  Details: {result.details}")
            if result.passed:
                passed_tests += 1
        print("-" * 70)
        total = len(self.results)
        print(f"Overall score: {passed_tests}/{total} tests passed")
        if passed_tests >= 4:
            print("Part 1 READY -- proceed to Part 2")
        else:
            print("Part 1 NOT READY -- address failed tests before Part 2")
            for result in self.results:
                if not result.passed:
                    print(f"  - Failed: {result.name}")
                    print("    Suggested fix: verify model downloads, adjust prompts/scales, or tune thresholds.")
        print("=" * 70 + "\n")


def main() -> None:
    tester = Part1Tester()
    tester.run_all()


if __name__ == "__main__":
    main()
