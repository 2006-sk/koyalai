#!/usr/bin/env python3
"""Part 1 SDXL ControlNet pipeline with Canny conditioning."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image
from safetensors.torch import load_file

from utils.image_utils import (
    load_image,
    resize_with_padding,
    save_image_with_metadata,
    save_metadata_json,
    save_side_by_side,
    timestamp_string,
)


def get_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"[device] CUDA detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("[device] MPS detected: Apple Silicon")
    else:
        device = "cpu"
        dtype = torch.float32
        print("[device] WARNING: Running on CPU — will be slow")
    return device, dtype


DEVICE, DTYPE = get_device()

PROMPT = (
    "masterpiece, best quality, anime style, one piece, eiichiro oda art style, "
    "bold black outlines, flat cel shading, vibrant colors, sharp details, adventure manga panel"
)
NEGATIVE_PROMPT = (
    "worst quality, low quality, jpeg artifacts, blurry, ugly, deformed, extra limbs, bad anatomy, "
    "realistic photo, 3d render, watermark, signature, gradient shading, heavy shadows"
)
DEFAULT_STRENGTH = 0.6
DEFAULT_CONTROLNET_SCALE = 0.7
GENERATION_RESOLUTION = (768, 768)
OUTPUT_RESOLUTION = (512, 512)


@dataclass
class PipelineResult:
    input_path: Path
    lineart_path: Path
    output_path: Path
    comparison_path: Path
    metadata_path: Path
    metadata: Dict[str, object]


def clear_device_cache() -> None:
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


def list_input_images(inputs_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(inputs_dir.glob(pattern)))
    return sorted(files)


def get_input_image_path(project_root: Path, provided: Optional[str]) -> Path:
    inputs_dir = project_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    if provided:
        path = Path(provided)
        if not path.is_absolute():
            path = (project_root / provided).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        return path
    images = list_input_images(inputs_dir)
    if not images:
        raise FileNotFoundError(f"No input image found in {inputs_dir}. Add an image or pass --input.")
    return images[0]


def get_canny_image(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb)


def build_pipeline(
    project_root: Path,
    device_preference: str = "auto",
    diagnostic: bool = False,
) -> tuple[StableDiffusionXLControlNetImg2ImgPipeline, str, torch.dtype]:
    _ = diagnostic
    model_root = project_root / "models"
    base_model_path = model_root / "sdxl_base"
    controlnet_path = model_root / "sdxl_controlnet"
    controlnet_v2 = controlnet_path / "diffusion_pytorch_model_V2.safetensors"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing SDXL base directory: {base_model_path}")
    if not controlnet_path.exists() or not controlnet_v2.exists():
        raise FileNotFoundError(f"Missing SDXL ControlNet V2 weights: {controlnet_v2}")

    device = DEVICE if device_preference == "auto" else device_preference
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path.as_posix(),
        torch_dtype=dtype,
        use_safetensors=True,
    )
    state_dict = load_file(controlnet_v2.as_posix())
    controlnet.load_state_dict(state_dict, strict=False)
    controlnet = controlnet.to(device)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model_path.as_posix(),
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    ).to(device)
    pipe.enable_model_cpu_offload()
    return pipe, device, dtype


def run_part1(
    input_image: Path,
    project_root: Optional[Path] = None,
    strength: float = DEFAULT_STRENGTH,
    controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 25,
    seed: int = 42,
    device_preference: str = "auto",
    diagnostic: bool = False,
) -> PipelineResult:
    root = (project_root or Path(__file__).resolve().parent).resolve()
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] Stage 1/5: Loading and resizing input image...")
    raw_image = load_image(input_image)
    input_768 = resize_with_padding(raw_image, size=GENERATION_RESOLUTION)
    input_512 = input_768.resize(OUTPUT_RESOLUTION, Image.Resampling.LANCZOS)

    print("[pipeline] Stage 2/5: Building Canny conditioning map...")
    canny_768 = get_canny_image(input_768)
    canny_512 = canny_768.resize(OUTPUT_RESOLUTION, Image.Resampling.LANCZOS)

    print("[pipeline] Stage 3/5: Initializing SDXL pipeline...")
    pipe, device, dtype = build_pipeline(root, device_preference=device_preference, diagnostic=diagnostic)
    clear_device_cache()

    print("[pipeline] Stage 4/5: Running ControlNet generation...")
    gen_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=input_768,
        control_image=canny_768,
        strength=strength,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    output_768 = result.images[0].resize(GENERATION_RESOLUTION, Image.Resampling.LANCZOS)
    output_512 = output_768.resize(OUTPUT_RESOLUTION, Image.Resampling.LANCZOS)

    print("[pipeline] Stage 5/5: Saving outputs and metadata...")
    run_id = f"{input_image.stem}_{timestamp_string()}"
    original_path = outputs_dir / f"{run_id}_original.png"
    lineart_path = outputs_dir / f"{run_id}_lineart.png"
    output_path = outputs_dir / f"{run_id}_styled.png"
    metadata_path = outputs_dir / f"{run_id}_metadata.json"

    metadata: Dict[str, object] = {
        "input_image": input_image.as_posix(),
        "resolution": {"width": OUTPUT_RESOLUTION[0], "height": OUTPUT_RESOLUTION[1]},
        "generation_resolution": {"width": GENERATION_RESOLUTION[0], "height": GENERATION_RESOLUTION[1]},
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "strength": strength,
        "controlnet_conditioning_scale": controlnet_scale,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "device": device,
        "dtype": str(dtype),
        "base_model_path": (root / "models" / "sdxl_base").as_posix(),
        "controlnet_path": (root / "models" / "sdxl_controlnet").as_posix(),
        "conditioning_type": "canny",
    }
    save_image_with_metadata(input_512, original_path, metadata)
    save_image_with_metadata(canny_512, lineart_path, metadata)
    save_image_with_metadata(output_512, output_path, metadata)
    comparison_path = save_side_by_side(input_512, canny_512, output_512, outputs_dir, run_id)
    save_metadata_json(metadata, metadata_path)
    clear_device_cache()

    print("[pipeline] Complete.")
    print(f"[pipeline] Original : {original_path}")
    print(f"[pipeline] Canny    : {lineart_path}")
    print(f"[pipeline] Styled   : {output_path}")
    print(f"[pipeline] Compare  : {comparison_path}")
    print(f"[pipeline] Metadata : {metadata_path}")
    return PipelineResult(
        input_path=original_path,
        lineart_path=lineart_path,
        output_path=output_path,
        comparison_path=comparison_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 One Piece converter pipeline.")
    parser.add_argument("--input", type=str, default=None, help="Path to input image.")
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--control-scale", type=float, default=DEFAULT_CONTROLNET_SCALE)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device: auto (default), cuda, mps, or cpu.",
    )
    parser.add_argument("--diagnostic", action="store_true", help="Enable diagnostic logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_path = get_input_image_path(root, args.input)
    run_part1(
        input_image=input_path,
        project_root=root,
        strength=args.strength,
        controlnet_scale=args.control_scale,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        device_preference=args.device,
        diagnostic=args.diagnostic,
    )


if __name__ == "__main__":
    main()
