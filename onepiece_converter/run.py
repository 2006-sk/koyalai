#!/usr/bin/env python3
"""Runtime entrypoint with Kaggle-compatible device configuration."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image

from utils.image_utils import (
    load_image,
    resize_with_padding,
    save_image_with_metadata,
    save_metadata_json,
    save_side_by_side,
    timestamp_string,
)
from utils.preprocessor import LineartPreprocessor


PROMPT = (
    "one piece anime, eiichiro oda art style, luffy style character, "
    "clean flat skin tone, bright cheerful manga, bold clean outlines, "
    "flat cel shading, white background, adventure manga panel, "
    "friendly expression, clean face, shounen anime"
)
NEGATIVE_PROMPT = (
    "realistic, photorealistic, 3d, horror, dark, scary, grotesque, "
    "berserk, attack on titan, dark manga, shadow face, heavy shading, "
    "gradient skin, wrinkles, ugly, deformed, noisy, blurry"
)


@dataclass
class RuntimeConfig:
    device: str
    dtype: torch.dtype
    model_dir: Path
    output_dir: Path
    is_kaggle: bool


def detect_runtime() -> RuntimeConfig:
    """Auto-detect runtime environment and configure device/paths."""
    if os.path.exists("/kaggle"):
        device = "cuda"
        dtype = torch.float16
        model_dir = Path("/kaggle/working/models")
        output_dir = Path("/kaggle/working/outputs")
        is_kaggle = True
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        model_dir = Path("./models")
        output_dir = Path("./outputs")
        is_kaggle = False
    else:
        device = "cpu"
        dtype = torch.float32
        model_dir = Path("./models")
        output_dir = Path("./outputs")
        is_kaggle = False

    return RuntimeConfig(
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        output_dir=output_dir,
        is_kaggle=is_kaggle,
    )


def list_input_images(inputs_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(inputs_dir.glob(pattern)))
    return sorted(files)


def resolve_input_path(provided: str | None) -> Path:
    if provided:
        path = Path(provided)
        if not path.is_absolute():
            path = Path(".") / path
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        return path

    inputs_dir = Path("./inputs")
    images = list_input_images(inputs_dir)
    if not images:
        raise FileNotFoundError(
            f"No input images in {inputs_dir.resolve()}. Add an image or pass --input."
        )
    return images[0]


def build_pipeline(config: RuntimeConfig) -> StableDiffusionControlNetImg2ImgPipeline:
    base_model_path = config.model_dir / "base_model"
    controlnet_path = config.model_dir / "controlnet_lineart"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing base model directory: {base_model_path}")
    if not controlnet_path.exists():
        raise FileNotFoundError(f"Missing controlnet directory: {controlnet_path}")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path.as_posix(),
        torch_dtype=config.dtype,
        use_safetensors=True,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model_path.as_posix(),
        controlnet=controlnet,
        torch_dtype=config.dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to(config.device)

    # Requested behavior: only use model CPU offload on MPS.
    if config.device == "mps":
        try:
            pipe.enable_model_cpu_offload()
        except Exception as exc:
            print(f"[run] Warning: enable_model_cpu_offload failed on MPS: {exc}")

    return pipe


def run_once(
    input_image: Path,
    config: RuntimeConfig,
    steps: int,
    guidance: float,
    strength: float,
    control_scale: float,
    seed: int,
) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    resized_input = resize_with_padding(load_image(input_image), size=(512, 512))
    lineart_pre = LineartPreprocessor(model_dir=config.model_dir / "lineart_annotators")
    lineart_image = lineart_pre.extract_lineart(resized_input).convert("RGB")

    pipe = build_pipeline(config)

    # Keep MPS-only workarounds away from CUDA.
    if config.device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    generator = torch.Generator(device=config.device if config.device != "mps" else "cpu")
    generator = generator.manual_seed(seed)

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=resized_input,
        control_image=lineart_image,
        strength=strength,
        controlnet_conditioning_scale=control_scale,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    output_image = result.images[0].resize((512, 512), Image.Resampling.LANCZOS)

    run_id = f"{input_image.stem}_{timestamp_string()}"
    original_path = config.output_dir / f"{run_id}_original.png"
    lineart_path = config.output_dir / f"{run_id}_lineart.png"
    styled_path = config.output_dir / f"{run_id}_styled.png"
    metadata_path = config.output_dir / f"{run_id}_metadata.json"
    comparison_path = save_side_by_side(
        resized_input, lineart_image, output_image, config.output_dir, run_id
    )

    metadata: Dict[str, object] = {
        "input_image": input_image.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "device": config.device,
        "dtype": str(config.dtype),
        "is_kaggle": config.is_kaggle,
        "model_dir": config.model_dir.as_posix(),
        "steps": steps,
        "guidance": guidance,
        "strength": strength,
        "control_scale": control_scale,
        "seed": seed,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
    }

    save_image_with_metadata(resized_input, original_path, metadata)
    save_image_with_metadata(lineart_image, lineart_path, metadata)
    save_image_with_metadata(output_image, styled_path, metadata)
    save_metadata_json(metadata, metadata_path)

    print(f"[run] device={config.device} dtype={config.dtype}")
    print(f"[run] input={input_image}")
    print(f"[run] output_dir={config.output_dir}")
    print(f"[run] styled={styled_path}")
    print(f"[run] comparison={comparison_path}")
    return {
        "original": original_path,
        "lineart": lineart_path,
        "styled": styled_path,
        "comparison": comparison_path,
        "metadata": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run One Piece converter (Kaggle-compatible).")
    parser.add_argument("--input", type=str, default=None, help="Input image path.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = detect_runtime()
    input_path = resolve_input_path(args.input)
    run_once(
        input_image=input_path,
        config=config,
        steps=args.steps,
        guidance=args.guidance,
        strength=args.strength,
        control_scale=args.control_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

