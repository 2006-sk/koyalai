#!/usr/bin/env python3
"""Part 2 pipeline: ControlNet structure + IP-Adapter identity conditioning."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import hf_hub_download
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from part1_pipeline import (
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_STRENGTH,
    DEVICE,
    DTYPE,
    clear_device_cache,
    get_input_image_path,
    run_part1,
)
from utils.face_utils import FaceExtractor, save_face_crop
from utils.image_utils import (
    load_image,
    resize_with_padding,
    save_image_with_metadata,
    save_metadata_json,
    timestamp_string,
)
from utils.preprocessor import LineartPreprocessor


IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_FILE = "models/ip-adapter_sd15.bin"
CLIP_ENCODER_REPO = "openai/clip-vit-large-patch14"
CLIP_ENCODER_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]

PART2_PROMPT = (
    "one piece anime, eiichiro oda style, shounen manga, "
    "clean flat skin tone, bright warm colors, bold black outlines, "
    "friendly neutral expression, mouth closed, clean face, "
    "bright white background, red shirt character, adventure manga"
)
PART2_NEGATIVE_PROMPT = (
    "angry, teeth, open mouth, dark skin, heavy shadows, "
    "scary, horror, menacing, realistic, photorealistic, "
    "gradient shading, wrinkles, deformed, blurry, ugly, "
    "extra objects on document, watermark"
)


@dataclass
class Part2Result:
    original_path: Path
    lineart_path: Path
    part1_path: Path
    part2_path: Path
    comparison_path: Path
    face_crop_path: Optional[Path]
    metadata_path: Path
    metadata: Dict[str, object]


def _download_ip_adapter_assets(models_dir: Path) -> tuple[Path, Path]:
    ip_dir = models_dir / "ip_adapter"
    ip_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = models_dir / "ip_adapter_encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    src_ip = Path(
        hf_hub_download(
            repo_id=IP_ADAPTER_REPO,
            filename=IP_ADAPTER_FILE,
            local_dir=ip_dir.as_posix(),
        )
    )
    target_ip = ip_dir / "ip-adapter_sd15.bin"
    if src_ip != target_ip:
        target_ip.write_bytes(src_ip.read_bytes())

    for filename in CLIP_ENCODER_FILES:
        try:
            hf_hub_download(
                repo_id=CLIP_ENCODER_REPO,
                filename=filename,
                local_dir=encoder_dir.as_posix(),
            )
        except Exception as exc:
            print(f"[part2] Warning: failed to download encoder file {filename}: {exc}")
    return target_ip, encoder_dir


def _build_part2_pipeline(project_root: Path) -> tuple[StableDiffusionControlNetImg2ImgPipeline, str, torch.dtype]:
    """Build a Part 2 pipeline without pre-IP-Adapter slicing hooks."""
    model_root = project_root / "models"
    base_model_path = model_root / "base_model"
    controlnet_path = model_root / "controlnet_lineart"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing base model directory: {base_model_path}")
    if not controlnet_path.exists():
        raise FileNotFoundError(f"Missing controlnet directory: {controlnet_path}")

    try:
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path),
            torch_dtype=DTYPE,
            variant="fp16",
            use_safetensors=True,
        )
    except Exception:
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path),
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
    try:
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            str(base_model_path),
            controlnet=controlnet,
            torch_dtype=DTYPE,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except Exception:
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            str(base_model_path),
            controlnet=controlnet,
            torch_dtype=DTYPE,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
    # IMPORTANT: do not enable attention/vae slicing before IP-Adapter load.
    pipe = pipe.to(DEVICE)
    return pipe, DEVICE, DTYPE


def _save_four_panel(
    original: Image.Image,
    lineart: Image.Image,
    part1_img: Image.Image,
    part2_img: Image.Image,
    output_dir: Path,
    stem: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (original.width * 4, original.height))
    canvas.paste(original, (0, 0))
    canvas.paste(lineart.convert("RGB"), (original.width, 0))
    canvas.paste(part1_img.convert("RGB"), (original.width * 2, 0))
    canvas.paste(part2_img.convert("RGB"), (original.width * 3, 0))
    out = output_dir / f"{stem}_part2_comparison.png"
    canvas.save(out)
    return out


def run_part2(
    input_image: Path,
    project_root: Optional[Path] = None,
    strength: float = DEFAULT_STRENGTH,
    controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
    ip_adapter_scale: float = 0.6,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
    seed: int = 42,
) -> Part2Result:
    root = (project_root or Path(__file__).resolve().parent).resolve()
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir = root / "models"

    print("[part2] Stage 1/7: Running Part 1 baseline...")
    part1_result = run_part1(
        input_image=input_image,
        project_root=root,
        strength=strength,
        controlnet_scale=controlnet_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    print("[part2] Stage 2/7: Loading input and lineart...")
    original_img = load_image(input_image)
    original_512 = resize_with_padding(original_img, size=(512, 512))
    lineart_pre = LineartPreprocessor(model_dir=models_dir / "lineart_annotators")
    lineart_img = lineart_pre.extract_lineart(original_512).convert("RGB")

    print("[part2] Stage 3/7: Detecting face for identity conditioning...")
    extractor = FaceExtractor()
    face_result = extractor.extract_face_crop(original_512, target_size=256)
    no_face_fallback = face_result.face_crop is None
    if no_face_fallback:
        print("No face detected — running without identity preservation")

    run_id = f"{input_image.stem}_{timestamp_string()}"
    face_crop_path: Optional[Path] = None
    if face_result.face_crop is not None:
        face_crop_path = outputs_dir / f"{run_id}_face_crop.png"
        save_face_crop(face_result.face_crop, face_crop_path)

    print("[part2] Stage 4/7: Loading ControlNet + IP-Adapter assets...")
    ip_file, _encoder_dir = _download_ip_adapter_assets(models_dir)
    pipe, device, dtype = _build_part2_pipeline(root)

    print("[part2] Stage 5/7: Attaching IP-Adapter...")
    if not no_face_fallback:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        ).to(DEVICE)
        pipe.image_encoder = image_encoder
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            image_encoder_folder=None,
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        # Only after IP-Adapter is loaded.
        pipe.vae.enable_slicing()

    print("[part2] Stage 6/7: Generating Part 2 output...")
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    generator_device = DEVICE if DEVICE != "mps" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    gen_start = time.time()
    if no_face_fallback:
        part2_img = load_image(part1_result.output_path)
    else:
        result = pipe(
            prompt=PART2_PROMPT,
            negative_prompt=PART2_NEGATIVE_PROMPT,
            image=original_512,
            control_image=lineart_img,
            ip_adapter_image=face_result.face_crop,
            strength=strength,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        part2_img = result.images[0].resize((512, 512), Image.Resampling.LANCZOS)
    gen_elapsed = time.time() - gen_start
    clear_device_cache()

    print("[part2] Stage 7/7: Saving Part 2 outputs...")
    original_path = outputs_dir / f"{run_id}_part2_original.png"
    lineart_path = outputs_dir / f"{run_id}_part2_lineart.png"
    part1_path = outputs_dir / f"{run_id}_part1_baseline.png"
    part2_path = outputs_dir / f"{run_id}_part2_styled.png"
    metadata_path = outputs_dir / f"{run_id}_part2_metadata.json"

    part1_img = load_image(part1_result.output_path).resize((512, 512), Image.Resampling.LANCZOS)
    save_image_with_metadata(original_512, original_path, {"stage": "part2"})
    save_image_with_metadata(lineart_img, lineart_path, {"stage": "part2"})
    save_image_with_metadata(part1_img, part1_path, {"stage": "part2"})
    save_image_with_metadata(part2_img, part2_path, {"stage": "part2"})
    comparison_path = _save_four_panel(
        original=original_512,
        lineart=lineart_img,
        part1_img=part1_img,
        part2_img=part2_img,
        output_dir=outputs_dir,
        stem=run_id,
    )

    metadata: Dict[str, object] = {
        "input_image": input_image.as_posix(),
        "device": device,
        "dtype": str(dtype),
        "seed": seed,
        "strength": strength,
        "controlnet_scale": controlnet_scale,
        "ip_adapter_scale": ip_adapter_scale,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "face_detected": not no_face_fallback,
        "face_area_ratio": face_result.face_area_ratio,
        "eyes_detected": face_result.eyes_detected,
        "face_laplacian_variance": face_result.laplacian_variance,
        "no_face_fallback": no_face_fallback,
        "ip_adapter_weights": ip_file.as_posix(),
        "ip_adapter_encoder_dir": "h94/IP-Adapter/models/image_encoder",
        "generation_time_s": gen_elapsed,
    }
    save_metadata_json(metadata, metadata_path)

    print("[part2] Complete.")
    print(f"[part2] Output: {part2_path}")
    print(f"[part2] Comparison: {comparison_path}")
    return Part2Result(
        original_path=original_path,
        lineart_path=lineart_path,
        part1_path=part1_path,
        part2_path=part2_path,
        comparison_path=comparison_path,
        face_crop_path=face_crop_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 identity-preserving pipeline.")
    parser.add_argument("--input", type=str, default=None, help="Path to input image.")
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--control-scale", type=float, default=DEFAULT_CONTROLNET_SCALE)
    parser.add_argument("--ip-scale", type=float, default=0.6)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_path = get_input_image_path(root, args.input)
    run_part2(
        input_image=input_path,
        project_root=root,
        strength=args.strength,
        controlnet_scale=args.control_scale,
        ip_adapter_scale=args.ip_scale,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

