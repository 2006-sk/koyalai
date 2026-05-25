#!/usr/bin/env python3
"""Part 3 pipeline: SDXL quality stage with graceful SD1.5 fallback."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from part1_pipeline import DEVICE, DTYPE, clear_device_cache, get_input_image_path, run_part1
from part2_pipeline import _build_part2_pipeline, _download_ip_adapter_assets, run_part2
from utils.color_grading import apply_arc_color_grading
from utils.face_utils import FaceExtractor
from utils.image_utils import load_image, resize_with_padding, save_metadata_json, timestamp_string
from utils.lora_utils import apply_lora_if_available, download_onepiece_lora
from utils.prompt_builder import analyze_scene, build_dynamic_prompt
from utils.spatial_utils import detect_person_map, save_person_map_visualization


SDXL_BASE_REPO = "cagliostrolab/animagine-xl-3.1"
SDXL_CONTROLNET_REPO = "xinsir/controlnet-tile-sdxl-1.0"
SDXL_SIZE = 768
SDXL_STEPS = 35
SDXL_GUIDANCE = 7.0

SDXL_POSITIVE = (
    "anime illustration, one piece style, eiichiro oda art, highly detailed, sharp lines, "
    "flat cel shading, bold outlines, vibrant colors, architectural details preserved, "
    "signs readable, all people preserved, consistent lighting"
)
SDXL_NEGATIVE = (
    "blurry, low quality, distorted faces, wrong anatomy, extra limbs, missing people, "
    "changed background, different architecture"
)

KAGGLE_SDXL_SETUP_SNIPPET = """from huggingface_hub import hf_hub_download
import os

os.makedirs("models/sdxl_base", exist_ok=True)
os.makedirs("models/sdxl_controlnet", exist_ok=True)

hf_hub_download(
    repo_id="xinsir/controlnet-tile-sdxl-1.0",
    filename="diffusion_pytorch_model.safetensors",
    local_dir="models/sdxl_controlnet",
)

for filename in [
    "model_index.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
]:
    hf_hub_download(
        repo_id="cagliostrolab/animagine-xl-3.1",
        filename=filename,
        local_dir="models/sdxl_base",
    )
"""


@dataclass
class Part3Result:
    part3_path: Path
    part3_pre_harmonized_path: Path
    comparison_path: Path
    person_map_path: Path
    metadata_path: Path
    metadata: Dict[str, object]


def _load_best_params(models_dir: Path) -> Dict[str, float]:
    best_path = models_dir / "best_params.json"
    if best_path.exists():
        try:
            data = json.loads(best_path.read_text(encoding="utf-8"))
            return {
                "ip_adapter_scale": float(data.get("ip_adapter_scale", 0.4)),
                "controlnet_scale": float(data.get("controlnet_scale", 0.7)),
                "denoising_strength": float(data.get("denoising_strength", 0.55)),
            }
        except Exception as exc:
            print(f"[part3] Warning: failed reading best_params.json: {exc}")
    return {"ip_adapter_scale": 0.4, "controlnet_scale": 0.7, "denoising_strength": 0.55}


def _save_five_panel(
    original: Image.Image,
    panel2: Image.Image,
    part1_img: Image.Image,
    part2_img: Image.Image,
    part3_img: Image.Image,
    output_path: Path,
) -> None:
    target_size = (512, 512)
    original = original.resize(target_size, Image.Resampling.LANCZOS)
    panel2 = panel2.resize(target_size, Image.Resampling.LANCZOS)
    part1_img = part1_img.resize(target_size, Image.Resampling.LANCZOS)
    part2_img = part2_img.resize(target_size, Image.Resampling.LANCZOS)
    part3_img = part3_img.resize(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size[0] * 5, target_size[1]))
    canvas.paste(original, (0, 0))
    canvas.paste(panel2, (target_size[0], 0))
    canvas.paste(part1_img, (target_size[0] * 2, 0))
    canvas.paste(part2_img, (target_size[0] * 3, 0))
    canvas.paste(part3_img, (target_size[0] * 4, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _sdxl_available(models_dir: Path) -> bool:
    required = [
        models_dir / "sdxl_base" / "model_index.json",
        models_dir / "sdxl_base" / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        models_dir / "sdxl_base" / "vae" / "diffusion_pytorch_model.fp16.safetensors",
        models_dir / "sdxl_base" / "text_encoder" / "model.fp16.safetensors",
        models_dir / "sdxl_base" / "text_encoder_2" / "model.fp16.safetensors",
        models_dir / "sdxl_controlnet" / "diffusion_pytorch_model.safetensors",
    ]
    return all(path.exists() for path in required)


def _build_sdxl_main_pipe(models_dir: Path):
    controlnet = ControlNetModel.from_pretrained(
        (models_dir / "sdxl_controlnet").as_posix(),
        torch_dtype=DTYPE,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        (models_dir / "sdxl_base").as_posix(),
        controlnet=controlnet,
        torch_dtype=DTYPE,
        use_safetensors=True,
        add_watermarker=False,
    )
    return pipe.to(DEVICE)


def _build_sdxl_harmonizer(models_dir: Path):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        (models_dir / "sdxl_base").as_posix(),
        torch_dtype=DTYPE,
        use_safetensors=True,
        add_watermarker=False,
    )
    return pipe.to(DEVICE)


def _build_sd15_harmonizer(project_root: Path):
    base_model_path = project_root / "models" / "base_model"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path.as_posix(),
        torch_dtype=DTYPE,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    return pipe.to(DEVICE)


def run_part3(
    input_image: Path,
    project_root: Optional[Path] = None,
    arc: str = "adventure",
) -> Part3Result:
    root = (project_root or Path(__file__).resolve().parent).resolve()
    models_dir = root / "models"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    params = _load_best_params(models_dir)
    ip_scale = params["ip_adapter_scale"]
    cn_scale = params["controlnet_scale"]
    denoising = params["denoising_strength"]
    print(f"[part3] Loaded best params: ip={ip_scale}, cn={cn_scale}, denoise={denoising}")

    print("[part3] Step 1/8: Running Part 1 and Part 2 baselines...")
    part1 = run_part1(
        input_image=input_image,
        project_root=root,
        controlnet_scale=cn_scale,
        strength=denoising,
        num_inference_steps=25,
    )
    part2 = run_part2(
        input_image=input_image,
        project_root=root,
        controlnet_scale=cn_scale,
        ip_adapter_scale=ip_scale,
        strength=denoising,
        num_inference_steps=25,
    )

    print("[part3] Step 2/8: Scene analysis...")
    original = resize_with_padding(load_image(input_image), size=(SDXL_SIZE, SDXL_SIZE))
    scene_context = analyze_scene(original)

    print("[part3] Step 3/8: Spatial person detection...")
    person_map = detect_person_map(original)
    person_count = int(person_map["person_count"])

    print("[part3] Step 4/8: Building dynamic prompt...")
    dynamic_positive, dynamic_negative = build_dynamic_prompt(scene_context, person_count, arc=arc)
    positive_prompt = f"{SDXL_POSITIVE}, {dynamic_positive}"
    negative_prompt = f"{SDXL_NEGATIVE}, {dynamic_negative}"

    print("[part3] Step 5/8: Main generation with LoRA...")
    face_extractor = FaceExtractor()
    face_res = face_extractor.extract_face_crop(original)
    ip_image = face_res.face_crop if face_res.face_crop is not None else original
    lora_path = download_onepiece_lora(models_dir)
    gen_device = DEVICE if DEVICE != "mps" else "cpu"

    used_sdxl = False
    control_panel = load_image(part1.lineart_path).resize((SDXL_SIZE, SDXL_SIZE), Image.Resampling.LANCZOS)
    if _sdxl_available(models_dir):
        try:
            pipe = _build_sdxl_main_pipe(models_dir)
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
            )
            pipe.set_ip_adapter_scale(ip_scale)
            lora_applied = apply_lora_if_available(pipe, lora_path, scale=0.7)
            generator = torch.Generator(device=gen_device).manual_seed(42)
            result = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=original,
                control_image=original,  # tile conditioning uses resized image directly
                strength=denoising,
                controlnet_conditioning_scale=cn_scale,
                num_inference_steps=SDXL_STEPS,
                guidance_scale=SDXL_GUIDANCE,
                generator=generator,
            )
            part3_pre = result.images[0].resize((SDXL_SIZE, SDXL_SIZE), Image.Resampling.LANCZOS)
            control_panel = original.copy()
            used_sdxl = True
        except Exception as exc:
            print(f"[part3] Warning: SDXL path failed ({exc})")
            print("SDXL not available, using SD1.5")
            used_sdxl = False
    else:
        print("SDXL not available, using SD1.5")
        print("[part3] Kaggle SDXL setup snippet:")
        print(KAGGLE_SDXL_SETUP_SNIPPET)

    if not used_sdxl:
        pipe, _device, _dtype = _build_part2_pipeline(root)
        ip_file, _enc = _download_ip_adapter_assets(models_dir)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        pipe.image_encoder = image_encoder
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
            image_encoder_folder=None,
        )
        pipe.set_ip_adapter_scale(ip_scale)
        lora_applied = apply_lora_if_available(pipe, lora_path, scale=0.7)
        generator = torch.Generator(device=gen_device).manual_seed(42)
        result = pipe(
            prompt=dynamic_positive,
            negative_prompt=dynamic_negative,
            image=original.resize((512, 512), Image.Resampling.LANCZOS),
            control_image=load_image(part1.lineart_path).resize((512, 512), Image.Resampling.LANCZOS),
            ip_adapter_image=ip_image.resize((256, 256), Image.Resampling.LANCZOS),
            strength=denoising,
            controlnet_conditioning_scale=cn_scale,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=generator,
        )
        part3_pre = result.images[0].resize((SDXL_SIZE, SDXL_SIZE), Image.Resampling.LANCZOS)
        ip_file = models_dir / "ip_adapter" / "ip-adapter_sd15.bin"

    print("[part3] Step 6/8: Harmonization pass...")
    harmonizer = _build_sdxl_harmonizer(models_dir) if used_sdxl else _build_sd15_harmonizer(root)
    _ = apply_lora_if_available(harmonizer, lora_path, scale=0.35)
    h_start = time.time()
    h_result = harmonizer(
        prompt=positive_prompt if used_sdxl else dynamic_positive,
        negative_prompt=negative_prompt if used_sdxl else dynamic_negative,
        image=part3_pre if used_sdxl else part3_pre.resize((512, 512), Image.Resampling.LANCZOS),
        strength=0.15,
        guidance_scale=6.5,
        num_inference_steps=12,
        generator=torch.Generator(device=gen_device).manual_seed(43),
    )
    harmonized = h_result.images[0].resize((SDXL_SIZE, SDXL_SIZE), Image.Resampling.LANCZOS)
    harmonization_time = time.time() - h_start
    clear_device_cache()

    print("[part3] Step 7/8: Arc color grading...")
    part3_final = apply_arc_color_grading(harmonized, arc=arc, reference_input=original)

    print("[part3] Step 8/8: Saving outputs...")
    run_id = f"{input_image.stem}_{timestamp_string()}"
    part3_path = outputs_dir / f"{run_id}_part3_styled.png"
    part3_pre_path = outputs_dir / f"{run_id}_part3_pre_harmonized.png"
    comparison_path = outputs_dir / f"{run_id}_part3_comparison.png"
    person_map_path = outputs_dir / f"{run_id}_person_map.png"
    metadata_path = outputs_dir / f"{run_id}_part3_metadata.json"

    part3_pre.save(part3_pre_path)
    part3_final.save(part3_path)
    save_person_map_visualization(original, person_map, person_map_path)
    part1_img = load_image(part1.output_path)
    part2_img = load_image(part2.part2_path)
    _save_five_panel(
        original=resize_with_padding(load_image(input_image), size=(512, 512)),
        panel2=control_panel.resize((512, 512), Image.Resampling.LANCZOS),
        part1_img=part1_img,
        part2_img=part2_img,
        part3_img=part3_final,
        output_path=comparison_path,
    )

    metadata: Dict[str, object] = {
        "input_image": input_image.as_posix(),
        "arc": arc,
        "best_params": params,
        "sdxl_used": used_sdxl,
        "positive_prompt": positive_prompt if used_sdxl else dynamic_positive,
        "negative_prompt": negative_prompt if used_sdxl else dynamic_negative,
        "scene_context": scene_context,
        "person_map": person_map,
        "lora_path": lora_path.as_posix() if lora_path else None,
        "lora_applied": lora_applied,
        "harmonization_time_s": harmonization_time,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_id": SDXL_BASE_REPO if used_sdxl else "sd15_fallback",
        "controlnet_id": SDXL_CONTROLNET_REPO if used_sdxl else "lllyasviel/control_v11p_sd15_lineart",
    }
    save_metadata_json(metadata, metadata_path)
    print(f"[part3] complete -> {part3_path}")

    return Part3Result(
        part3_path=part3_path,
        part3_pre_harmonized_path=part3_pre_path,
        comparison_path=comparison_path,
        person_map_path=person_map_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 One Piece converter pipeline.")
    parser.add_argument("--input", type=str, default=None, help="Path to input image.")
    parser.add_argument("--arc", type=str, default="adventure", choices=("adventure", "dramatic", "wano"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_path = get_input_image_path(root, args.input)
    run_part3(input_image=input_path, project_root=root, arc=args.arc)


if __name__ == "__main__":
    main()

