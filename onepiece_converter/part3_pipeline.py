#!/usr/bin/env python3
"""Part 3 pipeline: LoRA + spatial handling + harmonization + color grading."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageEnhance
from transformers import CLIPVisionModelWithProjection

from part1_pipeline import (
    DEVICE,
    DTYPE,
    PipelineResult,
    clear_device_cache,
    get_input_image_path,
    run_part1,
)
from part2_pipeline import (
    Part2Result,
    _build_part2_pipeline,
    _download_ip_adapter_assets,
    run_part2,
)
from utils.color_grading import ARC_PALETTES, apply_arc_color_grading
from utils.face_utils import FaceExtractor
from utils.image_utils import load_image, resize_with_padding, save_metadata_json, timestamp_string
from utils.lora_utils import apply_lora_if_available, download_onepiece_lora
from utils.preprocessor import LineartPreprocessor
from utils.prompt_builder import analyze_scene, build_dynamic_prompt
from utils.spatial_utils import detect_person_map, save_person_map_visualization


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
                "ip_adapter_scale": float(data.get("ip_adapter_scale", 0.5)),
                "controlnet_scale": float(data.get("controlnet_scale", 0.7)),
                "denoising_strength": float(data.get("denoising_strength", 0.55)),
                "lora_scale": float(data.get("lora_scale", 0.4)),
            }
        except Exception as exc:
            print(f"[part3] Warning: failed reading best_params.json: {exc}")
    return {
        "ip_adapter_scale": 0.5,
        "controlnet_scale": 0.7,
        "denoising_strength": 0.55,
        "lora_scale": 0.4,
    }


def _save_five_panel(
    original: Image.Image,
    lineart: Image.Image,
    part1_img: Image.Image,
    part2_img: Image.Image,
    part3_img: Image.Image,
    output_path: Path,
) -> None:
    canvas = Image.new("RGB", (original.width * 5, original.height))
    canvas.paste(original, (0, 0))
    canvas.paste(lineart, (original.width, 0))
    canvas.paste(part1_img, (original.width * 2, 0))
    canvas.paste(part2_img, (original.width * 3, 0))
    canvas.paste(part3_img, (original.width * 4, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _build_harmonizer(project_root: Path):
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
    part1_result: Optional[PipelineResult] = None,
    part2_result: Optional[Part2Result] = None,
    seed: int = 42,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    lora_scale: float = 0.4,
    ip_scale_p3: float = 0.55,
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
    if part1_result is None:
        part1 = run_part1(
            input_image=input_image,
            project_root=root,
            controlnet_scale=cn_scale,
            strength=denoising,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    else:
        print("[part3] Step 1/8: Reusing provided Part 1 result.")
        part1 = part1_result
    if part2_result is None:
        part2 = run_part2(
            input_image=input_image,
            project_root=root,
            controlnet_scale=cn_scale,
            ip_adapter_scale=ip_scale,
            strength=denoising,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    else:
        print("[part3] Step 1/8: Reusing provided Part 2 result.")
        part2 = part2_result

    print("[part3] Step 2/8: Scene analysis...")
    original = resize_with_padding(load_image(input_image), size=(512, 512))
    scene_context = analyze_scene(original)

    print("[part3] Step 3/8: Spatial person detection...")
    person_map = detect_person_map(original)
    person_count = int(person_map["person_count"])

    print("[part3] Step 4/8: Building dynamic prompt...")
    positive_prompt, negative_prompt = build_dynamic_prompt(scene_context, person_count, arc=arc)
    positive_prompt = (
        f"{positive_prompt}, bold black outlines on face, cel shaded skin, anime eyes, "
        "defined jawline, one piece character face, eiichiro oda facial features"
    )
    negative_prompt = (
        f"{negative_prompt}, photorealistic face, soft face, blurry face, 3d face, "
        "realistic skin, smooth gradient skin, photograph"
    )

    print("[part3] Step 5/8: Main generation with LoRA + ControlNet + IP-Adapter...")
    lineart_pre = LineartPreprocessor(model_dir=models_dir / "lineart_annotators")
    lineart = lineart_pre.extract_lineart(original).convert("RGB")

    lora_path = download_onepiece_lora(models_dir)
    pipe, _device, _dtype = _build_part2_pipeline(root)
    lora_applied = apply_lora_if_available(pipe, lora_path, scale=lora_scale)
    ip_file, _encoder_dir = _download_ip_adapter_assets(models_dir)
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
    pipe.set_ip_adapter_scale(ip_scale_p3)

    face_extractor = FaceExtractor()
    face_res = face_extractor.extract_face_crop(original)
    ip_image = face_res.face_crop if face_res.face_crop is not None else original

    gen_device = DEVICE if DEVICE != "mps" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    result = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=original,
        control_image=lineart,
        ip_adapter_image=ip_image,
        strength=denoising,
        controlnet_conditioning_scale=cn_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    part3_pre = result.images[0].resize((512, 512), Image.Resampling.LANCZOS)

    if face_res.bbox_xyxy is not None:
        part2_img = load_image(part2.part2_path).resize((512, 512), Image.Resampling.LANCZOS)
        x1, y1, x2, y2 = face_res.bbox_xyxy
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        pad_x = int(face_w * 0.2)
        pad_y = int(face_h * 0.2)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(512, x2 + pad_x)
        y2 = min(512, y2 + pad_y)
        w_pad = max(1, x2 - x1)
        h_pad = max(1, y2 - y1)
        face_patch = part2_img.crop((x1, y1, x2, y2))

        mask = np.zeros((h_pad, w_pad), dtype=np.uint8)
        cx, cy = w_pad // 2, h_pad // 2
        cv2.ellipse(mask, (cx, cy), (max(1, cx - 4), max(1, cy - 4)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_pil = Image.fromarray(mask).convert("L")
        part3_pre.paste(face_patch, (x1, y1), mask_pil)

    print("[part3] Step 6/8: Harmonization pass...")
    harmonizer = _build_harmonizer(root)
    _ = apply_lora_if_available(harmonizer, lora_path, scale=0.2)
    h_start = time.time()
    h_result = harmonizer(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=part3_pre,
        strength=0.15,
        guidance_scale=6.5,
        num_inference_steps=12,
        generator=torch.Generator(device=gen_device).manual_seed(seed + 1),
    )
    harmonized = h_result.images[0].resize((512, 512), Image.Resampling.LANCZOS)
    harmonization_time = time.time() - h_start
    clear_device_cache()

    print("[part3] Step 7/8: Arc color grading...")
    part3_final = apply_arc_color_grading(harmonized, arc=arc, reference_input=original)

    run_id = f"{input_image.stem}_{timestamp_string()}"
    part3_path = outputs_dir / f"{run_id}_part3_styled.png"
    part3_pre_path = outputs_dir / f"{run_id}_part3_pre_harmonized.png"
    pre_eye_path = outputs_dir / f"{run_id}_part3_pre_eye.png"
    comparison_path = outputs_dir / f"{run_id}_part3_comparison.png"
    person_map_path = outputs_dir / f"{run_id}_person_map.png"
    metadata_path = outputs_dir / f"{run_id}_part3_metadata.json"

    print("[part3] Step 8b: Anime eye enhancement...")
    pre_eye_enhanced = part3_final.copy()
    pre_eye_enhanced.save(pre_eye_path)
    part3_eye_enhanced = part3_final.copy()
    gray = cv2.cvtColor(np.array(part3_final.convert("RGB")), cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eye_fallback = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for fx, fy, fw, fh in faces:
        face_gray = gray[fy : fy + fh, fx : fx + fw]
        eyes = eye_detector.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        if len(eyes) == 0:
            eyes = eye_fallback.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        for ex, ey, ew, eh in eyes:
            eye_cx = fx + ex + (ew / 2.0)
            eye_cy = fy + ey + (eh / 2.0)
            exp_w = max(1, int(ew * 2.2))
            exp_h = max(1, int(eh * 2.8))
            x1 = max(0, int(eye_cx - exp_w / 2.0))
            y1 = max(0, int(eye_cy - exp_h / 2.0))
            x2 = min(512, int(eye_cx + exp_w / 2.0))
            y2 = min(512, int(eye_cy + exp_h / 2.0))
            if x2 <= x1 or y2 <= y1:
                continue
            region_w = x2 - x1
            region_h = y2 - y1
            eye_patch = part3_final.crop((x1, y1, x2, y2)).resize((128, 128), Image.Resampling.LANCZOS)
            eye_patch = ImageEnhance.Contrast(eye_patch).enhance(1.4)
            eye_patch = ImageEnhance.Color(eye_patch).enhance(1.3)
            eye_patch = ImageEnhance.Sharpness(eye_patch).enhance(2.0).convert("RGBA")
            draw = ImageDraw.Draw(eye_patch, "RGBA")
            draw.ellipse([80, 8, 96, 22], fill=(255, 255, 255, 200))
            draw.ellipse([20, 85, 30, 95], fill=(255, 255, 255, 120))
            eye_patch = eye_patch.convert("RGB").resize((region_w, region_h), Image.Resampling.LANCZOS)

            mask = np.zeros((region_h, region_w), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (region_w // 2, region_h // 2),
                (max(1, region_w // 2 - 2), max(1, region_h // 2 - 2)),
                0,
                0,
                360,
                255,
                -1,
            )
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            mask_pil = Image.fromarray(mask).convert("L")
            part3_eye_enhanced.paste(eye_patch, (x1, y1), mask_pil)
    part3_final = part3_eye_enhanced

    print("[part3] Step 8/8: Saving outputs...")
    part3_pre.save(part3_pre_path)
    part3_final.save(part3_path)
    save_person_map_visualization(original, person_map, person_map_path)

    part1_img = load_image(part1.output_path).resize((512, 512), Image.Resampling.LANCZOS)
    _save_five_panel(
        original=original,
        lineart=lineart,
        part1_img=part1_img,
        part2_img=pre_eye_enhanced,
        part3_img=part3_final,
        output_path=comparison_path,
    )

    metadata: Dict[str, object] = {
        "input_image": input_image.as_posix(),
        "arc": arc,
        "best_params": params,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "scene_context": scene_context,
        "person_map": person_map,
        "lora_path": lora_path.as_posix() if lora_path else None,
        "lora_applied": lora_applied,
        "ip_adapter_weights": ip_file.as_posix(),
        "pre_eye_enhanced_path": pre_eye_path.as_posix(),
        "harmonization_time_s": harmonization_time,
        "device": DEVICE,
        "dtype": str(DTYPE),
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


def run_ip_scale_search(
    input_image: Path,
    project_root: Optional[Path],
    arc: str = "adventure",
) -> Path:
    root = (project_root or Path(__file__).resolve().parent).resolve()
    models_dir = root / "models"
    output_dir = root / "outputs" / "ipsearch"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"name": "cfg_1", "ip": 0.2, "lora": 0.5},
        {"name": "cfg_2", "ip": 0.3, "lora": 0.5},
        {"name": "cfg_3", "ip": 0.4, "lora": 0.5},
        {"name": "cfg_4", "ip": 0.5, "lora": 0.5},
        {"name": "cfg_5", "ip": 0.6, "lora": 0.5},
        {"name": "cfg_6", "ip": 0.3, "lora": 0.35},
        {"name": "cfg_7", "ip": 0.4, "lora": 0.35},
        {"name": "cfg_8", "ip": 0.5, "lora": 0.35},
        {"name": "cfg_9", "ip": 0.6, "lora": 0.35},
        {"name": "cfg_10", "ip": 0.7, "lora": 0.35},
    ]

    original = resize_with_padding(load_image(input_image), size=(512, 512))
    scene_context = analyze_scene(original)
    person_map = detect_person_map(original)
    person_count = int(person_map["person_count"])
    positive_prompt, negative_prompt = build_dynamic_prompt(scene_context, person_count, arc=arc)

    lineart_pre = LineartPreprocessor(model_dir=models_dir / "lineart_annotators")
    lineart = lineart_pre.extract_lineart(original).convert("RGB")
    face_res = FaceExtractor().extract_face_crop(original)
    ip_image = face_res.face_crop if face_res.face_crop is not None else original

    lora_path = download_onepiece_lora(models_dir)
    pipe, _device, _dtype = _build_part2_pipeline(root)
    _ip_file, _encoder_dir = _download_ip_adapter_assets(models_dir)
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

    gen_device = DEVICE if DEVICE != "mps" else "cpu"
    outputs: list[tuple[dict[str, float | str], Path]] = []
    lora_fused = False
    for cfg in configs:
        if lora_fused:
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
            lora_fused = False
        if lora_path is not None:
            pipe.load_lora_weights(
                lora_path.parent.as_posix(),
                weight_name=lora_path.name,
            )
            pipe.fuse_lora(lora_scale=float(cfg["lora"]))
            lora_fused = True
        pipe.set_ip_adapter_scale(float(cfg["ip"]))

        generator = torch.Generator(device=gen_device).manual_seed(42)
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=original,
            control_image=lineart,
            ip_adapter_image=ip_image,
            strength=0.55,
            controlnet_conditioning_scale=0.7,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0].resize((512, 512), Image.Resampling.LANCZOS)
        out_path = output_dir / (
            f"{cfg['name']}_ip{float(cfg['ip']):g}_lora{float(cfg['lora']):g}.png"
        )
        image.save(out_path)
        outputs.append((cfg, out_path))
        print(f"[part3] ip-search saved {out_path}")

    grid = Image.new("RGB", (512 * 5, 512 * 2), (0, 0, 0))
    for idx, (cfg, out_path) in enumerate(outputs):
        cell = load_image(out_path).resize((512, 512), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(cell)
        label = f"{cfg['name']} ip={float(cfg['ip']):g} lora={float(cfg['lora']):g}"
        draw.rectangle((0, 0, 250, 30), fill=(0, 0, 0))
        draw.text((8, 8), label, fill=(255, 255, 255))
        x = (idx % 5) * 512
        y = (idx // 5) * 512
        grid.paste(cell, (x, y))

    grid_path = output_dir / "comparison_grid.png"
    grid.save(grid_path)
    print(f"[part3] ip-search comparison grid saved {grid_path}")
    clear_device_cache()
    return grid_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 One Piece converter pipeline.")
    parser.add_argument("--input", type=str, default=None, help="Path to input image.")
    parser.add_argument("--arc", type=str, default="adventure", choices=("adventure", "dramatic", "wano"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--lora-scale", type=float, default=0.4)
    parser.add_argument("--ip-scale-p3", type=float, default=0.55)
    parser.add_argument("--ip-search", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_path = get_input_image_path(root, args.input)
    if args.ip_search:
        run_ip_scale_search(input_path, root, args.arc)
        return
    run_part3(
        input_image=input_path,
        project_root=root,
        arc=args.arc,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        lora_scale=args.lora_scale,
        ip_scale_p3=args.ip_scale_p3,
    )


if __name__ == "__main__":
    main()

