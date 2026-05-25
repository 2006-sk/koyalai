#!/usr/bin/env python3
"""Part 1 ControlNet pipeline for structurally-faithful anime stylization."""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil
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
DEFAULT_STRENGTH = 0.65
DEFAULT_CONTROLNET_SCALE = 0.8
TARGET_RESOLUTION = (512, 512)


@dataclass
class PipelineResult:
    input_path: Path
    lineart_path: Path
    output_path: Path
    comparison_path: Path
    metadata_path: Path
    metadata: Dict[str, object]


def clear_device_cache() -> None:
    """Release cache memory for active accelerator."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


def component_device_dtype(module: torch.nn.Module) -> tuple[str, str]:
    """Return module parameter device and dtype."""
    param = next(module.parameters())
    return str(param.device), str(param.dtype)


def print_component_load_info(name: str, module: torch.nn.Module, elapsed_s: float) -> None:
    """Print diagnostic info for a loaded module."""
    device, dtype = component_device_dtype(module)
    print(
        f"[diagnostic] Loaded {name}: device={device}, dtype={dtype}, "
        f"load_time_s={elapsed_s:.2f}"
    )


def mps_memory_stats() -> tuple[str, str]:
    """Return MPS allocated and driver memory strings."""
    if not torch.backends.mps.is_available():
        return "N/A", "N/A"
    try:
        current = str(torch.mps.current_allocated_memory())
    except Exception:
        current = "unavailable"
    try:
        driver = str(torch.mps.driver_allocated_memory())
    except Exception:
        driver = "unavailable"
    return current, driver


def mps_allocated_bytes() -> int:
    """Return current allocated MPS memory in bytes or -1."""
    if not torch.backends.mps.is_available():
        return -1
    try:
        return int(torch.mps.current_allocated_memory())
    except Exception:
        return -1


def bytes_to_gb(value: int) -> float:
    """Convert bytes to GB with float precision."""
    if value < 0:
        return -1.0
    return value / float(1024**3)


def resolve_device_and_dtype(device_preference: str = "auto") -> tuple[str, torch.dtype]:
    """Select execution device with dtype fallback."""
    if device_preference == "cpu" or DEVICE == "cpu":
        print("[pipeline] Device override set to CPU.")
        return "cpu", DTYPE if DEVICE == "cpu" else torch.float32
    if device_preference == "mps" or DEVICE == "mps":
        if not torch.backends.mps.is_available() and device_preference == "mps":
            raise RuntimeError("Device override requested MPS, but MPS is not available.")
        print("[pipeline] Device override set to MPS.")
        return "mps", DTYPE
    if device_preference == "cuda" or DEVICE == "cuda":
        if not torch.cuda.is_available() and device_preference == "cuda":
            raise RuntimeError("Device override requested CUDA, but CUDA is not available.")
        print("[pipeline] CUDA detected. Using NVIDIA acceleration.")
        return "cuda", DTYPE
    print("[pipeline] MPS not available. Falling back to CPU.")
    return "cpu", torch.float32


def list_input_images(inputs_dir: Path) -> list[Path]:
    """Return supported images in the inputs directory."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(inputs_dir.glob(pattern)))
    return sorted(files)


def get_input_image_path(project_root: Path, provided: Optional[str]) -> Path:
    """Resolve input image path from argument or first available file."""
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
        raise FileNotFoundError(
            f"No input image found in {inputs_dir}. Add an image or pass --input."
        )
    return images[0]


def _load_pipeline_with_dtype(
    base_model_path: Path,
    controlnet_path: Path,
    device: str,
    dtype: torch.dtype,
    diagnostic: bool = False,
) -> StableDiffusionControlNetImg2ImgPipeline:
    """Load ControlNet pipeline for one dtype attempt."""
    print(f"[pipeline] Loading ControlNet with dtype={dtype}...")
    controlnet_start = time.time()
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path.as_posix(),
        torch_dtype=dtype,
        use_safetensors=True,
    )
    controlnet_elapsed = time.time() - controlnet_start
    if diagnostic:
        print_component_load_info("ControlNet", controlnet, controlnet_elapsed)

    print(f"[pipeline] Loading base model with dtype={dtype}...")
    base_start = time.time()
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model_path.as_posix(),
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    base_elapsed = time.time() - base_start
    if diagnostic:
        print_component_load_info("UNet", pipe.unet, base_elapsed)
        print_component_load_info("VAE", pipe.vae, base_elapsed)
        print_component_load_info("TextEncoder", pipe.text_encoder, base_elapsed)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe.to(device)


def build_pipeline(
    project_root: Path,
    device_preference: str = "auto",
    diagnostic: bool = False,
    fixes_applied: Optional[List[str]] = None,
) -> tuple[StableDiffusionControlNetImg2ImgPipeline, str, torch.dtype]:
    """Load ControlNet and base SD pipeline."""
    model_root = project_root / "models"
    base_model_path = model_root / "base_model"
    controlnet_path = model_root / "controlnet_lineart"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing base model directory: {base_model_path}")
    if not controlnet_path.exists():
        raise FileNotFoundError(f"Missing ControlNet directory: {controlnet_path}")

    device, dtype = resolve_device_and_dtype(device_preference=device_preference)
    pipe = _load_pipeline_with_dtype(
        base_model_path,
        controlnet_path,
        device,
        dtype,
        diagnostic=diagnostic,
    )

    # AUTO-FIX: 25min bug path when UNet stays CPU float32.
    unet_device, unet_dtype = component_device_dtype(pipe.unet)
    if unet_device == "cpu" and unet_dtype == "torch.float32" and DEVICE == "mps":
        print(
            "[diagnostic] AUTO-FIX applied: UNet on CPU float32 detected; "
            "reloading as float16 -> moving to MPS -> recasting UNet to float32."
        )
        pipe = _load_pipeline_with_dtype(
            base_model_path,
            controlnet_path,
            DEVICE,
            DTYPE,
            diagnostic=diagnostic,
        )
        pipe.unet.to(dtype=DTYPE)
        device = DEVICE
        dtype = DTYPE
        if fixes_applied is not None:
            fixes_applied.append("Reloaded float16->MPS and recast UNet to float32")

    # AUTO-FIX: ensure all major components are on selected accelerator.
    if DEVICE in ("mps", "cuda"):
        for name, module in (
            ("UNet", pipe.unet),
            ("VAE", pipe.vae),
            ("ControlNet", pipe.controlnet),
            ("TextEncoder", pipe.text_encoder),
        ):
            module_device, _module_dtype = component_device_dtype(module)
            if module_device == "cpu":
                module.to(DEVICE)
                print(f"[diagnostic] AUTO-FIX applied: {name} moved to {DEVICE}.")
                if fixes_applied is not None:
                    fixes_applied.append(f"{name} moved CPU->{DEVICE}")
        device = DEVICE

    return pipe, device, dtype


def run_part1(
    input_image: Path,
    project_root: Optional[Path] = None,
    strength: float = DEFAULT_STRENGTH,
    controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
    seed: int = 42,
    device_preference: str = "auto",
    diagnostic: bool = False,
) -> PipelineResult:
    """Execute the full Part 1 pipeline and save all outputs."""
    root = (project_root or Path(__file__).resolve().parent).resolve()
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fixes_applied: List[str] = []

    print("[pipeline] Stage 1/5: Loading and resizing input image...")
    raw_image = load_image(input_image)
    resized_input = resize_with_padding(raw_image, size=TARGET_RESOLUTION)

    print("[pipeline] Stage 2/5: Extracting lineart guidance...")
    lineart_pre = LineartPreprocessor(model_dir=root / "models" / "lineart_annotators")
    lineart_image = lineart_pre.extract_lineart(resized_input).convert("RGB")

    print("[pipeline] Stage 3/5: Initializing model pipeline...")
    if diagnostic:
        vm = psutil.virtual_memory()
        print("[diagnostic] System info")
        print(f"[diagnostic] torch_version={torch.__version__}")
        print(f"[diagnostic] mps_available={torch.backends.mps.is_available()}")
        print(f"[diagnostic] mps_built={torch.backends.mps.is_built()}")
        print(f"[diagnostic] total_ram_gb={bytes_to_gb(int(vm.total)):.2f}")
        print(f"[diagnostic] free_ram_gb={bytes_to_gb(int(vm.available)):.2f}")
    pipe, device, dtype = build_pipeline(
        root,
        device_preference=device_preference,
        diagnostic=diagnostic,
        fixes_applied=fixes_applied,
    )
    clear_device_cache()

    requested_steps = num_inference_steps
    if num_inference_steps > 20:
        num_inference_steps = 20
        print("[diagnostic] AUTO-FIX applied: steps reduced to 20.")
        fixes_applied.append("Reduced steps to 20")

    if resized_input.size[0] > 512 or resized_input.size[1] > 512:
        resized_input = resized_input.resize(TARGET_RESOLUTION, Image.Resampling.LANCZOS)
        print("[diagnostic] AUTO-FIX applied: input resized to 512x512.")
        fixes_applied.append("Input resized to 512x512")

    if lineart_image.size[0] > 512 or lineart_image.size[1] > 512:
        lineart_image = lineart_image.resize(TARGET_RESOLUTION, Image.Resampling.LANCZOS)
        print("[diagnostic] AUTO-FIX applied: conditioning image resized to 512x512.")
        fixes_applied.append("Conditioning image resized to 512x512")

    batch_size = 1
    if batch_size > 1:
        batch_size = 1
        print("[diagnostic] AUTO-FIX applied: batch size forced to 1.")
        fixes_applied.append("Batch size forced to 1")

    def generate_image(
        active_pipe: StableDiffusionControlNetImg2ImgPipeline,
        active_device: str,
    ) -> tuple[Image.Image, float]:
        """Run one generation pass and print output image stats."""
        generator_device = DEVICE if DEVICE != "mps" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        if diagnostic:
            unet_device, unet_dtype = component_device_dtype(active_pipe.unet)
            control_device, control_dtype = component_device_dtype(active_pipe.controlnet)
            vae_device, vae_dtype = component_device_dtype(active_pipe.vae)
            text_device, text_dtype = component_device_dtype(active_pipe.text_encoder)
            mps_current, mps_driver = mps_memory_stats()
            print("[diagnostic] Pre-generation state")
            print(f"[diagnostic] unet_device={unet_device}, unet_dtype={unet_dtype}")
            print(f"[diagnostic] controlnet_device={control_device}, controlnet_dtype={control_dtype}")
            print(f"[diagnostic] vae_device={vae_device}, vae_dtype={vae_dtype}")
            print(f"[diagnostic] text_encoder_device={text_device}, text_encoder_dtype={text_dtype}")
            print(f"[diagnostic] num_inference_steps={num_inference_steps}")
            print(f"[diagnostic] input_image_size={resized_input.size}")
            print(f"[diagnostic] conditioning_image_size={lineart_image.size}")
            print(f"[diagnostic] mps_current_allocated={mps_current}")
            print(f"[diagnostic] mps_driver_allocated={mps_driver}")

        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif DEVICE == "mps" and torch.backends.mps.is_available():
            mps_bytes = mps_allocated_bytes()
            if mps_bytes > int(6 * (1024**3)):
                print("[diagnostic] WARNING: MPS memory > 6GB before generation, clearing cache.")
                clear_device_cache()
                fixes_applied.append("Cleared MPS cache (>6GB before generation)")
            torch.mps.empty_cache()
            torch.mps.synchronize()

        step_start_time = time.time()

        def callback(
            callback_pipe: StableDiffusionControlNetImg2ImgPipeline,
            step: int,
            timestep: int,
            kwargs: Dict[str, object],
        ) -> Dict[str, object]:
            elapsed = time.time() - step_start_time
            unet_step_device, _ = component_device_dtype(callback_pipe.unet)
            print(
                f"[diagnostic] Step {step} (t={timestep}): {elapsed:.2f}s | "
                f"device check UNet: {unet_step_device}"
            )
            return kwargs

        generation_start = time.time()
        result = active_pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=resized_input,
            control_image=lineart_image,
            strength=strength,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=callback if diagnostic else None,
        )
        generation_elapsed = time.time() - generation_start
        image = result.images[0].resize(TARGET_RESOLUTION, Image.Resampling.LANCZOS)
        output_array = np.array(image)
        print(
            "[pipeline] Output stats "
            f"(device={active_device}): min={output_array.min()}, "
            f"max={output_array.max()}, mean={output_array.mean():.2f}"
        )
        if diagnostic:
            mps_current, mps_driver = mps_memory_stats()
            print(f"[diagnostic] total_generation_time_s={generation_elapsed:.2f}")
            print(f"[diagnostic] output_min={output_array.min()}")
            print(f"[diagnostic] output_max={output_array.max()}")
            print(f"[diagnostic] output_mean={output_array.mean():.2f}")
            print(f"[diagnostic] final_mps_allocated={mps_current}")
            print(f"[diagnostic] final_mps_driver_allocated={mps_driver}")
        return image, generation_elapsed

    print("[pipeline] Stage 4/5: Running ControlNet generation...")
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    total_generation_start = time.time()
    output_image, _generation_elapsed = generate_image(pipe, device)
    output_array = np.array(output_image)
    if float(output_array.mean()) < 5.0:
        print("AUTO-FIX: black output detected, retrying with float32")
        fixes_applied.append("Black output retry with float32")
        if DEVICE in ("mps", "cuda"):
            pipe = pipe.to(DEVICE)
            pipe.unet.to(dtype=DTYPE)
            pipe.vae.to(dtype=DTYPE)
            pipe.controlnet.to(dtype=DTYPE)
            pipe.text_encoder.to(dtype=DTYPE)
            output_image, _retry_elapsed = generate_image(pipe, DEVICE)
            device = DEVICE
        else:
            pipe = pipe.to(DEVICE)
            output_image, _retry_elapsed = generate_image(pipe, DEVICE)
            device = DEVICE
        dtype = DTYPE
        output_array = np.array(output_image)
    total_generation_time = time.time() - total_generation_start

    run_id = f"{input_image.stem}_{timestamp_string()}"
    original_path = outputs_dir / f"{run_id}_original.png"
    lineart_path = outputs_dir / f"{run_id}_lineart.png"
    output_path = outputs_dir / f"{run_id}_styled.png"
    metadata_path = outputs_dir / f"{run_id}_metadata.json"

    metadata: Dict[str, object] = {
        "input_image": input_image.as_posix(),
        "resolution": {"width": TARGET_RESOLUTION[0], "height": TARGET_RESOLUTION[1]},
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "strength": strength,
        "controlnet_conditioning_scale": controlnet_scale,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "device": device,
        "dtype": str(dtype),
        "base_model_path": (root / "models" / "base_model").as_posix(),
        "controlnet_path": (root / "models" / "controlnet_lineart").as_posix(),
        "fixes_applied": fixes_applied,
    }

    print("[pipeline] Stage 5/5: Saving outputs and metadata...")
    save_image_with_metadata(resized_input, original_path, metadata)
    save_image_with_metadata(lineart_image, lineart_path, metadata)
    save_image_with_metadata(output_image, output_path, metadata)
    comparison_path = save_side_by_side(resized_input, lineart_image, output_image, outputs_dir, run_id)
    save_metadata_json(metadata, metadata_path)

    clear_device_cache()
    print("[pipeline] Complete.")
    print(f"[pipeline] Original : {original_path}")
    print(f"[pipeline] Lineart  : {lineart_path}")
    print(f"[pipeline] Styled   : {output_path}")
    print(f"[pipeline] Compare  : {comparison_path}")
    print(f"[pipeline] Metadata : {metadata_path}")
    estimated_before = total_generation_time
    if requested_steps > 0 and num_inference_steps > 0:
        estimated_before = total_generation_time * (requested_steps / num_inference_steps)
    estimated_saved = max(0.0, estimated_before - total_generation_time)
    print(f"FIXES APPLIED: {fixes_applied if fixes_applied else ['none']}")
    print(f"TOTAL GENERATION TIME: {total_generation_time:.2f}s")
    print(
        "OUTPUT STATS: "
        f"min={output_array.min()}, max={output_array.max()}, mean={output_array.mean():.2f}"
    )
    print(f"ESTIMATED TIME SAVED VS BEFORE: {estimated_saved:.2f}s")

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
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input image. If omitted, first image in ./inputs is used.",
    )
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--control-scale", type=float, default=DEFAULT_CONTROLNET_SCALE)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device: auto (default), cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Enable diagnostic instrumentation logs.",
    )
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
