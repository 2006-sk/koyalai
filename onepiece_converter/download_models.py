#!/usr/bin/env python3
"""Download required SD1.5 and IP-Adapter assets into ./models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

MAX_RETRIES = 3

IP_ADAPTER_DIR = MODELS_DIR / "ip_adapter"
IP_ADAPTER_ENCODER_DIR = MODELS_DIR / "ip_adapter_encoder"


def ensure_project_dirs() -> None:
    """Create all required project folders."""
    for name in ("models", "inputs", "outputs", "utils"):
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


def download_file_with_retry(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download a single file with retries."""
    local_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[download] file {repo_id}:{filename} (attempt {attempt}/{MAX_RETRIES})")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir.as_posix(),
            )
            print(f"[download] Completed file: {filename}")
            return Path(downloaded)
        except Exception as exc:
            print(f"[download] Failed file attempt {attempt} for {filename}: {exc}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Unable to download {filename} after {MAX_RETRIES} attempts.") from exc
            time.sleep(2 * attempt)


def download_many(repo_id: str, filenames: Iterable[str], local_dir: Path) -> None:
    for filename in filenames:
        download_file_with_retry(repo_id=repo_id, filename=filename, local_dir=local_dir)


def download_sd15_base_model() -> None:
    """Download minimal fp16 SD1.5 base model files."""
    base_dir = MODELS_DIR / "base_model"
    base_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "feature_extractor/preprocessor_config.json",
    ]
    print("[download] Downloading SD1.5 base model files...")
    download_many("stablediffusionapi/anything-v5", base_files, base_dir)


def download_sd15_controlnet() -> None:
    """Download minimal SD1.5 lineart ControlNet files."""
    controlnet_dir = MODELS_DIR / "controlnet_lineart"
    controlnet_files = [
        "config.json",
        "diffusion_pytorch_model.fp16.safetensors",
    ]
    print("[download] Downloading SD1.5 lineart ControlNet files...")
    download_many("lllyasviel/control_v11p_sd15_lineart", controlnet_files, controlnet_dir)


def download_lineart_annotators() -> None:
    """Download annotator files needed by controlnet_aux lineart preprocessor."""
    annotator_dir = MODELS_DIR / "lineart_annotators"
    annotator_files = [
        "sk_model.pth",
        "sk_model2.pth",
    ]
    print("[download] Downloading lineart annotator files...")
    download_many("lllyasviel/Annotators", annotator_files, annotator_dir)


def download_ip_adapter_assets() -> None:
    """Download Part 2 IP-Adapter and encoder assets."""
    src_file = download_file_with_retry(
        repo_id="h94/IP-Adapter",
        filename="models/ip-adapter_sd15.bin",
        local_dir=IP_ADAPTER_DIR,
    )
    target_file = IP_ADAPTER_DIR / "ip-adapter_sd15.bin"
    if src_file != target_file:
        target_file.write_bytes(src_file.read_bytes())

    # CLIP image encoder files used by IP-Adapter.
    encoder_files = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    for filename in encoder_files:
        download_file_with_retry(
            repo_id="openai/clip-vit-large-patch14",
            filename=filename,
            local_dir=IP_ADAPTER_ENCODER_DIR,
        )


def main() -> None:
    print("[download] Preparing project directories...")
    ensure_project_dirs()
    print("[download] Starting SD1.5 model downloads...")
    download_sd15_base_model()
    download_sd15_controlnet()
    download_lineart_annotators()
    print("[download] Downloading IP-Adapter assets...")
    download_ip_adapter_assets()
    print("[download] All models are available.")


if __name__ == "__main__":
    main()
