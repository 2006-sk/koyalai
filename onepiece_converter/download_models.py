#!/usr/bin/env python3
"""Download all required models into ./models with retries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_TARGETS: Dict[str, str] = {
    "sdxl_base": "cagliostrolab/animagine-xl-4.0",
    "sdxl_controlnet": "xinsir/controlnet-canny-sdxl-1.0",
}

MAX_RETRIES = 3

IP_ADAPTER_DIR = MODELS_DIR / "ip_adapter_sdxl"
IP_ADAPTER_ENCODER_DIR = MODELS_DIR / "image_encoder"


def download_with_retry(repo_id: str, local_dir: Path) -> None:
    """Download a model repo with retries and progress bars."""
    local_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[download] {repo_id} -> {local_dir} (attempt {attempt}/{MAX_RETRIES})")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir.as_posix(),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"[download] Completed: {repo_id}")
            return
        except Exception as exc:
            print(f"[download] Failed attempt {attempt} for {repo_id}: {exc}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Unable to download {repo_id} after {MAX_RETRIES} attempts.") from exc
            time.sleep(2 * attempt)


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


def download_ip_adapter_assets() -> None:
    """Download Part 2 IP-Adapter and encoder assets."""
    src_file = download_file_with_retry(
        repo_id="h94/IP-Adapter",
        filename="sdxl_models/ip-adapter_sdxl.bin",
        local_dir=IP_ADAPTER_DIR,
    )
    target_file = IP_ADAPTER_DIR / "ip-adapter_sdxl.bin"
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
    print("[download] Starting model downloads...")
    for local_name, repo_id in MODEL_TARGETS.items():
        download_with_retry(repo_id, MODELS_DIR / local_name)
    print("[download] Downloading IP-Adapter assets...")
    download_ip_adapter_assets()
    print("[download] All models are available.")


if __name__ == "__main__":
    main()
