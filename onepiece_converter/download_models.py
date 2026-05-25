#!/usr/bin/env python3
"""Download all required models into ./models with retries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_TARGETS: Dict[str, str] = {
    # Anything-v5 compatible repo in diffusers format.
    "base_model": "stablediffusionapi/anything-v5",
    "controlnet_lineart": "lllyasviel/control_v11p_sd15_lineart",
    "lineart_annotators": "lllyasviel/Annotators",
}

MAX_RETRIES = 3


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


def main() -> None:
    print("[download] Preparing project directories...")
    ensure_project_dirs()
    print("[download] Starting model downloads...")
    for local_name, repo_id in MODEL_TARGETS.items():
        download_with_retry(repo_id, MODELS_DIR / local_name)
    print("[download] All models are available.")


if __name__ == "__main__":
    main()
