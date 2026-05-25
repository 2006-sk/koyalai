"""LoRA download and application helpers for Part 3."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


LORA_REPO = "kusnim1121/stable-diffusion-one-piece-lora"
LORA_FILE = "pytorch_lora_weights.safetensors"


def download_onepiece_lora(models_dir: Path) -> Optional[Path]:
    lora_dir = models_dir / "onepiece_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    try:
        src = Path(
            hf_hub_download(
                repo_id=LORA_REPO,
                filename=LORA_FILE,
                local_dir=lora_dir.as_posix(),
            )
        )
        dst = lora_dir / LORA_FILE
        if src != dst:
            dst.write_bytes(src.read_bytes())
        return dst
    except Exception as exc:
        print(f"[part3] Warning: LoRA download failed, continuing without LoRA: {exc}")
        return None


def apply_lora_if_available(pipe, lora_path: Optional[Path], scale: float) -> bool:
    if lora_path is None or not lora_path.exists():
        return False
    try:
        pipe.load_lora_weights(
            lora_path.parent.as_posix(),
            weight_name="pytorch_lora_weights.safetensors",
        )
        pipe.fuse_lora(lora_scale=scale)
        return True
    except Exception as exc:
        print(f"[part3] Warning: LoRA load failed, continuing without LoRA: {exc}")
        return False

