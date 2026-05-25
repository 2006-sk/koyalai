"""LoRA download and application helpers for Part 3."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


LORA_REPO = "KorAI/sdxl-base-1.0-onepiece-lora"
LORA_FILE = "pytorch_lora_weights.safetensors"


def download_onepiece_lora(models_dir: Path) -> Optional[Path]:
    lora_path = models_dir / "onepiece_lora" / LORA_FILE
    lora_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not lora_path.exists():
            hf_hub_download(
                repo_id=LORA_REPO,
                filename=LORA_FILE,
                local_dir=lora_path.parent.as_posix(),
            )
        return lora_path
    except Exception as exc:
        print(f"[part3] Warning: LoRA download failed, continuing without LoRA: {exc}")
        return None


def apply_lora_if_available(pipe, lora_path: Optional[Path], scale: float) -> bool:
    if lora_path is None or not lora_path.exists():
        return False
    try:
        pipe.load_lora_weights(
            lora_path.parent.as_posix(),
            weight_name=LORA_FILE,
        )
        pipe.set_adapters(["default_0"], adapter_weights=[scale])
        return True
    except Exception as exc:
        print(f"[part3] Warning: LoRA load failed, continuing without LoRA: {exc}")
        return False

