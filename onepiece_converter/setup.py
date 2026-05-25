#!/usr/bin/env python3
"""Install dependencies for the One Piece converter project."""

from __future__ import annotations

import subprocess
import sys
from typing import List


REQUIRED_PACKAGES: List[str] = [
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "controlnet_aux",
    "huggingface_hub",
    "Pillow",
    "opencv-python",
    "numpy",
    "torch",
    "deepface",
    "scikit-learn",
]


def check_python_version() -> None:
    """Ensure Python 3.11+ is being used."""
    if sys.version_info < (3, 11):
        raise RuntimeError(
            "Python 3.11+ is required. "
            f"Detected {sys.version_info.major}.{sys.version_info.minor}."
        )


def install_packages() -> None:
    """Install all required packages with pip."""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + REQUIRED_PACKAGES
    print("[setup] Installing dependencies...")
    print(f"[setup] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    print("[setup] Starting environment setup for One Piece converter...")
    check_python_version()
    install_packages()
    print("[setup] Dependency installation complete.")
    print("[setup] Next step: python download_models.py")


if __name__ == "__main__":
    main()
