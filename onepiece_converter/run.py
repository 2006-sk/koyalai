#!/usr/bin/env python3
"""Single-command runner for Part 1 -> Part 2 -> Part 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from part1_pipeline import get_input_image_path, run_part1
from part2_pipeline import run_part2
from part3_pipeline import run_part3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full One Piece converter pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Input image path.")
    parser.add_argument("--arc", type=str, default="adventure", choices=("adventure", "dramatic", "wano"))
    return parser.parse_args()


def _save_final_preview(
    input_path: Path,
    part1_path: Path,
    part2_path: Path,
    part3_path: Path,
    out_path: Path,
) -> None:
    original = Image.open(input_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    p1 = Image.open(part1_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    p2 = Image.open(part2_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    p3 = Image.open(part3_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512 * 4, 512))
    canvas.paste(original, (0, 0))
    canvas.paste(p1, (512, 0))
    canvas.paste(p2, (1024, 0))
    canvas.paste(p3, (1536, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_path = get_input_image_path(root, args.input)
    print("[run] Part 1 starting...")
    part1 = run_part1(input_image=input_path, project_root=root)
    print("[run] Part 2 starting...")
    part2 = run_part2(input_image=input_path, project_root=root)
    print("[run] Part 3 starting...")
    part3 = run_part3(input_image=input_path, project_root=root, arc=args.arc)

    final_dir = root / "outputs" / "final"
    final_image = final_dir / f"{input_path.stem}_final_part3.png"
    final_preview = final_dir / f"{input_path.stem}_full_progress.png"
    Image.open(part3.part3_path).save(final_image)
    _save_final_preview(
        input_path=input_path,
        part1_path=part1.output_path,
        part2_path=part2.part2_path,
        part3_path=part3.part3_path,
        out_path=final_preview,
    )
    print("[run] Done.")
    print(f"[run] Final image: {final_image}")
    print(f"[run] Progress preview: {final_preview}")


if __name__ == "__main__":
    main()

