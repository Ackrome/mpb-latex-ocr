"""Prediction entry point for formula images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from mpb_latex_ocr.data.dataset import FormulaImageTransform
from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.data.tokenizer import LatexTokenizer
from mpb_latex_ocr.models.lightning_module import LatexOCRModule

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict LaTeX from formula images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--image", action="append", required=True, help="Image file or directory.")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    args = parser.parse_args(argv)

    predictions = predict(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        for row in predictions:
            print(json.dumps(row, ensure_ascii=False))


@torch.inference_mode()
def predict(args: argparse.Namespace) -> list[dict[str, str]]:
    tokenizer = LatexTokenizer.load(args.tokenizer)
    module = LatexOCRModule.load_from_checkpoint(args.checkpoint, tokenizer=tokenizer)
    module.eval().to(args.device)
    transform = FormulaImageTransform(args.image_height, args.image_width, augment=False)

    rows: list[dict[str, str]] = []
    for image_path in resolve_images(args.image):
        image = Image.open(image_path)
        pixel_values = transform(image).unsqueeze(0).to(args.device)
        generated = module.model.generate(pixel_values, max_length=args.max_generation_length)
        latex = normalize_latex(tokenizer.decode(generated[0].detach().cpu().tolist()))
        rows.append({"image_path": str(image_path), "latex": latex})
    return rows


def resolve_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            images.extend(
                child for child in sorted(path.iterdir()) if child.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            raise ValueError(f"Unsupported image path: {path}")
    return images


if __name__ == "__main__":
    main()
