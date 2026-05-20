"""Evaluation entry point for trained checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from mpb_latex_ocr.data.datamodule import LatexOCRDataModule
from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.data.tokenizer import LatexTokenizer
from mpb_latex_ocr.metrics.edit_distance import normalized_edit_distance
from mpb_latex_ocr.models.lightning_module import LatexOCRModule


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LaTeX OCR checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-label-length", type=int, default=256)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictions-out", default=None)
    args = parser.parse_args(argv)

    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = LatexTokenizer.load(args.tokenizer)
    datamodule = LatexOCRDataModule(
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        image_root=args.image_root,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_label_length=args.max_label_length,
        augment=False,
    )
    datamodule.setup("test")
    loader = datamodule.test_dataloader() if args.split == "test" else datamodule.val_dataloader()
    if loader is None:
        raise ValueError(f"Manifest has no rows for split '{args.split}'.")

    module = LatexOCRModule.load_from_checkpoint(args.checkpoint, tokenizer=tokenizer)
    module.eval().to(args.device)

    rows: list[dict[str, Any]] = []
    exact_values: list[float] = []
    edit_values: list[float] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {args.split}")):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        images = batch["pixel_values"].to(args.device)
        generated = module.model.generate(images, max_length=args.max_generation_length)
        predictions = [tokenizer.decode(row.tolist()) for row in generated.cpu()]
        targets = [tokenizer.decode(row.tolist()) for row in batch["labels"]]

        for image_path, prediction, target in zip(
            batch["image_path"],
            predictions,
            targets,
            strict=True,
        ):
            prediction = normalize_latex(prediction)
            target = normalize_latex(target)
            exact = float(prediction == target)
            norm_edit = normalized_edit_distance(prediction, target)
            exact_values.append(exact)
            edit_values.append(norm_edit)
            rows.append(
                {
                    "image_path": image_path,
                    "prediction": prediction,
                    "target": target,
                    "exact_match": exact,
                    "norm_edit_distance": norm_edit,
                }
            )

    if args.predictions_out:
        output_path = Path(args.predictions_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "split": args.split,
        "num_samples": len(rows),
        "exact_match": sum(exact_values) / max(1, len(exact_values)),
        "norm_edit_distance": sum(edit_values) / max(1, len(edit_values)),
    }


if __name__ == "__main__":
    main()
