"""Evaluation entry point for trained checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from mpb_latex_ocr.data.datamodule import LatexOCRDataModule
from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.data.tokenizer import LatexTokenizer
from mpb_latex_ocr.metrics.edit_distance import normalized_edit_distance
from mpb_latex_ocr.metrics.render import compare_rendered_formulas
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
    parser.add_argument(
        "--render-metric",
        action="store_true",
        help="Compute a render-aware pixel-mask proxy metric. Slower than string metrics.",
    )
    parser.add_argument("--render-font-size", type=int, default=32)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--render-match-threshold", type=float, default=0.98)
    parser.add_argument(
        "--cdm-json-out",
        default=None,
        help="Optional JSON file with img_id/gt/pred records for official CDM tooling.",
    )
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
    cdm_rows: list[dict[str, str]] = []
    exact_values: list[float] = []
    edit_values: list[float] = []
    render_iou_values: list[float] = []
    render_f1_values: list[float] = []
    render_match_values: list[float] = []
    prediction_rendered_values: list[float] = []
    target_rendered_values: list[float] = []
    pair_rendered_values: list[float] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {args.split}")):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        images = batch["pixel_values"].to(args.device)
        generated = module.model.generate(images, max_length=args.max_generation_length)
        predictions = [tokenizer.decode(row.tolist()) for row in generated.cpu()]
        targets = target_texts_from_batch(batch)

        for sample_id, image_path, prediction, target in zip(
            batch["sample_id"],
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
            row = {
                "sample_id": sample_id,
                "image_path": image_path,
                "prediction": prediction,
                "target": target,
                "exact_match": exact,
                "norm_edit_distance": norm_edit,
            }

            if args.render_metric:
                render = compare_rendered_formulas(
                    prediction=prediction,
                    target=target,
                    font_size=args.render_font_size,
                    dpi=args.render_dpi,
                    match_threshold=args.render_match_threshold,
                )
                pair_rendered = render.prediction_rendered and render.target_rendered
                render_iou_values.append(render.iou if pair_rendered else 0.0)
                render_f1_values.append(render.f1 if pair_rendered else 0.0)
                render_match_values.append(float(render.match if pair_rendered else False))
                prediction_rendered_values.append(float(render.prediction_rendered))
                target_rendered_values.append(float(render.target_rendered))
                pair_rendered_values.append(float(pair_rendered))
                row.update(
                    {
                        "render_iou": render.iou,
                        "render_f1": render.f1,
                        "render_match": float(render.match),
                        "prediction_rendered": float(render.prediction_rendered),
                        "target_rendered": float(render.target_rendered),
                        "render_error": render.error,
                    }
                )

            rows.append(row)
            cdm_rows.append(
                {
                    "img_id": make_cdm_img_id(sample_id, image_path, len(cdm_rows)),
                    "gt": target,
                    "pred": prediction,
                }
            )

    if args.predictions_out:
        output_path = Path(args.predictions_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.cdm_json_out:
        output_path = Path(args.cdm_json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(cdm_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metrics = {
        "split": args.split,
        "num_samples": len(rows),
        "exact_match": sum(exact_values) / max(1, len(exact_values)),
        "norm_edit_distance": sum(edit_values) / max(1, len(edit_values)),
    }

    if args.render_metric:
        metrics.update(
            {
                "render_iou_with_failures": sum(render_iou_values)
                / max(1, len(render_iou_values)),
                "render_f1_with_failures": sum(render_f1_values) / max(1, len(render_f1_values)),
                "render_match": sum(render_match_values) / max(1, len(render_match_values)),
                "prediction_render_success": sum(prediction_rendered_values)
                / max(1, len(prediction_rendered_values)),
                "target_render_success": sum(target_rendered_values)
                / max(1, len(target_rendered_values)),
                "pair_render_success": sum(pair_rendered_values)
                / max(1, len(pair_rendered_values)),
            }
        )

    return metrics


def make_cdm_img_id(sample_id: str | None, image_path: str | None, index: int) -> str:
    raw = str(sample_id or "").strip()
    if not raw:
        raw = Path(str(image_path)).stem if image_path else f"sample_{index:06d}"
    if "/" in raw or "\\" in raw:
        raw = Path(raw).stem
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return raw or f"sample_{index:06d}"


def target_texts_from_batch(batch: dict[str, Any]) -> list[str]:
    return [normalize_latex(text) for text in batch["latex"]]


if __name__ == "__main__":
    main()
