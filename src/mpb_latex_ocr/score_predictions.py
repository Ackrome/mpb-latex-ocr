"""Score existing prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.metrics.edit_distance import normalized_edit_distance
from mpb_latex_ocr.metrics.render import compare_rendered_formulas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score an existing prediction JSONL file.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Input JSONL with prediction and target fields.",
    )
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--id-field", default="sample_id")
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--render-metric",
        action="store_true",
        help="Compute the fast render-aware pixel-mask proxy metric.",
    )
    parser.add_argument("--render-font-size", type=int, default=32)
    parser.add_argument("--render-dpi", type=int, default=200)
    parser.add_argument("--render-match-threshold", type=float, default=0.98)
    parser.add_argument(
        "--scored-out",
        default=None,
        help="Optional JSONL output with recomputed metrics added per sample.",
    )
    parser.add_argument(
        "--cdm-json-out",
        default=None,
        help="Optional JSON file with img_id/gt/pred records for official CDM tooling.",
    )
    args = parser.parse_args(argv)

    metrics = score_predictions(args)
    print(json.dumps(metrics, indent=2))


def score_predictions(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.predictions)
    rows = list(read_prediction_rows(input_path, max_samples=args.max_samples))
    if not rows:
        raise ValueError(f"No prediction rows found in {input_path}")

    scored_rows: list[dict[str, Any]] = []
    cdm_rows: list[dict[str, str]] = []
    exact_values: list[float] = []
    edit_values: list[float] = []
    render_iou_values: list[float] = []
    render_f1_values: list[float] = []
    render_match_values: list[float] = []
    prediction_rendered_values: list[float] = []
    target_rendered_values: list[float] = []
    pair_rendered_values: list[float] = []

    for index, row in enumerate(tqdm(rows, desc="Scoring predictions")):
        prediction = normalize_latex(str(row.get(args.prediction_field, "")))
        target = normalize_latex(str(row.get(args.target_field, "")))
        sample_id = make_cdm_img_id(
            row.get(args.id_field),
            row.get(args.image_field),
            index,
        )

        exact = float(prediction == target)
        norm_edit = normalized_edit_distance(prediction, target)
        exact_values.append(exact)
        edit_values.append(norm_edit)

        scored = dict(row)
        scored.update(
            {
                "prediction": prediction,
                "target": target,
                "exact_match": exact,
                "norm_edit_distance": norm_edit,
            }
        )

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
            scored.update(
                {
                    "render_iou": render.iou,
                    "render_f1": render.f1,
                    "render_match": float(render.match),
                    "prediction_rendered": float(render.prediction_rendered),
                    "target_rendered": float(render.target_rendered),
                    "render_error": render.error,
                }
            )

        scored_rows.append(scored)
        cdm_rows.append({"img_id": sample_id, "gt": target, "pred": prediction})

    if args.scored_out:
        output_path = Path(args.scored_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in scored_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.cdm_json_out:
        output_path = Path(args.cdm_json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(cdm_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metrics = {
        "num_samples": len(scored_rows),
        "exact_match": sum(exact_values) / len(exact_values),
        "norm_edit_distance": sum(edit_values) / len(edit_values),
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


def read_prediction_rows(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def make_cdm_img_id(sample_id: Any, image_path: Any, index: int) -> str:
    raw = str(sample_id or "").strip()
    if not raw:
        raw = Path(str(image_path)).stem if image_path else f"sample_{index:06d}"
    if "/" in raw or "\\" in raw:
        raw = Path(raw).stem
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return raw or f"sample_{index:06d}"


if __name__ == "__main__":
    main()
