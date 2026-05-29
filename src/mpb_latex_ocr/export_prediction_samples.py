"""Export portable image/prediction bundles from prediction JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Copy selected prediction images into a portable local-check bundle."
    )
    parser.add_argument("--predictions", required=True, help="Input prediction JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Output bundle directory.")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=[
            "first",
            "random",
            "best-render",
            "worst-render",
            "prediction-render-fail",
            "render-fail",
            "exact-mismatch",
        ],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-field", default="image_path")
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--id-field", default="sample_id")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="FROM=TO",
        help=(
            "Rewrite image paths before copying. Example: "
            "--path-map /kaggle/input/datasets/user/ds=data/kaggle/ds"
        ),
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--zip-out",
        default=None,
        help="Optional zip path. If omitted, no zip is written.",
    )
    args = parser.parse_args(argv)

    result = export_prediction_samples(args)
    print(json.dumps(result, indent=2))


def export_prediction_samples(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    path_maps = parse_path_maps(args.path_map)

    rows = read_jsonl(prediction_path)
    selected = select_rows(rows, mode=args.mode, count=args.num_samples, seed=args.seed)

    manifest_rows: list[dict[str, str]] = []
    portable_prediction_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []

    for bundle_index, row in enumerate(selected):
        raw_image_path = row.get(args.image_field)
        if raw_image_path is None:
            missing_images.append(f"<missing {args.image_field}>")
            if args.skip_missing:
                continue
            raise ValueError(f"Prediction row has no image field '{args.image_field}': {row}")

        source_path = remap_image_path(str(raw_image_path), path_maps)
        if not source_path.exists():
            missing_images.append(str(source_path))
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Image does not exist: {source_path}")

        sample_id = safe_sample_id(row.get(args.id_field), source_path, bundle_index)
        target_name = unique_image_name(image_dir, bundle_index, sample_id, source_path.suffix)
        target_path = image_dir / target_name
        shutil.copy2(source_path, target_path)

        relative_image_path = str(Path("images") / target_name).replace("\\", "/")
        target = str(row.get(args.target_field, ""))

        manifest_rows.append(
            {
                "image_path": relative_image_path,
                "latex": target,
                "split": args.split,
                "sample_id": sample_id,
            }
        )

        portable_row = dict(row)
        portable_row["original_image_path"] = str(raw_image_path)
        portable_row[args.image_field] = relative_image_path
        portable_row["sample_id"] = sample_id
        portable_prediction_rows.append(portable_row)

    manifest_path = output_dir / "manifest.csv"
    predictions_out = output_dir / "predictions.jsonl"
    metadata_path = output_dir / "metadata.json"

    write_manifest(manifest_path, manifest_rows)
    write_jsonl(predictions_out, portable_prediction_rows)

    metadata = {
        "source_predictions": str(prediction_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "exported_samples": len(portable_prediction_rows),
        "missing_images": missing_images,
        "manifest": str(manifest_path),
        "predictions": str(predictions_out),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = None
    if args.zip_out:
        zip_path = Path(args.zip_out)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        write_zip(output_dir, zip_path)
        metadata["zip"] = str(zip_path)

    return metadata


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def parse_path_maps(values: list[str]) -> list[tuple[str, Path]]:
    maps: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --path-map value, expected FROM=TO: {value}")
        source, target = value.split("=", 1)
        source = normalize_path_prefix(source)
        if not source:
            raise ValueError(f"Invalid empty source in --path-map: {value}")
        maps.append((source, Path(target).expanduser()))
    maps.sort(key=lambda item: len(item[0]), reverse=True)
    return maps


def remap_image_path(raw_image_path: str, path_maps: list[tuple[str, Path]]) -> Path:
    normalized = normalize_path_prefix(raw_image_path)
    for source_prefix, target_prefix in path_maps:
        if normalized == source_prefix:
            return target_prefix
        if normalized.startswith(source_prefix + "/"):
            relative = normalized[len(source_prefix) + 1 :]
            return target_prefix / Path(relative)
    return Path(raw_image_path)


def normalize_path_prefix(value: str) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "latex", "split", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)


def select_rows(
    rows: list[dict[str, Any]],
    mode: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    count = max(0, min(count, len(rows)))
    selected = list(rows)

    if mode == "first":
        return selected[:count]
    if mode == "random":
        random.Random(seed).shuffle(selected)
        return selected[:count]
    if mode == "best-render":
        selected.sort(key=lambda row: numeric(row.get("render_f1"), default=-1.0), reverse=True)
        return selected[:count]
    if mode == "worst-render":
        selected.sort(key=lambda row: numeric(row.get("render_f1"), default=1.0))
        return selected[:count]
    if mode == "prediction-render-fail":
        selected = [row for row in selected if numeric(row.get("prediction_rendered"), 1.0) < 1.0]
        return selected[:count]
    if mode == "render-fail":
        selected = [
            row
            for row in selected
            if numeric(row.get("prediction_rendered"), 1.0) < 1.0
            or numeric(row.get("target_rendered"), 1.0) < 1.0
        ]
        return selected[:count]
    if mode == "exact-mismatch":
        selected = [row for row in selected if numeric(row.get("exact_match"), 0.0) < 1.0]
        selected.sort(key=lambda row: numeric(row.get("norm_edit_distance"), 0.0), reverse=True)
        return selected[:count]

    raise ValueError(f"Unknown sample mode: {mode}")


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_sample_id(raw_sample_id: Any, source_path: Path, index: int) -> str:
    raw = str(raw_sample_id or source_path.stem or f"sample_{index:06d}")
    raw = Path(raw).stem if "/" in raw or "\\" in raw else raw
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return value or f"sample_{index:06d}"


def unique_image_name(image_dir: Path, index: int, sample_id: str, suffix: str) -> str:
    suffix = suffix if suffix else ".png"
    base = f"{index:06d}_{sample_id}"
    candidate = f"{base}{suffix}"
    counter = 1
    while (image_dir / candidate).exists():
        candidate = f"{base}_{counter}{suffix}"
        counter += 1
    return candidate


def write_zip(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


if __name__ == "__main__":
    main()
