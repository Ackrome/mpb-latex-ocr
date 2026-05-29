"""Kaggle Model2 input preparation helpers.

This module turns the known public Kaggle datasets used by the Model2 notebook
into the two formats the project already trains on:

- a combined crop-to-LaTeX OCR manifest for model1
- a YOLO data.yaml tree for the page-level formula detector
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from mpb_latex_ocr.prepare_manifest import (
    IMAGE_EXTENSIONS,
    ManifestRow,
    assign_missing_splits,
    build_image_index,
    normalize_latex,
    normalize_split,
    resolve_image_path,
)

DEFAULT_OCR_MANIFEST_NAME = "latex-ocr-manifest.csv"


def prepare_kaggle_model2_inputs(
    input_roots: list[str | Path],
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    max_ocr_samples: int | None = 50000,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """Prepare attached/downloaded Kaggle datasets for the Model2 notebook."""

    roots = [Path(root).resolve() for root in input_roots if Path(root).exists()]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path or output_dir / DEFAULT_OCR_MANIFEST_NAME).resolve()

    ocr_rows = collect_ocr_rows(roots)
    ocr_rows = assign_missing_splits(
        ocr_rows,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    ocr_rows = ensure_eval_splits(
        ocr_rows,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    ocr_rows = sample_rows(ocr_rows, max_samples=max_ocr_samples, seed=seed)

    result: dict[str, object] = {
        "input_roots": [str(root) for root in roots],
        "ocr_manifest": None,
        "ocr_rows": 0,
        "ocr_split_counts": {},
        "detector_data_yaml": None,
        "page_image_source": None,
    }

    if ocr_rows:
        write_ocr_manifest(ocr_rows, manifest_path)
        result["ocr_manifest"] = str(manifest_path)
        result["ocr_rows"] = len(ocr_rows)
        result["ocr_split_counts"] = split_counts(ocr_rows)

    detector_data_yaml = find_yolo_data_yaml(roots)
    if detector_data_yaml is None:
        detector_data_yaml = convert_first_coco_detection_dataset(
            roots=roots,
            output_dir=output_dir / "synthetic_detection_yolo",
            seed=seed,
        )
    if detector_data_yaml is not None:
        result["detector_data_yaml"] = str(detector_data_yaml)
        page_source = yolo_split_path(detector_data_yaml, "val") or yolo_split_path(
            detector_data_yaml, "train"
        )
        result["page_image_source"] = str(page_source) if page_source is not None else None

    return result


def collect_ocr_rows(input_roots: list[Path]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for root in input_roots:
        rows.extend(rows_from_im2latex_pairs(root))
        rows.extend(rows_from_tabbed_labels(root))
        rows.extend(rows_from_same_stem_text_labels(root))
    return deduplicate_rows(rows)


def rows_from_im2latex_pairs(search_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for image_list in search_root.rglob("corresponding_png_images.txt"):
        formula_list = image_list.parent / "final_png_formulas.txt"
        if not formula_list.exists():
            continue
        dataset_root = image_list.parent
        image_index = build_image_index(dataset_root)
        image_names = read_text_lines(image_list)
        formulas = read_text_lines(formula_list)
        if len(image_names) != len(formulas):
            raise ValueError(
                f"Paired Im2LaTeX files differ in length: {image_list}={len(image_names)}, "
                f"{formula_list}={len(formulas)}"
            )
        for index, (image_name, formula) in enumerate(zip(image_names, formulas, strict=True)):
            image_path = resolve_image_path(image_name, dataset_root, image_index)
            if image_path is None:
                continue
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    latex=normalize_latex(formula),
                    split=None,
                    sample_id=f"im2latex:{index}",
                )
            )
    return rows


def rows_from_tabbed_labels(search_root: Path) -> list[ManifestRow]:
    """Read datasets like 25k_math_equation/dataset25k/labels.txt."""

    rows: list[ManifestRow] = []
    for labels_path in search_root.rglob("labels.txt"):
        records = []
        for line in read_text_lines(labels_path):
            if "\t" not in line:
                continue
            image_value, latex = line.split("\t", 1)
            if image_value.strip() and latex.strip():
                records.append((image_value.strip(), latex.strip()))
        if not records:
            continue

        dataset_root = labels_path.parent
        image_index = build_image_index(dataset_root)
        for index, (image_value, latex) in enumerate(records):
            image_path = resolve_image_path(image_value, dataset_root, image_index)
            if image_path is None:
                image_path = resolve_image_path(f"images/{image_value}", dataset_root, image_index)
            if image_path is None:
                continue
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    latex=normalize_latex(latex),
                    split=split_from_path(image_path, dataset_root),
                    sample_id=f"{labels_path.parent.name}:labels:{index}",
                )
            )
    return rows


def rows_from_same_stem_text_labels(search_root: Path) -> list[ManifestRow]:
    """Read PNG/TXT shard datasets like willcsc/mathwriting."""

    skipped_names = {
        "corresponding_png_images.txt",
        "final_png_formulas.txt",
        "labels.txt",
        "image_names.txt",
        "images.txt",
    }
    rows: list[ManifestRow] = []
    for text_path in search_root.rglob("*.txt"):
        if text_path.name.lower() in skipped_names:
            continue
        image_path = same_stem_image(text_path)
        if image_path is None:
            continue
        label = text_path.read_text(encoding="utf-8", errors="replace").strip()
        if not label:
            continue
        rows.append(
            ManifestRow(
                image_path=image_path,
                latex=normalize_latex(label),
                split=split_from_path(text_path, search_root),
                sample_id=f"same-stem:{text_path.relative_to(search_root).as_posix()}",
            )
        )
    return rows


def same_stem_image(text_path: Path) -> Path | None:
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = text_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def split_from_path(path: Path, root: Path) -> str | None:
    try:
        parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    except ValueError:
        parts = [part.lower() for part in path.parts[:-1]]
    for part in parts:
        normalized = normalize_split(part)
        if normalized is not None:
            return normalized
        if part == "synthetic":
            return "train"
    return None


def ensure_eval_splits(
    rows: list[ManifestRow],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[ManifestRow]:
    if not rows:
        return rows

    assigned = list(rows)
    counts = split_counts(assigned)
    rng = random.Random(seed)

    for split, fraction in (("val", val_fraction), ("test", test_fraction)):
        if counts.get(split, 0) > 0:
            continue
        train_indices = [index for index, row in enumerate(assigned) if row.split == "train"]
        if len(train_indices) < 3:
            continue
        move_count = max(1, int(len(assigned) * fraction))
        move_count = min(move_count, max(1, len(train_indices) // 5))
        selected = set(rng.sample(train_indices, move_count))
        assigned = [
            replace(row, split=split) if index in selected else row
            for index, row in enumerate(assigned)
        ]
        counts = split_counts(assigned)

    return assigned


def sample_rows(
    rows: list[ManifestRow],
    max_samples: int | None,
    seed: int,
) -> list[ManifestRow]:
    if max_samples is None or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    groups: dict[str, list[tuple[int, ManifestRow]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.split or "train"].append((index, row))

    allocations: dict[str, int] = {}
    for split, group in groups.items():
        proportional = int(round(max_samples * (len(group) / len(rows))))
        allocations[split] = min(len(group), max(1, proportional))

    while sum(allocations.values()) > max_samples:
        split = max(allocations, key=allocations.get)
        if allocations[split] <= 1:
            break
        allocations[split] -= 1

    while sum(allocations.values()) < max_samples:
        candidates = [
            split for split, group in groups.items() if allocations[split] < len(group)
        ]
        if not candidates:
            break
        split = max(candidates, key=lambda key: len(groups[key]) - allocations[key])
        allocations[split] += 1

    selected: list[tuple[int, ManifestRow]] = []
    for split, group in groups.items():
        selected.extend(rng.sample(group, allocations[split]))
    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def write_ocr_manifest(rows: list[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "latex", "split", "sample_id"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "image_path": str(row.image_path.resolve()),
                    "latex": row.latex,
                    "split": row.split or "train",
                    "sample_id": row.sample_id or str(index),
                }
            )


def find_yolo_data_yaml(input_roots: list[Path]) -> Path | None:
    for root in input_roots:
        for path in root.rglob("data.yaml"):
            data = load_yaml(path)
            if not data:
                continue
            if all(key in data for key in ("train", "val", "names")):
                train_path = yolo_split_path(path, "train")
                val_path = yolo_split_path(path, "val")
                if path_has_images(train_path) and path_has_images(val_path):
                    return path
    return None


def convert_first_coco_detection_dataset(
    roots: list[Path],
    output_dir: Path,
    seed: int = 42,
) -> Path | None:
    for root in roots:
        for annotation_path in root.rglob("coco_ann.json"):
            return convert_coco_detection_dataset(
                annotation_path=annotation_path,
                output_dir=output_dir,
                seed=seed,
            )
    return None


def convert_coco_detection_dataset(
    annotation_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.05,
) -> Path:
    annotation_path = Path(annotation_path).resolve()
    dataset_root = annotation_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = [image for image in payload.get("images", []) if isinstance(image, dict)]
    annotations = [
        annotation
        for annotation in payload.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("bbox")
    ]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    usable_images = []
    for image in images:
        source_path = resolve_coco_image(dataset_root, str(image.get("file_name", "")))
        if source_path is None:
            continue
        usable_images.append((image, source_path))
    if not usable_images:
        raise ValueError(f"No COCO images from {annotation_path} could be resolved.")

    rng = random.Random(seed)
    rng.shuffle(usable_images)
    total = len(usable_images)
    test_count = int(total * test_fraction)
    val_count = max(1, int(total * val_fraction)) if total > 1 else 0
    split_items = {
        "test": usable_images[:test_count],
        "val": usable_images[test_count : test_count + val_count],
        "train": usable_images[test_count + val_count :],
    }
    if not split_items["train"] and split_items["val"]:
        split_items["train"].append(split_items["val"].pop())

    for split, items in split_items.items():
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, source_path in items:
            image_id = int(image["id"])
            target_image = unique_target_path(image_dir, source_path.name, image_id)
            link_or_copy(source_path, target_image)
            label_path = label_dir / f"{target_image.stem}.txt"
            lines = [
                yolo_line(
                    annotation,
                    image_width=float(image["width"]),
                    image_height=float(image["height"]),
                )
                for annotation in annotations_by_image.get(image_id, [])
            ]
            lines = [line for line in lines if line is not None]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: formula",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def yolo_line(annotation: dict[str, Any], image_width: float, image_height: float) -> str | None:
    x, y, width, height = [float(value) for value in annotation["bbox"][:4]]
    x1 = min(max(0.0, x), image_width)
    y1 = min(max(0.0, y), image_height)
    x2 = min(max(0.0, x + width), image_width)
    y2 = min(max(0.0, y + height), image_height)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0 or image_width <= 0 or image_height <= 0:
        return None
    cx = (x1 + box_width / 2.0) / image_width
    cy = (y1 + box_height / 2.0) / image_height
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height
    return f"0 {cx:.8f} {cy:.8f} {normalized_width:.8f} {normalized_height:.8f}"


def resolve_coco_image(dataset_root: Path, file_name: str) -> Path | None:
    normalized = file_name.replace("\\", "/").lstrip("./")
    candidates = [
        dataset_root / normalized,
        dataset_root / "equation_imgs" / Path(normalized).name,
        dataset_root / Path(normalized).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def unique_target_path(directory: Path, name: str, image_id: int) -> Path:
    target = directory / name
    if not target.exists():
        return target
    return directory / f"{image_id}_{name}"


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def yolo_split_path(data_yaml_path: Path, split: str) -> Path | None:
    data = load_yaml(data_yaml_path)
    raw_value = data.get(split) if data else None
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else None
    if not raw_value:
        return None
    path = Path(str(raw_value))
    if path.is_absolute():
        return path
    root = yolo_root(data_yaml_path, data)
    return root / path


def yolo_root(data_yaml_path: Path, data: dict[str, Any]) -> Path:
    raw_root = data.get("path")
    if raw_root:
        root = Path(str(raw_root))
        if not root.is_absolute():
            root = data_yaml_path.parent / root
        return root
    return data_yaml_path.parent


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def path_has_images(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in IMAGE_EXTENSIONS
    return any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.rglob("*")
    )


def deduplicate_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    seen: set[tuple[str, str]] = set()
    unique: list[ManifestRow] = []
    for row in rows:
        key = (str(row.image_path.resolve()), row.latex)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def split_counts(rows: list[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = row.split or "train"
        counts[split] = counts.get(split, 0) + 1
    return counts


def read_text_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="\n") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle Model2 attached datasets.")
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-ocr-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    result = prepare_kaggle_model2_inputs(
        input_roots=[Path(root) for root in args.input_root],
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest) if args.manifest else None,
        max_ocr_samples=args.max_ocr_samples,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
