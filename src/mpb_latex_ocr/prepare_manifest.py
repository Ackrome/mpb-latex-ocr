"""Prepare formula OCR manifests from Kaggle-style datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_COLUMNS = (
    "image_path",
    "image",
    "path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "png",
    "img",
    "uuid",
)
LATEX_COLUMNS = (
    "latex",
    "formula",
    "equation",
    "label",
    "text",
    "ground_truth",
    "gt",
    "target",
)
SPLIT_COLUMNS = ("split", "set", "subset", "stage")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "validate": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "te": "test",
}
COMMENT_RE = re.compile(r"(?<!\\)%.*")


@dataclass(frozen=True)
class ManifestRow:
    image_path: Path
    latex: str
    split: str | None = None
    sample_id: str | None = None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle-style formula OCR datasets into a training manifest."
    )
    parser.add_argument("--input-root", required=True, help="Dataset root, e.g. /kaggle/input/...")
    parser.add_argument("--output", required=True, help="Output CSV manifest path.")
    parser.add_argument(
        "--format",
        choices=["auto", "table", "paired-files", "im2latex-lst"],
        default="auto",
        help="Input format. Use auto for common Kaggle datasets.",
    )
    parser.add_argument("--table-path", default=None, help="Specific CSV/TSV/JSON/JSONL metadata file.")
    parser.add_argument("--image-col", default=None, help="Image column for table inputs.")
    parser.add_argument("--latex-col", default=None, help="LaTeX label column for table inputs.")
    parser.add_argument("--split-col", default=None, help="Split column for table inputs.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths. Useful when image_root should be null.",
    )
    args = parser.parse_args(argv)

    rows = prepare_manifest(
        input_root=Path(args.input_root),
        output_path=Path(args.output),
        input_format=args.format,
        table_path=Path(args.table_path) if args.table_path else None,
        image_col=args.image_col,
        latex_col=args.latex_col,
        split_col=args.split_col,
        max_samples=args.max_samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        absolute_paths=args.absolute_paths,
    )
    counts = _split_counts(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print("Splits:", ", ".join(f"{split}={count}" for split, count in sorted(counts.items())))


def prepare_manifest(
    input_root: Path,
    output_path: Path,
    input_format: str = "auto",
    table_path: Path | None = None,
    image_col: str | None = None,
    latex_col: str | None = None,
    split_col: str | None = None,
    max_samples: int | None = None,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
    absolute_paths: bool = False,
) -> list[ManifestRow]:
    input_root = input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    image_index = build_image_index(input_root)
    if not image_index:
        raise ValueError(f"No image files found under {input_root}")

    rows = _load_rows(
        input_root=input_root,
        input_format=input_format,
        image_index=image_index,
        table_path=table_path,
        image_col=image_col,
        latex_col=latex_col,
        split_col=split_col,
    )
    if not rows:
        raise ValueError(f"No usable image/LaTeX pairs found under {input_root}")

    rows = assign_missing_splits(
        rows,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    rows = _sample_rows(rows, max_samples=max_samples, seed=seed)
    write_manifest(rows, output_path, input_root=input_root, absolute_paths=absolute_paths)
    return rows


def _load_rows(
    input_root: Path,
    input_format: str,
    image_index: dict[str, Path],
    table_path: Path | None,
    image_col: str | None,
    latex_col: str | None,
    split_col: str | None,
) -> list[ManifestRow]:
    errors: list[str] = []
    loaders = {
        "paired-files": lambda: rows_from_paired_files(input_root, image_index),
        "table": lambda: rows_from_table_files(
            input_root,
            image_index,
            table_path=table_path,
            image_col=image_col,
            latex_col=latex_col,
            split_col=split_col,
        ),
        "im2latex-lst": lambda: rows_from_im2latex_lst(input_root, image_index),
    }

    if input_format != "auto":
        return loaders[input_format]()

    for name in ("paired-files", "table", "im2latex-lst"):
        try:
            rows = loaders[name]()
        except Exception as exc:  # Keep auto-detection moving across known formats.
            errors.append(f"{name}: {exc}")
            continue
        if rows:
            print(f"Detected dataset format: {name}")
            return rows

    raise ValueError("Could not auto-detect dataset format. Tried: " + " | ".join(errors))


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        keys = {
            path.name,
            path.name.lower(),
            path.stem,
            path.stem.lower(),
            str(path.relative_to(root)).replace("\\", "/"),
            str(path.relative_to(root)).replace("\\", "/").lower(),
        }
        for key in keys:
            index.setdefault(key, path)
    return index


def rows_from_paired_files(input_root: Path, image_index: dict[str, Path]) -> list[ManifestRow]:
    image_list = _find_first(input_root, ("corresponding_png_images.txt", "image_names.txt", "images.txt"))
    formula_list = _find_first(input_root, ("final_png_formulas.txt", "formulas.txt", "labels.txt"))
    if image_list is None or formula_list is None:
        return []

    image_names = _read_text_lines(image_list)
    formulas = _read_text_lines(formula_list)
    if len(image_names) != len(formulas):
        raise ValueError(
            f"Paired files have different lengths: {image_list}={len(image_names)}, "
            f"{formula_list}={len(formulas)}"
        )

    rows: list[ManifestRow] = []
    for index, (image_name, formula) in enumerate(zip(image_names, formulas, strict=True)):
        image_path = resolve_image_path(image_name, input_root, image_index)
        if image_path is None:
            continue
        rows.append(
            ManifestRow(
                image_path=image_path,
                latex=normalize_latex(formula),
                sample_id=str(index),
            )
        )
    return rows


def rows_from_table_files(
    input_root: Path,
    image_index: dict[str, Path],
    table_path: Path | None = None,
    image_col: str | None = None,
    latex_col: str | None = None,
    split_col: str | None = None,
) -> list[ManifestRow]:
    table_paths = [table_path] if table_path else _candidate_table_paths(input_root)
    rows: list[ManifestRow] = []
    last_error: Exception | None = None

    for path in table_paths:
        if path is None:
            continue
        try:
            table_rows = list(read_table_records(path))
            if not table_rows:
                continue
            rows.extend(
                rows_from_records(
                    table_rows,
                    input_root=input_root,
                    image_index=image_index,
                    image_col=image_col,
                    latex_col=latex_col,
                    split_col=split_col,
                )
            )
        except Exception as exc:
            last_error = exc
            continue
        if rows:
            return rows

    if table_path and last_error:
        raise last_error
    return rows


def rows_from_records(
    records: list[dict[str, Any]],
    input_root: Path,
    image_index: dict[str, Path],
    image_col: str | None = None,
    latex_col: str | None = None,
    split_col: str | None = None,
) -> list[ManifestRow]:
    fieldnames = sorted({str(key) for row in records for key in row})
    image_col = image_col or _pick_column(fieldnames, IMAGE_COLUMNS)
    latex_col = latex_col or _pick_column(fieldnames, LATEX_COLUMNS)
    split_col = split_col or _pick_column(fieldnames, SPLIT_COLUMNS, required=False)
    id_col = _pick_column(fieldnames, ("sample_id", "id", "uuid"), required=False)

    if not image_col:
        raise ValueError(f"Could not find image column in {fieldnames}")
    if not latex_col:
        raise ValueError(f"Could not find LaTeX column in {fieldnames}")

    rows: list[ManifestRow] = []
    for index, record in enumerate(records):
        latex = record.get(latex_col)
        raw_image = record.get(image_col)
        if latex is None or raw_image is None:
            continue
        image_path = resolve_image_path(str(raw_image), input_root, image_index)
        if image_path is None:
            continue
        split = normalize_split(str(record.get(split_col, ""))) if split_col else None
        sample_id = str(record.get(id_col, index)) if id_col else str(index)
        rows.append(
            ManifestRow(
                image_path=image_path,
                latex=normalize_latex(str(latex)),
                split=split,
                sample_id=sample_id,
            )
        )
    return rows


def rows_from_im2latex_lst(input_root: Path, image_index: dict[str, Path]) -> list[ManifestRow]:
    formula_path = _find_formula_list(input_root)
    if formula_path is None:
        return []

    formulas = _read_text_lines(formula_path)
    split_files = _find_split_lists(input_root)
    if not split_files:
        return []

    rows: list[ManifestRow] = []
    for split, path in split_files:
        for line_index, line in enumerate(_read_text_lines(path)):
            parsed = _parse_im2latex_split_line(line, formulas, input_root, image_index)
            if parsed is None:
                continue
            image_path, formula, sample_key = parsed
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    latex=normalize_latex(formula),
                    split=split,
                    sample_id=f"{path.stem}:{sample_key or line_index}",
                )
            )
    return rows


def read_table_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        yield from _iter_json_records(payload)
        return

    raise ValueError(f"Unsupported metadata file: {path}")


def _iter_json_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_json_records(item)
    elif isinstance(payload, dict):
        if any(key.lower() in LATEX_COLUMNS for key in payload):
            yield payload
        else:
            for value in payload.values():
                yield from _iter_json_records(value)


def assign_missing_splits(
    rows: list[ManifestRow],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[ManifestRow]:
    normalized = [replace(row, split=normalize_split(row.split)) for row in rows]
    missing = [index for index, row in enumerate(normalized) if row.split is None]
    if not missing:
        return normalized

    rng = random.Random(seed)
    shuffled = missing[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * test_fraction)
    val_count = int(total * val_fraction)
    test_ids = set(shuffled[:test_count])
    val_ids = set(shuffled[test_count : test_count + val_count])

    assigned: list[ManifestRow] = []
    for index, row in enumerate(normalized):
        if row.split is not None:
            assigned.append(row)
        elif index in test_ids:
            assigned.append(replace(row, split="test"))
        elif index in val_ids:
            assigned.append(replace(row, split="val"))
        else:
            assigned.append(replace(row, split="train"))
    return assigned


def write_manifest(
    rows: list[ManifestRow],
    output_path: Path,
    input_root: Path,
    absolute_paths: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "latex", "split", "sample_id"])
        writer.writeheader()
        for index, row in enumerate(rows):
            image_path = row.image_path.resolve()
            if absolute_paths:
                image_value = str(image_path)
            else:
                image_value = str(image_path.relative_to(input_root)).replace("\\", "/")
            writer.writerow(
                {
                    "image_path": image_value,
                    "latex": row.latex,
                    "split": row.split or "train",
                    "sample_id": row.sample_id or str(index),
                }
            )


def resolve_image_path(raw_value: str, input_root: Path, image_index: dict[str, Path]) -> Path | None:
    raw_value = raw_value.strip().strip('"').strip("'")
    if not raw_value:
        return None

    path = Path(raw_value)
    if path.is_absolute() and path.exists():
        return path

    direct = input_root / path
    if direct.exists():
        return direct

    normalized = raw_value.replace("\\", "/")
    keys = [
        normalized,
        normalized.lower(),
        Path(normalized).name,
        Path(normalized).name.lower(),
        Path(normalized).stem,
        Path(normalized).stem.lower(),
    ]
    for key in keys:
        if key in image_index:
            return image_index[key]

    for extension in IMAGE_EXTENSIONS:
        key = f"{raw_value}{extension}"
        if key in image_index:
            return image_index[key]
        if key.lower() in image_index:
            return image_index[key.lower()]
    return None


def normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return SPLIT_ALIASES.get(normalized)


def normalize_latex(text: str) -> str:
    """Small self-contained target normalizer for manifest preparation.

    Keep this duplicated locally so `latex-ocr-prepare-manifest` can run even in
    minimal Kaggle import states before the full training package is imported.
    """

    value = str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u2212", "-")
    value = "\n".join(COMMENT_RE.sub("", line) for line in value.splitlines())

    replacements = {
        "\\displaystyle": "",
        "\\textstyle": "",
        "\\scriptstyle": "",
        "\\scriptscriptstyle": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "\\left": "",
        "\\right": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*([{}_^=+\-*/(),\[\]])\s*", r"\1", value)
    value = re.sub(r"(\\[A-Za-z]+)\s+(?=[{}_^=+\-*/(),\[\]])", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_im2latex_split_line(
    line: str,
    formulas: list[str],
    input_root: Path,
    image_index: dict[str, Path],
) -> tuple[Path, str, str | None] | None:
    tokens = line.split()
    if not tokens:
        return None

    image_path: Path | None = None
    formula: str | None = None
    formula_index: int | None = None

    for token in tokens:
        candidate = resolve_image_path(token, input_root, image_index)
        if candidate is not None:
            image_path = candidate
            break

    for token in tokens:
        if token.isdigit():
            candidate_index = int(token)
            if 0 <= candidate_index < len(formulas):
                formula_index = candidate_index
                formula = formulas[candidate_index]
                break
            if 1 <= candidate_index <= len(formulas):
                formula_index = candidate_index - 1
                formula = formulas[candidate_index - 1]
                break

    if image_path is None or formula is None:
        return None
    return image_path, formula, str(formula_index) if formula_index is not None else None


def _candidate_table_paths(input_root: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.csv", "*.tsv", "*.jsonl", "*.json"):
        paths.extend(input_root.rglob(suffix))
    paths.sort(key=lambda path: (path.name.lower().startswith(("vocab", "map")), len(path.parts), path.name))
    return paths


def _find_first(input_root: Path, names: tuple[str, ...]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in input_root.rglob("*"):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def _find_formula_list(input_root: Path) -> Path | None:
    candidates = [
        path
        for path in input_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".lst", ".txt"}
        and "formula" in path.name.lower()
        and path.name.lower() != "final_png_formulas.txt"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_size, reverse=True)
    return candidates[0]


def _find_split_lists(input_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".lst", ".txt"}:
            continue
        name = path.name.lower()
        if "formula" in name:
            continue
        split = None
        if "train" in name:
            split = "train"
        elif "valid" in name or "val" in name or "validate" in name:
            split = "val"
        elif "test" in name:
            split = "test"
        if split:
            found.append((split, path))
    return found


def _read_text_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="\n") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _pick_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:
    by_lower = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    if required:
        return None
    return None


def _sample_rows(rows: list[ManifestRow], max_samples: int | None, seed: int) -> list[ManifestRow]:
    if max_samples is None or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    groups: dict[str, list[tuple[int, ManifestRow]]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.split or "train", []).append((index, row))

    if len(groups) == 1:
        indices = sorted(rng.sample(range(len(rows)), max_samples))
        return [rows[index] for index in indices]

    allocations: dict[str, int] = {}
    for split, group in groups.items():
        proportional = int(round(max_samples * (len(group) / len(rows))))
        allocations[split] = min(len(group), max(1, proportional))

    while sum(allocations.values()) > max_samples:
        candidates = [
            split
            for split, count in allocations.items()
            if count > 1
        ]
        if not candidates:
            break
        split = max(candidates, key=lambda key: allocations[key])
        allocations[split] -= 1

    while sum(allocations.values()) < max_samples:
        candidates = [
            split
            for split, group in groups.items()
            if allocations[split] < len(group)
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


def _split_counts(rows: list[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = row.split or "train"
        counts[split] = counts.get(split, 0) + 1
    return counts


if __name__ == "__main__":
    main()
