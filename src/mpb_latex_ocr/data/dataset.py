"""Datasets and image transforms for formula OCR."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.data.tokenizer import LatexTokenizer


@dataclass(frozen=True)
class FormulaSample:
    image_path: str
    latex: str
    split: str = "train"
    sample_id: str | None = None


class FormulaImageTransform:
    """Resize with aspect preservation, pad to a fixed canvas, and normalize."""

    def __init__(self, height: int, width: int, augment: bool = False):
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")

        if self.augment:
            image = self._augment(image)

        image.thumbnail((self.width, self.height), Image.Resampling.BICUBIC)
        canvas = Image.new("L", (self.width, self.height), color=255)
        left = (self.width - image.width) // 2
        top = (self.height - image.height) // 2
        canvas.paste(image, (left, top))

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return torch.from_numpy(array).unsqueeze(0)

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        if random.random() < 0.25:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.20:
            image = image.rotate(
                random.uniform(-1.5, 1.5),
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=255,
            )
        return image


class LatexFormulaDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[FormulaSample],
        tokenizer: LatexTokenizer,
        image_root: str | Path | None,
        image_height: int,
        image_width: int,
        max_label_length: int,
        augment: bool = False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        self.max_label_length = max_label_length
        self.transform = FormulaImageTransform(image_height, image_width, augment=augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample.image_path)
        image = Image.open(image_path)
        label_ids = self.tokenizer.encode(
            sample.latex,
            add_special_tokens=True,
            max_length=self.max_label_length,
        )

        return {
            "pixel_values": self.transform(image),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "latex": normalize_latex(sample.latex),
            "image_path": str(image_path),
            "sample_id": sample.sample_id or str(index),
        }

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if self.image_root is not None:
            return self.image_root / path
        return path


def collate_formula_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    max_length = max(int(item["labels"].numel()) for item in batch)
    labels = torch.full((len(batch), max_length), pad_id, dtype=torch.long)

    for row, item in enumerate(batch):
        item_labels = item["labels"]
        labels[row, : item_labels.numel()] = item_labels

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "latex": [item["latex"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }


def read_manifest(path: str | Path) -> list[FormulaSample]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_manifest(path)
    if path.suffix.lower() == ".csv":
        return _read_csv_manifest(path)
    raise ValueError(f"Unsupported manifest type: {path}. Use .csv or .jsonl.")


def split_samples(samples: list[FormulaSample], split: str) -> list[FormulaSample]:
    split = split.lower()
    return [sample for sample in samples if sample.split.lower() == split]


def _read_csv_manifest(path: Path) -> list[FormulaSample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [_sample_from_mapping(row, index) for index, row in enumerate(rows)]


def _read_jsonl_manifest(path: Path) -> list[FormulaSample]:
    samples: list[FormulaSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            samples.append(_sample_from_mapping(json.loads(line), index))
    return samples


def _sample_from_mapping(row: dict[str, Any], index: int) -> FormulaSample:
    image_path = row.get("image_path") or row.get("image") or row.get("path")
    latex = row.get("latex") or row.get("label") or row.get("text")
    if not image_path or latex is None:
        raise ValueError("Manifest rows must include image_path and latex columns.")
    return FormulaSample(
        image_path=str(image_path),
        latex=normalize_latex(str(latex)),
        split=str(row.get("split", "train")),
        sample_id=str(row.get("sample_id") or row.get("id") or index),
    )
