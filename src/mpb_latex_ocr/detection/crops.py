"""Crop formula regions from detector boxes."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from mpb_latex_ocr.detection.types import BBox, CropRecord, Detection

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_image_paths(paths: list[str | Path]) -> list[Path]:
    images: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            images.extend(
                child.resolve()
                for child in sorted(path.iterdir())
                if child.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path.resolve())
        else:
            raise ValueError(f"Unsupported image path: {path}")
    return images


def sort_detections_reading_order(
    detections: list[Detection],
    row_tolerance: float = 24.0,
) -> list[Detection]:
    """Sort boxes top-to-bottom, then left-to-right within each visual row."""
    pending = sorted(detections, key=lambda item: (_center_y(item.bbox_xyxy), item.bbox_xyxy[0]))
    rows: list[list[Detection]] = []
    row_centers: list[float] = []

    for detection in pending:
        center_y = _center_y(detection.bbox_xyxy)
        for row_index, row_center in enumerate(row_centers):
            if abs(center_y - row_center) <= row_tolerance:
                rows[row_index].append(detection)
                row_centers[row_index] = _mean_center_y(rows[row_index])
                break
        else:
            rows.append([detection])
            row_centers.append(center_y)

    ordered: list[Detection] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item.bbox_xyxy[0]))
    return ordered


def crop_detections(
    image_path: str | Path,
    detections: list[Detection],
    output_dir: str | Path,
    *,
    page_index: int = 0,
    crop_padding_px: int = 4,
    crop_padding_ratio: float = 0.02,
    row_tolerance: float = 24.0,
    min_size: int = 2,
) -> list[CropRecord]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        width, height = image.size
        ordered = sort_detections_reading_order(detections, row_tolerance=row_tolerance)
        records: list[CropRecord] = []

        for crop_index, detection in enumerate(ordered):
            bbox = expand_and_clamp_bbox(
                detection.bbox_xyxy,
                image_width=width,
                image_height=height,
                padding_px=crop_padding_px,
                padding_ratio=crop_padding_ratio,
            )
            x1, y1, x2, y2 = bbox
            if x2 - x1 < min_size or y2 - y1 < min_size:
                continue

            crop = image.crop(bbox)
            crop_name = f"{image_path.stem}_{crop_index:03d}{image_path.suffix.lower()}"
            crop_path = output_dir / crop_name
            crop.save(crop_path)
            records.append(
                CropRecord(
                    source_image_path=image_path,
                    crop_path=crop_path,
                    bbox_xyxy=bbox,
                    original_bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    page_index=page_index,
                    crop_index=crop_index,
                )
            )

    return records


def write_jsonl(rows: list[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def expand_and_clamp_bbox(
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
    padding_px: int = 4,
    padding_ratio: float = 0.02,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    pad_x = max(float(padding_px), box_width * float(padding_ratio))
    pad_y = max(float(padding_px), box_height * float(padding_ratio))

    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(image_width, int(x2 + pad_x + 0.9999))
    bottom = min(image_height, int(y2 + pad_y + 0.9999))
    return left, top, right, bottom


def _center_y(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _mean_center_y(detections: list[Detection]) -> float:
    return sum(_center_y(detection.bbox_xyxy) for detection in detections) / len(detections)
