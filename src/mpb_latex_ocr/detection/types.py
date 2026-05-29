"""Structured records shared by detector and page-level OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    image_path: Path
    bbox_xyxy: BBox
    confidence: float
    class_id: int = 0
    class_name: str = "formula"

    def to_json(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(frozen=True)
class CropRecord:
    source_image_path: Path
    crop_path: Path
    bbox_xyxy: tuple[int, int, int, int]
    original_bbox_xyxy: BBox
    confidence: float
    class_id: int
    class_name: str
    page_index: int
    crop_index: int

    @property
    def sample_id(self) -> str:
        return f"{self.source_image_path.stem}_{self.crop_index:03d}"

    def to_json(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "source_image_path": str(self.source_image_path),
            "crop_path": str(self.crop_path),
            "bbox_xyxy": list(self.bbox_xyxy),
            "original_bbox_xyxy": list(self.original_bbox_xyxy),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "page_index": self.page_index,
            "crop_index": self.crop_index,
        }
