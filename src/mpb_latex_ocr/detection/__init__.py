"""Formula region detection helpers."""

from mpb_latex_ocr.detection.crops import (
    crop_detections,
    resolve_image_paths,
    sort_detections_reading_order,
)
from mpb_latex_ocr.detection.types import CropRecord, Detection

__all__ = [
    "CropRecord",
    "Detection",
    "crop_detections",
    "resolve_image_paths",
    "sort_detections_reading_order",
]
