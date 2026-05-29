"""YOLO adapter for formula region detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mpb_latex_ocr.detection.types import Detection


def detect_images(
    *,
    weights: str | Path,
    image_paths: list[Path],
    image_size: int = 960,
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    class_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    class_names: list[str] | tuple[str, ...] | set[str] | None = None,
    verbose: bool = False,
) -> dict[Path, list[Detection]]:
    if not image_paths:
        return {}

    yolo_cls = _load_yolo_class()
    model = yolo_cls(str(weights))
    predict_kwargs: dict[str, Any] = {
        "source": [str(path) for path in image_paths],
        "imgsz": image_size,
        "conf": confidence,
        "iou": iou,
        "save": False,
        "verbose": verbose,
    }
    if device and device != "auto":
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    detections_by_image: dict[Path, list[Detection]] = {path: [] for path in image_paths}
    fallback_by_index = list(image_paths)
    allowed_class_ids = {int(class_id) for class_id in class_ids} if class_ids else None
    allowed_class_names = {_normalize_label(name) for name in class_names} if class_names else None

    for index, result in enumerate(results):
        result_path = Path(getattr(result, "path", fallback_by_index[index])).resolve()
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(_tensor_scalar(getattr(box, "cls", 0)))
            class_name = _class_name(names, class_id)
            if allowed_class_ids is not None and class_id not in allowed_class_ids:
                continue
            if (
                allowed_class_names is not None
                and _normalize_label(class_name) not in allowed_class_names
            ):
                continue
            detections_by_image.setdefault(result_path, []).append(
                Detection(
                    image_path=result_path,
                    bbox_xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),  # type: ignore[arg-type]
                    confidence=float(_tensor_scalar(getattr(box, "conf", 0.0))),
                    class_id=class_id,
                    class_name=class_name,
                )
            )

    return detections_by_image


def train_detector(
    *,
    data_yaml: str | Path,
    output_dir: str | Path,
    run_name: str,
    weights: str = "yolo26n.pt",
    image_size: int = 960,
    epochs: int = 50,
    batch_size: int = 8,
    patience: int = 15,
    device: str | None = "auto",
    workers: int = 4,
    seed: int = 42,
) -> Path:
    yolo_cls = _load_yolo_class()
    model = yolo_cls(weights)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "project": str(output_dir),
        "name": run_name,
        "imgsz": image_size,
        "epochs": epochs,
        "batch": batch_size,
        "patience": patience,
        "workers": workers,
        "seed": seed,
        "exist_ok": True,
        "pretrained": True,
    }
    if device and device != "auto":
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    result_dir = Path(getattr(results, "save_dir", Path(output_dir) / run_name))
    best = result_dir / "weights" / "best.pt"
    if best.exists():
        return best
    return best


def _load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO detector support requires optional dependencies. "
            'Install them with: pip install -e ".[detector]"'
        ) from exc
    return YOLO


def _tensor_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)) and value:
        return _tensor_scalar(value[0])
    return float(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", "-").replace(" ", "-")
