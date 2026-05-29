"""Full-page formula OCR pipeline: YOLO detector crops followed by OCR decoding."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch

from mpb_latex_ocr.detection.crops import crop_detections, resolve_image_paths, write_jsonl
from mpb_latex_ocr.detection.types import CropRecord, Detection
from mpb_latex_ocr.detection.yolo import detect_images
from mpb_latex_ocr.predict import predict_image_paths

DetectFn = Callable[..., dict[Path, list[Detection]]]
PredictFn = Callable[..., list[dict[str, str]]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detect formula regions in page images and run LaTeX OCR on each crop."
    )
    parser.add_argument(
        "--detector-weights",
        required=True,
        help="YOLO formula detector .pt weights.",
    )
    parser.add_argument("--checkpoint", required=True, help="OCR checkpoint.")
    parser.add_argument("--tokenizer", required=True, help="OCR tokenizer JSON.")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Page image file or directory.",
    )
    parser.add_argument("--output-dir", default="outputs/page_predict")
    parser.add_argument("--predictions-out", default=None, help="JSONL output path.")
    parser.add_argument("--detector-image-size", type=int, default=960)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument(
        "--detector-class-id",
        action="append",
        type=int,
        dest="detector_class_ids",
        help="Keep only this YOLO detector class id. Repeat for multiple classes.",
    )
    parser.add_argument(
        "--detector-class-name",
        action="append",
        dest="detector_class_names",
        help="Keep only this YOLO detector class name. Repeat for multiple classes.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Shared device. Numeric values are treated as CUDA device indexes for OCR.",
    )
    parser.add_argument("--detector-device", default=None, help="Optional YOLO-only device.")
    parser.add_argument("--ocr-device", default=None, help="Optional OCR-only torch device.")
    parser.add_argument("--crop-padding-px", type=int, default=4)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.02)
    parser.add_argument("--row-tolerance", type=float, default=24.0)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-generation-length", type=int, default=256)
    args = parser.parse_args(argv)

    rows = run_page_prediction(args)
    output_path = (
        Path(args.predictions_out)
        if args.predictions_out
        else Path(args.output_dir) / "predictions.jsonl"
    )
    write_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} formula predictions to {output_path}")


def run_page_prediction(
    args: argparse.Namespace,
    *,
    detect_fn: DetectFn = detect_images,
    predict_fn: PredictFn = predict_image_paths,
) -> list[dict[str, object]]:
    output_dir = Path(args.output_dir)
    crop_root = output_dir / "crops"
    image_paths = resolve_image_paths(args.image)
    detector_device = getattr(args, "detector_device", None) or args.device
    ocr_device = getattr(args, "ocr_device", None) or normalize_torch_device(args.device)
    detections_by_image = detect_fn(
        weights=args.detector_weights,
        image_paths=image_paths,
        image_size=args.detector_image_size,
        confidence=args.detector_confidence,
        iou=args.detector_iou,
        device=detector_device,
        class_ids=getattr(args, "detector_class_ids", None),
        class_names=getattr(args, "detector_class_names", None),
    )

    crop_records: list[CropRecord] = []
    for page_index, image_path in enumerate(image_paths):
        crop_records.extend(
            crop_detections(
                image_path=image_path,
                detections=detections_by_image.get(image_path, []),
                output_dir=crop_root / image_path.stem,
                page_index=page_index,
                crop_padding_px=args.crop_padding_px,
                crop_padding_ratio=args.crop_padding_ratio,
                row_tolerance=args.row_tolerance,
            )
        )

    write_jsonl(
        [record.to_json() for record in crop_records],
        output_dir / "crops.jsonl",
    )

    if not crop_records:
        return []

    crop_paths = [record.crop_path for record in crop_records]
    ocr_rows = predict_fn(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        image_paths=crop_paths,
        image_height=args.image_height,
        image_width=args.image_width,
        max_generation_length=args.max_generation_length,
        device=ocr_device,
    )
    latex_by_crop = {Path(row["image_path"]): row["latex"] for row in ocr_rows}

    rows: list[dict[str, object]] = []
    for record in crop_records:
        row = record.to_json()
        row["latex"] = latex_by_crop.get(record.crop_path, "")
        rows.append(row)
    return rows


def normalize_torch_device(device: str | None) -> str:
    if not device or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


if __name__ == "__main__":
    main()
