"""Detect formula regions and export cropped images."""

from __future__ import annotations

import argparse
from pathlib import Path

from mpb_latex_ocr.detection.crops import crop_detections, resolve_image_paths, write_jsonl
from mpb_latex_ocr.detection.yolo import detect_images


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect formula regions with YOLO and save crops.")
    parser.add_argument("--weights", required=True, help="YOLO .pt weights.")
    parser.add_argument("--image", action="append", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", default="outputs/detection/crops")
    parser.add_argument("--metadata-out", default=None, help="JSONL crop metadata output.")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--class-id",
        action="append",
        type=int,
        dest="class_ids",
        help="Keep only this YOLO class id. Repeat for multiple classes.",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="Keep only this YOLO class name. Repeat for multiple classes.",
    )
    parser.add_argument("--crop-padding-px", type=int, default=4)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.02)
    parser.add_argument("--row-tolerance", type=float, default=24.0)
    args = parser.parse_args(argv)

    records = detect_and_crop(args)
    output_path = (
        Path(args.metadata_out) if args.metadata_out else Path(args.output_dir) / "crops.jsonl"
    )
    write_jsonl([record.to_json() for record in records], output_path)
    print(f"Wrote {len(records)} crops to {args.output_dir}")
    print(f"Metadata: {output_path}")


def detect_and_crop(args: argparse.Namespace):
    image_paths = resolve_image_paths(args.image)
    detections_by_image = detect_images(
        weights=args.weights,
        image_paths=image_paths,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        batch_size=args.batch_size,
        device=args.device,
        class_ids=args.class_ids,
        class_names=args.class_names,
    )

    records = []
    output_dir = Path(args.output_dir)
    for page_index, image_path in enumerate(image_paths):
        image_output_dir = output_dir / image_path.stem
        records.extend(
            crop_detections(
                image_path=image_path,
                detections=detections_by_image.get(image_path, []),
                output_dir=image_output_dir,
                page_index=page_index,
                crop_padding_px=args.crop_padding_px,
                crop_padding_ratio=args.crop_padding_ratio,
                row_tolerance=args.row_tolerance,
            )
        )
    return records


if __name__ == "__main__":
    main()
