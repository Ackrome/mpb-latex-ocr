from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from mpb_latex_ocr.kaggle_model2_prepare import (
    convert_coco_detection_dataset,
    prepare_kaggle_model2_inputs,
)


def _image(path: Path, size: tuple[int, int] = (20, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_prepare_kaggle_model2_inputs_combines_ocr_and_converts_coco(tmp_path: Path) -> None:
    input_root = tmp_path / "input"

    im2latex_root = input_root / "im2latex-230k" / "PRINTED_TEX_230k"
    _image(im2latex_root / "generated_png_images" / "im2latex.png")
    (im2latex_root / "corresponding_png_images.txt").write_text("im2latex.png\n", encoding="utf-8")
    (im2latex_root / "final_png_formulas.txt").write_text("\\frac { a } { b }\n", encoding="utf-8")

    mathwriting_root = input_root / "mathwriting" / "shards" / "train" / "shard-000000"
    _image(mathwriting_root / "000000000.png")
    (mathwriting_root / "000000000.txt").write_text("\\dot{y}=\\frac{dy}{dt}\n", encoding="utf-8")

    equation_root = input_root / "25k-math-equation" / "dataset25k"
    _image(equation_root / "images" / "eq_00000.png")
    (equation_root / "labels.txt").write_text("eq_00000.png\t\\int_0^1 x dx\n", encoding="utf-8")

    detection_root = input_root / "synthetic-mathemtical-expression-detection"
    _image(detection_root / "equation_imgs" / "page.png", size=(100, 50))
    (detection_root / "coco_ann.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "page.png", "width": 100, "height": 50}],
                "annotations": [{"id": 7, "image_id": 1, "bbox": [10, 5, 20, 10]}],
                "categories": [{"id": 0, "name": "text"}],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "working" / "latex-ocr-manifest.csv"
    result = prepare_kaggle_model2_inputs(
        input_roots=[input_root],
        output_dir=tmp_path / "working" / "prepared",
        manifest_path=manifest_path,
        max_ocr_samples=None,
        seed=123,
    )

    assert result["ocr_manifest"] == str(manifest_path.resolve())
    assert result["ocr_rows"] == 3
    assert Path(str(result["detector_data_yaml"])).exists()
    assert Path(str(result["page_image_source"])).exists()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["latex"] for row in rows} == {
        "\\frac{a}{b}",
        "\\dot{y}=\\frac{dy}{dt}",
        "\\int_0^1 x dx",
    }
    assert all(Path(row["image_path"]).is_absolute() for row in rows)


def test_convert_coco_detection_dataset_writes_yolo_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _image(source / "equation_imgs" / "page.png", size=(100, 50))
    annotations = source / "coco_ann.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "./page.png", "width": 100, "height": 50}],
                "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 5, 20, 10]}],
            }
        ),
        encoding="utf-8",
    )

    data_yaml = convert_coco_detection_dataset(
        annotation_path=annotations,
        output_dir=tmp_path / "yolo",
        seed=1,
        val_fraction=0.0,
        test_fraction=0.0,
    )

    assert data_yaml.exists()
    label_path = tmp_path / "yolo" / "train" / "labels" / "page.txt"
    assert label_path.read_text(encoding="utf-8").strip() == (
        "0 0.20000000 0.20000000 0.20000000 0.20000000"
    )
