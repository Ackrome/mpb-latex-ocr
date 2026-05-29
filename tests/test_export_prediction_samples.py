import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from mpb_latex_ocr.export_prediction_samples import (
    export_prediction_samples,
    parse_path_maps,
    remap_image_path,
)


def test_export_prediction_samples_copies_images_and_rewrites_paths(tmp_path: Path):
    source_image = tmp_path / "source.png"
    Image.new("L", (8, 8), color=255).save(source_image)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "row/one",
                "image_path": str(source_image),
                "target": "x",
                "prediction": "x",
                "render_f1": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle"
    zip_out = tmp_path / "bundle.zip"
    args = argparse.Namespace(
        predictions=str(predictions),
        output_dir=str(output_dir),
        num_samples=1,
        mode="first",
        seed=42,
        image_field="image_path",
        target_field="target",
        prediction_field="prediction",
        id_field="sample_id",
        split="test",
        path_map=[],
        skip_missing=False,
        zip_out=str(zip_out),
    )

    metadata = export_prediction_samples(args)

    assert metadata["exported_samples"] == 1
    assert zip_out.exists()
    manifest_rows = list(csv.DictReader((output_dir / "manifest.csv").open(encoding="utf-8")))
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["latex"] == "x"
    assert manifest_rows[0]["image_path"].startswith("images/")
    assert (output_dir / manifest_rows[0]["image_path"]).exists()

    exported_rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exported_rows[0]["image_path"] == manifest_rows[0]["image_path"]
    assert exported_rows[0]["original_image_path"] == str(source_image)


def test_remap_image_path_uses_longest_matching_prefix(tmp_path: Path):
    path_maps = parse_path_maps(
        [
            f"/kaggle/input/datasets/user={tmp_path / 'too_high'}",
            f"/kaggle/input/datasets/user/im2latex-230k={tmp_path / 'dataset'}",
        ]
    )

    remapped = remap_image_path(
        "/kaggle/input/datasets/user/im2latex-230k/PRINTED_TEX_230k/images/a.png",
        path_maps,
    )

    assert remapped == tmp_path / "dataset" / "PRINTED_TEX_230k" / "images" / "a.png"
