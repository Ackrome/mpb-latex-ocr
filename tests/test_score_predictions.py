import argparse
import json
from pathlib import Path

from mpb_latex_ocr.score_predictions import make_cdm_img_id, score_predictions


def test_score_predictions_jsonl_and_cdm_export(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a", "prediction": "x", "target": "x"}),
                json.dumps({"sample_id": "b", "prediction": "x", "target": "y"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scored_out = tmp_path / "scored.jsonl"
    cdm_out = tmp_path / "cdm.json"
    args = argparse.Namespace(
        predictions=str(predictions),
        prediction_field="prediction",
        target_field="target",
        id_field="sample_id",
        image_field="image_path",
        max_samples=None,
        render_metric=False,
        render_font_size=32,
        render_dpi=200,
        render_match_threshold=0.98,
        scored_out=str(scored_out),
        cdm_json_out=str(cdm_out),
    )

    metrics = score_predictions(args)

    assert metrics["num_samples"] == 2
    assert metrics["exact_match"] == 0.5
    assert scored_out.exists()
    cdm_rows = json.loads(cdm_out.read_text(encoding="utf-8"))
    assert cdm_rows == [
        {"img_id": "a", "gt": "x", "pred": "x"},
        {"img_id": "b", "gt": "y", "pred": "x"},
    ]


def test_make_cdm_img_id_uses_safe_image_stem_when_sample_id_missing():
    image_path = "/kaggle/input/data/generated_png_images/bf114eb35be316b.png"
    assert make_cdm_img_id(None, image_path, 0) == "bf114eb35be316b"
