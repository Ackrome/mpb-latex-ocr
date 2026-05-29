import argparse
import json
from pathlib import Path

from PIL import Image

from mpb_latex_ocr.detection import yolo as yolo_module
from mpb_latex_ocr.detection.crops import crop_detections, sort_detections_reading_order
from mpb_latex_ocr.detection.types import Detection
from mpb_latex_ocr.page_predict import normalize_torch_device, run_page_prediction


def test_sort_detections_reading_order_groups_visual_rows(tmp_path: Path):
    image_path = tmp_path / "page.png"
    detections = [
        Detection(image_path, (90, 52, 120, 70), 0.9),
        Detection(image_path, (50, 10, 80, 30), 0.9),
        Detection(image_path, (10, 14, 40, 34), 0.9),
    ]

    ordered = sort_detections_reading_order(detections, row_tolerance=12)

    assert [item.bbox_xyxy[0] for item in ordered] == [10, 50, 90]


def test_crop_detections_expands_clamps_and_writes_crops(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 60), color="white").save(image_path)
    detections = [
        Detection(image_path, (2, 2, 20, 20), 0.95, class_name="formula"),
    ]

    records = crop_detections(
        image_path=image_path,
        detections=detections,
        output_dir=tmp_path / "crops",
        crop_padding_px=8,
        crop_padding_ratio=0.0,
    )

    assert len(records) == 1
    assert records[0].bbox_xyxy == (0, 0, 28, 28)
    assert records[0].crop_path.exists()
    with Image.open(records[0].crop_path) as crop:
        assert crop.size == (28, 28)


def test_page_prediction_composes_detector_crops_and_ocr(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 60), color="white").save(image_path)
    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        detector_weights="detector.pt",
        checkpoint="ocr.ckpt",
        tokenizer="tokenizer.json",
        image=[str(image_path)],
        output_dir=str(output_dir),
        detector_image_size=960,
        detector_confidence=0.25,
        detector_iou=0.45,
        detector_batch_size=1,
        device="cpu",
        crop_padding_px=0,
        crop_padding_ratio=0.0,
        row_tolerance=24.0,
        image_height=128,
        image_width=512,
        max_generation_length=256,
    )

    def fake_detect_fn(**kwargs):
        assert kwargs["weights"] == "detector.pt"
        assert kwargs["batch_size"] == 1
        return {
            image_path: [
                Detection(image_path, (5, 5, 30, 25), 0.88, class_id=0, class_name="formula")
            ]
        }

    def fake_predict_fn(**kwargs):
        crop_paths = kwargs["image_paths"]
        assert len(crop_paths) == 1
        return [{"image_path": str(crop_paths[0]), "latex": r"x^2"}]

    rows = run_page_prediction(args, detect_fn=fake_detect_fn, predict_fn=fake_predict_fn)

    assert len(rows) == 1
    assert rows[0]["latex"] == r"x^2"
    assert rows[0]["source_image_path"] == str(image_path)
    assert Path(str(rows[0]["crop_path"])).exists()

    crop_metadata = [
        json.loads(line)
        for line in (output_dir / "crops.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert crop_metadata[0]["class_name"] == "formula"


def test_detect_images_filters_by_class_name(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 60), color="white").save(image_path)

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeXyxy:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, index):
            assert index == 0
            return self

        def tolist(self):
            return self.values

    class FakeBox:
        def __init__(self, class_id, values):
            self.cls = FakeTensor(class_id)
            self.conf = FakeTensor(0.9)
            self.xyxy = FakeXyxy(values)

    class FakeResult:
        path = str(image_path)
        names = {0: "Equation", 1: "Figure"}
        boxes = [
            FakeBox(0, [1, 2, 30, 40]),
            FakeBox(1, [10, 20, 50, 60]),
        ]

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights
            self.kwargs = None

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [FakeResult()]

    yolo_instances = []

    class TrackingFakeYOLO(FakeYOLO):
        def __init__(self, weights):
            super().__init__(weights)
            yolo_instances.append(self)

    monkeypatch.setattr(yolo_module, "_load_yolo_class", lambda: TrackingFakeYOLO)

    rows = yolo_module.detect_images(
        weights="detector.pt",
        image_paths=[image_path],
        batch_size=1,
        class_names=["equation"],
    )

    assert yolo_instances[0].kwargs["batch"] == 1
    assert len(rows[image_path]) == 1
    assert rows[image_path][0].class_name == "Equation"
    assert rows[image_path][0].bbox_xyxy == (1.0, 2.0, 30.0, 40.0)


def test_train_detector_forwards_multi_gpu_device_string(monkeypatch, tmp_path: Path):
    yolo_instances = []

    class FakeResults:
        save_dir = tmp_path / "runs" / "dual"

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights
            self.kwargs = None
            yolo_instances.append(self)

        def train(self, **kwargs):
            self.kwargs = kwargs
            return FakeResults()

    monkeypatch.setattr(yolo_module, "_load_yolo_class", lambda: FakeYOLO)

    best = yolo_module.train_detector(
        data_yaml=tmp_path / "data.yaml",
        output_dir=tmp_path / "runs",
        run_name="dual",
        device="0,1",
    )

    assert yolo_instances[0].kwargs["device"] == "0,1"
    assert best == tmp_path / "runs" / "dual" / "weights" / "best.pt"


def test_numeric_page_predict_device_maps_to_torch_cuda_index():
    assert normalize_torch_device("0") == "cuda:0"
