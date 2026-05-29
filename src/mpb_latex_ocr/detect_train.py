"""Train a YOLO formula-region detector."""

from __future__ import annotations

import argparse
from typing import Any

from mpb_latex_ocr.detection.yolo import train_detector
from mpb_latex_ocr.utils.config import load_config

DEFAULT_CONFIG = "configs/detection/yolo_formula.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for formula regions.")
    parser.add_argument(
        "--config",
        action="append",
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. train.epochs=10 train.batch_size=4",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config or [DEFAULT_CONFIG], args.overrides)
    best = train_from_config(cfg)
    print(f"Best detector weights: {best}")


def train_from_config(cfg: dict[str, Any]):
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    return train_detector(
        data_yaml=data["yaml_path"],
        output_dir=train["output_dir"],
        run_name=train["run_name"],
        weights=model.get("weights", "yolo26n.pt"),
        image_size=int(model.get("image_size", 960)),
        epochs=int(train.get("epochs", 50)),
        batch_size=int(train.get("batch_size", 8)),
        patience=int(train.get("patience", 15)),
        device=train.get("device", "auto"),
        workers=int(train.get("workers", 4)),
        seed=int(train.get("seed", 42)),
    )


if __name__ == "__main__":
    main()
