"""Training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import mlflow.pytorch
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from mpb_latex_ocr.data.datamodule import LatexOCRDataModule, build_or_load_tokenizer
from mpb_latex_ocr.models.lightning_module import LatexOCRModule
from mpb_latex_ocr.utils.config import ensure_dir, flatten_config, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a LaTeX OCR model.")
    parser.add_argument(
        "--config",
        action="append",
        help="YAML config path. Can be passed multiple times; later files override earlier files.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. trainer.max_epochs=1 data.batch_size=4",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    train(cfg)


def train(cfg: dict[str, Any]) -> None:
    L.seed_everything(int(cfg.get("seed", 42)), workers=True)
    torch.set_float32_matmul_precision("high")

    paths = cfg["paths"]
    output_dir = ensure_dir(paths["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tokenizer = build_or_load_tokenizer(
        manifest_path=paths["manifest"],
        tokenizer_path=paths["tokenizer_path"],
        min_freq=int(cfg["tokenizer"].get("min_freq", 1)),
        max_vocab_size=cfg["tokenizer"].get("max_vocab_size"),
        force_rebuild=bool(cfg["tokenizer"].get("force_rebuild", False)),
    )

    datamodule = LatexOCRDataModule(
        manifest_path=paths["manifest"],
        tokenizer=tokenizer,
        image_root=paths.get("image_root"),
        **cfg["data"],
    )
    module = LatexOCRModule(
        tokenizer=tokenizer,
        model_config=cfg["model"],
        optimizer_config=cfg["optimizer"],
        generation_config=cfg.get("generation", {}),
    )

    logger = MLFlowLogger(
        experiment_name=cfg["mlflow"].get("experiment_name", "latex-ocr"),
        tracking_uri=cfg["mlflow"].get("tracking_uri", "file:./mlruns"),
    )
    logger.log_hyperparams(flatten_config(cfg))

    if cfg["mlflow"].get("autolog", True):
        mlflow.pytorch.autolog(
            log_models=False,
            log_datasets=False,
            checkpoint=False,
            silent=True,
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch={epoch:03d}-val_ned={val_norm_edit_distance:.4f}",
            monitor=cfg["callbacks"].get("monitor", "val_norm_edit_distance"),
            mode=cfg["callbacks"].get("mode", "min"),
            save_top_k=int(cfg["callbacks"].get("save_top_k", 3)),
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg["callbacks"].get("early_stopping", True):
        callbacks.append(
            EarlyStopping(
                monitor=cfg["callbacks"].get("monitor", "val_norm_edit_distance"),
                mode=cfg["callbacks"].get("mode", "min"),
                patience=int(cfg["callbacks"].get("patience", 8)),
            )
        )

    trainer_cfg = dict(cfg["trainer"])
    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=logger,
        **trainer_cfg,
    )
    trainer.fit(module, datamodule=datamodule)

    if trainer.global_rank == 0:
        logger.experiment.log_artifact(logger.run_id, str(config_path))
        logger.experiment.log_artifact(logger.run_id, str(paths["tokenizer_path"]))

        best_path = callbacks[0].best_model_path
        if best_path:
            print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
