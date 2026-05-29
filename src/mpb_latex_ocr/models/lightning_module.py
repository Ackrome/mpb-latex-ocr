"""PyTorch Lightning module for LaTeX OCR."""

from __future__ import annotations

import math
from typing import Any

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.data.tokenizer import LatexTokenizer
from mpb_latex_ocr.metrics.edit_distance import normalized_edit_distance
from mpb_latex_ocr.models.encoder_decoder import TransformerOCRModel


class LatexOCRModule(L.LightningModule):
    def __init__(
        self,
        tokenizer: LatexTokenizer,
        model_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        generation_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_config = dict(model_config)
        self.optimizer_config = dict(optimizer_config)
        self.generation_config = dict(generation_config or {})
        self.save_hyperparameters(
            {
                "model_config": self.model_config,
                "optimizer_config": self.optimizer_config,
                "generation_config": self.generation_config,
            }
        )

        self.model = TransformerOCRModel(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            **self.model_config,
        )

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(images, decoder_input_ids)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        max_batches = int(self.generation_config.get("validation_batches", 0))
        if max_batches <= 0 or batch_idx < max_batches:
            exact_match, norm_edit = self._generation_metrics(batch)
            self.log(
                "val_exact_match",
                exact_match,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val_norm_edit_distance",
                norm_edit,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        exact_match, norm_edit = self._generation_metrics(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_exact_match", exact_match, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_norm_edit_distance", norm_edit, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.optimizer_config.get("lr", 3e-4)),
            weight_decay=float(self.optimizer_config.get("weight_decay", 0.01)),
        )

        if self.optimizer_config.get("scheduler", "cosine") != "cosine":
            return optimizer

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * float(self.optimizer_config.get("warmup_fraction", 0.05)))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _loss(self, batch: dict[str, Any]) -> torch.Tensor:
        labels = batch["labels"]
        decoder_input_ids = labels[:, :-1].contiguous()
        targets = labels[:, 1:].contiguous()
        logits = self.model(batch["pixel_values"], decoder_input_ids)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_id,
        )

    @torch.inference_mode()
    def _generation_metrics(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = int(self.generation_config.get("max_length", self.model.max_seq_len))
        generated_ids = self.model.generate(batch["pixel_values"], max_length=max_length)
        predictions = [self.tokenizer.decode(row.tolist()) for row in generated_ids]
        targets = [normalize_latex(text) for text in batch["latex"]]

        exact_values: list[float] = []
        edit_values: list[float] = []
        for prediction, target in zip(predictions, targets, strict=True):
            prediction = normalize_latex(prediction)
            target = normalize_latex(target)
            exact_values.append(float(prediction == target))
            edit_values.append(normalized_edit_distance(prediction, target))

        device = batch["pixel_values"].device
        return (
            torch.tensor(sum(exact_values) / len(exact_values), device=device),
            torch.tensor(sum(edit_values) / len(edit_values), device=device),
        )
