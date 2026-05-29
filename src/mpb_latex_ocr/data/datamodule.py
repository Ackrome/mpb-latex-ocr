"""Lightning DataModule for formula OCR manifests."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import lightning.pytorch as L
from torch.utils.data import DataLoader

from mpb_latex_ocr.data.dataset import (
    LatexFormulaDataset,
    collate_formula_batch,
    read_manifest,
    split_samples,
)
from mpb_latex_ocr.data.tokenizer import LatexTokenizer


class LatexOCRDataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: LatexTokenizer,
        image_root: str | Path | None = None,
        image_height: int = 128,
        image_width: int = 512,
        batch_size: int = 16,
        num_workers: int = 2,
        max_label_length: int = 256,
        augment: bool = True,
        augmentation_profile: str = "printed",
        augmentation_strength: float = 1.0,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_label_length = max_label_length
        self.augment = augment
        self.augmentation_profile = augmentation_profile
        self.augmentation_strength = augmentation_strength

        self.train_dataset: LatexFormulaDataset | None = None
        self.val_dataset: LatexFormulaDataset | None = None
        self.test_dataset: LatexFormulaDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        samples = read_manifest(self.manifest_path)
        train_samples = split_samples(samples, "train")
        val_samples = split_samples(samples, "val")
        test_samples = split_samples(samples, "test")

        self.train_dataset = LatexFormulaDataset(
            train_samples,
            tokenizer=self.tokenizer,
            image_root=self.image_root,
            image_height=self.image_height,
            image_width=self.image_width,
            max_label_length=self.max_label_length,
            augment=self.augment,
            augmentation_profile=self.augmentation_profile,
            augmentation_strength=self.augmentation_strength,
        )
        self.val_dataset = LatexFormulaDataset(
            val_samples,
            tokenizer=self.tokenizer,
            image_root=self.image_root,
            image_height=self.image_height,
            image_width=self.image_width,
            max_label_length=self.max_label_length,
            augment=False,
        )
        self.test_dataset = LatexFormulaDataset(
            test_samples,
            tokenizer=self.tokenizer,
            image_root=self.image_root,
            image_height=self.image_height,
            image_width=self.image_width,
            max_label_length=self.max_label_length,
            augment=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before train_dataloader().")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None
        return self._loader(self.test_dataset, shuffle=False)

    def _loader(self, dataset: LatexFormulaDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(collate_formula_batch, pad_id=self.tokenizer.pad_id),
        )


def build_or_load_tokenizer(
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    force_rebuild: bool = False,
) -> LatexTokenizer:
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists() and not force_rebuild:
        return LatexTokenizer.load(tokenizer_path)

    samples = read_manifest(manifest_path)
    train_texts = [sample.latex for sample in samples if sample.split.lower() == "train"]
    if not train_texts:
        raise ValueError("Cannot train tokenizer: manifest has no train split rows.")

    tokenizer = LatexTokenizer.train(
        train_texts,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )
    tokenizer.save(tokenizer_path)
    return tokenizer
