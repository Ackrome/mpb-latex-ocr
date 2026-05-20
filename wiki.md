# MPB LaTeX OCR Wiki

## Goal

This project trains models that convert cropped equation images into normalized LaTeX strings. Start with this crop-level formula OCR task before attempting full-page PDF, Markdown, or compilable-LaTeX reconstruction.

## Environment Setup

Use Python 3.10 or newer.

For the local RTX 5070 Ti, install the current CUDA PyTorch wheel first. The official PyTorch installer currently lists CUDA 12.8 as a stable option:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

Verify CUDA:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'@ | python -
```

For Kaggle, install the package after uploading or cloning the repo, then use the Kaggle hardware config:

```bash
pip install -e .
```

Kaggle datasets are mounted read-only under `/kaggle/input`. Training outputs should go under `/kaggle/working` so Kaggle saves them as notebook artifacts.

For Google Colab, select a GPU runtime first, then install the package from a cloned copy of the repo:

```bash
pip install -e ".[dev]"
```

For an A6000 48 GB server, use the same package install but select the CUDA wheel appropriate for that server's driver.

## Feature: Toy Dataset Generator

Use this to prove the training loop works before downloading real formula datasets.

```powershell
latex-ocr-make-toy-data --output-dir data/toy --num-samples 240
```

Equivalent module form:

```powershell
python -m mpb_latex_ocr.make_toy_data --output-dir data/toy --num-samples 240
```

The toy generator renders formulas with Matplotlib mathtext, so the PNGs contain visual equations rather than raw LaTeX source text. It does not require a system LaTeX installation.

Optional rendering controls:

```powershell
latex-ocr-make-toy-data `
  --output-dir data/toy `
  --num-samples 240 `
  --image-width 512 `
  --image-height 128 `
  --font-size 32 `
  --dpi 200
```

It creates:

- `data/toy/images/*.png`
- `data/toy/manifest.csv`

The manifest columns are:

- `image_path`: relative or absolute image path
- `latex`: target LaTeX string
- `split`: `train`, `val`, or `test`
- `sample_id`: optional stable id

## Feature: Lightning Training With MLflow

Run a local smoke training job:

```powershell
latex-ocr-train --config configs/train.yaml --config configs/hardware/rtx5070ti.yaml trainer.max_epochs=3
```

The command writes:

- checkpoints under `outputs/baseline/checkpoints`
- tokenizer at `outputs/baseline/tokenizer.json`
- resolved config at `outputs/baseline/resolved_config.json`
- MLflow runs under `mlruns`

Open MLflow UI:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Then open `http://127.0.0.1:5000`.

## Feature: Hardware Profiles

Hardware profiles are regular config overlays. Later files override earlier files.

Local RTX 5070 Ti:

```powershell
latex-ocr-train --config configs/train.yaml --config configs/hardware/rtx5070ti.yaml
```

Kaggle:

```bash
latex-ocr-train --config configs/train.yaml --config configs/hardware/kaggle.yaml
```

Google Colab GPU:

```bash
latex-ocr-train --config configs/train.yaml --config configs/hardware/colab_gpu.yaml
```

A6000 48 GB:

```bash
latex-ocr-train --config configs/train.yaml --config configs/hardware/a6000_48gb.yaml
```

You can override any value at the command line:

```powershell
latex-ocr-train --config configs/train.yaml data.batch_size=8 trainer.max_epochs=1
```

## Feature: Colab GPU Notebook

Use `notebooks/colab_train.ipynb` when you want to run the smoke-training workflow in Google Colab.

Before running the notebook:

1. Open `Runtime > Change runtime type`.
2. Select `GPU`.
3. Set `REPO_URL` in the notebook if the repo is not already cloned to `/content/mpb-latex-ocr`.
4. Keep `USE_DRIVE = True` if you want checkpoints and MLflow runs to survive runtime resets.

The notebook runs these steps:

- checks the assigned GPU with `nvidia-smi`
- clones or uses the project directory
- optionally mounts Google Drive
- installs the project
- creates toy data
- trains with `configs/hardware/colab_gpu.yaml`
- evaluates the latest checkpoint

The default Colab config writes to `/content`, but the notebook overrides `paths.output_dir`, `paths.tokenizer_path`, and `mlflow.tracking_uri` to Google Drive when Drive persistence is enabled.

## Feature: Kaggle Dataset Manifest Preparation

Use `latex-ocr-prepare-manifest` to convert attached Kaggle datasets into the project manifest format.

Common auto-detected formats:

- `corresponding_png_images.txt` plus `final_png_formulas.txt`, used by common IM2LATEX-230k-style datasets
- CSV/TSV/JSON/JSONL files with image and LaTeX columns
- some Im2LaTeX `.lst` split files plus a formula list

Example for an attached Kaggle dataset:

```bash
latex-ocr-prepare-manifest \
  --input-root /kaggle/input/im2latex-230k/PRINTED_TEX_230k \
  --output /kaggle/working/latex-ocr-manifest.csv \
  --format auto \
  --absolute-paths \
  --max-samples 50000 \
  --val-fraction 0.05 \
  --test-fraction 0.05
```

`--max-samples` uses split-aware sampling after split assignment, so small Kaggle smoke runs still keep validation and test examples when possible.

Then train against the generated manifest:

```bash
latex-ocr-train \
  --config configs/train.yaml \
  --config configs/hardware/kaggle.yaml \
  --config configs/datasets/kaggle_manifest.yaml \
  trainer.max_epochs=5
```

If auto-detection picks the wrong metadata file, specify the table columns:

```bash
latex-ocr-prepare-manifest \
  --input-root /kaggle/input/YOUR_DATASET \
  --output /kaggle/working/latex-ocr-manifest.csv \
  --format table \
  --table-path /kaggle/input/YOUR_DATASET/labels.csv \
  --image-col image_path \
  --latex-col latex \
  --split-col split \
  --absolute-paths
```

## Feature: Kaggle Training Notebook

Use `notebooks/kaggle_train.ipynb` when training on Kaggle-hosted datasets.

Before running it:

1. Create or open a Kaggle notebook.
2. Enable a GPU accelerator.
3. Attach a formula OCR dataset from the right sidebar.
4. Set `REPO_URL` or copy this repo to `/kaggle/working/mpb-latex-ocr`.
5. Set `KAGGLE_DATASET_ROOT` to the attached dataset folder under `/kaggle/input`.

The notebook prepares `/kaggle/working/latex-ocr-manifest.csv`, trains with `configs/hardware/kaggle.yaml`, evaluates the test split, and leaves artifacts in `/kaggle/working`.

## Feature: Evaluation

Evaluate a checkpoint on the test split:

```powershell
latex-ocr-evaluate `
  --checkpoint outputs/baseline/checkpoints/YOUR_CHECKPOINT.ckpt `
  --manifest data/toy/manifest.csv `
  --image-root data/toy `
  --tokenizer outputs/baseline/tokenizer.json `
  --split test `
  --predictions-out outputs/baseline/test_predictions.jsonl
```

Metrics currently include:

- exact match
- normalized edit distance

These are baseline metrics. For serious formula OCR, add render-aware metrics later.

## Feature: Single-Image Prediction

Predict LaTeX for one image:

```powershell
latex-ocr-predict `
  --checkpoint outputs/baseline/checkpoints/YOUR_CHECKPOINT.ckpt `
  --tokenizer outputs/baseline/tokenizer.json `
  --image data/toy/images/formula_00000.png
```

Predict a whole directory:

```powershell
latex-ocr-predict `
  --checkpoint outputs/baseline/checkpoints/YOUR_CHECKPOINT.ckpt `
  --tokenizer outputs/baseline/tokenizer.json `
  --image data/toy/images `
  --output outputs/baseline/predictions.jsonl
```

## Real Dataset Plan

Use this order:

1. Toy generated data to validate code.
2. Im2LaTeX-100K to validate a classic baseline.
3. UniMER-1M subset for real training.
4. UniMER-Test for serious evaluation.
5. CROHME or MathWriting if handwriting is a target.

Keep each dataset as a manifest with the same columns used by the toy generator. Do not mix train, validation, and test sources without recording the source and split in the manifest.

## Current Baseline Architecture

The current model is a compact CNN encoder plus Transformer decoder. It is designed to be easy to train and debug, not to be state of the art.

Next architecture upgrades should be:

1. Pretrained vision encoder.
2. Hugging Face `VisionEncoderDecoderModel`.
3. UniMERNet-style Swin encoder plus mBART-like decoder.
4. Beam search and render-aware reranking.

## Experiment Rules

For every real experiment, log or save:

- resolved config
- tokenizer
- dataset manifest hash
- checkpoint
- validation metrics
- test predictions
- failure cases

Use MLflow to compare runs. Do not compare models trained with different target normalization unless the normalization is explicitly documented.
