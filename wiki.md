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
export PYTHONPATH=/kaggle/working/mpb-latex-ocr/src:$PYTHONPATH
pip install --force-reinstall --no-deps -e ".[dev]"
pip install lightning mlflow omegaconf pillow matplotlib numpy tqdm pytest
python scripts/kaggle_preflight.py
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

## Feature: Plot MLflow Training Curves Without Rerunning

Use `latex-ocr-plot-mlruns` or the module form to create static PNG plots directly from an MLflow file-store directory. This does not rerun training or evaluation.

List available runs:

```powershell
.\.venv\Scripts\python.exe -m mpb_latex_ocr.plot_mlruns `
  --mlruns-dir mlruns `
  --list-runs
```

Plot a specific run:

```powershell
.\.venv\Scripts\python.exe -m mpb_latex_ocr.plot_mlruns `
  --mlruns-dir mlruns `
  --run-id YOUR_RUN_ID `
  --output-dir outputs\training_curves
```

Plot the newest discovered run:

```powershell
.\.venv\Scripts\python.exe -m mpb_latex_ocr.plot_mlruns `
  --mlruns-dir mlruns `
  --run-id latest `
  --output-dir outputs\training_curves
```

Outputs:

```text
outputs/training_curves/
  loss_curves.png
  validation_metrics.png
  learning_rate.png
  all_metrics.png
  metrics_long.csv
  summary.json
```

For Kaggle training curves, download the Kaggle artifact folder `/kaggle/working/mlruns` and place it at `mlruns/` or another local path. If you only downloaded checkpoints, tokenizer, config, and `test_predictions.jsonl`, you can plot final prediction metrics but not recover the full train/validation loss curves. Checkpoint filenames may contain one validation metric such as `val_ned`, but not the whole curve.

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
3. The notebook defaults `REPO_URL` to `https://github.com/Ackrome/mpb-latex-ocr.git`; change it only if you are using a fork.
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
python -m mpb_latex_ocr.prepare_manifest \
  --input-root /kaggle/input/im2latex-230k/PRINTED_TEX_230k \
  --output /kaggle/working/latex-ocr-manifest.csv \
  --format auto \
  --absolute-paths \
  --max-samples 50000 \
  --val-fraction 0.05 \
  --test-fraction 0.05
```

`--max-samples` uses split-aware sampling after split assignment, so small Kaggle smoke runs still keep validation and test examples when possible.

## Feature: MathWriting InkML Preparation

Use this for Google MathWriting-style handwriting data. MathWriting examples are InkML stroke files, not ready-made PNG crops, so the manifest command rasterizes strokes into PNGs and writes the normal OCR manifest.

Example:

```powershell
latex-ocr-prepare-manifest `
  --input-root data/mathwriting/raw `
  --format mathwriting-inkml `
  --output data/mathwriting/manifest.csv `
  --render-dir data/mathwriting/rendered `
  --mathwriting-image-width 1024 `
  --mathwriting-image-height 256 `
  --mathwriting-stroke-width 4 `
  --absolute-paths
```

The parser reads InkML `<trace>` strokes and the `normalizedLabel` annotation when present. Split folders are mapped as:

- `train` and `synthetic` -> `train`
- `valid` or `validation` -> `val`
- `test` -> `test`
- `symbols` are skipped by default; add `--mathwriting-include-symbols` if single-symbol training examples are useful

Train with the handwriting augmentation profile and the deeper CNN preset:

```powershell
latex-ocr-train `
  --config configs/train.yaml `
  --config configs/model/deep_cnn.yaml `
  --config configs/datasets/mathwriting.yaml `
  paths.manifest=data/mathwriting/manifest.csv
```

The `configs/datasets/mathwriting.yaml` overlay sets a larger OCR canvas, enables `augmentation_profile: handwriting`, and writes outputs to `outputs/mathwriting_deep_cnn`.

The handwriting augmentation profile adds the printed-profile contrast/brightness/rotation jitter plus small affine shear/translation, stroke thickening/thinning, light blur, and pixel noise. Keep validation/test augmentation off; the data module already does this.

Then train against the generated manifest:

```bash
python -m mpb_latex_ocr.train \
  --config configs/train.yaml \
  --config configs/hardware/kaggle.yaml \
  --config configs/datasets/kaggle_manifest.yaml \
  trainer.max_epochs=5
```

If auto-detection picks the wrong metadata file, specify the table columns:

```bash
python -m mpb_latex_ocr.prepare_manifest \
  --input-root /kaggle/input/YOUR_DATASET \
  --output /kaggle/working/latex-ocr-manifest.csv \
  --format table \
  --table-path /kaggle/input/YOUR_DATASET/labels.csv \
  --image-col image_path \
  --latex-col latex \
  --split-col split \
  --absolute-paths
```

If Kaggle raises `ModuleNotFoundError` after you update the repo, run:

```bash
cd /kaggle/working/mpb-latex-ocr
find src/mpb_latex_ocr -maxdepth 2 -type f | sort
python scripts/kaggle_preflight.py
```

If `src/mpb_latex_ocr/data/__init__.py` is missing, Kaggle is using an incomplete or stale project copy. Delete `/kaggle/working/mpb-latex-ocr`, clone or upload the current repo again, and rerun the install cell. The notebook also sets `PYTHONPATH=/kaggle/working/mpb-latex-ocr/src` and uses `python -m ...` commands so it picks up the current working copy rather than a stale console entry point.

## Feature: Kaggle Training Notebook

Use `notebooks/kaggle_train.ipynb` when training on Kaggle-hosted datasets.

Before running it:

1. Create or open a Kaggle notebook.
2. Enable a GPU accelerator.
3. Attach a formula OCR dataset from the right sidebar.
4. The notebook defaults `REPO_URL` to `https://github.com/Ackrome/mpb-latex-ocr.git`; change it only if you are using a fork, or copy this repo to `/kaggle/working/mpb-latex-ocr`.
5. Set `KAGGLE_DATASET_ROOT` to the attached dataset folder under `/kaggle/input`.

The notebook prepares `/kaggle/working/latex-ocr-manifest.csv`, trains with `configs/hardware/kaggle.yaml`, evaluates the test split, and leaves artifacts in `/kaggle/working`.

## Feature: Kaggle Model2 Training Notebook

Use `notebooks/kaggle_train_model2.ipynb` when training the page-level model2 pipeline on Kaggle.

Model2 has two trainable parts with different datasets:

- model1 OCR: cropped formula image to LaTeX, trained from Im2LaTeX-style crop datasets
- YOLO26 detector: page/photo image to formula bounding boxes, trained from YOLO-format box datasets

Default run-all workflow:

1. Create or open a Kaggle notebook.
2. Enable a GPU accelerator.
3. Attach these Kaggle datasets from the notebook sidebar:
   - `willcsc/mathwriting`
   - `gregoryeritsyan/im2latex-230k`
   - `shiva22btcse0007/25k-math-equation`
   - `anismekacher/synthetic-mathemtical-expression-detection`
4. Keep the notebook defaults and press Kaggle's `Run All`.
5. The notebook defaults `REPO_URL` to `https://github.com/Ackrome/mpb-latex-ocr.git`; change it only if you are using a fork, or copy this repo to `/kaggle/working/mpb-latex-ocr`.

The notebook's auto-prep step creates `/kaggle/working/latex-ocr-manifest.csv` from the attached crop-to-LaTeX datasets:

- Im2LaTeX paired files: `corresponding_png_images.txt` plus `final_png_formulas.txt`
- MathWriting PNG/TXT shards under `shards/...`
- 25k equation labels from `dataset25k/labels.txt`

It also converts the synthetic detection dataset's `coco_ann.json` plus `equation_imgs/` into a YOLO tree under:

```text
/kaggle/working/latex-ocr-runs/prepared_inputs/synthetic_detection_yolo/
  data.yaml
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
```

The main notebook switches are:

```python
TRAIN_MODEL1_OCR = True
TRAIN_MODEL2_DETECTOR = True
RUN_MODEL2_PAGE_INFERENCE = True
AUTO_DISCOVER_INPUTS = True
AUTO_PREPARE_MODEL2_INPUTS = True
AUTO_DOWNLOAD_DATASETS = False
DETECTOR_MODEL_WEIGHTS = "yolo26n.pt"
```

`AUTO_DOWNLOAD_DATASETS = False` is intentional for the attached-dataset workflow. Set it to `True` only if you want the notebook to try downloading missing public Kaggle datasets.

Manual overrides remain available for existing artifacts or different datasets:

```python
TRAIN_MODEL1_OCR = False
EXISTING_MODEL1_CHECKPOINT = Path("/kaggle/input/YOUR_MODEL1_RUN/best.ckpt")
EXISTING_MODEL1_TOKENIZER = Path("/kaggle/input/YOUR_MODEL1_RUN/tokenizer.json")
DETECTOR_DATA_YAML = Path("/kaggle/input/YOUR_FORMULA_DETECTOR/data.yaml")
PAGE_IMAGE_SOURCE = Path("/kaggle/input/YOUR_PAGE_IMAGES")
```

The notebook writes:

```text
/kaggle/working/latex-ocr-runs/
  detection/yolo/formula_regions/weights/best.pt
  model2_detector_preview/crops.jsonl
  model2_detector_preview/crops/
  model2_page_predict/crops.jsonl
  model2_page_predict/predictions.jsonl
  model2_page_predict/crops/
```

If `TRAIN_MODEL1_OCR = True`, it also writes the model1 artifacts under:

```text
/kaggle/working/latex-ocr-runs/baseline/
  checkpoints/
  tokenizer.json
  resolved_config.json
```

After evaluation, run the `Visual Check On Im2LaTeX Samples` section to inspect real crops from the attached dataset such as:

```text
/kaggle/input/.../im2latex-230k/PRINTED_TEX_230k
```

The visual check reads `test_predictions.jsonl`, displays input crops, target renders, prediction renders, and raw strings, and prints a compact metric summary. Use `VISUAL_SAMPLE_MODE` to choose examples:

- `random`: representative random samples
- `worst_render`: lowest render-F1 samples for failure analysis
- `best_render`: highest render-F1 samples
- `first`: the first rows in the prediction file

Then run the `Export Portable Im2LaTeX Sample Bundle` section if you want to inspect the exact same dataset images locally. It copies selected image files referenced by `test_predictions.jsonl` into:

```text
/kaggle/working/latex-ocr-runs/baseline/im2latex_sample_bundle/
  images/
  manifest.csv
  predictions.jsonl
  metadata.json
```

and writes:

```text
/kaggle/working/latex-ocr-runs/baseline/im2latex_sample_bundle.zip
```

Download the zip from Kaggle artifacts.

## Feature: Local Im2LaTeX Sample Bundle Check

Use `notebooks/local_im2latex_sample_check.ipynb` to inspect exact Im2LaTeX images downloaded from Kaggle and rerun your local checkpoint against them.

After downloading `im2latex_sample_bundle.zip`, extract it into:

```text
data/im2latex_sample_bundle/
```

PowerShell example:

```powershell
Expand-Archive outputs\im2latex_sample_bundle.zip data\im2latex_sample_bundle -Force
```

The expected local bundle layout is:

```text
data/im2latex_sample_bundle/
  images/
  manifest.csv
  predictions.jsonl
  metadata.json
```

Open the notebook:

```powershell
jupyter notebook notebooks/local_im2latex_sample_check.ipynb
```

The notebook does two checks:

- Summarizes and visualizes the exported Kaggle/server predictions from `data/im2latex_sample_bundle/predictions.jsonl`.
- Optionally reruns the local checkpoint from `outputs/checkpoints` with `outputs/tokenizer.json` on the same exact images and writes local predictions to `outputs/im2latex_bundle_check/test_predictions.jsonl`.

The last visual cell writes one readable figure per selected example under:

```text
outputs/im2latex_bundle_check/visual_examples/
```

Each figure has a left label rail and a readable vertical stack: input crop, target render, prediction render, and an enlarged raw target/prediction string panel. Set `VISUAL_SAMPLE_COUNT`, `VISUAL_SAMPLE_MODE`, and `VISUAL_SEED` in the setup cell before running it.

Use this when the toy-renderer check looks bad but you need to know whether the downloaded model still behaves correctly on real `PRINTED_TEX_230k` images.

You can also build the local bundle without rerunning the Kaggle notebook if you already downloaded `outputs/test_predictions.jsonl`. Download the Kaggle dataset locally with the Kaggle API:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\python.exe -m kaggle datasets download `
  -d gregoryeritsyan/im2latex-230k `
  -p data\kaggle\im2latex-230k `
  --unzip
```

This requires Kaggle credentials in `%USERPROFILE%\.kaggle\kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

Then materialize a portable local sample bundle from your existing prediction file. Prefer the module form because it works even if editable console scripts are stale:

```powershell
.\.venv\Scripts\python.exe -m mpb_latex_ocr.export_prediction_samples `
  --predictions outputs\test_predictions.jsonl `
  --output-dir data\im2latex_sample_bundle `
  --num-samples 96 `
  --mode random `
  --path-map "/kaggle/input/datasets/gregoryeritsyan/im2latex-230k=data/kaggle/im2latex-230k" `
  --zip-out outputs\im2latex_sample_bundle.zip
```

If you want the shorter `latex-ocr-export-prediction-samples` command, reinstall the project after pulling this feature:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`--path-map` rewrites the original Kaggle image path stored in `test_predictions.jsonl` to the local dataset directory before copying images. For example:

```text
/kaggle/input/datasets/gregoryeritsyan/im2latex-230k/PRINTED_TEX_230k/generated_png_images/a.png
```

becomes:

```text
data/kaggle/im2latex-230k/PRINTED_TEX_230k/generated_png_images/a.png
```

The local notebook includes the same direct-download/build section behind `DOWNLOAD_KAGGLE_DATASET` and `BUILD_BUNDLE_FROM_LOCAL_DATASET` toggles.

The default local bundle is only 96 examples because it is meant for fast visual inspection and CDM debugging. That number comes from either `--num-samples 96` in `latex-ocr-export-prediction-samples` or `DIRECT_BUNDLE_SAMPLE_COUNT = 96` in the notebook. Increase it for a larger bundle:

```powershell
.\.venv\Scripts\python.exe -m mpb_latex_ocr.export_prediction_samples `
  --predictions outputs\test_predictions.jsonl `
  --output-dir data\im2latex_sample_bundle `
  --num-samples 1000 `
  --mode random `
  --path-map "/kaggle/input/datasets/gregoryeritsyan/im2latex-230k=data/kaggle/im2latex-230k" `
  --zip-out outputs\im2latex_sample_bundle.zip
```

Use the full `outputs/cdm_predictions.json` for full-test official CDM. Do not confuse that full prediction file with the local portable image bundle; the bundle intentionally copies only selected images.

The notebook also has an `Official CDM Metrics` section. It uses `outputs/im2latex_bundle_check/cdm_predictions.json`, runs `scripts/cdm/run_official_cdm_windows.ps1` when metrics are missing or `FORCE_RECOMPUTE_OFFICIAL_CDM=True`, and displays summary metrics plus the worst per-image CDM rows. The default output path is:

```text
outputs/im2latex_bundle_check/official_cdm_windows/cdm_predictions/metrics_res.json
```

## Feature: Local Acquired Checkpoint Toy Check Notebook

Use `notebooks/local_checkpoint_toy_check.ipynb` after downloading a trained run from Kaggle, Colab, or a server. It verifies that the acquired checkpoint bundle loads locally, runs inference on a small generated toy formula set, computes string and render-proxy metrics, and displays visual prediction examples.

Default expected local layout:

```text
outputs/
  checkpoints/
    epoch=041-val_ned=0.0746.ckpt
  tokenizer.json
  resolved_config.json
```

The notebook selects the checkpoint with the lowest `val_ned=...` value in the filename. If filenames do not contain `val_ned`, it falls back to the newest checkpoint.

Run the notebook from the project root or open it in VS Code/Jupyter:

```powershell
jupyter notebook notebooks/local_checkpoint_toy_check.ipynb
```

The notebook writes:

```text
data/toy_acquired_checkpoint_check/
  images/
  manifest.csv
outputs/toy_check/
  test_predictions.jsonl
  cdm_predictions.json
```

Interpretation:

- This is a sanity check, not the final benchmark. The toy images are rendered by the in-repo Matplotlib toy generator and may not match the Kaggle dataset render style.
- The notebook also summarizes `outputs/test_predictions.jsonl` when present. Prefer that real dataset summary over toy metrics when judging a Kaggle-trained checkpoint.
- If the checkpoint fails to load, the `.ckpt`, `tokenizer.json`, or source code version is mismatched.
- If predictions are empty or invalid, inspect tokenizer coverage in the notebook and confirm the selected checkpoint is the intended one.
- If toy metrics are lower than real test metrics but predictions are valid and visually plausible, trust the real dataset evaluation more.
- If you previously evaluated with an older repo version, rerun evaluation after this update. Evaluation now compares predictions against the manifest `latex` text directly instead of decoding targets through the tokenizer, which matters for out-of-vocabulary validation or toy formulas.

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

Enable render-aware proxy metrics:

```powershell
latex-ocr-evaluate `
  --checkpoint outputs/baseline/checkpoints/YOUR_CHECKPOINT.ckpt `
  --manifest data/toy/manifest.csv `
  --image-root data/toy `
  --tokenizer outputs/baseline/tokenizer.json `
  --split test `
  --render-metric `
  --predictions-out outputs/baseline/test_predictions.jsonl `
  --cdm-json-out outputs/baseline/cdm_predictions.json
```

Render-aware output metrics:

- `render_iou_with_failures`: pixel-mask IoU after rendering prediction and target
- `render_f1_with_failures`: pixel-mask Dice/F1 after rendering prediction and target
- `render_match`: share of samples with render F1 at or above `--render-match-threshold`
- `prediction_render_success`: share of predictions that Matplotlib mathtext could render
- `target_render_success`: share of targets that Matplotlib mathtext could render
- `pair_render_success`: share of samples where both prediction and target rendered

This is a practical render-aware proxy, not official CDM. It is useful for fast local/Kaggle evaluation because it catches visually equivalent strings better than edit distance and penalizes invalid LaTeX that cannot render.

For official CDM-style reporting, use `--cdm-json-out` to export a JSON list with `img_id`, `gt`, and `pred` records. Run the official CDM tooling on that file in an environment with its rendering dependencies. Keep both scores in reports: string metrics for debugging, render proxy for quick model selection, and official CDM for final benchmark claims.

## Feature: Official CDM Evaluation

Official CDM is maintained in the UniMERNet repository under `cdm/`. It is heavier than the local render proxy because it renders formulas, extracts character boxes, and performs spatial character matching. The official README recommends Linux and requires Node.js, ImageMagick, `pdflatex`/TeX Live, and Python requirements, or using the provided Dockerfile.

First export CDM input:

```powershell
latex-ocr-score-predictions `
  --predictions C:\Users\Rog G16\Downloads\test_predictions.jsonl `
  --cdm-json-out outputs/baseline/cdm_predictions_safe.json
```

The exported `img_id` values are filename-safe image stems, not absolute paths.

Recommended Docker flow:

```bash
git clone https://github.com/opendatalab/UniMERNet.git
cd UniMERNet/cdm
docker build -f DockerFile -t cdm:latest .
docker run --rm -it -v /ABS/PATH/TO/mpb-latex-ocr/outputs/baseline:/work cdm:latest bash
```

Inside the container:

```bash
cd /path/to/UniMERNet/cdm
python evaluation.py \
  -i /work/cdm_predictions_safe.json \
  -o /work/official_cdm \
  -p 8
```

On Windows PowerShell, the Docker mount is typically:

```powershell
docker run --rm -it -v "C:\Projects\mpb-latex-ocr\outputs\baseline:/work" cdm:latest bash
```

Official CDM writes:

- `official_cdm/cdm_predictions_safe/metrics_res.json`
- visual matching images under `official_cdm/cdm_predictions_safe/vis_match`

Use `mean_score` from `metrics_res.json` as the official CDM F1-style score, and `exp_rate` as the exact-render-style rate where CDM score equals 1.

Windows Docker/WSL shortcut:

If Docker Desktop is running on Windows, but no normal Ubuntu WSL distro is installed, use the lightweight Dockerfile in this repo:

```powershell
git clone https://github.com/opendatalab/UniMERNet.git outputs/tools/UniMERNet
docker build `
  -f scripts/cdm/Dockerfile.cdm-lite `
  -t cdm-lite:latest `
  outputs/tools/UniMERNet/cdm
docker run --rm `
  -v "C:\Projects\mpb-latex-ocr\outputs\baseline:/work" `
  cdm-lite:latest `
  python /code/evaluation.py `
    -i /work/cdm_predictions_safe.json `
    -o /work/official_cdm_lite `
    -p 4
```

The lightweight image uses apt ImageMagick and a TeX Live subset. It is faster to build than the official Dockerfile, but may miss rare LaTeX packages. Use the official Dockerfile or `texlive-full` if final CDM rendering fails.

Direct Windows PowerShell flow:

Official CDM can also run directly on Windows if Node.js, MiKTeX `pdflatex`, ImageMagick, Ghostscript, and the Python dependencies are available. The repo includes a helper that patches the local UniMERNet CDM checkout for Windows path quoting and current `scikit-image` compatibility before running `evaluation.py`.

First-time setup:

```powershell
git clone --depth 1 https://github.com/opendatalab/UniMERNet.git outputs\tools\UniMERNet
.\.venv\Scripts\python.exe -m pip install scikit-image opencv-python
conda create -p outputs\tools\conda_ghostscript -c conda-forge ghostscript -y
```

Run CDM on the local 96-sample Im2LaTeX bundle:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cdm\run_official_cdm_windows.ps1 `
  -InputJson outputs\im2latex_bundle_check\cdm_predictions.json `
  -OutputDir outputs\im2latex_bundle_check\official_cdm_windows `
  -Pools 4
```

The metrics path is:

```text
outputs/im2latex_bundle_check/official_cdm_windows/cdm_predictions/metrics_res.json
```

To run the full downloaded Kaggle prediction file, point `-InputJson` to `outputs\cdm_predictions.json`. This is much slower because CDM renders both target and prediction for every row.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cdm\run_official_cdm_windows.ps1 `
  -InputJson outputs\cdm_predictions.json `
  -OutputDir outputs\official_cdm_windows_full `
  -Pools 8
```

Direct WSL Ubuntu flow:

Install a normal Ubuntu WSL distro first. Docker Desktop's internal `docker-desktop` distro is not enough.

```powershell
wsl --install -d Ubuntu-24.04
```

After Ubuntu opens and finishes first-time setup, run from PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Projects/mpb-latex-ocr && bash scripts/cdm/run_official_cdm_wsl.sh"
```

The script reads:

```text
outputs/baseline/cdm_predictions_safe.json
```

and writes:

```text
outputs/baseline/official_cdm_wsl/cdm_predictions_safe/metrics_res.json
```

You can override paths:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Projects/mpb-latex-ocr && CDM_INPUT=/mnt/c/Projects/mpb-latex-ocr/outputs/baseline/cdm_predictions_safe.json CDM_POOLS=8 bash scripts/cdm/run_official_cdm_wsl.sh"
```

Kaggle notebook flow:

Kaggle notebooks cannot run Docker, so `notebooks/kaggle_train.ipynb` includes an optional direct-install CDM section after evaluation. Keep `RUN_OFFICIAL_CDM = False` for normal training, then set it to `True` after `cdm_predictions.json` exists.

The notebook cell does the following:

- installs Node.js, ImageMagick, poppler utilities, and a TeX Live subset with `apt-get`
- relaxes ImageMagick PDF policy when possible
- clones `https://github.com/opendatalab/UniMERNet`
- installs CDM Python dependencies
- runs `python evaluation.py -i /kaggle/working/latex-ocr-runs/baseline/cdm_predictions.json -o /kaggle/working/latex-ocr-runs/baseline/official_cdm -p 4`

If CDM rendering fails on missing LaTeX packages, replace the lighter TeX package list in the notebook with `texlive-full`. That is slower and larger, but closer to the official README requirement.

## Feature: Scoring Existing Prediction Files

Use `latex-ocr-score-predictions` when you already have `test_predictions.jsonl` and do not want to rerun model inference.

```powershell
latex-ocr-score-predictions `
  --predictions C:\Users\Rog G16\Downloads\test_predictions.jsonl `
  --render-metric `
  --scored-out outputs/baseline/test_predictions_scored.jsonl `
  --cdm-json-out outputs/baseline/cdm_predictions.json
```

For Kaggle:

```bash
python -m mpb_latex_ocr.score_predictions \
  --predictions /kaggle/working/latex-ocr-runs/baseline/test_predictions.jsonl \
  --render-metric \
  --scored-out /kaggle/working/latex-ocr-runs/baseline/test_predictions_scored.jsonl \
  --cdm-json-out /kaggle/working/latex-ocr-runs/baseline/cdm_predictions.json
```

This recomputes exact match and normalized edit distance, adds render-aware proxy metrics when `--render-metric` is enabled, and writes official-CDM-style `img_id`, `gt`, `pred` pairs. It does not run the official CDM algorithm itself.

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

## Feature: YOLO Formula Region Detection Pipeline

Use this when the input is a page or photo that may contain one or more formulas. YOLO26 finds formula regions first, saves crops, then the existing crop-level OCR model predicts LaTeX for each crop.

This is a pipeline feature. It does not replace the current OCR architecture; the current CNN plus Transformer OCR model remains model1. If the OCR architecture changes later, put that under model2.

Install the optional detector dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[detector,dev]"
```

Train a formula detector from a YOLO-format dataset:

```powershell
latex-ocr-detect-train --config configs/detection/yolo_formula.yaml
```

The detector config expects a YOLO `data.yaml` with a formula class, for example:

```yaml
path: C:/datasets/formula_regions
train: train/images
val: val/images
test: test/images
names:
  - formula
```

Override detector training settings at the command line:

```powershell
latex-ocr-detect-train `
  --config configs/detection/yolo_formula.yaml `
  train.epochs=10 `
  train.batch_size=4 `
  train.device=0
```

Detect formula regions and save crops without running OCR:

```powershell
latex-ocr-detect `
  --weights outputs/detection/yolo/formula_regions/weights/best.pt `
  --image data/pages `
  --class-name formula `
  --output-dir outputs/detection/page_crops `
  --metadata-out outputs/detection/page_crops/crops.jsonl
```

Run the full page-to-LaTeX pipeline:

```powershell
latex-ocr-page-predict `
  --detector-weights outputs/detection/yolo/formula_regions/weights/best.pt `
  --checkpoint outputs/baseline/checkpoints/YOUR_CHECKPOINT.ckpt `
  --tokenizer outputs/baseline/tokenizer.json `
  --image data/pages `
  --detector-class-name formula `
  --output-dir outputs/page_predict `
  --predictions-out outputs/page_predict/predictions.jsonl
```

Outputs:

- `outputs/page_predict/crops/`: cropped formula images in reading order
- `outputs/page_predict/crops.jsonl`: crop metadata with source image, bbox, confidence, and crop path
- `outputs/page_predict/predictions.jsonl`: the same crop metadata plus predicted `latex`

Tuning options:

- `--detector-confidence`: raise it to reduce false positives, lower it to recover missed formulas
- `--crop-padding-px` and `--crop-padding-ratio`: add whitespace around detector boxes before OCR
- `--row-tolerance`: controls top-to-bottom, left-to-right crop ordering for multi-formula pages
- `--image-height` and `--image-width`: must match the OCR checkpoint's expected crop size
- `--detector-device` and `--ocr-device`: split devices when YOLO and OCR need different syntax
- `--class-name` / `--detector-class-name`: restrict multi-class detector outputs to formula/equation classes before OCR

## Real Dataset Plan

Use this order:

1. Toy generated data to validate code.
2. Im2LaTeX-100K to validate a classic baseline.
3. UniMER-1M subset for real training.
4. UniMER-Test for serious evaluation.
5. CROHME or MathWriting if handwriting is a target. MathWriting is now supported through InkML rasterization plus handwriting augmentations.

Keep each dataset as a manifest with the same columns used by the toy generator. Do not mix train, validation, and test sources without recording the source and split in the manifest.

## Current Model Architecture Figures

Model1 is the compact CNN encoder plus Transformer decoder. It is designed to be easy to train and debug, not to be state of the art.

The default model1 config is preserved. For higher-capacity OCR experiments, use `configs/model/deep_cnn.yaml`; it widens the CNN encoder and adds residual blocks through `encoder_depths` while keeping the same decoder/checkpoint interface.

Model2 is the YOLO formula detector plus model1 OCR page pipeline. It changes the input scope from cropped formulas to page images, but does not replace model1.

Use the paper-style model descriptions and diagrams here:

```text
docs/model_architecture.md
docs/figures/model1_cnn_transformer_architecture.svg
docs/figures/model2_yolo_ocr_pipeline.svg
docs/assets/model_architecture.png
```

`docs/figures/model1_cnn_transformer_architecture.svg` shows the implemented CNN encoder, flattened 1024-token visual memory, four pre-norm Transformer decoder layers, output logits, training objective, and parameter breakdown.

`docs/figures/model2_yolo_ocr_pipeline.svg` shows the YOLO detector training branch, page-level inference graph, crop metadata flow, and the boundary showing that model2 calls model1 rather than overwriting it.

The older `docs/figures/baseline_architecture.svg` remains available for backward compatibility.

Regenerate the PNG architecture figure from the current saved config and tokenizer:

```powershell
.\.venv\Scripts\python.exe scripts\make_model_architecture_figure.py `
  --config outputs\resolved_config.json `
  --tokenizer outputs\tokenizer.json `
  --output docs\assets\model_architecture.png
```

The current downloaded run uses:

- input shape: grayscale `1 x 128 x 512`
- encoder: four convolutional stages ending in a `256 x 16 x 64` feature map
- visual memory: `1024` tokens of width `256`
- decoder: 4-layer Transformer decoder, 8 heads, FFN width 1024
- max generation length: 256 tokens
- current tokenizer vocabulary: 543 tokens
- current parameter count: 5,012,959

This is an image-to-sequence recognition model, not a segmentation/U-Net model.

Deep CNN OCR training overlay:

```powershell
latex-ocr-train `
  --config configs/train.yaml `
  --config configs/model/deep_cnn.yaml
```

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
