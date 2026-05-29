#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/c/Projects/mpb-latex-ocr}"
CDM_INPUT="${CDM_INPUT:-$REPO_ROOT/outputs/baseline/cdm_predictions_safe.json}"
CDM_OUTPUT="${CDM_OUTPUT:-$REPO_ROOT/outputs/baseline/official_cdm_wsl}"
CDM_REPO="${CDM_REPO:-$REPO_ROOT/outputs/tools/UniMERNet}"
CDM_POOLS="${CDM_POOLS:-4}"

if [[ ! -f "$CDM_INPUT" ]]; then
  echo "Missing CDM input: $CDM_INPUT" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  git \
  python3 \
  python3-pip \
  nodejs \
  npm \
  imagemagick \
  ghostscript \
  poppler-utils \
  texlive-latex-base \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-science \
  cm-super

sudo sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true

if [[ ! -d "$CDM_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/opendatalab/UniMERNet.git "$CDM_REPO"
fi

python3 -m pip install --user --upgrade pip
python3 -m pip install --user "numpy<2.0.0" tqdm matplotlib opencv-python "scikit-image<=0.20.0"

mkdir -p "$CDM_OUTPUT"
python3 "$CDM_REPO/cdm/evaluation.py" \
  -i "$CDM_INPUT" \
  -o "$CDM_OUTPUT" \
  -p "$CDM_POOLS"

METRICS_PATH="$CDM_OUTPUT/$(basename "$CDM_INPUT" .json)/metrics_res.json"
echo "Official CDM metrics: $METRICS_PATH"
cat "$METRICS_PATH"
