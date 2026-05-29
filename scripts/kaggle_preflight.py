"""Kaggle source-tree and import preflight checks."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REQUIRED_PATHS = [
    "src/mpb_latex_ocr/__init__.py",
    "src/mpb_latex_ocr/data/__init__.py",
    "src/mpb_latex_ocr/data/datamodule.py",
    "src/mpb_latex_ocr/data/dataset.py",
    "src/mpb_latex_ocr/data/tokenizer.py",
    "src/mpb_latex_ocr/models/lightning_module.py",
    "src/mpb_latex_ocr/prepare_manifest.py",
    "src/mpb_latex_ocr/kaggle_model2_prepare.py",
    "configs/train.yaml",
    "configs/hardware/kaggle.yaml",
    "configs/datasets/kaggle_manifest.yaml",
]


def main() -> None:
    project_dir = Path(os.environ.get("PROJECT_DIR", "/kaggle/working/mpb-latex-ocr")).resolve()
    src_dir = project_dir / "src"
    sys.path.insert(0, str(src_dir))

    print("PROJECT_DIR:", project_dir)
    print("SRC_DIR:", src_dir)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

    missing = [relative for relative in REQUIRED_PATHS if not (project_dir / relative).exists()]
    if missing:
        print("\nMissing required project files:")
        for relative in missing:
            print(" -", relative)
        print("\nCurrent mpb_latex_ocr tree, if present:")
        package_dir = src_dir / "mpb_latex_ocr"
        if package_dir.exists():
            for path in sorted(package_dir.rglob("*"))[:80]:
                print(" -", path.relative_to(project_dir))
        else:
            print(" - src/mpb_latex_ocr does not exist")
        raise SystemExit(
            "\nKaggle is using an incomplete or stale project copy. "
            "Refresh /kaggle/working/mpb-latex-ocr from the current repo, then reinstall."
        )

    package = importlib.import_module("mpb_latex_ocr")
    data = importlib.import_module("mpb_latex_ocr.data")
    prepare_manifest = importlib.import_module("mpb_latex_ocr.prepare_manifest")

    print("mpb_latex_ocr:", package.__file__)
    print("mpb_latex_ocr.data:", data.__file__)
    print("prepare_manifest:", prepare_manifest.__file__)
    print("preflight: ok")


if __name__ == "__main__":
    main()
