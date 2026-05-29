from __future__ import annotations

import argparse
from pathlib import Path


def replace_or_skip(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def patch_latex2bbox(cdm_root: Path) -> int:
    path = cdm_root / "modules" / "latex2bbox_color.py"
    changed = 0
    changed += replace_or_skip(
        path,
        '    cmd = "magick -density 200 -quality 100 %s %s"%(pdf_filename, png_filename)\n',
        '    cmd = \'magick -density 200 -quality 100 "%s" "%s"\'%(pdf_filename, png_filename)\n',
    )
    changed += replace_or_skip(
        path,
        "    pre_name = output_path.replace('/', '_').replace('.','_') + '_' + basename\n",
        '    pre_name = re.sub(r"[^A-Za-z0-9_-]+", "_", output_path) + \'_\' + basename\n',
    )
    changed += replace_or_skip(
        path,
        '    run_cmd(f"pdflatex -interaction=nonstopmode -output-directory={temp_dir} {tex_filename} >/dev/null")\n',
        '    null_device = "NUL" if os.name == "nt" else "/dev/null"\n'
        "    run_cmd(\n"
        '        f\'pdflatex -interaction=nonstopmode -output-directory="{temp_dir}" \'\n'
        '        f\'"{tex_filename}" >{null_device}\'\n'
        "    )\n",
    )
    return changed


def patch_evaluation(cdm_root: Path) -> int:
    path = cdm_root / "evaluation.py"
    return replace_or_skip(
        path,
        "                model, inliers_1 = ransac((src[inliers==False], dst[inliers==False]), SimpleAffineTransform, min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, random_state=42)\n",
        "                try:\n"
        "                    model, inliers_1 = ransac((src[inliers==False], dst[inliers==False]), SimpleAffineTransform, min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, random_state=42)\n"
        "                except TypeError:\n"
        "                    model, inliers_1 = ransac((src[inliers==False], dst[inliers==False]), SimpleAffineTransform, min_samples=min_samples, residual_threshold=residual_threshold, max_trials=max_trials, rng=42)\n",
    )


def patch_visual_matcher(cdm_root: Path) -> int:
    path = cdm_root / "modules" / "visual_matcher.py"
    changed = 0
    changed += replace_or_skip(
        path,
        "    def estimate(self, src, dst):\n",
        "    @classmethod\n"
        "    def from_estimate(cls, src, dst):\n"
        "        model = cls()\n"
        "        if src.shape[0] == 0:\n"
        "            return None\n"
        "        success = model.estimate(src, dst)\n"
        "        if success is False:\n"
        "            return None\n"
        "        return model\n"
        "\n"
        "    def estimate(self, src, dst):\n",
    )
    changed += replace_or_skip(
        path,
        "        self.scale = np.mean(dst_dists) / (np.mean(src_dists) + 1e-10)\n",
        "        self.scale = np.mean(dst_dists) / (np.mean(src_dists) + 1e-10)\n"
        "        return np.isfinite(self.scale).all()\n",
    )
    changed += replace_or_skip(
        path,
        "        inverse_transform = AffineTransform(-self.translation, 1.0/self.scale)\n",
        "        inverse_transform = SimpleAffineTransform(-self.translation, 1.0/self.scale)\n",
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdm-root", default="outputs/tools/UniMERNet/cdm")
    args = parser.parse_args()

    cdm_root = Path(args.cdm_root).resolve()
    if not (cdm_root / "evaluation.py").exists():
        raise FileNotFoundError(f"CDM root does not look valid: {cdm_root}")

    changed = 0
    changed += patch_latex2bbox(cdm_root)
    changed += patch_evaluation(cdm_root)
    changed += patch_visual_matcher(cdm_root)
    print(f"patched {changed} UniMERNet CDM anchors under {cdm_root}")


if __name__ == "__main__":
    main()
