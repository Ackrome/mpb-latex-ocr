"""Create a tiny synthetic rendered-formula dataset for smoke tests."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import random
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from mpb_latex_ocr.data.latex_normalize import normalize_latex

FORMULAS = [
    r"x^2+y^2=z^2",
    r"\frac{a}{b}+c",
    r"\int_0^1 x^2 dx",
    r"\sum_{i=1}^n i=\frac{n(n+1)}{2}",
    r"\alpha+\beta=\gamma",
    r"e^{i\pi}+1=0",
    r"\sqrt{x+1}",
    r"\lim_{x\to 0}\frac{\sin x}{x}=1",
    r"A=\pi r^2",
    r"\nabla\cdot E=\frac{\rho}{\epsilon_0}",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate toy LaTeX OCR smoke-test data.")
    parser.add_argument("--output-dir", default="data/toy")
    parser.add_argument("--num-samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    make_toy_data(
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        seed=args.seed,
        image_width=args.image_width,
        image_height=args.image_height,
        font_size=args.font_size,
        dpi=args.dpi,
    )


def make_toy_data(
    output_dir: Path,
    num_samples: int,
    seed: int,
    image_width: int = 512,
    image_height: int = 128,
    font_size: int = 32,
    dpi: int = 200,
) -> None:
    random.seed(seed)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    rows: list[dict[str, str]] = []
    for index in range(num_samples):
        formula = normalize_latex(random.choice(FORMULAS))
        split = _split_for_index(index, num_samples)
        image_name = f"formula_{index:05d}.png"
        render_formula(
            formula=formula,
            output_path=image_dir / image_name,
            image_width=image_width,
            image_height=image_height,
            font_size=font_size,
            dpi=dpi,
        )
        rows.append(
            {
                "image_path": f"images/{image_name}",
                "latex": formula,
                "split": split,
                "sample_id": str(index),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "latex", "split", "sample_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {num_samples} samples to {manifest_path}")


def render_formula(
    formula: str,
    output_path: Path,
    image_width: int = 512,
    image_height: int = 128,
    font_size: int = 32,
    dpi: int = 200,
    padding: int = 10,
) -> None:
    """Render a LaTeX-like math expression with Matplotlib mathtext.

    Mathtext covers the commands used by this toy generator without requiring a
    local TeX distribution. The target string in the manifest remains the raw
    normalized LaTeX, while the image contains the rendered formula.
    """

    rendered = _render_mathtext_to_image(formula, font_size=font_size, dpi=dpi)
    rendered.thumbnail(
        (max(1, image_width - 2 * padding), max(1, image_height - 2 * padding)),
        Image.Resampling.LANCZOS,
    )

    image = Image.new("L", (image_width, image_height), color=255)
    left = (image_width - rendered.width) // 2
    top = (image_height - rendered.height) // 2
    image.paste(rendered, (left, top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _render_mathtext_to_image(formula: str, font_size: int, dpi: int) -> Image.Image:
    math_expression = f"${formula}$"
    figure = Figure(figsize=(8, 2), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    text = axes.text(
        0.5,
        0.5,
        math_expression,
        color="black",
        fontsize=font_size,
        horizontalalignment="center",
        verticalalignment="center",
    )

    try:
        canvas.draw()
        renderer = canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer).expanded(1.08, 1.20)
        bbox_inches = bbox.transformed(figure.dpi_scale_trans.inverted())

        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=0.03,
            facecolor="white",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not render formula with matplotlib mathtext: {formula}") from exc
    finally:
        figure.clear()

    buffer.seek(0)
    return Image.open(buffer).convert("L")


def _split_for_index(index: int, total: int) -> str:
    ratio = index / max(1, total)
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


if __name__ == "__main__":
    main()
