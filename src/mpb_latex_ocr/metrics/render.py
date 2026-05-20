"""Render-aware formula metrics.

This module provides a practical pixel-level proxy for render-aware evaluation.
It is not an official implementation of CDM. Official CDM computes spatial
character matching from specially rendered formulas; this proxy renders both
strings and compares their binary ink masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from mpb_latex_ocr.data.latex_normalize import normalize_latex


@dataclass(frozen=True)
class RenderComparison:
    iou: float
    f1: float
    match: bool
    prediction_rendered: bool
    target_rendered: bool
    error: str | None = None


def compare_rendered_formulas(
    prediction: str,
    target: str,
    font_size: int = 32,
    dpi: int = 200,
    threshold: int = 245,
    match_threshold: float = 0.98,
) -> RenderComparison:
    """Compare two LaTeX formulas by rendering and matching ink masks."""

    try:
        prediction_mask = render_formula_mask(
            prediction,
            font_size=font_size,
            dpi=dpi,
            threshold=threshold,
        )
    except Exception as exc:
        return RenderComparison(
            iou=0.0,
            f1=0.0,
            match=False,
            prediction_rendered=False,
            target_rendered=True,
            error=f"prediction render failed: {exc}",
        )

    try:
        target_mask = render_formula_mask(
            target,
            font_size=font_size,
            dpi=dpi,
            threshold=threshold,
        )
    except Exception as exc:
        return RenderComparison(
            iou=0.0,
            f1=0.0,
            match=False,
            prediction_rendered=True,
            target_rendered=False,
            error=f"target render failed: {exc}",
        )

    prediction_canvas, target_canvas = align_masks(prediction_mask, target_mask)
    intersection = np.logical_and(prediction_canvas, target_canvas).sum()
    prediction_area = prediction_canvas.sum()
    target_area = target_canvas.sum()
    union = np.logical_or(prediction_canvas, target_canvas).sum()

    if union == 0:
        iou = 1.0
        f1 = 1.0
    else:
        iou = float(intersection / union)
        denominator = prediction_area + target_area
        f1 = float(1.0 if denominator == 0 else (2.0 * intersection) / denominator)

    return RenderComparison(
        iou=iou,
        f1=f1,
        match=f1 >= match_threshold,
        prediction_rendered=True,
        target_rendered=True,
        error=None,
    )


def render_formula_mask(
    formula: str,
    font_size: int = 32,
    dpi: int = 200,
    threshold: int = 245,
) -> np.ndarray:
    image = render_formula_image(formula, font_size=font_size, dpi=dpi)
    array = np.asarray(image.convert("L"))
    mask = array < threshold
    return crop_mask(mask)


def render_formula_image(formula: str, font_size: int = 32, dpi: int = 200) -> Image.Image:
    expression = _to_mathtext_expression(normalize_latex(formula))
    figure = Figure(figsize=(8, 2), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    text = axes.text(
        0.5,
        0.5,
        expression,
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
    finally:
        figure.clear()

    buffer.seek(0)
    return Image.open(buffer).convert("L")


def crop_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1), dtype=bool)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def align_masks(left: np.ndarray, right: np.ndarray, padding: int = 4) -> tuple[np.ndarray, np.ndarray]:
    height = int(max(left.shape[0], right.shape[0]) + 2 * padding)
    width = int(max(left.shape[1], right.shape[1]) + 2 * padding)
    left_canvas = np.zeros((height, width), dtype=bool)
    right_canvas = np.zeros((height, width), dtype=bool)
    _paste_center(left_canvas, left)
    _paste_center(right_canvas, right)
    return left_canvas, right_canvas


def _paste_center(canvas: np.ndarray, mask: np.ndarray) -> None:
    top = (canvas.shape[0] - mask.shape[0]) // 2
    left = (canvas.shape[1] - mask.shape[1]) // 2
    canvas[top : top + mask.shape[0], left : left + mask.shape[1]] = mask


def _to_mathtext_expression(formula: str) -> str:
    value = str(formula).strip()
    if value.startswith("$") and value.endswith("$") and len(value) >= 2:
        value = value[1:-1].strip()
    return f"${value}$"
