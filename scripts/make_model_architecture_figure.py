"""Generate the MPB LaTeX OCR architecture figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the model architecture figure.")
    parser.add_argument("--config", default="outputs/resolved_config.json")
    parser.add_argument("--tokenizer", default="outputs/tokenizer.json")
    parser.add_argument("--output", default="docs/assets/model_architecture.png")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = json.loads(Path(args.tokenizer).read_text(encoding="utf-8"))
    vocab_size = len(tokenizer["token_to_id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    make_figure(config=config, vocab_size=vocab_size, output=output)
    print(f"Wrote {output}")


def make_figure(config: dict, vocab_size: int, output: Path) -> None:
    data = config["data"]
    model = config["model"]
    image_h = int(data["image_height"])
    image_w = int(data["image_width"])
    d_model = int(model["d_model"])
    max_seq_len = int(model["max_seq_len"])
    decoder_layers = int(model["decoder_layers"])
    nhead = int(model["nhead"])
    dim_feedforward = int(model["dim_feedforward"])
    dropout = float(model["dropout"])
    channels = [int(value) for value in model["encoder_channels"]]

    encoder_shapes = [
        f"B x 1 x {image_h} x {image_w}",
        f"B x {channels[0]} x {image_h // 2} x {image_w // 2}",
        f"B x {channels[1]} x {image_h // 4} x {image_w // 4}",
        f"B x {channels[2]} x {image_h // 8} x {image_w // 8}",
        f"B x {d_model} x {image_h // 8} x {image_w // 8}",
        f"B x {(image_h // 8) * (image_w // 8)} x {d_model}",
    ]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    title = "MPB LaTeX OCR Baseline: CNN Visual Encoder + Autoregressive Transformer Decoder"
    ax.text(8, 7.55, title, ha="center", va="center", fontsize=17, fontweight="bold")

    encoder_y = 5.4
    decoder_y = 2.15
    box_h = 1.05

    x_positions = [0.45, 2.5, 4.55, 6.6, 8.65, 10.7]
    labels = [
        ("Input crop", encoder_shapes[0]),
        ("Conv 3x3 s=2\nBN + GELU", encoder_shapes[1]),
        ("Conv 3x3 s=2\nBN + GELU", encoder_shapes[2]),
        ("Conv 3x3 s=2\nBN + GELU", encoder_shapes[3]),
        (f"Conv 3x3 s=1\nBN + GELU + Dropout2d({dropout:g})", encoder_shapes[4]),
        ("Flatten HW\nLinear 256->256", encoder_shapes[5]),
    ]

    for x, (header, shape) in zip(x_positions, labels, strict=True):
        block(ax, x, encoder_y, 1.55, box_h, header, shape, face="#E8F2FF", edge="#2B6CB0")

    for left, right in zip(x_positions[:-1], x_positions[1:], strict=True):
        arrow(ax, left + 1.55, encoder_y + box_h / 2, right, encoder_y + box_h / 2, color="#2B6CB0")

    ax.text(6.25, 6.75, "Visual encoder", ha="center", fontsize=13, fontweight="bold", color="#1A365D")

    token_x = 0.55
    block(
        ax,
        token_x,
        decoder_y,
        2.1,
        box_h,
        "Target prefix tokens",
        f"B x T, T <= {max_seq_len}\nBOS + previous LaTeX tokens",
        face="#FFF7E6",
        edge="#C05621",
    )
    block(
        ax,
        3.25,
        decoder_y,
        2.2,
        box_h,
        "Token + position embedding",
        f"vocab={vocab_size}, d={d_model}\nlearned absolute positions",
        face="#FFF7E6",
        edge="#C05621",
    )
    block(
        ax,
        6.0,
        decoder_y,
        3.0,
        box_h,
        f"{decoder_layers}x Transformer decoder layer",
        f"causal self-attn, cross-attn to visual tokens\n{nhead} heads, FFN={dim_feedforward}, GELU, dropout={dropout:g}",
        face="#F0FFF4",
        edge="#2F855A",
    )
    block(
        ax,
        9.75,
        decoder_y,
        2.05,
        box_h,
        "LayerNorm + Linear",
        f"d={d_model} -> vocab={vocab_size}",
        face="#F0FFF4",
        edge="#2F855A",
    )
    block(
        ax,
        12.45,
        decoder_y,
        2.75,
        box_h,
        "LaTeX token logits",
        "training: teacher-forced CE loss\ninference: greedy next-token decoding",
        face="#F7FAFC",
        edge="#4A5568",
    )

    arrow(ax, token_x + 2.1, decoder_y + box_h / 2, 3.25, decoder_y + box_h / 2, color="#C05621")
    arrow(ax, 5.45, decoder_y + box_h / 2, 6.0, decoder_y + box_h / 2, color="#2F855A")
    arrow(ax, 9.0, decoder_y + box_h / 2, 9.75, decoder_y + box_h / 2, color="#2F855A")
    arrow(ax, 11.8, decoder_y + box_h / 2, 12.45, decoder_y + box_h / 2, color="#4A5568")

    # Cross-attention path from visual memory to decoder.
    ax.add_patch(
        FancyArrowPatch(
            (11.5, encoder_y),
            (7.5, decoder_y + box_h),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color="#805AD5",
            connectionstyle="arc3,rad=-0.25",
        )
    )
    ax.text(
        11.95,
        4.25,
        "cross-attention memory\n1024 visual tokens",
        fontsize=10,
        color="#553C9A",
        ha="left",
    )

    footer = (
        "Images are grayscale, aspect-preserving resized and white-padded to "
        f"{image_h}x{image_w}, then normalized to [-1, 1]."
    )
    ax.text(8, 0.85, footer, ha="center", va="center", fontsize=10, color="#2D3748")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def block(ax, x: float, y: float, w: float, h: float, header: str, body: str, face: str, edge: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.055",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, header, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center", fontsize=8.2)


def arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2.0,
            color=color,
        )
    )


if __name__ == "__main__":
    main()
