from pathlib import Path

from PIL import Image

from mpb_latex_ocr.prepare_manifest import prepare_manifest


def test_prepare_manifest_from_im2latex_230k_style_files(tmp_path: Path):
    root = tmp_path / "PRINTED_TEX_230k"
    images = root / "generated_png_images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"fake")
    (images / "b.png").write_bytes(b"fake")
    (root / "corresponding_png_images.txt").write_text("a.png\nb.png\n", encoding="utf-8")
    (root / "final_png_formulas.txt").write_text(
        "\\frac { a } { b }\nx ^ 2\n",
        encoding="utf-8",
    )

    output = tmp_path / "manifest.csv"
    rows = prepare_manifest(
        input_root=root,
        output_path=output,
        input_format="auto",
        val_fraction=0.0,
        test_fraction=0.0,
    )

    text = output.read_text(encoding="utf-8")
    assert len(rows) == 2
    assert "generated_png_images/a.png" in text
    assert "\\frac{a}{b}" in text


def test_prepare_manifest_from_table_file(tmp_path: Path):
    root = tmp_path / "dataset"
    images = root / "paper"
    images.mkdir(parents=True)
    (images / "sample.jpeg").write_bytes(b"fake")
    (root / "labels.csv").write_text(
        "image_path,equation,split\npaper/sample.jpeg,x^2+y^2,test\n",
        encoding="utf-8",
    )

    output = tmp_path / "manifest.csv"
    rows = prepare_manifest(input_root=root, output_path=output, input_format="auto")

    text = output.read_text(encoding="utf-8")
    assert len(rows) == 1
    assert "paper/sample.jpeg" in text
    assert "test" in text


def test_prepare_manifest_from_mathwriting_inkml_renders_images(tmp_path: Path):
    root = tmp_path / "mathwriting"
    train = root / "train"
    valid = root / "valid"
    train.mkdir(parents=True)
    valid.mkdir(parents=True)
    (train / "a.inkml").write_text(
        """<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="normalizedLabel">x ^ 2</annotation>
  <trace id="0">0 0, 10 10, 20 0</trace>
</ink>
""",
        encoding="utf-8",
    )
    (valid / "b.inkml").write_text(
        """<ink>
  <annotation type="normalizedLabel">\\frac { a } { b }</annotation>
  <trace>0 10, 5 0, 10 10</trace>
</ink>
""",
        encoding="utf-8",
    )

    output = tmp_path / "manifest.csv"
    render_dir = tmp_path / "rendered"
    rows = prepare_manifest(
        input_root=root,
        output_path=output,
        input_format="auto",
        render_dir=render_dir,
        val_fraction=0.0,
        test_fraction=0.0,
        mathwriting_image_width=128,
        mathwriting_image_height=64,
    )

    text = output.read_text(encoding="utf-8")
    assert len(rows) == 2
    assert "x^2" in text
    assert "\\frac{a}{b}" in text
    assert rows[0].image_path.exists()
    assert rows[1].split == "val"
    with Image.open(rows[0].image_path) as image:
        assert image.size == (128, 64)
