from mpb_latex_ocr.data.latex_normalize import normalize_latex
from mpb_latex_ocr.data.tokenizer import LatexTokenizer


def test_normalize_latex_removes_visual_size_commands():
    assert normalize_latex(r"\displaystyle \dfrac { a } { b }") == r"\frac{a}{b}"


def test_tokenizer_round_trip_common_formula():
    tokenizer = LatexTokenizer.train([r"\frac{a}{b}+c"])
    ids = tokenizer.encode(r"\frac{a}{b}+c")
    assert tokenizer.decode(ids) == r"\frac{a}{b}+c"
