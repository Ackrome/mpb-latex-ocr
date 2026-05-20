from mpb_latex_ocr.metrics.render import compare_rendered_formulas, render_formula_mask


def test_render_formula_mask_has_ink():
    mask = render_formula_mask(r"\frac{a}{b}+c")
    assert mask.any()


def test_render_comparison_identical_formula_matches():
    result = compare_rendered_formulas(r"\frac{a}{b}+c", r"\frac{a}{b}+c")
    assert result.prediction_rendered
    assert result.target_rendered
    assert result.f1 == 1.0
    assert result.match


def test_render_comparison_normalizes_visual_variants():
    result = compare_rendered_formulas(r"\dfrac { a } { b }", r"\frac{a}{b}")
    assert result.f1 == 1.0
    assert result.match


def test_render_comparison_different_formula_is_lower_than_identical():
    same = compare_rendered_formulas("x", "x")
    different = compare_rendered_formulas("x", "y")
    assert different.f1 < same.f1
