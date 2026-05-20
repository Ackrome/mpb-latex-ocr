from mpb_latex_ocr.metrics.edit_distance import levenshtein_distance, normalized_edit_distance


def test_levenshtein_distance():
    assert levenshtein_distance("kitten", "sitting") == 3


def test_normalized_edit_distance():
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("", "abcd") == 1.0
