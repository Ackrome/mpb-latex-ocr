import torch

from mpb_latex_ocr.data.tokenizer import LatexTokenizer
from mpb_latex_ocr.evaluate import target_texts_from_batch


def test_target_texts_from_batch_uses_manifest_latex_not_decoded_labels():
    tokenizer = LatexTokenizer.train(["x^2+y^2=z^2"])
    target = r"\lim_{x\to 0}\frac{\sin x}{x}=1"
    labels = torch.tensor([tokenizer.encode(target)])

    decoded_target = tokenizer.decode(labels[0].tolist())
    assert decoded_target != target
    assert target_texts_from_batch({"labels": labels, "latex": [target]}) == [target]
