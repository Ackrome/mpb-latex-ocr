"""Conservative LaTeX normalization for OCR targets."""

from __future__ import annotations

import re

_COMMENT_RE = re.compile(r"(?<!\\)%.*")


def strip_latex_comments(text: str) -> str:
    """Remove unescaped percent comments line by line."""

    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def normalize_latex(text: str) -> str:
    """Normalize common visual-only and whitespace differences in math LaTeX.

    This is deliberately conservative. It removes noise that should not change a
    rendered formula, but it does not try to prove mathematical equivalence.
    """

    if text is None:
        return ""

    value = str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u2212", "-")
    value = strip_latex_comments(value)

    replacements = {
        "\\displaystyle": "",
        "\\textstyle": "",
        "\\scriptstyle": "",
        "\\scriptscriptstyle": "",
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "\\left": "",
        "\\right": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*([{}_^=+\-*/(),\[\]])\s*", r"\1", value)
    value = re.sub(r"(\\[A-Za-z]+)\s+(?=[{}_^=+\-*/(),\[\]])", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value
