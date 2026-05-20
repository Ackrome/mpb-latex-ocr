"""Small regex tokenizer for LaTeX formula OCR baselines."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mpb_latex_ocr.data.latex_normalize import normalize_latex

TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|\\.|[A-Za-z]+|\d+(?:\.\d+)?|[{}_^=+\-*/(),\[\]|]|[^\s]"
)


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"


class LatexTokenizer:
    """Regex tokenizer with explicit BOS/EOS/PAD/UNK ids."""

    def __init__(self, token_to_id: dict[str, int], special_tokens: SpecialTokens | None = None):
        self.special_tokens = special_tokens or SpecialTokens()
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        for token in (
            self.special_tokens.pad,
            self.special_tokens.bos,
            self.special_tokens.eos,
            self.special_tokens.unk,
        ):
            if token not in self.token_to_id:
                raise ValueError(f"Missing required special token: {token}")

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens.pad]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.special_tokens.bos]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.special_tokens.eos]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special_tokens.unk]

    def __len__(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "LatexTokenizer":
        special = SpecialTokens()
        token_to_id = {
            special.pad: 0,
            special.bos: 1,
            special.eos: 2,
            special.unk: 3,
        }

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize_latex(text))

        items = [(token, count) for token, count in counts.items() if count >= min_freq]
        items.sort(key=lambda item: (-item[1], item[0]))

        if max_vocab_size is not None:
            room = max(0, max_vocab_size - len(token_to_id))
            items = items[:room]

        for token, _ in items:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id=token_to_id, special_tokens=special)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize_latex(text)]
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
            if add_special_tokens and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}

        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in special_ids:
                continue
            token = self.id_to_token.get(idx, self.special_tokens.unk)
            if skip_special_tokens and token == self.special_tokens.unk:
                continue
            tokens.append(token)

        return detokenize_latex(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens.__dict__,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LatexTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_id={str(token): int(idx) for token, idx in payload["token_to_id"].items()},
            special_tokens=SpecialTokens(**payload.get("special_tokens", {})),
        )


def tokenize_latex(text: str) -> list[str]:
    value = normalize_latex(text)
    return TOKEN_RE.findall(value)


def detokenize_latex(tokens: Iterable[str]) -> str:
    output: list[str] = []
    previous = ""

    for token in tokens:
        if previous and _needs_space(previous, token):
            output.append(" ")
        output.append(token)
        previous = token

    return normalize_latex("".join(output))


def _needs_space(previous: str, current: str) -> bool:
    return bool(re.fullmatch(r"\\[A-Za-z]+", previous) and re.match(r"[A-Za-z0-9]", current))
