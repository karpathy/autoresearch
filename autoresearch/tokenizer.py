"""Karpathy-style character-level tokenizer.

Minimal and offline — no external downloads. Mirrors the tokenizer used in
nanoGPT's ``char`` dataset path: a sorted stable mapping from characters in
the training corpus to integer ids.
"""
from __future__ import annotations

from typing import Iterable, List


class CharTokenizer:
    """Character-level tokenizer.

    Build from a corpus of text — the vocabulary is the sorted set of distinct
    characters. Unknown characters raise ``KeyError`` on encode; decode never
    fails because ids are always in range by construction.
    """

    def __init__(self, chars: Iterable[str]):
        unique_sorted = sorted(set(chars))
        if not unique_sorted:
            raise ValueError("CharTokenizer needs at least one character")
        self.itos: List[str] = list(unique_sorted)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        stoi = self.stoi
        return [stoi[c] for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        itos = self.itos
        return "".join(itos[i] for i in ids)
