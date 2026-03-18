from __future__ import annotations

import re
from typing import Iterable, Sequence


REDACTION_TOKEN = "[REDACTED]"


def redact_text(text: str, literals: Iterable[str] | None = None, patterns: Iterable[str] | None = None) -> str:
    spans: list[tuple[int, int]] = []
    if literals:
        for literal in literals:
            if not literal:
                continue
            regex = re.compile(re.escape(literal), re.IGNORECASE)
            spans.extend((match.start(), match.end()) for match in regex.finditer(text))
    if patterns:
        for pattern in patterns:
            if not pattern:
                continue
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid redaction pattern: {pattern!r}") from exc
            spans.extend((match.start(), match.end()) for match in regex.finditer(text))
    merged = _merge_spans(spans)
    if not merged:
        return text
    return _build_redacted_text(text, merged)


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((span for span in spans if span[0] < span[1]), key=lambda span: span[0])
    if not ordered:
        return []
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_redacted_text(text: str, spans: Sequence[tuple[int, int]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            pieces.append(text[cursor:start])
        pieces.append(REDACTION_TOKEN)
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces)
