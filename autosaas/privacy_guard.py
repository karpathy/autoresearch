from __future__ import annotations

import re
from typing import Iterable


REDACTION_TOKEN = "[REDACTED]"


def redact_text(text: str, literals: Iterable[str] | None = None, patterns: Iterable[str] | None = None) -> str:
    result = text
    if literals:
        ordered_literals = sorted({lit for lit in literals if lit}, key=len, reverse=True)
        for literal in ordered_literals:
            escaped = re.escape(literal)
            result = re.sub(escaped, REDACTION_TOKEN, result, flags=re.IGNORECASE)
    if patterns:
        for pattern in patterns:
            if not pattern:
                continue
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid redaction pattern: {pattern!r}") from exc
            result = compiled.sub(REDACTION_TOKEN, result)
    return result
