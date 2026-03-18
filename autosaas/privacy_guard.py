from __future__ import annotations

import re
from typing import Iterable


REDACTION_TOKEN = "[REDACTED]"


def redact_text(text: str, literals: Iterable[str] | None = None, patterns: Iterable[str] | None = None) -> str:
    result = text
    if literals:
        for literal in literals:
            if not literal:
                continue
            escaped = re.escape(literal)
            result = re.sub(escaped, REDACTION_TOKEN, result, flags=re.IGNORECASE)
    if patterns:
        for pattern in patterns:
            result = re.sub(pattern, REDACTION_TOKEN, result)
    return result
