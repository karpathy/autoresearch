import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autosaas.privacy_guard import redact_text


def test_redact_text_replaces_sensitive_literals():
    text = "tenant acme on acme-prod.internal uses sk_live_123"
    redacted = redact_text(
        text,
        literals=["acme", "acme-prod.internal"],
        patterns=[r"sk_live_[A-Za-z0-9]+"],
    )
    assert "acme" not in redacted.lower()
    assert "prod.internal" not in redacted.lower()
    assert "sk_live_123" not in redacted
    assert redacted.count("[REDACTED]") >= 2
    assert "tenant" in redacted


def test_redact_text_invalid_pattern_raises():
    with pytest.raises(ValueError):
        redact_text("tenant", patterns=["["])


def test_redact_text_merges_overlapping_literal_spans():
    text = "abcd"
    redacted = redact_text(text, literals=["abc", "bcd"])
    assert redacted == "[REDACTED]"
