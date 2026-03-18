import sys
from pathlib import Path

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
    assert "sk_live_123" not in redacted
