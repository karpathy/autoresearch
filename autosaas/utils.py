from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def utc_now_iso() -> str:
    """UTC timestamp suitable for TSV logging (no external calls, no telemetry)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Generate a local run id for correlating logs/reports."""
    return uuid4().hex

