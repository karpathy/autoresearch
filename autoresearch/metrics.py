"""Structured metrics sidecar.

Replaces the brittle stdout-regex parsing path with a JSON file written by the
training loop at each checkpoint. The regex path (see legacy ``train.py``)
remains as a fallback but is no longer primary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Regex fallback — tolerant of minor formatting changes in the legacy summary.
_SUMMARY_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^\s]+)\s*$")


def write_metrics(path: Union[str, Path], metrics: Dict[str, Any]) -> Path:
    """Atomically write a metrics JSON sidecar.

    Uses write-then-replace so readers never see a half-written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def read_metrics(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read a metrics JSON sidecar, or return ``None`` if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def parse_stdout_summary(text: str) -> Dict[str, Any]:
    """Fallback parser for the legacy ``---``/``key: value`` stdout block.

    Only used if the JSON sidecar is absent. Returns a dict (possibly empty);
    values are coerced to float/int when possible, else kept as strings.
    """
    out: Dict[str, Any] = {}
    in_summary = False
    for line in text.splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if not in_summary:
            continue
        m = _SUMMARY_RE.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        try:
            val: Any = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                val = raw
        out[key] = val
    return out
