"""Loopback-only live dashboard. Mutations are delete, archive, and restore of one chosen lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any





def _read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback

def _merged_usage(run: dict[str, Any], usage_path: Path) -> dict[str, Any]:
    result_usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    file_usage = _read_json(usage_path, {}) or {}
    merged = dict(file_usage) if isinstance(file_usage, dict) else {}
    for key, value in result_usage.items():
        if (key in {"requests", "cost_usd", "confirmed_cost_usd", "unknown_requests", "unknown_cost_usd"}
                and isinstance(value, (int, float)) and not isinstance(value, bool)):
            existing = merged.get(key)
            merged[key] = max(value, existing) if isinstance(existing, (int, float)) and not isinstance(existing, bool) else value
        else:
            merged[key] = value
    return merged
