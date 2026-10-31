"""Record actual request spend. This module never blocks a call."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any


_USAGE_LOCK = threading.Lock()


def record(path: Path, *, cost: float | None = None, requests: int = 1) -> dict[str, Any]:
    """Merge one observed request into ``usage.json``.

    ``cost=None`` means the provider did not confirm a price. That is counted
    and shown later; it does not stop the next request.
    """
    if isinstance(requests, bool) or not isinstance(requests, int) or requests < 0:
        raise ValueError("requests must be a non-negative integer")
    if cost is not None and (
        isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0
    ):
        raise ValueError("cost must be a finite non-negative number or None")
    path = Path(path)
    with _USAGE_LOCK:
        from .store import _sqlite_document
        target = _sqlite_document(path)
        if target is not None and target[2] == "usage":
            store, lab_id, _, run_id, repeat_index = target
            store.append_usage(lab_id, run_id, repeat_index, {
                "requests": requests,
                "cost_usd": float(cost) if cost is not None else None,
                "confirmed_cost_usd": float(cost) if cost is not None else 0.0,
                "unknown_requests": requests if cost is None else 0,
            })
            totals = store.usage_totals(lab_id, run_id, repeat_index=repeat_index)
            return {
                "requests": int(totals.get("requests", 0)),
                "unknown_requests": int(totals.get("unknown_requests", 0)),
                "cost_usd": float(totals.get("cost_usd", 0.0)),
                "confirmed_cost_usd": float(totals.get("cost_usd", 0.0)),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        current: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except (OSError, ValueError):
                current = {}
        current["requests"] = int(current.get("requests") or 0) + requests
        if cost is None:
            current["unknown_requests"] = int(current.get("unknown_requests") or 0) + requests
        else:
            current["cost_usd"] = float(current.get("cost_usd") or 0) + float(cost)
            current["confirmed_cost_usd"] = float(current.get("confirmed_cost_usd") or 0) + float(cost)
        current.setdefault("cost_usd", float(current.get("cost_usd") or 0))
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return current
