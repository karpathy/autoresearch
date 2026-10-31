"""Per-case checkpoint logs for resumable evaluator runs.

The evaluator (or runtime) passes explicit ``Path`` arguments. Typical paths
come from environment variables set for each run directory:

- ``AUTORESEARCH_CASE_LOG`` — finished case records (``cases.jsonl``).
- ``AUTORESEARCH_ATTEMPTS_LOG`` — per-HTTP attempt lines (``attempts.jsonl``).
- ``AUTORESEARCH_WIRE_LOG`` — wire hook output (``sent-requests.jsonl``).
"""

from __future__ import annotations

import json
import math
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any


_CURRENT_CASE_ID: ContextVar[str | None] = ContextVar("autoresearch_case_id", default=None)
_APPEND_LOCK = threading.Lock()
_FALLBACK_LOCK = threading.Lock()
_FALLBACK_CASE_ID: str | None = None


@contextmanager
def case_scope(case_id: str):
    """Expose a stable case identifier to HTTP hooks during one evaluator case."""
    global _FALLBACK_CASE_ID
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")
    token = _CURRENT_CASE_ID.set(case_id)
    with _FALLBACK_LOCK:
        previous_fallback = _FALLBACK_CASE_ID
        _FALLBACK_CASE_ID = case_id
    try:
        yield
    finally:
        _CURRENT_CASE_ID.reset(token)
        with _FALLBACK_LOCK:
            _FALLBACK_CASE_ID = previous_fallback


def current_case_id() -> str | None:
    """Return the case id active in this execution context, if any."""
    cid = _CURRENT_CASE_ID.get()
    if cid is not None:
        return cid
    with _FALLBACK_LOCK:
        return _FALLBACK_CASE_ID


def completed_ids(path: Path) -> set[str]:
    """Return case ids that have at least one record with ``status == \"ok\"``."""
    path = Path(path)
    from .store import _sqlite_document
    target = _sqlite_document(path)
    if target is not None and target[2] == "cases":
        store, lab_id, _, run_id, repeat_index = target
        return {str(item["id"]) for item in store.load_checkpoints(lab_id, run_id, repeat_index=repeat_index)
                if item.get("status") == "ok" and isinstance(item.get("id"), str)}
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("status") != "ok":
                continue
            case_id = record.get("id")
            if isinstance(case_id, str):
                done.add(case_id)
    return done


def pending_ids(all_ids: list[str], done: set[str]) -> list[str]:
    """Return ``all_ids`` in order, omitting ids present in ``done``."""
    return [case_id for case_id in all_ids if case_id not in done]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    from .store import _sqlite_document
    target = _sqlite_document(path)
    if target is not None and target[2] in {"cases", "attempts"}:
        store, lab_id, kind, run_id, repeat_index = target
        payload = {**record, "repeat_index": repeat_index}
        if kind == "cases":
            case_id = payload.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("checkpoint records require a stable case id")
            store.append_checkpoint(lab_id, run_id, case_id, payload)
        else:
            import uuid
            attempt_id = payload.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                attempt_id = str(uuid.uuid4())
                payload["attempt_id"] = attempt_id
            event = payload.get("event")
            if not isinstance(event, str) or not event:
                event = "attempt"
                payload["event"] = event
            store.append_attempt(lab_id, run_id, attempt_id, event, payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with _APPEND_LOCK:
        separator = b""
        if path.is_file() and path.stat().st_size:
            with path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                if existing.read(1) != b"\n":
                    separator = b"\n"
        with path.open("ab") as handle:
            handle.write(separator + line)
            handle.flush()
            os.fsync(handle.fileno())


def append_case(path: Path, record: dict) -> None:
    """Append one finished-case line (``id``, ``status``, ``usage``)."""
    _append_jsonl(path, record)


def append_attempt(path: Path, record: dict) -> None:
    """Append one HTTP attempt line; does not mark a case complete."""
    _append_jsonl(path, record)


def summarize_usage(attempts_path: Path) -> dict[str, Any]:
    """Aggregate attempts, pairing durable start/finish events by attempt id."""
    attempts_path = Path(attempts_path)
    requests = 0
    unknown_requests = 0
    cost_usd = 0.0
    lifecycle: dict[str, dict[str, Any]] = {}
    from .store import _sqlite_document
    target = _sqlite_document(attempts_path)
    records: list[dict[str, Any]] = []
    if target is not None and target[2] == "attempts":
        store, lab_id, _, run_id, repeat_index = target
        records = store.load_attempts(lab_id, run_id, repeat_index=repeat_index)
    elif attempts_path.is_file():
        with attempts_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line.strip())
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    if not records:
        return {
            "requests": 0, "unknown_requests": 0, "cost_usd": 0.0,
            "confirmed_cost_usd": 0.0,
        }
    standalone: list[dict[str, Any]] = []
    for record in records:
        event = record.get("event")
        attempt_id = record.get("attempt_id")
        if event == "wire":
            continue
        if event in {"started", "finished"} and isinstance(attempt_id, str) and attempt_id:
            lifecycle.setdefault(attempt_id, {})[event] = record
        elif event != "usage":
            # Legacy SDK rows are single completed attempts; usage rows are
            # aggregates and are handled only when no lifecycle journal exists.
            standalone.append(record)
    usage_rows = [record for record in records if record.get("event") == "usage"]
    if not lifecycle:
        standalone.extend(usage_rows)
    for record in standalone:
        requests += 1
        raw_cost = record.get("cost_usd")
        if raw_cost is None:
            unknown_requests += 1
        elif (isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float))
              or not math.isfinite(raw_cost) or raw_cost < 0):
            unknown_requests += 1
        else:
            cost_usd += float(raw_cost)
    for events in lifecycle.values():
        requests += 1
        record = events.get("finished")
        raw_cost = record.get("cost_usd") if record else None
        if raw_cost is None:
            unknown_requests += 1
        elif (isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float))
              or not math.isfinite(raw_cost) or raw_cost < 0):
            unknown_requests += 1
        else:
            cost_usd += float(raw_cost)
    cost_usd = round(cost_usd, 12)
    return {
        "requests": requests,
        "unknown_requests": unknown_requests,
        "cost_usd": None if unknown_requests else cost_usd,
        "confirmed_cost_usd": cost_usd,
    }
