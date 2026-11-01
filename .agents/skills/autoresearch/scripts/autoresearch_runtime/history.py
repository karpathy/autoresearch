"""Read a lab's recorded hypothesis history from SQLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_history(state_dir: Path) -> dict[str, Any]:
    from .sqlite_store import EntityNotFoundError, SQLiteStore
    empty: dict[str, Any] = {"schema": 1, "documents": [], "hypotheses": []}
    store = SQLiteStore()
    if not store.is_initialized():
        return empty
    try:
        lab_id = store.resolve_lab_id(state_dir)
    except EntityNotFoundError:
        return empty
    return store.get_document(lab_id, "history_catalog", empty)
