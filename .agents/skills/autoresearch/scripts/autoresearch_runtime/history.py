"""Read old experiment evidence without treating it as a current measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def _catalog_path(state_dir: Path) -> Path:
    return state_dir / "history.json"


def read_history(state_dir: Path) -> dict[str, Any]:
    from .sqlite_store import EntityNotFoundError, SQLiteStore
    store = SQLiteStore()
    if store.is_initialized():
        try:
            lab_id = store.resolve_lab_id(state_dir)
        except EntityNotFoundError:
            pass
        else:
            return store.get_document(lab_id, "history_catalog", {"schema": 1, "documents": [], "hypotheses": []})
    path = _catalog_path(state_dir)
    if not path.exists():
        return {"schema": 1, "documents": [], "hypotheses": []}
    return json.loads(path.read_text(encoding="utf-8"))
