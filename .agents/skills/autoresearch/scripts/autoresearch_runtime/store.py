"""Reads and writes lab documents as JSON and SQLite, with SHA-256 hashing."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any





def _sqlite_document(path: Path) -> tuple[Any, Any, str, str | None, int | None] | None:
    """Resolve a virtual control-file path to its canonical SQLite row."""
    from .sqlite_store import EntityNotFoundError, SQLiteStore, StoreIntegrityError, _is_link_or_reparse

    def document_parts(candidate: Path) -> tuple[Path, str, str | None, int | None] | None:
        name = candidate.name
        if name == "lab.json":
            return candidate.parent, "config", None, None
        if name in {"state.json", "history.json", "worker.json"}:
            return candidate.parent, {"state.json": "state", "history.json": "history", "worker.json": "worker"}[name], None, None
        if name == "index.json" and candidate.parent.name == "vault":
            return candidate.parent.parent, "vault_index", None, None
        if name == "manifest.json" and candidate.parent.parent.name == "runs":
            state_dir = candidate.parent.parent.parent
            return state_dir, "manifest", candidate.parent.name, None
        if name in {"result.json", "attempts.jsonl", "cases.jsonl", "usage.json", "net-stats.json"}:
            repeat_name = candidate.parent.name
            if candidate.parent.parent.parent.name == "runs" and repeat_name.startswith("repeat_"):
                try:
                    repeat_index = int(repeat_name.removeprefix("repeat_"))
                except ValueError:
                    return None
                state_dir = candidate.parent.parent.parent.parent
                kinds = {
                    "result.json": "result", "attempts.jsonl": "attempts",
                    "cases.jsonl": "cases", "usage.json": "usage",
                    "net-stats.json": "net_stats",
                }
                return state_dir, kinds[name], candidate.parent.parent.name, repeat_index
        return None

    def reject_reparse_descendant(root: Path, candidate: Path) -> None:
        """Do not let virtual SQLite document components redirect to another row."""
        root_path = Path(os.path.abspath(root))
        candidate_path = Path(os.path.abspath(candidate))
        configured_home = Path(
            os.environ.get("AUTORESEARCH_HOME") or (Path.home() / ".autoresearch")
        ).expanduser()
        bases = [Path(os.path.abspath(configured_home)), Path(os.path.abspath(store.home))]
        base = next((item for item in bases if root_path == item or item in root_path.parents), None)
        if base is None:
            base = Path(root_path.anchor)
        try:
            relative = candidate_path.relative_to(base)
        except ValueError as exc:
            raise StoreIntegrityError("SQLite virtual document path escapes its original lab root") from exc
        current = base
        for component in relative.parts:
            current = current / component
            if os.path.lexists(current) and _is_link_or_reparse(current):
                raise StoreIntegrityError(
                    "SQLite virtual document path crosses a symlink or reparse point; redirection is blocked"
                )

    original_path = Path(os.path.abspath(Path(path).expanduser()))
    original_parts = document_parts(original_path)
    store = SQLiteStore()
    if not store.is_initialized():
        return None
    if original_parts is None:
        original_state_dir = None
        original_kind = original_run_alias = None
        original_repeat_index = None
        original_lab_id = None
    else:
        original_state_dir, original_kind, original_run_alias, original_repeat_index = original_parts
        try:
            original_lab_id = store.resolve_lab_id(original_state_dir)
        except EntityNotFoundError:
            original_lab_id = None
        if original_lab_id is not None:
            reject_reparse_descendant(original_state_dir, original_path)

    path = original_path.resolve()
    resolved_parts = document_parts(path)
    if resolved_parts is None:
        if original_lab_id is not None:
            raise StoreIntegrityError("SQLite virtual document path no longer identifies its original document")
        return None
    state_dir, kind, run_alias, repeat_index = resolved_parts
    try:
        lab_id = store.resolve_lab_id(state_dir)
    except EntityNotFoundError:
        if original_lab_id is not None:
            raise StoreIntegrityError("SQLite virtual document path resolves outside its original lab")
        return None
    if original_lab_id is None:
        raise StoreIntegrityError("SQLite virtual document path redirects into a registered lab")
    if lab_id != original_lab_id:
        raise StoreIntegrityError("SQLite virtual document path resolves outside its original lab")
    try:
        run_id = str(store.resolve_run_id(lab_id, run_alias)) if run_alias is not None else None
    except EntityNotFoundError:
        # A new run may still be a filesystem scratch target. Reparse paths
        # were rejected above, so a missing alias cannot redirect a DB write.
        return None
    original_run_id = None
    if original_run_alias is not None:
        try:
            original_run_id = str(store.resolve_run_id(original_lab_id, original_run_alias))
        except EntityNotFoundError:
            pass
    if (
        original_kind != kind
        or original_repeat_index != repeat_index
        or original_run_id != run_id
    ):
        raise StoreIntegrityError("SQLite virtual document path resolves to a different logical document")
    return store, lab_id, kind, run_id, repeat_index


def _sqlite_read(path: Path, fallback: Any = None) -> tuple[bool, Any]:
    target = _sqlite_document(path)
    if target is None:
        return False, fallback
    store, lab_id, kind, run_id, repeat_index = target
    if kind == "config":
        return True, store.load_projection(lab_id, "config")
    if kind == "state":
        return True, store.load_state(lab_id)
    if kind == "history":
        return True, store.get_document(lab_id, "history_catalog", fallback)
    if kind == "worker":
        return True, store.load_worker(lab_id) or fallback
    if kind == "vault_index":
        return True, store.get_document(lab_id, "vault_index", fallback)
    if kind == "manifest":
        return True, store.load_manifest(lab_id, run_id) or fallback
    if kind == "result":
        return True, store.load_result(lab_id, run_id, repeat_index) or fallback
    if kind == "attempts":
        records = store.load_attempts(lab_id, run_id, repeat_index=repeat_index)
        return True, sorted(records, key=lambda item: (
            item.get("created_at", ""), item.get("attempt_id", ""),
            0 if item.get("event") == "started" else 1 if item.get("event") == "finished" else 2,
        ))
    if kind == "cases":
        return True, store.load_checkpoints(lab_id, run_id, repeat_index=repeat_index)
    if kind == "usage":
        totals = store.usage_totals(lab_id, run_id, repeat_index=repeat_index)
        saved = store.get_document(lab_id, path, None)
        result = dict(saved) if isinstance(saved, dict) else {}
        if not totals.get("requests") and not result:
            return True, fallback
        # A request whose cost was never reported is unknown, not free.
        unknown_costs = totals.get("unknown_costs", 0)
        result.update({
            "requests": totals.get("requests", 0),
            "unknown_requests": max(totals.get("unknown_requests", 0), int(result.get("unknown_requests") or 0)),
            "cost_usd": None if unknown_costs else totals.get("cost_usd", 0.0),
            "confirmed_cost_usd": totals.get("cost_usd", 0.0),
        })
        if unknown_costs:
            result["unknown_costs"] = unknown_costs
        return True, result
    if kind == "net_stats":
        return True, store.get_document(lab_id, path, fallback)
    return False, fallback


def _sqlite_write(path: Path, value: Any) -> bool:
    target = _sqlite_document(path)
    if target is None:
        return False
    store, lab_id, kind, run_id, repeat_index = target
    if kind == "config":
        store.save_config(lab_id, value)
    elif kind == "state":
        store.save_state(lab_id, value)
    elif kind == "history":
        store.set_document(lab_id, "history_catalog", value)
    elif kind == "vault_index":
        store.set_document(lab_id, "vault_index", value)
    elif kind == "manifest":
        store.save_manifest(lab_id, run_id, value)
    elif kind == "result":
        store.append_result(lab_id, run_id, repeat_index, value)
    elif kind == "attempts":
        for item in value if isinstance(value, list) else []:
            if isinstance(item, dict):
                event = str(item.get("event") or "attempt")
                attempt_id = str(item.get("attempt_id") or item.get("id") or f"{event}-{repeat_index}")
                store.append_attempt(lab_id, run_id, attempt_id, event, {**item, "repeat_index": repeat_index})
    elif kind == "cases":
        for item in value if isinstance(value, list) else []:
            if isinstance(item, dict) and item.get("id"):
                store.append_checkpoint(lab_id, run_id, str(item["id"]), {**item, "repeat_index": repeat_index})
    elif kind == "usage":
        store.set_document(lab_id, path, value)
    elif kind == "net_stats":
        store.set_document(lab_id, path, value)
    return True


def _sha256(path: Path) -> str:
    target = _sqlite_document(path)
    if target is not None and target[2] == "config":
        return hashlib.sha256(target[0].load_config_bytes(target[1])).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _write_json(path: Path, value: dict[str, Any]) -> None:
    if _sqlite_write(path, value):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def _read_json(path: Path, fallback: Any = None) -> Any:
    # Evaluator output is an intentional crash-recovery scratch file until the
    # validated payload has been committed to SQLite by `_evaluate_once`.
    if Path(path).name == "result.json" and Path(path).is_file():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return fallback
    routed, value = _sqlite_read(path, fallback)
    if routed:
        return value
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
