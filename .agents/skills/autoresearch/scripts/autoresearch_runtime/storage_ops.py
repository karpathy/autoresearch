"""Explicit SQLite storage operations that do not migrate legacy lab files."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from . import paths
from .sqlite_store import SQLiteStore


def export_lab_json(*, home: str | Path | None = None, lab: str, output: str | Path) -> dict[str, Any]:
    """Write an explicit, self-contained JSON export without changing SQLite."""
    home_path = Path(home).expanduser().resolve() if home is not None else paths.resolve_home()
    store = SQLiteStore(home_path)
    if not store.is_initialized():
        raise ValueError("SQLite storage is not initialized")
    lab_id = store.resolve_lab_id(lab)
    target = Path(output).expanduser().absolute()
    if target.exists():
        raise FileExistsError(f"Export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    config_raw = store.load_config_bytes(lab_id)
    run_items: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list_runs(lab_id, limit=100, offset=offset)
        for run_item in page["items"]:
            run_id = run_item["run_id"]
            run = store.get_run(lab_id, run_id)
            if run is None:
                raise ValueError(f"Run disappeared during export: {run_id}")
            repeats = []
            for summary in run["repeats"]:
                repeat_index = int(summary["repeat_index"])
                repeats.append({
                    "summary": summary,
                    "result": store.load_result(lab_id, run_id, repeat_index),
                    "checkpoints": store.load_checkpoints(lab_id, run_id, repeat_index=repeat_index),
                    "attempts": store.load_attempts(lab_id, run_id, repeat_index=repeat_index),
                })
            run_items.append({**run_item, "payload": run["payload"], "manifest": run["manifest"], "repeats": repeats})
        if page["next_cursor"] is None or not page["items"]:
            break
        offset += len(page["items"])

    artifact_items: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = store.list_artifacts(lab_id, limit=100, offset=offset)
        for artifact in batch:
            artifact_items.append({
                **artifact,
                "content_base64": base64.b64encode(store.load_artifact(artifact["artifact_id"])).decode("ascii"),
            })
        if len(batch) < 100:
            break
        offset += len(batch)

    export = {
        "format": "autoresearch-sqlite-export-v1",
        "lab": store.get_lab(lab_id),
        "config_bytes_base64": base64.b64encode(config_raw).decode("ascii"),
        "config": store.load_projection(lab_id, "config"),
        "state": store.load_projection(lab_id, "state"),
        "history": store.load_history(lab_id),
        "migration_binding": store.get_document(lab_id, "migration_binding"),
        "runs": run_items,
        "artifacts": artifact_items,
    }
    encoded = json.dumps(export, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"
    created = False
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(encoded)
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise
    return {"lab_id": str(lab_id), "output": str(target.resolve()),
            "run_count": len(run_items), "artifact_count": len(artifact_items),
            "size_bytes": target.stat().st_size}


def reconcile_storage(
    *, home: str | Path | None = None, lab: str | None = None, apply: bool = False,
) -> dict[str, Any]:
    """Preview or repair only SQLite statuses contradicted by terminal manifests."""
    home_path = Path(home).expanduser().resolve() if home is not None else paths.resolve_home()
    store = SQLiteStore(home_path)
    if not store.is_initialized():
        return {"applied": bool(apply), "home": str(home_path), "items": [], "repaired": 0,
                "repairable": 0, "blocked": 0}
    if lab is not None:
        lab_ids = [store.resolve_lab_id(lab)]
    else:
        lab_ids = []
        offset = 0
        while True:
            page = store.list_labs(limit=100, offset=offset)
            lab_ids.extend(item["lab_id"] for item in page["items"])
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
    items: list[dict[str, Any]] = []
    for lab_id in lab_ids:
        findings = store.reconcile_terminal_run_statuses(lab_id, apply=apply)
        item_status = ("reconciled" if apply else "repairable") if findings else "clean"
        items.append({"lab_id": str(lab_id), "status": item_status, "findings": findings})
    return {
        "applied": bool(apply),
        "home": str(home_path),
        "items": items,
        "repaired": sum(len(item["findings"]) for item in items if item["status"] == "reconciled"),
        "repairable": sum(len(item["findings"]) for item in items if item["status"] == "repairable"),
        "blocked": 0,
    }


__all__ = ["export_lab_json", "reconcile_storage"]
