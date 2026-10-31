"""Conservative archival for labs with no accepted baseline or active run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import resolve_lab_path
from .errors import LabError
from .paths import lab_storage_roots, resolve_home


FAILED = {"invalid", "crash", "timeout", "budget", "interrupted", "rejected"}


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lab_dirs(root: Path) -> list[Path]:
    paths = list((root / "projects").glob("*/*/lab.json")) + list(root.glob("*/lab.json"))
    return sorted({path.parent.resolve() for path in paths
                   if path.parent.resolve().is_relative_to(root.resolve())
                   and "archive" not in path.relative_to(root).parts})


def _sqlite_lab_dirs(root: Path) -> list[Path]:
    from .sqlite_store import SQLiteStore

    roots = {resolve_home(), root.resolve()}
    found: set[Path] = set()
    for home in roots:
        store = SQLiteStore(home)
        if not store.is_initialized():
            continue
        offset = 0
        while True:
            page = store.list_labs(limit=100, offset=offset)
            for item in page["items"]:
                state_dir = Path(item["state_dir"]).resolve()
                if state_dir.is_relative_to(root.resolve()):
                    found.add(state_dir)
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
    return sorted(found)


def _contains_path(value: Any, target: Path, base: Path) -> bool:
    if isinstance(value, dict):
        return any(_contains_path(item, target, base) for item in value.values())
    if isinstance(value, list):
        return any(_contains_path(item, target, base) for item in value)
    if not isinstance(value, str):
        return False
    try:
        path = Path(value)
        candidates = [path.resolve()]
        if not path.is_absolute():
            candidates.append((base / path).resolve())
        return any(candidate == target or target in candidate.parents for candidate in candidates)
    except (OSError, ValueError):
        return False


def _references(root: Path, target: Path) -> list[str]:
    found = []
    for path in root.rglob("*.json"):
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        if path.name == "catalog.json" or target in path.resolve().parents:
            continue
        value = _json(path)
        if value is not None and _contains_path(value, target, path.parent):
            found.append(str(path))
    return found


def inspect_archive_candidates(root: Path) -> dict[str, Any]:
    """Return a dry-run plan; any live run pauses all physical archival."""
    root = root.resolve()
    sqlite_labs = _sqlite_lab_dirs(root)
    if sqlite_labs:
        return {
            "blocked": "SQLite-backed labs cannot be archived or restored in this phase",
            "running": [], "eligible": [], "sqlite_labs": [str(path) for path in sqlite_labs],
        }
    dirs = _lab_dirs(root)
    running = []
    for state_dir in dirs:
        for path in (state_dir / "runs").glob("*/manifest.json"):
            manifest = _json(path) or {}
            if manifest.get("status") == "running":
                running.append(str(path))
    if running:
        return {"blocked": "A lab is still running", "running": running, "eligible": []}
    eligible = []
    for state_dir in dirs:
        state = _json(state_dir / "state.json")
        if not isinstance(state, dict):
            continue
        manifests = [_json(path) or {} for path in (state_dir / "runs").glob("*/manifest.json")]
        has_baseline = bool(state.get("baseline_id")) or any(
            item.get("id", "").startswith("baseline") and item.get("status") == "accepted"
            for item in manifests
        )
        failed = bool(state.get("setup_pending")) or (not has_baseline
                                                       and all(item.get("status") in FAILED for item in manifests))
        if not failed:
            continue
        refs = _references(root, state_dir)
        if refs:
            continue
        meta = _json(state_dir / "lab-meta.json") or {}
        config = _json(state_dir / "lab.json") or {}
        source_root = config.get("source_root")
        project_name = meta.get("project_name") or (Path(source_root).name if isinstance(source_root, str) else state_dir.parent.name)
        eligible.append({
            "path": str(state_dir), "project": project_name,
            "status": "setup_pending" if state.get("setup_pending") else "failed",
            "manifests": len(manifests),
        })
    return {"blocked": None, "running": [], "eligible": eligible}


def archive_idle_labs(root: Path | None = None, *, apply: bool = False) -> dict[str, Any]:
    """Preview legacy archive candidates. Physical archive moves are retired."""
    if apply:
        raise LabError(
            "Physical archive moves for legacy lab folders are no longer available",
            code="E_LEGACY_READ_ONLY",
            hint=(
                "Export a legacy lab with `--config <legacy-lab.json> storage export --output archive.zip`, "
                "then start fresh with `prepare` and a new AUTORESEARCH_HOME."
            ),
        )
    requested = (root or resolve_home()).resolve()
    home = resolve_home()
    registered = lab_storage_roots(home)
    if requested == home or requested in registered:
        roots = registered
        visible_roots = registered if requested == home else [requested]
    else:
        roots = [requested]
        visible_roots = [requested]
    previews = [inspect_archive_candidates(storage) for storage in roots]
    blocked = next((plan for plan in previews if plan.get("blocked")), None)
    if blocked is not None:
        running = [item for plan in previews for item in plan.get("running", [])]
        return {**blocked, "running": running, "eligible": [], "archived": []}
    combined = {"blocked": None, "running": [], "eligible": [], "archived": []}
    for plan in previews:
        combined["eligible"].extend(plan.get("eligible", []))
    combined["eligible"] = [
        item for item in combined["eligible"]
        if any(Path(item["path"]).resolve().is_relative_to(storage.resolve()) for storage in visible_roots)
    ]
    return combined
