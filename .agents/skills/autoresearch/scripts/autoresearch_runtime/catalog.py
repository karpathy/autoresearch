"""Project and lab catalog helpers for grouped, human-readable lab storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dashboard import assign_short_labels
from .paths import home_root_id, lab_storage_roots, resolve_home, root_id_for


def project_directory(root: Path, project: Path) -> Path:
    name = re.sub(r"[^a-zA-Z0-9]+", "-", project.name).strip("-").lower() or "project"
    identity = hashlib.sha256(str(project.resolve()).casefold().encode("utf-8")).hexdigest()[:10]
    return root / "projects" / f"{name}-{identity}"


def resolve_lab_path(root: Path, project: dict[str, Any], value: str) -> Path:
    """Resolve both the original project-relative catalog format and root-relative paths."""
    root = root.resolve()
    path = Path(value)
    candidates = [path] if path.is_absolute() else []
    project_id = project.get("id")
    if isinstance(project_id, str) and project_id:
        candidates.append(root / "projects" / project_id / path)
    if not path.is_absolute():
        candidates.append(root / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.exists():
            return resolved
    return candidates[0].resolve()


def _brief_goal(state_dir: Path) -> str:
    """Extract the human objective from the frozen brief when the lab label is a path/id."""
    path = state_dir / "program.snapshot.md"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ""
    in_goal = False
    for line in lines:
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            in_goal = heading.group(1).strip().casefold() in {"цель", "goal", "objective"}
            continue
        if not in_goal:
            continue
        match = re.match(r"^\s*(?:[-*]\s*)?(?:что улучшить|цель|goal|objective)\s*:\s*(.*?)\s*$",
                         line, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip().strip("`\"'")
            if value and not value.casefold().startswith(("опишите", "describe", "укажите", "specify")):
                return value
    return ""


def _machine_label(value: Any, *identifiers: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    candidate = value.strip().casefold()
    if any(isinstance(item, str) and item.strip().casefold() == candidate for item in identifiers):
        return True
    return bool(re.fullmatch(r"[0-9a-f]{10,}", candidate))


def _human_goal(state_dir: Path, metadata: dict[str, Any], existing: dict[str, Any],
                project_name: str, metric: str = "") -> str:
    brief_goal = _brief_goal(state_dir)
    if brief_goal:
        return brief_goal
    identifiers = (metadata.get("contract_id"), existing.get("id"), state_dir.name)
    for value in (metadata.get("goal"), existing.get("goal"), existing.get("name")):
        if isinstance(value, str) and value.strip() and not _machine_label(value, *identifiers):
            return value.strip()
    return f"{project_name} · {metric}" if metric else project_name


def _catalog_lab_directories(root: Path) -> list[Path]:
    paths = list((root / "projects").glob("*/*/lab.json")) + list(root.glob("*/lab.json"))
    return sorted({path.parent.resolve() for path in paths
                   if path.parent.resolve().is_relative_to(root.resolve())
                   and "archive" not in path.relative_to(root).parts})


def _running_manifests(directories: list[Path]) -> list[str]:
    running = []
    for state_dir in directories:
        for manifest_path in (state_dir / "runs").glob("*/manifest.json"):
            try:
                if json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "running":
                    running.append(str(manifest_path))
            except (OSError, ValueError):
                continue
    return running


def lab_status(state_dir: Path) -> str:
    try:
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        manifests = [json.loads(path.read_text(encoding="utf-8"))
                     for path in (state_dir / "runs").glob("*/manifest.json")]
    except (OSError, ValueError):
        return "incomplete"
    if any(item.get("status") == "running" for item in manifests):
        return "running"
    if any(item.get("status") == "incomparable" and str(item.get("id", "")).startswith("baseline")
           for item in manifests):
        return "incomparable"
    has_baseline = bool(state.get("baseline_id")) or any(
        item.get("status") == "accepted" and str(item.get("id", "")).startswith("baseline")
        for item in manifests
    )
    if state.get("setup_pending") and not has_baseline:
        return "setup_pending"
    if has_baseline:
        trials = [item for item in manifests if item.get("id", "").startswith("trial-")]
        if not trials:
            return "ready"
        latest = max(trials, key=lambda item: item.get("created_at_utc", ""))
        return str(latest.get("status", "completed"))
    if manifests:
        return str(manifests[-1].get("status", "failed"))
    return "setup_pending"


def lab_lifecycle(state_dir: Path) -> str:
    """Session phase only: running, preparation, ready, or finished. Not the last trial verdict."""
    try:
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        manifests = [json.loads(path.read_text(encoding="utf-8"))
                     for path in (state_dir / "runs").glob("*/manifest.json")]
    except (OSError, ValueError):
        return "setup_pending"
    if any(item.get("status") == "running" for item in manifests):
        return "running"
    has_baseline = bool(state.get("baseline_id")) or any(
        item.get("status") == "accepted" and str(item.get("id", "")).startswith("baseline")
        for item in manifests
    )
    if not has_baseline:
        return "setup_pending"
    trials = [item for item in manifests if str(item.get("id", "")).startswith("trial-")]
    return "completed" if trials else "ready"


def record_lab(root: Path, project: Path, state_dir: Path, *, goal: str = "",
               metric: str = "", conditions: dict[str, Any] | None = None,
               contract_id: str = "") -> dict[str, Any]:
    root = root.resolve()
    project = project.resolve()
    project_dir = project_directory(root, project)
    lab_relative = state_dir.resolve().relative_to(root).as_posix()
    metadata_path = state_dir / "lab-meta.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except ValueError:
            metadata = {}
    else:
        metadata = {}
    metadata.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    effective_contract_id = contract_id or metadata.get("contract_id", state_dir.name)
    brief_goal = _brief_goal(state_dir)
    saved_goal = goal.strip() or brief_goal or metadata.get("goal", "")
    if _machine_label(saved_goal, effective_contract_id, state_dir.name):
        saved_goal = ""
    metadata.update({
        "project_name": project.name,
        "project_path": str(project),
        "goal": saved_goal,
        "primary_metric": metric or metadata.get("primary_metric", ""),
        "conditions": conditions if conditions is not None else metadata.get("conditions", {}),
        "contract_id": effective_contract_id,
    })
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)

    catalog_path = root / "catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        catalog = {"schema": 1, "projects": [], "archived": []}
    projects = catalog.setdefault("projects", [])
    project_id = project_dir.name
    record = next((item for item in projects if item.get("id") == project_id), None)
    if record is None:
        record = {"id": project_id, "name": project.name, "source_root": str(project), "labs": []}
        projects.append(record)
    labs = record.setdefault("labs", [])
    lab = next((item for item in labs if isinstance(item, dict)
                and resolve_lab_path(root, record, str(item.get("path", ""))) == state_dir.resolve()), None)
    display_goal = _human_goal(state_dir, metadata, lab or {}, project.name,
                               metadata.get("primary_metric", ""))
    if metadata.get("goal") != display_goal:
        metadata["goal"] = display_goal
        temporary = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, metadata_path)
    values = {
        "id": metadata["contract_id"], "path": lab_relative,
        "name": display_goal, "goal": display_goal,
        "primary_metric": metadata.get("primary_metric", ""),
        "conditions": metadata.get("conditions", {}),
        "created_at_utc": metadata["created_at_utc"],
        "status": lab_status(state_dir), "archived": False,
    }
    if lab is None:
        labs.append(values)
    else:
        lab.update(values)
    catalog["schema"] = 1
    catalog.setdefault("archived", [])
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_name(catalog_path.name + ".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, catalog_path)
    return catalog


def refresh_catalog(root: Path) -> dict[str, Any]:
    """Refresh statuses and include legacy flat lab directories in the index."""
    root = root.resolve()
    catalog_path = root / "catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        catalog = {"schema": 1, "projects": [], "archived": []}
    indexed = {str(resolve_lab_path(root, project, lab["path"]))
               for project in catalog.get("projects", []) for lab in project.get("labs", [])
               if isinstance(project, dict) and isinstance(lab, dict)
               and isinstance(lab.get("path"), str)}
    configs = list((root / "projects").glob("*/*/lab.json")) + list(root.glob("*/lab.json"))
    for config_path in configs:
        state_dir = config_path.parent.resolve()
        if not state_dir.is_relative_to(root):
            continue
        if str(state_dir) in indexed or "archive" in state_dir.relative_to(root).parts:
            continue
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            project = Path(raw.get("source_root", "Project")).resolve()
        except (OSError, ValueError):
            continue
        metadata = json.loads((state_dir / "lab-meta.json").read_text(encoding="utf-8")) if (state_dir / "lab-meta.json").is_file() else {}
        existing_goal = metadata.get("goal", "")
        if _machine_label(existing_goal, metadata.get("contract_id"), state_dir.name):
            existing_goal = ""
        record_lab(root, project, state_dir, goal=_brief_goal(state_dir) or existing_goal,
                   metric=metadata.get("primary_metric", ""), contract_id=metadata.get("contract_id", state_dir.name))
        # Reload after record_lab performs its atomic update.
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        indexed.add(str(state_dir))
    for project in catalog.get("projects", []):
        for lab in project.get("labs", []):
            state_dir = resolve_lab_path(root, project, lab["path"])
            if state_dir.is_dir():
                meta = json.loads((state_dir / "lab-meta.json").read_text(encoding="utf-8")) if (state_dir / "lab-meta.json").is_file() else {}
                goal = _human_goal(state_dir, meta, lab, str(project.get("name", "Project")),
                                   str(meta.get("primary_metric") or lab.get("primary_metric") or ""))
                if goal:
                    lab["goal"] = goal
                    lab["name"] = goal
                lab["status"] = lab_status(state_dir)
    return catalog


def save_catalog(root: Path, catalog: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "catalog.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_lab_status(state_dir: Path) -> None:
    state_dir = state_dir.resolve()
    try:
        from .sqlite_store import EntityNotFoundError, SQLiteStore
        store = SQLiteStore()
        if store.is_initialized():
            try:
                store.resolve_lab_id(state_dir)
            except EntityNotFoundError:
                pass
            else:
                # SQLite owns status for registered labs; a nearby legacy catalog
                # must not become a second writable source of truth.
                return
    except ImportError:
        pass
    projects_dir = next((parent for parent in state_dir.parents if parent.name == "projects"), None)
    root = projects_dir.parent if projects_dir is not None else state_dir.parent.resolve()
    path = root / "catalog.json"
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    changed = False
    for project in catalog.get("projects", []):
        for lab in project.get("labs", []):
            if resolve_lab_path(root, project, str(lab.get("path", ""))) == state_dir:
                status = lab_status(state_dir)
                if lab.get("status") != status:
                    lab["status"] = status
                    changed = True
    if changed:
        save_catalog(root, catalog)


def repair_catalog(root: Path, *, apply: bool = False) -> dict[str, Any]:
    """Canonicalize catalog references without moving or rewriting lab data."""
    if apply:
        from .errors import LabError

        raise LabError(
            "Legacy catalog metadata is read-only; export legacy labs and start new SQLite labs",
            code="E_LEGACY_READ_ONLY",
            hint=("Use `storage export --config <legacy lab.json> --output <archive.zip>`, "
                  "then create a fresh lab with `prepare` under a dedicated AUTORESEARCH_HOME."),
        )
    root = root.resolve()
    active_dirs = _catalog_lab_directories(root)
    running = _running_manifests(active_dirs)
    if running:
        return {"applied": False, "blocked": "A lab is still running", "running": running,
                "active_count": len(active_dirs), "archived_count": 0, "changes": []}
    catalog_path = root / "catalog.json"
    try:
        original = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        original = {"schema": 1, "projects": [], "archived": []}
    catalog = json.loads(json.dumps(original))
    catalog.setdefault("projects", [])
    catalog.setdefault("archived", [])

    archive_dirs = sorted({path.parent.resolve() for path in (root / "archive").glob("*/*/lab.json")
                           if path.parent.resolve().is_relative_to(root.resolve())})
    changes: list[dict[str, str]] = []
    active_by_dir = {str(path): path for path in active_dirs}
    archived_by_dir = {str(path): path for path in archive_dirs}
    archived_names = {(path.parent.name, path.name): path for path in archive_dirs}
    projects = catalog["projects"] if isinstance(catalog["projects"], list) else []
    # Remove active rows that already point into archive or are duplicates; preserve metadata on matches.
    seen: set[str] = set()
    for project in projects:
        if not isinstance(project, dict):
            continue
        kept = []
        for lab in project.get("labs", []) if isinstance(project.get("labs", []), list) else []:
            if not isinstance(lab, dict) or not isinstance(lab.get("path"), str):
                continue
            resolved = resolve_lab_path(root, project, lab["path"])
            key = str(resolved)
            archived_match = archived_by_dir.get(key)
            archive_group = archived_names.get((str(project.get("id", "")), str(lab.get("id", ""))))
            if archived_match is None:
                archive_group = archive_group or archived_names.get((str(project.get("id", "")), Path(lab["path"]).name))
                archived_match = archive_group
            if archived_match is not None:
                changes.append({"action": "remove_archived_active_entry", "path": lab["path"]})
                continue
            if key not in active_by_dir or key in seen:
                if key in seen:
                    changes.append({"action": "deduplicate", "path": lab["path"]})
                continue
            seen.add(key)
            canonical = resolved.relative_to(root).as_posix()
            if lab["path"] != canonical:
                changes.append({"action": "canonicalize", "from": lab["path"], "to": canonical})
            lab["path"] = canonical
            meta_path = resolved / "lab-meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            goal = _human_goal(resolved, meta, lab, str(project.get("name", "Project")),
                               str(meta.get("primary_metric") or lab.get("primary_metric") or ""))
            if goal:
                if lab.get("goal") != goal or lab.get("name") != goal:
                    changes.append({"action": "restore_goal", "path": canonical, "goal": goal})
                lab["goal"] = goal
                lab["name"] = goal
            lab["status"] = lab_lifecycle(resolved)
            kept.append(lab)
        project["labs"] = kept

    # Add any physical active lab missed by the catalog using its own frozen metadata.
    for directory in active_dirs:
        key = str(directory)
        if key in seen:
            continue
        config_path = directory / "lab.json"
        meta_path = directory / "lab-meta.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        source = Path(str(config.get("source_root", "Project")))
        project_id = directory.parent.name if directory.parent.parent == root / "projects" else directory.parent.name
        project = next((item for item in projects if isinstance(item, dict) and item.get("id") == project_id), None)
        if project is None:
            project = {"id": project_id, "name": meta.get("project_name") or source.name,
                       "source_root": str(source), "labs": []}
            projects.append(project)
        goal = _human_goal(directory, meta, {}, str(meta.get("project_name") or source.name),
                           str(meta.get("primary_metric", "")))
        project.setdefault("labs", []).append({
            "id": meta.get("contract_id", directory.name), "path": directory.relative_to(root).as_posix(),
            "name": goal, "goal": goal, "primary_metric": meta.get("primary_metric", ""),
            "conditions": meta.get("conditions", {}), "created_at_utc": meta.get("created_at_utc", ""),
            "status": lab_status(directory), "archived": False,
        })
        changes.append({"action": "add_missing_active_entry", "path": directory.relative_to(root).as_posix()})

    archived_records = catalog["archived"] if isinstance(catalog["archived"], list) else []
    archive_index = {}
    for record in archived_records:
        if not isinstance(record, dict):
            continue
        value = record.get("archived_path", record.get("path", ""))
        if value:
            archive_index.setdefault(str((root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()), record)
    for directory in archive_dirs:
        key = str(directory)
        if key not in archive_index:
            meta_path = directory / "lab-meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            project_name = meta.get("project_name", directory.parent.name)
            goal = _human_goal(directory, meta, {}, str(project_name), str(meta.get("primary_metric", "")))
            archived_records.append({"path": directory.relative_to(root).as_posix(),
                                     "archived_path": directory.relative_to(root).as_posix(),
                                     "project": project_name, "name": goal, "goal": goal,
                                     "status": "archived"})
            archive_index[key] = archived_records[-1]
            changes.append({"action": "add_missing_archive_entry", "path": directory.relative_to(root).as_posix()})
        else:
            record = archive_index[key]
            record["archived_path"] = directory.relative_to(root).as_posix()
            record.pop("path", None)
            meta_path = directory / "lab-meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            project_name = str(meta.get("project_name") or record.get("project") or directory.parent.name)
            goal = _human_goal(directory, meta, record, project_name, str(meta.get("primary_metric", "")))
            if goal and (record.get("name") != goal or record.get("goal") != goal):
                changes.append({"action": "restore_archive_goal", "path": directory.relative_to(root).as_posix(),
                                "goal": goal})
                record["name"] = goal
                record["goal"] = goal
    catalog["projects"] = projects
    catalog["archived"] = archived_records
    active_count = sum(len(project.get("labs", [])) for project in projects if isinstance(project, dict))
    archived_count = len(archive_index)
    return {"applied": False, "blocked": None, "active_count": active_count,
            "archived_count": archived_count, "changes": changes}


def list_catalogs(home: Path | None = None, *, program: Path | None = None) -> dict[str, Any]:
    """Refresh and return every per-root catalog visible from AUTORESEARCH_HOME."""
    home = (home or resolve_home()).resolve()
    catalogs = []
    for storage in lab_storage_roots(home, program=program):
        catalogs.append({
            "root_id": root_id_for(storage, home),
            "path": str(storage),
            "catalog": refresh_catalog(storage),
        })
    return {"home": str(home), "roots": catalogs}


def aggregate_dashboard_status(home: Path, *, program: Path | None = None) -> dict[str, Any]:
    """Merge dashboard summaries from every registered lab storage root."""
    import math
    import time

    from . import dash_model

    home = home.resolve()
    projects_by_id: dict[str, dict[str, Any]] = {}
    archived: list[dict[str, Any]] = []
    for storage in lab_storage_roots(home, program=program):
        partial = dash_model._compute_dashboard_status(storage)
        root_id = root_id_for(storage, home)
        for project in partial.get("projects", []):
            if not isinstance(project, dict):
                continue
            project_id = str(project.get("id", ""))
            bucket = projects_by_id.setdefault(project_id, {
                "id": project_id,
                "name": project.get("name", "Project"),
                "labs": [],
            })
            for lab in project.get("labs", []):
                if isinstance(lab, dict):
                    tagged = dict(lab)
                    tagged["root_id"] = root_id
                    bucket["labs"].append(tagged)
        for item in partial.get("archived", []):
            if isinstance(item, dict):
                tagged = dict(item)
                tagged["root_id"] = root_id
                archived.append(tagged)
    projects = list(projects_by_id.values())
    for project in projects:
        assign_short_labels(project.get("labs", []))
    assign_short_labels(archived)
    archived_totals = {
        "requests": sum(int(item.get("requests", 0) or 0) for item in archived),
        "cost_usd": round(math.fsum(float(item.get("cost_usd", 0) or 0) for item in archived), 10),
        "unknown_requests": sum(int(item.get("unknown_requests", 0) or 0) for item in archived),
        "unknown_cost_usd": round(math.fsum(float(item.get("unknown_cost_usd", 0) or 0) for item in archived), 10),
    }
    payload = {"projects": projects, "archived": archived, "archived_totals": archived_totals,
                 "updated_at": time.time(), "home_root_id": home_root_id(home)}
    payload["revision"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return payload
