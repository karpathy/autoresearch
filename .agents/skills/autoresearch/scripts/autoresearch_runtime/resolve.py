"""Pick a lab.json from the home catalog, active pointer, or explicit flags."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .catalog import resolve_lab_path
from .errors import LabError
from .paths import lab_storage_roots, project_id, resolve_home


def _load_catalog(home: Path) -> dict[str, Any]:
    path = home / "catalog.json"
    if not path.is_file():
        return {"projects": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabError(f"Catalog is unreadable: {error}") from error
    return payload if isinstance(payload, dict) else {"projects": []}


def _active_config(home: Path, project: Path) -> Path | None:
    pointer = home / ".active" / f"{project_id(project)}.json"
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    config = payload.get("config")
    if not isinstance(config, str):
        return None
    path = Path(config).resolve()
    return path if path.is_file() else None


def _storage_roots(home: Path) -> list[Path]:
    return lab_storage_roots(home)


def _labs_for_project(home: Path, project: Path) -> list[tuple[Path, dict[str, Any]]]:
    project_resolved = project.resolve()
    matches: list[tuple[Path, dict[str, Any]]] = []
    for storage in _storage_roots(home):
        catalog = _load_catalog(storage)
        for record in catalog.get("projects", []):
            if not isinstance(record, dict):
                continue
            source = record.get("source_root")
            if isinstance(source, str) and Path(source).resolve() == project_resolved:
                for lab in record.get("labs", []):
                    if not isinstance(lab, dict) or lab.get("archived"):
                        continue
                    config_dir = resolve_lab_path(storage, record, str(lab.get("path", "")))
                    config_path = config_dir / "lab.json"
                    if config_path.is_file():
                        matches.append((config_path.resolve(), lab))
    return matches


def _labs_matching_id(storage: Path, catalog: dict[str, Any], needle: str) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for record in catalog.get("projects", []):
        if not isinstance(record, dict):
            continue
        for item in record.get("labs", []):
            if not isinstance(item, dict) or item.get("archived"):
                continue
            identifiers = {str(item.get(key, "")).casefold() for key in ("id", "name", "path")}
            if needle not in identifiers and needle not in str(item.get("path", "")).casefold():
                continue
            config_dir = resolve_lab_path(storage, record, str(item.get("path", "")))
            config_path = config_dir / "lab.json"
            if config_path.is_file():
                matches.append((config_path.resolve(), item, record))
    return matches


def _find_lab_id(home: Path, lab: str) -> Path:
    needle = lab.strip().casefold()
    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for storage in _storage_roots(home):
        matches.extend(_labs_matching_id(storage, _load_catalog(storage), needle))
    if not matches:
        raise LabError(f"No lab matches --lab {lab!r}")
    if len(matches) > 1:
        lines = [f"  {path} ({item.get('id')})" for path, item, _ in matches]
        raise LabError("Several labs match --lab; pick --config:\n" + "\n".join(lines),
                       code="E_LAB_AMBIGUOUS")
    return matches[0][0]


def resolve_config_path(
    *,
    config: Path | None,
    lab: str | None,
    project: Path | None,
    labs_root: Path | None,
) -> Path:
    if config is not None:
        return config
    home = (labs_root or resolve_home()).resolve()
    if lab:
        return _find_lab_id(home, lab)
    target = (project or Path.cwd()).resolve()
    active = _active_config(home, target)
    if active is not None:
        return active
    matches = _labs_for_project(home, target)
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        lines = [f"  {path} ({meta.get('id')})" for path, meta in matches]
        raise LabError(
            f"Several labs match project {target}; use --config or --lab:\n" + "\n".join(lines),
            code="E_LAB_AMBIGUOUS",
            hint="prepare --resume or pass --config explicitly",
        )
    identity = hashlib.sha256(str(target).casefold().encode("utf-8")).hexdigest()[:10]
    legacy = home / f"{target.name}-{identity}" / "lab.json"
    if legacy.is_file():
        return legacy
    if project is None and (target / "lab.json").is_file():
        return target / "lab.json"  # a lab.json in the current directory keeps working
    message = (f"No lab for {target}; prepare a new SQLite lab first" if project is not None
               else f"No lab found for {target}")
    raise LabError(
        message,
        code="E_NO_LAB",
        hint="Run prepare with a reviewed spec and dedicated AUTORESEARCH_HOME, or pass --config <lab.json>, --lab <id>, or --project <path>",
    )


def resolve_config_ref(
    *,
    config: Path | None,
    lab: str | None,
    project: Path | None,
    labs_root: Path | None,
) -> Path | str:
    """Resolve legacy catalog IDs locally and leave SQLite refs to the runtime facade."""
    if config is not None:
        return config
    if lab:
        home = (labs_root or resolve_home()).resolve()
        selected_path = Path(lab).expanduser()
        path_like = (
            selected_path.is_absolute()
            or "/" in lab
            or "\\" in lab
            or selected_path.name.casefold() == "lab.json"
        )
        try:
            return _find_lab_id(home, lab)
        except LabError as error:
            # SQLite UUIDs, path aliases, and IDs imported from a DB are resolved by
            # config.load_config. Preserve ambiguity errors from the legacy catalog.
            if not str(error).startswith("No lab matches --lab"):
                raise
            from .sqlite_store import AmbiguousAliasError, EntityNotFoundError, SQLiteStore

            store = SQLiteStore()
            if store.is_initialized():
                try:
                    return str(store.resolve_lab_id(lab))
                except AmbiguousAliasError as ambiguous:
                    raise LabError(
                        "Several SQLite labs match --lab; pass a unique --lab or --config",
                        code="E_LAB_AMBIGUOUS",
                        hint="Use an unambiguous lab reference or a known `lab.json` path with --config.",
                    ) from ambiguous
                except EntityNotFoundError:
                    pass
            if path_like:
                return selected_path
            raise LabError(
                f"No lab matches --lab {lab!r}",
                code="E_LAB_NOT_FOUND",
                hint="Pass --config <known-lab.json>, use a valid --lab reference, or run prepare for a new lab.",
            ) from error
    return resolve_config_path(config=None, lab=None, project=project, labs_root=labs_root)
