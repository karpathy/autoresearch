"""Where lab state lives. Never inside the project being measured."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def resolve_home() -> Path:
    """User state directory. AUTORESEARCH_HOME overrides the default ~/.autoresearch."""
    override = os.environ.get("AUTORESEARCH_HOME")
    path = Path(override) if override else Path.home() / ".autoresearch"
    return path.resolve()


def _path_key(path: Path | str) -> str:
    """Return a path identity using the current platform's case rules."""
    resolved = str(Path(path).resolve())
    # Preserve Windows keys created by older releases, which used casefold().
    # POSIX paths stay case-sensitive because normcase() is a no-op there.
    return resolved.casefold() if os.name == "nt" else os.path.normcase(resolved)


def project_id(project: Path) -> str:
    resolved = project.resolve()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", resolved.name).strip("-").lower() or "project"
    digest = hashlib.sha256(_path_key(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _roots_file(home: Path | None = None) -> Path:
    return (home or resolve_home()) / "roots.json"


def load_roots(home: Path | None = None) -> list[dict[str, Any]]:
    home = (home or resolve_home()).resolve()
    path = _roots_file(home)
    legacy: list[dict[str, Any]] = []
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        roots = payload.get("roots", []) if isinstance(payload, dict) else []
        legacy = [item for item in roots if isinstance(item, dict) and isinstance(item.get("path"), str)]
    try:
        from .sqlite_store import SQLiteStore
        store = SQLiteStore(home)
        registered = store.load_roots() if store.is_initialized() else []
    except ImportError:
        registered = []
    combined: dict[str, dict[str, Any]] = {}
    for item in [*legacy, *registered]:
        key = _path_key(item["path"])
        combined[key] = item
    return list(combined.values())


def save_roots(roots: list[dict[str, Any]], home: Path | None = None) -> None:
    home = (home or resolve_home()).resolve()
    try:
        from .sqlite_store import SQLiteStore
        store = SQLiteStore(home)
        if store.is_initialized():
            for item in roots:
                if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("path"), str):
                    store.register_root(item["id"], item["path"], {
                        key: value for key, value in item.items() if key not in {"id", "path"}
                    })
            return
    except ImportError:
        pass
    home.mkdir(parents=True, exist_ok=True)
    _roots_file(home).write_text(
        json.dumps({"schema": 1, "roots": roots}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def all_roots(home: Path | None = None) -> list[Path]:
    home = (home or resolve_home()).resolve()
    found = [home]
    for item in load_roots(home):
        path = Path(item["path"]).resolve()
        if path not in found:
            found.append(path)
    return found


def _looks_like_lab_store(path: Path) -> bool:
    return path.is_dir() and ((path / "catalog.json").is_file() or (path / "projects").is_dir()
                              or (path / "autoresearch.sqlite3").is_file())


def discover_legacy_stores(seeds: list[Path] | None = None) -> list[Path]:
    """Find `.labs` directories created before state moved to AUTORESEARCH_HOME."""
    if seeds is None:
        seeds = [Path.cwd().resolve(), Path(__file__).resolve()]
    found: list[Path] = []
    seen: set[str] = set()
    for seed in seeds:
        directory = seed if seed.is_dir() else seed.parent
        for base in [directory, *directory.parents]:
            candidate = (base / ".labs").resolve()
            key = _path_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            if _looks_like_lab_store(candidate):
                found.append(candidate)
    return found


def adopt_legacy_stores(home: Path | None = None, *, seeds: list[Path] | None = None,
                        extra_seeds: list[Path] | None = None) -> list[dict[str, Any]]:
    """Remember existing `.labs` folders in roots.json. Their files are not moved or rewritten."""
    home = (home or resolve_home()).resolve()
    # An explicit AUTORESEARCH_HOME means "use exactly this store". Only the default
    # home is allowed to pick up old `.labs` folders found next to the current work.
    if seeds is None and os.environ.get("AUTORESEARCH_HOME"):
        return []
    if seeds is None and extra_seeds:
        # Places the caller knows about (for example program.parent) count as well as cwd.
        seeds = [Path.cwd().resolve(), Path(__file__).resolve(), *[Path(item).resolve() for item in extra_seeds]]
    adopted = []
    for store in discover_legacy_stores(seeds):
        if store == home:
            continue
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", store.parent.name).strip("-").lower() or "labs"
        digest = hashlib.sha256(_path_key(store).encode("utf-8")).hexdigest()[:8]
        adopted.append(add_root(store, home=home, root_id=f"{slug}-labs-{digest}"))
    return adopted


def add_root(path: Path, *, home: Path | None = None, root_id: str | None = None) -> dict[str, Any]:
    home = (home or resolve_home()).resolve()
    target = path.resolve()
    try:
        from .sqlite_store import SQLiteStore
        store = SQLiteStore(home)
        if store.is_initialized():
            registered = store.load_roots()
            for item in registered:
                if Path(item["path"]).resolve() == target:
                    return item
            prior = next((item for item in load_roots(home)
                          if Path(item["path"]).resolve() == target), None)
            record = prior or {"id": root_id or project_id(target), "path": str(target)}
            id_collision = next((item for item in registered
                                 if item.get("id") == record["id"]
                                 and Path(item["path"]).resolve() != target), None)
            if id_collision is not None:
                raise ValueError(f"Root id is already registered for another path: {record['id']}")
            store.register_root(str(record["id"]), target, {
                key: value for key, value in record.items() if key not in {"id", "path"}
            })
            return record
    except ImportError:
        pass
    roots = load_roots(home)
    for item in roots:
        if Path(item["path"]).resolve() == target:
            return item
    record_id = root_id or project_id(target)
    id_collision = next((item for item in roots
                         if item.get("id") == record_id
                         and Path(item["path"]).resolve() != target), None)
    if id_collision is not None:
        raise ValueError(f"Root id is already registered for another path: {record_id}")
    record = {"id": record_id, "path": str(target)}
    roots.append(record)
    save_roots(roots, home)
    return record


def brief_path(project: Path, home: Path | None = None) -> Path:
    """Canonical brief file for a measured project under AUTORESEARCH_HOME."""
    return (home or resolve_home()).resolve() / "briefs" / project_id(project) / "brief.md"


def home_root_id(home: Path | None = None) -> str:
    home = (home or resolve_home()).resolve()
    return hashlib.sha256(_path_key(home).encode("utf-8")).hexdigest()[:16]


def root_id_for(storage_root: Path, home: Path | None = None) -> str:
    storage_root = storage_root.resolve()
    home = (home or resolve_home()).resolve()
    if storage_root == home:
        return home_root_id(home)
    for item in load_roots(home):
        if Path(item["path"]).resolve() == storage_root:
            return str(item.get("id") or storage_root.name)
    return hashlib.sha256(_path_key(storage_root).encode("utf-8")).hexdigest()[:16]


def resolve_root_by_id(root_id: str, home: Path | None = None) -> Path | None:
    home = (home or resolve_home()).resolve()
    if root_id == home_root_id(home):
        return home
    for item in load_roots(home):
        if str(item.get("id")) == root_id:
            return Path(item["path"]).resolve()
    return None


def lab_storage_roots(home: Path | None = None, *, program: Path | None = None) -> list[Path]:
    """Home plus registered extra roots and optional legacy program.parent/.labs."""
    home = (home or resolve_home()).resolve()
    found = list(all_roots(home))
    if program is not None:
        legacy = (program.parent / ".labs").resolve()
        if legacy not in found:
            found.append(legacy)
    return found


def storage_root_for_path(path: Path, home: Path | None = None, *, program: Path | None = None) -> Path | None:
    target = path.resolve()
    for root in lab_storage_roots(home, program=program):
        try:
            if target.is_relative_to(root.resolve()):
                return root.resolve()
        except ValueError:
            continue
    return None
