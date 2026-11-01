"""Captures the source for a lab snapshot and initializes the lab project."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


from .errors import LabError
from .store import _sha256, _write_json


def _discover_source(root: Path, exclude_globs: tuple[str, ...] = ()) -> tuple[str, ...]:
    from .config import SERVICE_PARTS, _matches_exclude_glob, _relative_file, _service_file
    from .gitutil import source_git
    try:
        source_git(root, "rev-parse", "--show-toplevel")
        found = source_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", binary=True)
        names = [os.fsdecode(item) for item in found.split(b"\0") if item]
    except LabError:
        names = []
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name.lower() not in SERVICE_PARTS
                       and not (Path(directory) / name).is_symlink()]
            names.extend((Path(directory) / name).relative_to(root).as_posix() for name in files)
    selected = set()
    for name in names:
        relative = _relative_file(name)
        file = root / relative
        if _service_file(relative) or _matches_exclude_glob(relative, exclude_globs):
            continue
        if file.is_symlink() or not file.is_file():
            continue
        if not file.resolve().is_relative_to(root):
            continue
        selected.add(relative)
    return tuple(sorted(selected))


def _included_untracked(root: Path, selected: tuple[str, ...]) -> tuple[str, ...]:
    from .config import _relative_file
    from .gitutil import source_git

    selected_set = set(selected)
    try:
        raw = source_git(root, "ls-files", "--others", "--exclude-standard", "-z", binary=True)
    except LabError:
        return ()
    included: set[str] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            relative = _relative_file(os.fsdecode(item).replace("\\", "/"))
        except LabError:
            continue
        if relative in selected_set:
            included.add(relative)
    return tuple(sorted(included))

def _source_meta(project: Path) -> dict[str, Any]:
    """Record where the snapshot came from. Missing git metadata does not block a lab."""
    from .gitutil import source_git
    empty = {
        "source_head": None, "source_branch": None, "source_toplevel": None,
        "project_prefix": "", "source_dirty": False,
    }
    try:
        toplevel = Path(str(source_git(project, "rev-parse", "--show-toplevel")))
        # Git's own prefix needs no commit, so a repository with an unborn HEAD still records where the project sits.
        prefix = str(source_git(project, "rev-parse", "--show-prefix")).strip("/")
        dirty = bool(str(source_git(project, "status", "--porcelain")).strip())
    except (LabError, ValueError):
        return empty
    head = branch = None
    try:
        head = source_git(project, "rev-parse", "HEAD")
        branch = source_git(project, "rev-parse", "--abbrev-ref", "HEAD")
    except LabError:
        pass
    return {
        "source_head": head, "source_branch": branch, "source_toplevel": str(toplevel),
        "project_prefix": prefix, "source_dirty": dirty,
    }


def init_lab(config: LabConfig) -> dict[str, Any]:
    from .config import LabConfig
    from .gitutil import _git
    from .vault import _store_vault
    existing = {item.name for item in config.state_dir.iterdir()} if config.state_dir.exists() else set()
    existing.discard(".lock")
    existing.discard(".lock.guard")
    allowed = {"lab.json", "history.json", "inputs", "program.snapshot.md"} if config.pending and config.path.parent == config.state_dir else set()
    if existing - allowed:
        raise LabError("State directory is not empty")
    candidate = config.state_dir / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    for name in config.source_files:
        destination = candidate / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.source_root / name, destination)
    _git(candidate, "init", "-q")
    # Keep snapshot bytes identical to the source tree. Windows autocrlf would
    # rewrite text and make a later historical replay fail its hash check.
    _git(candidate, "config", "core.autocrlf", "false")
    _git(candidate, "config", "core.safecrlf", "false")
    _git(candidate, "add", "-A", "-f", "--", ".")
    _git(candidate, "-c", "user.name=AutoResearch", "-c", "user.email=lab@local", "commit", "-q", "-m", "baseline snapshot")
    meta = _source_meta(config.source_root)
    if not isinstance(config.config_bytes, bytes):
        raise LabError("SQLite lab requires the exact validated config bytes")
    config_hash = hashlib.sha256(config.config_bytes).hexdigest()
    config_copy_sha256 = config_hash
    state = {
        "schema": 2,
        "setup_pending": config.pending,
        "source_root": str(config.source_root),
        **meta,
        "included_untracked": list(_included_untracked(config.source_root, config.source_files)),
        "config_sha256": config_hash,
        "config_copy_sha256": config_copy_sha256,
        "source": [{"path": name, "sha256": _sha256(config.source_root / name)} for name in config.source_files],
        "inputs": [{"path": str(item), "sha256": _sha256(item)} for item in config.input_files],
        "best_commit": _git(candidate, "rev-parse", "HEAD"),
        "best_id": None,
        "baseline_id": None,
        "epochs": [],
        "current_epoch": 1,
    }
    if config.baseline_validation is not None:
        state["baseline_verified"] = False
    state_path = config.state_dir / "state.json"
    from .sqlite_store import SQLiteStore
    try:
        raw_config = json.loads(config.config_bytes.decode("utf-8"))
    except (AttributeError, UnicodeError, ValueError) as error:
        raise LabError("SQLite lab config bytes are invalid") from error
    passport = config.passport or {}
    create_config = {**raw_config, "confirmed": True, "passport": passport,
                     "pins": passport.get("pins", [])}
    SQLiteStore().create_lab(
        create_config, state, candidate,
        source_config_bytes=config.config_bytes,
    )
    _write_json(state_path, state)
    _store_vault(config)
    return state
