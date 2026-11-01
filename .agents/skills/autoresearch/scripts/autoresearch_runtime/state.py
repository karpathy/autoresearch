"""Loads lab state and verifies frozen files, passport pins, and manifest stamps."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


from .errors import LabError
from .store import _sha256, _write_json


def load_state(config: LabConfig) -> dict[str, Any]:
    from .config import LabConfig
    from .measurement import migrate_state
    try:
        from .sqlite_store import SQLiteStore
        raw_state = SQLiteStore().load_state(config.lab_id)
        state = migrate_state(raw_state)
    except (OSError, ValueError) as error:
        raise LabError(f"Lab is not initialized: {error}") from error
    return state

def _runs(config: LabConfig) -> list[dict[str, Any]]:
    from .config import LabConfig
    from .measurement import migrate_manifest

    from .sqlite_store import SQLiteStore
    store = SQLiteStore()
    manifests = []
    cursor = None
    while True:
        page = store.list_runs(config.lab_id, limit=100, cursor=cursor)
        manifests.extend(store.load_manifest(config.lab_id, item["run_id"]) for item in page["items"])
        cursor = page.get("next_cursor")
        if cursor is None:
            break
    return sorted((migrate_manifest(item) for item in manifests if item),
                  key=lambda item: str(item.get("id", "")))

def _check_external_frozen(config: LabConfig, state: dict[str, Any]) -> None:
    from .config import LabConfig
    if _sha256(config.path) != state["config_sha256"]:
        raise LabError("Configuration changed after initialization")
    for item in state["inputs"]:
        file = Path(item["path"])
        if not file.is_file() or _sha256(file) != item["sha256"]:
            raise LabError(f"Input changed: {file}", code="input_tampered")

def _finalize_setup(config: LabConfig, state: dict[str, Any]) -> None:
    from .config import LabConfig
    from .gitutil import _git
    from .vault import _store_vault
    if not state.get("setup_pending"):
        return
    if state.get("source_root") != str(config.source_root):
        raise LabError("Project path changed after setup; create a new lab")
    candidate = config.state_dir / "candidate"
    for item in state.get("inputs", []):
        file = Path(item["path"])
        if not file.is_file() or _sha256(file) != item["sha256"]:
            raise LabError(f"Input changed: {file}", code="input_tampered")
    _check_candidate_unchanged(config, {item["path"]: item["sha256"] for item in state["source"]})
    if _git(candidate, "status", "--porcelain"):
        raise LabError("Baseline candidate changed after setup")
    state["inputs"] = [{"path": str(item), "sha256": _sha256(item)} for item in config.input_files]
    state["config_sha256"] = _sha256(config.path)
    _write_json(config.state_dir / "state.json", state)
    _store_vault(config)

_IGNORED_CACHE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox"}


def _check_frozen(config: LabConfig, state: dict[str, Any]) -> None:
    from .config import LabConfig, _service_file
    from .gitutil import _git
    import logging
    _check_external_frozen(config, state)
    candidate = config.state_dir / "candidate"
    if _git(candidate, "rev-parse", "HEAD") != state["best_commit"]:
        raise LabError("Candidate Git HEAD changed outside the lab")
    entries = [file for file in candidate.rglob("*") if not any(part in _IGNORED_CACHE_DIRS for part in file.relative_to(candidate).parts)]
    if any(file.is_symlink() for file in entries):
        raise LabError("Candidate symlink is not allowed")
    protected_hashes = {item["sha256"] for item in state["inputs"]}
    protected_copies = [file for file in entries if file.is_file() and _sha256(file) in protected_hashes]
    if protected_copies:
        for file in protected_copies:
            try:
                file.unlink(missing_ok=True)
            except OSError:
                pass
        raise LabError("Candidate contains a copy of a protected input")
    current = {file.relative_to(candidate).as_posix() for file in entries if file.is_file()}
    if config.editable_files is None and any(_service_file(name) for name in current):
        raise LabError("Candidate contains a service or secret file")
    if config.editable_files is not None and current != set(config.source_files):
        raise LabError("Candidate file set changed outside the declared snapshot")
    for item in state["source"]:
        if config.editable_files is not None and item["path"] not in config.editable_files:
            file = candidate / item["path"]
            if _sha256(file) != item["sha256"]:
                raise LabError(f"Protected file changed: {item['path']}")

def _fingerprints(config: LabConfig, state: dict[str, Any]) -> dict[str, Any]:
    from .config import LabConfig
    candidate = config.state_dir / "candidate"
    names = {file.relative_to(candidate).as_posix() for file in candidate.rglob("*")
             if file.is_file() and not any(part in _IGNORED_CACHE_DIRS for part in file.relative_to(candidate).parts)}
    fingerprints = {
        "config_sha256": state["config_sha256"],
        "input_hashes": {item["path"]: item["sha256"] for item in state["inputs"]},
        "candidate_hashes": {
            name: _sha256(candidate / name) if not (candidate / name).is_symlink() else None
            for name in sorted(names)
        },
    }
    pin_sha256 = _passport_pins_sha256(config)
    if pin_sha256 is not None:
        fingerprints["passport_pins_sha256"] = pin_sha256
    return fingerprints


def _passport_pins_sha256(config: LabConfig) -> str | None:
    from .sqlite_store import SQLiteStore
    pins = SQLiteStore().load_pins(config.lab_id)
    immutable = [{
        "name": item["name"],
        "sha256": item["sha256"],
        "source_path": item.get("source_path"),
        "payload": item.get("payload", {}),
    } for item in pins]
    payload = json.dumps(immutable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_passport_pins(config: LabConfig, manifest: dict[str, Any]) -> None:
    expected = manifest.get("passport_pins_sha256")
    actual = _passport_pins_sha256(config)
    if not isinstance(expected, str) or expected != actual:
        raise LabError("Passport pins changed after the run snapshot; refusing to finalize or resume")


def _verify_migration_baseline_trust(config: LabConfig, state: dict[str, Any]) -> None:
    if not state.get("baseline_id"):
        return
    from .sqlite_store import SQLiteStore
    binding = SQLiteStore().get_document(config.lab_id, "migration_binding", None)
    if not isinstance(binding, dict):
        return
    trust = binding.get("baseline_trust")
    if not isinstance(trust, dict) or trust.get("usable_for_comparison") is not True:
        source_id = trust.get("source_baseline_id") if isinstance(trust, dict) else None
        if isinstance(source_id, str) and source_id:
            from .sqlite_store import EntityNotFoundError
            store = SQLiteStore()
            current_id = str(state["baseline_id"])
            try:
                source_run = store.resolve_run_id(config.lab_id, source_id)
            except EntityNotFoundError:
                source_run = source_id
            try:
                current_run = store.resolve_run_id(config.lab_id, current_id)
            except EntityNotFoundError:
                current_run = current_id
            # Keep the imported source verdict immutable. A later native baseline
            # has its own accepted/verified manifest and no longer uses that source.
            if str(source_run) != str(current_run):
                return
        raise LabError(
            "Imported baseline is not verified for comparison; explicitly validate a new baseline before trial",
            code="baseline_comparison_unverified",
        )

def _check_candidate_unchanged(config: LabConfig, expected: dict[str, str]) -> None:
    from .config import LabConfig
    candidate = config.state_dir / "candidate"
    entries = [file for file in candidate.rglob("*") if not any(part in _IGNORED_CACHE_DIRS for part in file.relative_to(candidate).parts)]
    if any(file.is_symlink() for file in entries):
        raise LabError("Candidate changed during evaluation: symlink appeared")
    current = {file.relative_to(candidate).as_posix() for file in entries if file.is_file()}
    if current != set(expected) or any(_sha256(candidate / name) != digest for name, digest in expected.items()):
        raise LabError("Candidate changed during evaluation")

def stamp_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Record the skill build and the manifest schema. Old files on disk are not rewritten."""
    from .integrity import skill_stamp

    version, digest = skill_stamp()
    manifest["schema"] = 1
    manifest["skill_version"] = version
    manifest["skill_sha"] = digest
    return manifest


def _save_manifest(config: LabConfig, manifest: dict[str, Any]) -> None:
    from .config import LabConfig
    stamp_manifest(manifest)
    from .sqlite_store import SQLiteStore
    store = SQLiteStore()
    alias = str(manifest["id"])
    existing = store.get_run(config.lab_id, alias)
    if existing is None:
        parent = manifest.get("parent")
        parent_id = str(store.resolve_run_id(config.lab_id, parent)) if parent else None
        store.create_run(
            config.lab_id,
            "baseline" if alias.startswith("baseline") else "trial",
            alias,
            int(manifest.get("epoch") or 1),
            {"parent_run_id": parent_id, "candidate_commit": manifest.get("base_commit")},
        )
    store.save_manifest(config.lab_id, alias, manifest)
