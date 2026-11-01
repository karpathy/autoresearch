"""Measurement epochs, probe ids, and manifest migrations (in memory only)."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LabConfig, load_config
from .errors import LabError
from .store import _sha256, _write_json

_CONTRACT_SPEC_KEYS = frozenset({
    "metrics", "repeats", "evaluator", "input_files", "baseline_validation", "timeout_seconds",
    "goal_criteria", "evaluator_capabilities", "expected_attestations",
})


def migrate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") == 1:
        return manifest
    updated = dict(manifest)
    updated["schema"] = 1
    if "epoch" not in updated:
        updated["epoch"] = 1
    return updated


def migrate_state(state: dict[str, Any]) -> dict[str, Any]:
    if "epochs" not in state or not isinstance(state.get("epochs"), list):
        state["epochs"] = []
    if "current_epoch" not in state:
        state["current_epoch"] = state["epochs"][-1]["id"] if state["epochs"] else 1
    return state


def epoch_fingerprint(config: LabConfig, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluator": list(config.evaluator),
        "input_hashes": {item["path"]: item["sha256"] for item in state.get("inputs", [])},
        "config_sha256": state["config_sha256"],
    }


def find_epoch(state: dict[str, Any], epoch_id: int) -> dict[str, Any] | None:
    for item in state.get("epochs", []):
        if isinstance(item, dict) and item.get("id") == epoch_id:
            return item
    return None


def open_epoch(
    state: dict[str, Any],
    config: LabConfig,
    *,
    reason: str | None = None,
    previous_epoch_id: int | None = None,
    metrics: dict[str, Any] | None = None,
    repeats: int | None = None,
) -> int:
    migrate_state(state)
    superseded_id = previous_epoch_id
    if superseded_id is None:
        if state["epochs"]:
            superseded_id = int(state["current_epoch"])
        elif reason:
            superseded_id = int(state.get("current_epoch") or 1)
    metrics_record = metrics if metrics is not None else {"primary": config.primary, "direction": config.direction}
    repeats_record = repeats if repeats is not None else config.repeats
    # An upgraded lab has no epochs list, but its existing runs already are epoch 1.
    # Supersede must advance past that implicit era instead of allocating 1 again.
    if not state["epochs"] and reason:
        implicit = int(state.get("current_epoch") or 1)
        now = datetime.now(timezone.utc).isoformat()
        state["epochs"].append({
            "id": implicit,
            "started_at": None,
            "superseded_at": now,
            "supersede_reason": reason,
            **epoch_fingerprint(config, state),
        })
    epoch_id = (max(int(item.get("id") or 0) for item in state["epochs"]) + 1) if state["epochs"] else 1
    if state["epochs"] and state["epochs"][-1].get("id") != epoch_id and state["epochs"][-1].get("superseded_at") is None:
        previous = state["epochs"][-1]
        previous["superseded_at"] = datetime.now(timezone.utc).isoformat()
        previous["supersede_reason"] = reason or ""
    record = {
        "id": epoch_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "superseded_at": None,
        "supersede_reason": None,
        "previous_epoch_id": superseded_id,
        "metrics": metrics_record,
        "repeats": repeats_record,
        **epoch_fingerprint(config, state),
    }
    state["epochs"].append(record)
    state["current_epoch"] = epoch_id
    return epoch_id


def _resnapshot_candidate_source(config: LabConfig, state: dict[str, Any], *, reason: str) -> None:
    """Move a strict lab to its declared source set without losing the old snapshot."""
    if config.editable_files is None:
        return

    candidate_path = config.state_dir / "candidate"
    if candidate_path.is_symlink() or not candidate_path.is_dir():
        raise LabError("Candidate directory is missing or linked")
    candidate = candidate_path.resolve()

    old_source = state.get("source")
    if not isinstance(old_source, list):
        raise LabError("Cannot resnapshot a lab without its previous source snapshot")
    old_hashes: dict[str, str] = {}
    for item in old_source:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise LabError("Previous source snapshot is invalid")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise LabError("Previous source snapshot is invalid")
        if item["path"] in old_hashes:
            raise LabError("Previous source snapshot has duplicate paths")
        old_hashes[item["path"]] = digest.lower()

    candidate_files: dict[str, Path] = {}
    for path in candidate.rglob("*"):
        relative = path.relative_to(candidate)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise LabError("Candidate symlink is not allowed")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(candidate):
            raise LabError("Candidate source path leaves the candidate directory")
        candidate_files[relative.as_posix()] = path

    declared = set(config.source_files)
    missing = declared - candidate_files.keys()
    if missing:
        raise LabError(f"Declared candidate source is missing: {', '.join(sorted(missing))}")

    obsolete = set(candidate_files) - declared
    for name in sorted(obsolete):
        if name not in old_hashes or _sha256(candidate_files[name]) != old_hashes[name]:
            raise LabError(f"Cannot remove candidate file outside the unchanged previous source snapshot: {name}")

    new_source = [
        {"path": name, "sha256": _sha256(candidate_files[name])}
        for name in sorted(declared)
    ]
    previous_snapshot = [dict(item) for item in old_source]
    if previous_snapshot != new_source:
        snapshots = state.setdefault("source_snapshots", [])
        if not isinstance(snapshots, list):
            raise LabError("Source snapshot history is invalid")
        snapshots.append({
            "epoch_id": int(state.get("current_epoch") or 1),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source_root": state.get("source_root"),
            "source_head": state.get("source_head"),
            "files": previous_snapshot,
        })

    # Delete only files proven unchanged from the prior immutable snapshot.
    for name in sorted(obsolete):
        candidate_files[name].unlink()
    state["source"] = new_source


def open_contract_epoch(config_path: Path, spec: dict[str, Any], *, reason: str) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason.strip():
        raise LabError("epoch open requires --reason")
    reason = reason.strip()
    if not isinstance(spec, dict):
        raise LabError("epoch spec must be a JSON object")
    unknown = set(spec) - _CONTRACT_SPEC_KEYS
    if unknown:
        raise LabError(f"Unsupported epoch spec keys: {', '.join(sorted(unknown))}")

    resolved = config_path.resolve()
    config = load_config(resolved)
    from .state import _runs, load_state

    state = load_state(config)
    migrate_state(state)
    if any(item.get("status") == "running" for item in _runs(config)):
        raise LabError("Interrupted run exists; run recover first")

    if getattr(config, "storage_mode", "legacy") == "sqlite":
        from .sqlite_store import SQLiteStore
        raw = json.loads(SQLiteStore().load_config_bytes(config.lab_id).decode("utf-8"))
    else:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    for key, value in spec.items():
        raw[key] = value
    raw_config_bytes = (json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Keep relative paths anchored to the same directory and validate before replacing the contract.
    with tempfile.NamedTemporaryFile(dir=resolved.parent, suffix=".json", delete=False) as temporary:
        proposed = Path(temporary.name)
    try:
        proposed.write_bytes(raw_config_bytes)
        validated_config = load_config(proposed, read_only=True)
    finally:
        proposed.unlink(missing_ok=True)
    from .runtimeops import mark_operation_finalizing
    mark_operation_finalizing()
    sqlite_mode = getattr(config, "storage_mode", "legacy") == "sqlite"
    if not sqlite_mode:
        _write_json(resolved, raw)
        config = load_config(resolved)
    else:
        # Validation uses a temporary file so relative paths are resolved normally,
        # then retain the SQLite lab identity and virtual config location.
        config = replace(
            validated_config,
            path=config.path,
            source_root=config.source_root,
            state_dir=config.state_dir,
            storage_mode="sqlite",
            lab_id=config.lab_id,
            config_bytes=raw_config_bytes,
            passport=config.passport,
        )
    updated_hash = hashlib.sha256(raw_config_bytes).hexdigest() if sqlite_mode else _sha256(resolved)
    _resnapshot_candidate_source(config, state, reason=reason)
    # Дашборд читает копию внутри лаборатории и сверяет config_copy_sha256.
    copy_path = config.state_dir / "lab.json"
    if not sqlite_mode and resolved != copy_path.resolve():
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved, copy_path)
    state["inputs"] = [{"path": str(item), "sha256": _sha256(item)} for item in config.input_files]
    state["config_sha256"] = updated_hash
    state["config_copy_sha256"] = updated_hash if sqlite_mode else _sha256(copy_path)
    epoch_id = open_epoch(
        state,
        config,
        reason=reason,
        metrics={"primary": config.primary, "direction": config.direction},
        repeats=config.repeats,
    )
    state["baseline_id"] = None
    state["best_id"] = None
    state["baseline_verified"] = False
    if sqlite_mode:
        from .sqlite_store import SQLiteStore
        SQLiteStore().save_config_and_state(config.lab_id, raw, state, raw_config_bytes=raw_config_bytes)
    else:
        _write_json(config.state_dir / "state.json", state)
    epoch_record = state["epochs"][-1]
    return {
        "epoch_id": epoch_id,
        "previous_epoch_id": epoch_record.get("previous_epoch_id"),
        "config_sha256": updated_hash,
        "repeats": config.repeats,
        "metrics": {"primary": config.primary, "direction": config.direction},
    }


def ensure_epoch(state: dict[str, Any], config: LabConfig) -> int:
    migrate_state(state)
    if not state["epochs"]:
        return open_epoch(state, config)
    current = int(state["current_epoch"])
    if find_epoch(state, current) is None:
        state["current_epoch"] = state["epochs"][-1]["id"]
    return int(state["current_epoch"])


def record_noise_floor(state: dict[str, Any], config: LabConfig, runs: list[dict[str, Any]]) -> None:
    migrate_state(state)
    epoch = find_epoch(state, int(state["current_epoch"]))
    if epoch is None:
        return
    values = [float(run["metrics"][config.primary]) for run in runs if isinstance(run.get("metrics"), dict)]
    if not values:
        return
    spread = max(values) - min(values)
    floor = {"primary": spread, "metric": config.primary}
    epoch["noise_floor"] = floor
    state["noise_floor"] = floor


def manifest_epoch_id(manifest: dict[str, Any], state: dict[str, Any]) -> int:
    return int(manifest.get("epoch") or state.get("current_epoch") or 1)


def epoch_snapshot_for_manifest(manifest: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    epoch_id = manifest_epoch_id(manifest, state)
    epoch = find_epoch(state, epoch_id)
    if epoch is not None:
        return epoch
    if manifest.get("input_hashes") is not None and manifest.get("config_sha256"):
        return {
            "evaluator": manifest.get("evaluator"),
            "input_hashes": manifest.get("input_hashes"),
            "config_sha256": manifest.get("config_sha256"),
        }
    return None


def incomparable_with_current(manifest: dict[str, Any], state: dict[str, Any], config: LabConfig) -> str | None:
    migrate_state(state)
    epoch_id = manifest_epoch_id(manifest, state)
    current_epoch = int(state.get("current_epoch") or 1)
    if epoch_id != current_epoch:
        return "incomparable_with_current: measurement epoch changed"
    current = epoch_fingerprint(config, state)
    snapshot = {
        "config_sha256": manifest.get("config_sha256"),
        "input_hashes": manifest.get("input_hashes"),
        "evaluator": manifest.get("evaluator"),
    }
    epoch = find_epoch(state, epoch_id)
    if epoch is not None:
        for key in ("config_sha256", "input_hashes", "evaluator"):
            if snapshot.get(key) is None:
                snapshot[key] = epoch.get(key)
    for key in ("config_sha256", "input_hashes"):
        if snapshot.get(key) != current.get(key):
            return f"incomparable_with_current: {key} differs from the current epoch"
    if list(snapshot.get("evaluator") or []) != list(current.get("evaluator") or []):
        return "incomparable_with_current: evaluator differs from the current epoch"
    return None


def next_sequence_id(storage_dir: os.PathLike[str] | str, prefix: str) -> str:
    directory = os.fspath(storage_dir)
    os.makedirs(directory, exist_ok=True)
    counter_path = os.path.join(directory, f".{prefix}.counter")
    lock_path = os.path.join(directory, f".{prefix}.counter.lock")
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                stale = time.time() - os.path.getmtime(lock_path) > 30
            except OSError:
                stale = True
            if stale:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
            else:
                time.sleep(0.01)
    else:
        raise LabError(f"Could not allocate {prefix} id: counter lock busy")
    try:
        value = 0
        if os.path.isfile(counter_path):
            with open(counter_path, encoding="utf-8") as stream:
                text = stream.read().strip()
                if text.isdigit():
                    value = int(text)
        # Imported probe directories exist before the counter file does.
        if os.path.isdir(directory):
            head = prefix + "-"
            for name in os.listdir(directory):
                if not name.startswith(head):
                    continue
                suffix = name[len(head):]
                if suffix.isdigit():
                    value = max(value, int(suffix))
        value += 1
        with open(counter_path, "w", encoding="utf-8") as stream:
            stream.write(str(value))
        return f"{prefix}-{value:04d}"
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def check_guard_thresholds(
    metrics: dict[str, Any],
    guard_thresholds: dict[str, dict[str, float]] | list[dict[str, Any]] | None,
) -> tuple[bool, str | None]:
    """Check explicit min_threshold and max_threshold bounds on guard metrics."""
    if not guard_thresholds or not metrics:
        return True, None

    normalized_guards: dict[str, dict[str, float]] = {}
    if isinstance(guard_thresholds, dict):
        normalized_guards = guard_thresholds
    elif isinstance(guard_thresholds, (list, tuple)):
        for item in guard_thresholds:
            if isinstance(item, dict) and "name" in item:
                name = item["name"]
                thresh = {}
                if "min_threshold" in item and item["min_threshold"] is not None:
                    thresh["min_threshold"] = float(item["min_threshold"])
                if "max_threshold" in item and item["max_threshold"] is not None:
                    thresh["max_threshold"] = float(item["max_threshold"])
                if thresh:
                    normalized_guards[name] = thresh

    for name, bounds in normalized_guards.items():
        if name not in metrics:
            continue
        val = metrics[name]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            return False, f"guard {name} has invalid non-numeric value: {val!r}"
        val = float(val)
        min_t = bounds.get("min_threshold")
        if min_t is not None and val < min_t:
            return False, f"guard {name} below min_threshold ({val} < {min_t})"
        max_t = bounds.get("max_threshold")
        if max_t is not None and val > max_t:
            return False, f"guard {name} above max_threshold ({val} > {max_t})"

    return True, None
