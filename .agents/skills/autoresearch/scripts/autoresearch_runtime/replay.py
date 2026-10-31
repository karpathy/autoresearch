"""Small, data-agnostic experiment runner.

An evaluator owns the domain data and writes one result.json per repeat. This
module only manages files, comparable metrics, and experiment history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .dashboard import render_report_page
from .catalog import update_lab_status

from .config import LabConfig
from .errors import LabError
from .evaluator import _evaluate_once
from .gitutil import _git
from .measurement import epoch_snapshot_for_manifest, migrate_state, next_sequence_id
from .state import _runs, load_state
from .store import _sha256, _write_json


def _tree_hashes(root: Path) -> dict[str, str | None]:
    names = {file.relative_to(root).as_posix() for file in root.rglob("*")
             if file.is_file() and ".git" not in file.relative_to(root).parts}
    return {
        name: None if (root / name).is_symlink() else _sha256(root / name)
        for name in sorted(names)
    }

def _input_blockers(config: LabConfig, manifest: dict[str, Any]) -> list[dict[str, str]]:
    blockers = []
    expected = manifest.get("input_hashes") or {}
    if not isinstance(expected, dict):
        return [{"path": "", "reason": "manifest input hashes are unreadable"}]
    for path_str, digest in expected.items():
        path = Path(str(path_str))
        if path.is_file() and _sha256(path) == digest:
            continue
        blob = config.state_dir / "vault" / "blobs" / str(digest)
        selector = (f"--lab {config.lab_id}" if getattr(config, "storage_mode", "legacy") == "sqlite"
                    else f"--config {config.path}")
        command = f"python lab.py {selector} vault restore --path {path}"
        if blob.is_file():
            blockers.append({
                "path": str(path),
                "reason": "protected input is missing or its bytes changed",
                "command": command,
            })
        else:
            blockers.append({
                "path": str(path),
                "reason": "protected input is missing or changed and the vault has no copy",
            })
    return blockers

_GAP_STATUSES = {"invalid", "crash", "timeout", "budget", "interrupted"}

def replay_plan(config: LabConfig) -> dict[str, Any]:
    """List hypotheses whose chart point is missing, without measuring anything."""
    gaps = []
    for manifest in _runs(config):
        run_id = str(manifest.get("id", ""))
        if not run_id.startswith("trial-"):
            continue
        runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
        incomplete = len(runs) < config.repeats or manifest.get("status") in _GAP_STATUSES
        if not incomplete:
            continue
        blockers = _input_blockers(config, manifest)
        patch = config.state_dir / "runs" / run_id / "change.patch"
        if manifest.get("patch_available", True) and not patch.is_file():
            blockers.append({"path": str(patch), "reason": "change.patch is missing"})
        if not manifest.get("base_commit"):
            blockers.append({"path": "", "reason": "base_commit is missing"})
        selector = (f"--lab {config.lab_id}" if getattr(config, "storage_mode", "legacy") == "sqlite"
                    else f"--config {config.path}")
        gaps.append({
            "id": run_id,
            "status": manifest.get("status"),
            "ready": not blockers,
            "blockers": blockers,
            "command": f"python lab.py {selector} replay --trial {run_id}",
        })
    return {"gaps": gaps}

def _replay_checkout(config: LabConfig, manifest: dict[str, Any], checkout: Path) -> None:
    live = config.state_dir / "candidate"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    _git(checkout.parent, "clone", "-q",
         "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
         str(live), checkout.name)
    _git(checkout, "config", "core.autocrlf", "false")
    _git(checkout, "config", "core.safecrlf", "false")
    _git(checkout, "checkout", "-q", "--detach", str(manifest["base_commit"]))
    patch = config.state_dir / "runs" / str(manifest["id"]) / "change.patch"
    if patch.is_file() and patch.stat().st_size:
        _git(checkout, "apply", "--binary", "--whitespace=nowarn", str(patch))

def run_replay(config: LabConfig, trial_id: str) -> dict[str, Any]:
    """Re-measure one saved hypothesis in an isolated checkout. The live candidate stays put."""
    from .runtimeops import require_durable_operation_context
    require_durable_operation_context(config, "replay")
    if not trial_id.startswith("trial-"):
        raise LabError("Replay needs a trial id")
    manifests = {item.get("id"): item for item in _runs(config)}
    manifest = manifests.get(trial_id)
    if not isinstance(manifest, dict):
        raise LabError(f"Unknown trial: {trial_id}")
    live = (config.state_dir / "candidate").resolve()
    head_before = _git(live, "rev-parse", "HEAD")
    worktree_before = _git(live, "status", "--porcelain")
    blockers = _input_blockers(config, manifest)
    replay_root = config.state_dir / "replays" / trial_id
    sqlite_mode = getattr(config, "storage_mode", "legacy") == "sqlite"
    if sqlite_mode:
        from .sqlite_store import SQLiteStore
        store = SQLiteStore()
        page = store.list_runs(config.lab_id, limit=100, offset=0)
        replay_aliases = [item.get("alias", "") for item in page["items"]
                          if item.get("kind") == "replay"]
        offset = len(page["items"])
        while offset < page["total"]:
            page = store.list_runs(config.lab_id, limit=100, offset=offset)
            replay_aliases.extend(item.get("alias", "") for item in page["items"]
                                  if item.get("kind") == "replay")
            offset += len(page["items"])
        numbers = [int(alias[7:]) for alias in replay_aliases
                   if isinstance(alias, str) and alias.startswith("replay-") and alias[7:].isdigit()]
        replay_id = f"replay-{max(numbers, default=0) + 1:04d}"
    else:
        replay_id = next_sequence_id(replay_root, "replay")
    state = migrate_state(load_state(config))
    epoch = epoch_snapshot_for_manifest(manifest, state)
    frozen_evaluator = tuple(epoch.get("evaluator") or config.evaluator) if epoch else config.evaluator
    recheck_dir = (config.state_dir / "runs" / replay_id if sqlite_mode else
                   config.state_dir / "runs" / trial_id / "rechecks" / replay_id)
    if not sqlite_mode:
        recheck_dir.mkdir(parents=True, exist_ok=True)
    proof: dict[str, Any] = {
        "schema": 1,
        "id": replay_id,
        "type": "replay",
        "trial_id": trial_id,
        "lab_id": str(config.lab_id) if sqlite_mode else config.state_dir.name,
        "replay_id": replay_id,
        "recheck_kind": "exact",
        "base_commit": manifest.get("base_commit"),
        "baseline_commit": manifest.get("base_commit"),
        "status": "blocked",
        "candidate_hashes": manifest.get("candidate_hashes"),
        "input_hashes": manifest.get("input_hashes"),
        "repeats": config.repeats,
        "adoptable": False,
    }
    if sqlite_mode:
        replay_run_id = store.create_run(config.lab_id, "replay", replay_id,
                                         int(state.get("current_epoch") or 1), {
                                             "parent_run_id": str(store.resolve_run_id(config.lab_id, trial_id)),
                                             "candidate_commit": state.get("best_commit"),
                                         })
        proof["run_id"] = str(replay_run_id)
        proof["epoch"] = int(state.get("current_epoch") or 1)
        _write_json(recheck_dir / "manifest.json", proof)
    patch = config.state_dir / "runs" / trial_id / "change.patch"
    if patch.is_file():
        proof["patch_sha256"] = _sha256(patch)
    if blockers or not manifest.get("base_commit"):
        proof["blockers"] = blockers or [{"reason": "base_commit is missing"}]
        _write_json(recheck_dir / ("manifest.json" if sqlite_mode else "verification.json"), proof)
        return proof
    checkout = config.state_dir / "replays" / trial_id / replay_id / "checkout"
    try:
        _replay_checkout(config, manifest, checkout)
        actual = _tree_hashes(checkout)
        expected = manifest.get("candidate_hashes") or {}
        if actual != expected:
            proof["status"] = "failed"
            proof["reason"] = "checkout bytes do not match the saved candidate hashes"
            proof["hash_mismatch"] = sorted(
                name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name)
            )
            _write_json(recheck_dir / ("manifest.json" if sqlite_mode else "verification.json"), proof)
            return proof
        runs = []
        limited = 0
        for index in range(config.repeats):
            run_dir = recheck_dir / f"repeat_{index + 1:02d}"
            status, result, reason = _evaluate_once(
                config, run_dir, config.timeout_seconds, candidate=checkout,
                evaluator=frozen_evaluator,
            )
            if status != "ok":
                proof.update(status="failed", reason=reason)
                _write_json(recheck_dir / ("manifest.json" if sqlite_mode else "verification.json"), proof)
                return proof
            if _tree_hashes(checkout) != expected:
                proof.update(status="failed", reason="checkout changed during evaluation")
                _write_json(recheck_dir / ("manifest.json" if sqlite_mode else "verification.json"), proof)
                return proof
            if not sqlite_mode:
                _write_json(run_dir / "result.json", {
                    key: result[key] for key in ("metrics", "usage", "details") if key in result
                })
            runs.append(result)
            limited += int((result.get("net") or {}).get("rate_limited_responses") or 0)
        proof.update(status="complete", runs_recorded=len(runs),
                     unknown_cost_attempts=limited, cost_status="confirmed" if not limited else "partial")
        _write_json(recheck_dir / ("manifest.json" if sqlite_mode else "verification.json"), proof)
        return proof
    finally:
        if _git(live, "rev-parse", "HEAD") != head_before or _git(live, "status", "--porcelain") != worktree_before:
            raise LabError("Replay touched the live candidate")
