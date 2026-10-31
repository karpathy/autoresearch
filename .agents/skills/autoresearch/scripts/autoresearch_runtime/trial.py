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
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .dashboard import render_report_page
from .history import read_history
from .catalog import update_lab_status

from .config import LabConfig
from .errors import LabError
from .evaluator import _budget_overrun, _evaluate_once, _remaining, _spent
from .gitutil import _git, _git_bytes, _restore_candidate
from .runtime_env import hidden_process_kwargs
from .state import _check_candidate_unchanged, _check_external_frozen, _check_frozen, _fingerprints, _runs, _save_manifest, _verify_migration_baseline_trust, _verify_passport_pins, load_state
from .store import _read_json, _sha256, _write_json


def _aggregate_metric(
    results: list[dict[str, Any]], metric: str, direction: str, statistic: str = "worst",
) -> float:
    values = [float(run["metrics"][metric]) for run in results]
    stat = str(statistic).lower() if statistic else "worst"
    if stat == "mean":
        return sum(values) / len(values)
    if stat == "median":
        return float(statistics.median(values))
    return min(values) if direction == "max" else max(values)


def _worst(results: list[dict[str, Any]], metric: str, direction: str) -> float:
    return _aggregate_metric(results, metric, direction, statistic="worst")

def assess_candidate(
    config: LabConfig, baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
) -> tuple[bool, str]:
    if len(baseline) != config.repeats or len(candidate) != config.repeats:
        return False, "incomplete repeats"
    statistic = str((config.acceptance or {}).get("statistic") or "worst").lower()
    best = _aggregate_metric(baseline, config.primary, config.direction, statistic=statistic)
    proposed = _aggregate_metric(candidate, config.primary, config.direction, statistic=statistic)
    acceptance = config.acceptance
    if acceptance and acceptance.get("min_gain") is not None:
        margin = float(acceptance["min_gain"])
        threshold = best + margin if config.direction == "max" else best - margin
        if not (proposed > threshold if config.direction == "max" else proposed < threshold):
            return False, "primary did not improve"
        confirm = int(acceptance.get("confirm_repeats") or 1)
        if confirm > 1:
            confirmed = 0
            for run in candidate:
                value = float(run["metrics"][config.primary])
                if value > threshold if config.direction == "max" else value < threshold:
                    confirmed += 1
            if confirmed < confirm:
                return False, "insufficient confirmed repeats"
    elif not (proposed > best if config.direction == "max" else proposed < best):
        return False, "primary did not improve"
    for name, direction in config.guards:
        old = _aggregate_metric(baseline, name, direction, statistic=statistic)
        new = _aggregate_metric(candidate, name, direction, statistic=statistic)
        if (new < old if direction == "max" else new > old):
            return False, f"guard {name} regressed"
    if getattr(config, "guard_thresholds", None):
        from .measurement import check_guard_thresholds
        for run in candidate:
            ok, reason = check_guard_thresholds(run.get("metrics", {}), config.guard_thresholds)
            if not ok:
                return False, reason or "guard threshold violated"
    if getattr(config, "target_threshold", None) is not None:
        target = float(config.target_threshold)
        if (proposed < target if config.direction == "max" else proposed > target):
            return False, "primary did not meet target_threshold"
    return True, "improved"


def _working_tree_id(config: LabConfig, base_commit: str) -> str:
    """Compute the candidate tree with a temporary Git index, leaving the real index untouched."""
    candidate = config.state_dir / "candidate"
    descriptor, index_name = tempfile.mkstemp(prefix=".finalizing-index-", dir=config.state_dir)
    os.close(descriptor)
    index_path = Path(index_name)
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        for arguments in (
            ["git", "read-tree", base_commit],
            ["git", "add", "-A", "--", "."],
        ):
            completed = subprocess.run(arguments, cwd=candidate, env=environment,
                                       capture_output=True, text=True, encoding="utf-8", errors="replace", **hidden_process_kwargs())
            if completed.returncode:
                raise LabError(f"git {' '.join(arguments[1:])}: {completed.stderr.strip()}")
        completed = subprocess.run(["git", "write-tree"], cwd=candidate, env=environment,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", **hidden_process_kwargs())
        if completed.returncode:
            raise LabError(f"git write-tree: {completed.stderr.strip()}")
        return completed.stdout.strip()
    finally:
        index_path.unlink(missing_ok=True)
        index_path.with_name(index_path.name + ".lock").unlink(missing_ok=True)


def validate_trial_request(config: LabConfig, hypothesis: Any) -> dict[str, Any]:
    """Validate a SQLite trial before an operation becomes durable; never run the evaluator."""
    from .commands.common import normalize_trial_hypothesis

    try:
        normalized = normalize_trial_hypothesis(hypothesis)
    except ValueError as error:
        raise LabError(str(error), code="E_CONTRACT_HYPOTHESIS") from error
    state = load_state(config)
    _verify_migration_baseline_trust(config, state)
    if state.get("baseline_id") is None or (
        config.baseline_validation is not None and state.get("baseline_verified") is not True
    ):
        raise LabError("Run a verified baseline first", code="E_BASELINE_REQUIRED")
    manifests = _runs(config)
    if any(item.get("status") == "running" for item in manifests):
        raise LabError("A trial is already running", code="E_LOCKED")
    _check_external_frozen(config, state)
    candidate = config.state_dir / "candidate"
    base_commit = state.get("best_commit")
    if not isinstance(base_commit, str) or _git(candidate, "rev-parse", "HEAD") != base_commit:
        raise LabError("Candidate Git HEAD changed outside the lab")
    _check_frozen(config, state)
    if _working_tree_id(config, base_commit) == _git(candidate, "rev-parse", f"{base_commit}^{{tree}}"):
        raise LabError("No candidate change")
    if config.budget["max_trials"] is not None:
        trial_count = sum(str(item.get("id", "")).startswith("trial-") for item in manifests)
        if trial_count >= config.budget["max_trials"]:
            raise LabError("Trial budget exhausted", code="E_BUDGET")
    return normalized


def _first_repeat_rejection(
    config: LabConfig, best_runs: list[dict[str, Any]], result: dict[str, Any],
) -> str | None:
    if config.repeats <= 1:
        return None
    if isinstance(config.acceptance, dict):
        if (
            config.acceptance.get("stop_on_first_failure") is False
            or str(config.acceptance.get("statistic") or "worst").lower() in ("mean", "median")
        ):
            return None
    old_primary = _worst(best_runs, config.primary, config.direction)
    primary_now = result["metrics"][config.primary]
    if not (primary_now > old_primary if config.direction == "max" else primary_now < old_primary):
        return "primary did not improve"
    for name, direction in config.guards:
        guard_old = _worst(best_runs, name, direction)
        if (result["metrics"][name] < guard_old if direction == "max"
                else result["metrics"][name] > guard_old):
            return f"guard {name} regressed"
    return None


def _complete_trial_finalization(
    config: LabConfig, state: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    from .sqlite_store import SQLiteStore

    _verify_passport_pins(config, manifest)
    finalization = manifest.get("finalization")
    if not isinstance(finalization, dict):
        raise LabError("Finalizing trial is missing its durable decision")
    decision = finalization.get("decision")
    if decision not in {"accepted", "rejected", "budget", "incomparable", "invalid",
                        "crash", "timeout", "interrupted"}:
        raise LabError("Finalizing trial has an invalid decision")
    candidate = config.state_dir / "candidate"
    base_commit = finalization.get("base_commit")
    if not isinstance(base_commit, str) or not base_commit:
        raise LabError("Finalizing trial is missing its base commit")

    if decision == "accepted":
        target_tree = finalization.get("target_tree")
        if not isinstance(target_tree, str) or not target_tree:
            raise LabError("Accepted finalizing trial is missing its target tree")
        _check_candidate_unchanged(config, manifest["candidate_hashes"])
        head = _git(candidate, "rev-parse", "HEAD")
        head_tree = _git(candidate, "rev-parse", f"{head}^{{tree}}")
        if head_tree == target_tree:
            commit = head
            parent = _git(candidate, "rev-parse", f"{head}^")
            if parent != base_commit:
                raise LabError("Finalizing candidate commit has an unexpected parent")
        elif head == base_commit:
            _git(candidate, "add", "-A", "--", ".")
            current_tree = _git(candidate, "write-tree")
            if current_tree != target_tree:
                raise LabError("Candidate tree changed after the finalizing decision")
            _git(candidate, "-c", "user.name=AutoResearch", "-c", "user.email=lab@local",
                 "commit", "-q", "-m", f"{manifest['id']}: {manifest['title']}")
            commit = _git(candidate, "rev-parse", "HEAD")
            if _git(candidate, "rev-parse", f"{commit}^{{tree}}") != target_tree:
                raise LabError("Candidate commit does not match the saved finalizing tree")
        else:
            raise LabError("Candidate HEAD changed after the finalizing decision")
        state["best_commit"] = commit
        state["best_id"] = manifest["id"]
        manifest["commit"] = commit
    else:
        restore_commit = finalization.get("restore_commit")
        if not isinstance(restore_commit, str) or not restore_commit:
            raise LabError("Rejected finalizing trial is missing its restore commit")
        head = _git(candidate, "rev-parse", "HEAD")
        if head not in {base_commit, restore_commit}:
            raise LabError("Candidate HEAD changed after the finalizing decision")
        try:
            _check_candidate_unchanged(config, manifest["candidate_hashes"])
            trial_snapshot_is_live = True
        except LabError:
            trial_snapshot_is_live = False
        already_restored = (
            head == restore_commit
            and not _git(candidate, "status", "--porcelain", "--untracked-files=all").strip()
            and not _git(candidate, "clean", "-nd").strip()
        )
        if not already_restored:
            expected_tree = finalization.get("pre_restore_tree")
            if isinstance(expected_tree, str) and expected_tree:
                if _working_tree_id(config, base_commit) != expected_tree:
                    raise LabError("Candidate worktree changed after the finalizing decision")
            elif not trial_snapshot_is_live:
                # Older finalizing records lack a decision-time fingerprint, so
                # retain their stricter original-snapshot check.
                raise LabError("Candidate worktree changed after the finalizing decision")
        _restore_candidate(config, restore_commit)

    manifest["status"] = decision
    finalization["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if getattr(config, "storage_mode", "legacy") == "sqlite":
        SQLiteStore().finalize_run(config.lab_id, manifest["id"], state, manifest)
    else:
        if decision == "accepted":
            _write_json(config.state_dir / "state.json", state)
        _save_manifest(config, manifest)
    return manifest


def _finalize_trial(
    config: LabConfig, state: dict[str, Any], manifest: dict[str, Any], best: dict[str, Any],
) -> dict[str, Any]:
    from .measurement import incomparable_with_current

    _verify_passport_pins(config, manifest)
    if manifest.get("status") == "finalizing":
        return _complete_trial_finalization(config, state, manifest)

    if manifest["status"] == "running":
        overrun = _budget_overrun(config)
        if overrun:
            manifest.update(status="budget", reason=overrun)
    if manifest["status"] == "running":
        mismatch = incomparable_with_current(manifest, state, config)
        if mismatch:
            manifest.update(
                status="incomparable", reason=mismatch, comparison_status="incomparable",
            )
        else:
            accepted, reason = assess_candidate(config, best["runs"], manifest["runs"])
            manifest.update(status="accepted" if accepted else "rejected", reason=reason)
    decision = manifest["status"]
    candidate = config.state_dir / "candidate"
    finalization = {
        "decision": decision,
        "base_commit": manifest.get("base_commit"),
        "restore_commit": state.get("best_commit"),
    }
    if decision == "accepted":
        _check_candidate_unchanged(config, manifest["candidate_hashes"])
        finalization["target_tree"] = _working_tree_id(config, str(manifest.get("base_commit")))
    else:
        base_commit = finalization.get("base_commit")
        if not isinstance(base_commit, str) or not base_commit:
            raise LabError("Finalizing trial is missing its base commit")
        # Capture evaluator-created edits/artifacts before persisting the decision.
        # Resume can then distinguish that known snapshot from later foreign edits.
        finalization["pre_restore_tree"] = _working_tree_id(config, base_commit)
    manifest["finalization"] = finalization
    manifest["status"] = "finalizing"
    from .runtimeops import mark_operation_finalizing
    mark_operation_finalizing()
    _save_manifest(config, manifest)
    return _complete_trial_finalization(config, state, manifest)


def run_trial(config: LabConfig, hypothesis: dict[str, Any] | None = None, *, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    from .runtimeops import require_durable_operation_context
    require_durable_operation_context(config, "trial")
    from .measurement import ensure_epoch, migrate_state

    state = load_state(config)
    migrate_state(state)
    epoch_id = ensure_epoch(state, config)
    _verify_migration_baseline_trust(config, state)
    if state["baseline_id"] is None or (
        config.baseline_validation is not None and state.get("baseline_verified") is not True
    ):
        raise LabError("Run a verified baseline first")
    manifests = _runs(config)
    if any(item["status"] == "running" for item in manifests):
        recover_lab(config, restore_candidate=False)
        state = load_state(config)
        manifests = _runs(config)
    _check_external_frozen(config, state)
    candidate = config.state_dir / "candidate"
    if _git(candidate, "rev-parse", "HEAD") != state["best_commit"]:
        raise LabError("Candidate Git HEAD changed outside the lab")

    _git(candidate, "add", "-N", "--", ".")
    patch_bytes = _git_bytes(candidate, "diff", "--binary", "HEAD")
    patch = bool(patch_bytes)

    if isinstance(hypothesis, dict):
        hypothesis = dict(hypothesis)
        if not hypothesis.get("observation"):
            for k in ("rationale", "reason", "why", "context", "description"):
                if isinstance(hypothesis.get(k), str) and hypothesis[k].strip():
                    hypothesis["observation"] = hypothesis[k]
                    break
        if not hypothesis.get("expected_effect"):
            for k in ("expected_impact", "expected_result", "impact", "effect", "target_impact"):
                if isinstance(hypothesis.get(k), str) and hypothesis[k].strip():
                    hypothesis["expected_effect"] = hypothesis[k]
                    break

    is_empty_or_omitted = hypothesis is None or (
        isinstance(hypothesis, dict)
        and not any(isinstance(v, str) and v.strip() for v in hypothesis.values())
    )
    if is_empty_or_omitted and patch:
        diff_stat = _git(candidate, "diff", "--stat", "HEAD").strip()
        changed_names = [line.strip() for line in _git(candidate, "diff", "--name-only", "HEAD").splitlines() if line.strip()]
        file_summary = ", ".join(changed_names[:3])
        if len(changed_names) > 3:
            file_summary += f" (+{len(changed_names) - 3} more)"
        hypothesis = {
            "title": f"Auto: candidate change in {file_summary}" if file_summary else "Auto: candidate change",
            "observation": f"Candidate git diff stat:\n{diff_stat}" if diff_stat else "Candidate has local changes",
            "expected_effect": "Improve primary metric without regressing guard metrics",
        }
    elif isinstance(hypothesis, dict) and patch:
        diff_stat = _git(candidate, "diff", "--stat", "HEAD").strip()
        changed_names = [line.strip() for line in _git(candidate, "diff", "--name-only", "HEAD").splitlines() if line.strip()]
        file_summary = ", ".join(changed_names[:3])
        if len(changed_names) > 3:
            file_summary += f" (+{len(changed_names) - 3} more)"
        if not (isinstance(hypothesis.get("title"), str) and hypothesis["title"].strip()):
            hypothesis["title"] = f"Auto: candidate change in {file_summary}" if file_summary else "Auto: candidate change"
        if not (isinstance(hypothesis.get("observation"), str) and hypothesis["observation"].strip()):
            hypothesis["observation"] = f"Candidate git diff stat:\n{diff_stat}" if diff_stat else "Candidate has local changes"
        if not (isinstance(hypothesis.get("expected_effect"), str) and hypothesis["expected_effect"].strip()):
            hypothesis["expected_effect"] = "Improve primary metric without regressing guard metrics"

    if not isinstance(hypothesis, dict) or any(not isinstance(hypothesis.get(key), str) or not hypothesis[key].strip() for key in ("title", "observation", "expected_effect")):
        raise LabError("Hypothesis needs title, observation, and expected_effect")
    if "evidence" in hypothesis:
        evidence = hypothesis["evidence"]
        valid_evidence = (
            isinstance(evidence, str) and bool(evidence.strip())
        ) or (
            isinstance(evidence, list)
            and bool(evidence)
            and all(isinstance(item, str) and item.strip() for item in evidence)
        )
        if not valid_evidence:
            raise LabError("Hypothesis evidence must be a non-empty string or list of non-empty strings")
    if "falsification" in hypothesis and (
        not isinstance(hypothesis["falsification"], str) or not hypothesis["falsification"].strip()
    ):
        raise LabError("Hypothesis falsification must be a non-empty string")
    # Optional note only. The runtime does not check harness, model, or receipts.
    agent_note = hypothesis.get("agent") if isinstance(hypothesis.get("agent"), dict) else None
    legacy_id = hypothesis.get("legacy_id")
    recheck_kind = hypothesis.get("recheck_kind")
    if legacy_id is not None:
        if (not isinstance(legacy_id, str) or legacy_id not in {
            item["id"] for item in read_history(config.state_dir)["hypotheses"]
        } or recheck_kind not in {"exact", "reconstructed"}):
            raise LabError("Unknown legacy hypothesis or invalid recheck kind")
    elif recheck_kind is not None:
        raise LabError("Recheck kind needs a legacy hypothesis")
    number = sum(item["id"].startswith("trial-") for item in manifests) + 1
    if config.budget["max_trials"] is not None and number > config.budget["max_trials"]:
        raise LabError("Trial budget exhausted")
    _remaining(config)
    try:
        _check_frozen(config, state)
        candidate_error = None
        candidate_error_code = None
    except LabError as error:
        candidate_error = str(error)
        candidate_error_code = error.code
    if not patch and candidate_error is None:
        raise LabError("No candidate change")
    best = next(item for item in manifests if item["id"] == state["best_id"])
    trial_id = f"trial-{number:04d}"
    trial_dir = config.state_dir / "runs" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    if patch_bytes:
        (trial_dir / "change.patch").write_bytes(patch_bytes)
    manifest: dict[str, Any] = {
        "id": trial_id, "title": hypothesis["title"].strip(),
        "observation": hypothesis["observation"].strip(),
        "expected_effect": hypothesis["expected_effect"].strip(),
        "status": "running", "runs": [], "parent": state["best_id"],
        "base_commit": state["best_commit"], "epoch": epoch_id,
        "evaluator": list(config.evaluator),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_available": bool(patch),
        **_fingerprints(config, state),
    }
    if legacy_id is not None:
        manifest.update(legacy_id=legacy_id, recheck_kind=recheck_kind)
    if agent_note is not None:
        manifest["agent"] = agent_note
    for key in ("evidence", "falsification", "rationale", "target_files"):
        if key in hypothesis and hypothesis[key] is not None:
            manifest[key] = hypothesis[key]
    if candidate_error:
        invalid = {"status": "invalid", "reason": candidate_error}
        if candidate_error_code == "input_tampered":
            invalid["reason_code"] = "input_tampered"
        manifest.update(invalid)
        _restore_candidate(config, state["best_commit"])
        _save_manifest(config, manifest)
        return manifest
    _save_manifest(config, manifest)
    complete_all = config.repeat_policy == "complete_all"
    for index in range(config.repeats):
        try:
            timeout = min(config.timeout_seconds, _remaining(config))
        except LabError as error:
            manifest.update(status="budget", reason=str(error))
            break
        status, result, reason = _evaluate_once(
            config, trial_dir / f"repeat_{index + 1:02d}", timeout, extra_env=extra_env,
        )
        manifest["runs"].append(result)
        _save_manifest(config, manifest)
        if status != "ok":
            manifest.update(status=status, reason=reason)
            if isinstance(result, dict) and result.get("error_tail"):
                manifest["error_tail"] = result["error_tail"]
            if not complete_all:
                break
            continue
        try:
            _check_candidate_unchanged(config, manifest["candidate_hashes"])
            _check_frozen(config, state)
        except LabError as error:
            invalid = {"status": "invalid", "reason": str(error)}
            if error.code == "input_tampered":
                invalid["reason_code"] = "input_tampered"
            manifest.update(invalid)
            break
        # A failed first repeat cannot satisfy the worst-of-repeats gate.
        if index == 0:
            first_rejection = _first_repeat_rejection(config, best["runs"], result)
            if first_rejection:
                manifest.update(status="rejected", reason=first_rejection)
                if not complete_all:
                    break
    return _finalize_trial(config, state, manifest, best)

def recover_lab(config: LabConfig, *, restore_candidate: bool = True) -> list[str]:
    """Mark unfinished experiments and restore the last accepted snapshot."""
    if getattr(config, "storage_mode", "legacy") != "sqlite":
        raise LabError(
            "Legacy labs are read-only and cannot be recovered in place",
            code="E_LEGACY_READ_ONLY",
            hint=("Export the legacy lab with `storage export --config <legacy lab.json> --output <archive.zip>`, "
                  "then create a fresh SQLite lab with `prepare` under a dedicated AUTORESEARCH_HOME."),
        )
    from .sqlite_store import SQLiteStore

    state = load_state(config)
    if _sha256(config.path) != state["config_sha256"]:
        raise LabError("Configuration changed after initialization")
    interrupted = [item for item in _runs(config) if item.get("status") in {"created", "queued", "running"}]
    if not interrupted:
        return []
    store = SQLiteStore() if config.storage_mode == "sqlite" else None
    for item in interrupted:
        if state["best_id"] == item["id"]:
            state["best_id"] = item.get("parent")
            state["best_commit"] = item["base_commit"]
        if state["baseline_id"] == item["id"]:
            state["baseline_id"] = None
        item.update(status="interrupted", reason="Run interrupted before a decision")
        if store is not None:
            # Persist the manifest and recovered state together; a later reconcile
            # can safely close repeats if the process stops before the next write.
            store.finalize_run(config.lab_id, item["id"], state, item)
            store.interrupt_running_repeats(config.lab_id, item["id"])
        else:
            _write_json(config.state_dir / "runs" / item["id"] / "manifest.json", item)
    if restore_candidate:
        _restore_candidate(config, state["best_commit"])
    else:
        candidate = (config.state_dir / "candidate").resolve()
        try:
            _git(candidate, "reset", "--mixed", state["best_commit"])
        except Exception:
            pass
    if store is None:
        _write_json(config.state_dir / "state.json", state)
    update_lab_status(config.state_dir)
    return [item["id"] for item in interrupted]

def status_lab(config: LabConfig) -> dict[str, Any]:
    state = load_state(config)
    manifests = [] if state.get("setup_pending", False) else _runs(config)
    history = read_history(config.state_dir)
    best = next((item for item in manifests if item["id"] == state["best_id"]), None)
    baseline = next((item for item in manifests if item["id"] == state["baseline_id"]), None)
    return {
        "setup_pending": state.get("setup_pending", False),
        "missing": (["evaluator"] if not config.evaluator else [])
                   + (["metrics.primary"] if not config.primary else []),
        "baseline_id": state["baseline_id"],
        "baseline_verified": state.get("baseline_verified", state["baseline_id"] is not None),
        "best_id": state["best_id"],
        "baseline_score": _worst(baseline["runs"], config.primary, config.direction) if baseline else None,
        "best_score": _worst(best["runs"], config.primary, config.direction) if best else None,
        "trials": sum(item["id"].startswith("trial-") for item in manifests),
        "spent": _spent(manifests, config.state_dir),
        "budget": config.budget,
        "runs": [{"id": item["id"], "status": item["status"]} for item in manifests],
        "historical_hypotheses": len(history["hypotheses"]),
        "history_documents": len(history["documents"]),
        "dashboard": str(config.state_dir / "dashboard.html"),
    }
