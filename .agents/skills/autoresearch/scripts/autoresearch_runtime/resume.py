"""Resume one interrupted baseline or trial from its saved repeat checkpoints."""

from __future__ import annotations

import json
import math
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline import _finalize_baseline
from .config import LabConfig
from .errors import LabError
from .evaluator import _evaluate_once, _merge_attempt_usage, _remaining, _validate_result
from .gitutil import _git, _restore_candidate
from .runtime_env import hidden_process_kwargs
from .state import (
    _check_candidate_unchanged,
    _check_external_frozen,
    _check_frozen,
    _runs,
    _save_manifest,
    load_state,
)
from .store import _read_json, _sha256, _sqlite_document
from .trial import _finalize_trial, _first_repeat_rejection, recover_lab


_RESUMABLE_STATUSES = {"interrupted", "crash", "timeout"}
_JSONL_FILES = ("cases.jsonl", "attempts.jsonl", "sent-requests.jsonl")


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read complete object rows, preserving an unterminated tail as uncertain."""
    target = _sqlite_document(path)
    if target is not None:
        store, lab_id, kind, run_id, repeat_index = target
        if kind == "cases":
            return store.load_checkpoints(lab_id, run_id, repeat_index=repeat_index), False
        if kind == "attempts":
            records = store.load_attempts(lab_id, run_id, repeat_index=repeat_index)
            records.sort(key=lambda item: (
                item.get("created_at", ""), item.get("attempt_id", ""),
                0 if item.get("event") == "started" else 1 if item.get("event") == "finished" else 2,
            ))
            return records, False
        if kind == "wire":
            return store.load_wire_events(lab_id, run_id, repeat_index=repeat_index), False
    if not path.is_file():
        return [], False
    raw = path.read_bytes()
    if not raw:
        return [], False
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    uncertain_tail = not raw.endswith(b"\n")
    for index, item in enumerate(lines):
        if not item.endswith(b"\n"):
            # A non-newline-terminated final fragment is evidence, not a row to repair.
            continue
        try:
            text = item[:-1].decode("utf-8", errors="strict").strip()
        except UnicodeError as error:
            raise LabError(f"Corrupt UTF-8 in {path.name}:{index + 1}") from error
        if not text:
            continue
        try:
            value = json.loads(text)
        except (UnicodeError, ValueError) as error:
            raise LabError(f"Corrupt JSONL record {index + 1} in {path.name}: {error}") from error
        if not isinstance(value, dict):
            raise LabError(f"JSONL record {index + 1} in {path.name} must be an object")
        records.append(value)
    return records, uncertain_tail


def _validate_attempt_events(records: list[dict[str, Any]], path: Path) -> None:
    started: dict[str, str] = {}
    finished: set[str] = set()
    for index, record in enumerate(records, start=1):
        if "event" not in record:
            continue  # Legacy response/usage rows were written only after a response.
        event = record["event"]
        if event in {"attempt", "usage", "wire"}:
            # SQLite keeps these indexed events beside the lifecycle journal;
            # only started/finished rows participate in resume certainty checks.
            continue
        if event not in {"started", "finished"}:
            raise LabError(f"Unknown HTTP attempt event in {path.name}:{index}")
        attempt_id = record.get("attempt_id")
        case_id = record.get("case_id")
        if not isinstance(attempt_id, str) or not attempt_id or not isinstance(case_id, str) or not case_id:
            raise LabError(f"HTTP attempt identity is missing in {path.name}:{index}")
        if event == "started":
            if attempt_id in started or attempt_id in finished:
                raise LabError(f"Duplicate HTTP attempt start in {path.name}:{index}")
            started[attempt_id] = case_id
        else:
            if attempt_id not in started or attempt_id in finished:
                raise LabError(f"HTTP attempt finish has no unique start in {path.name}:{index}")
            if started[attempt_id] != case_id:
                raise LabError(f"HTTP attempt case changed in {path.name}:{index}")
            finished.add(attempt_id)
    unresolved = set(started) - finished
    if unresolved:
        raise LabError(f"Uncertain unfinished HTTP attempt in {path.name}: {sorted(unresolved)[0]}")


def _duration_seconds(record: dict[str, Any] | None) -> float:
    value = record.get("duration_seconds", 0.0) if isinstance(record, dict) else 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise LabError("Resume repeat duration is invalid")
    return float(value)


def _validate_resume_journals(config: LabConfig, run_id: str) -> None:
    for repeat_index in range(1, config.repeats + 1):
        repeat_dir = config.state_dir / "runs" / run_id / f"repeat_{repeat_index:02d}"
        for name in _JSONL_FILES:
            path = repeat_dir / name
            records, uncertain = _read_jsonl(path)
            if uncertain:
                raise LabError(f"Uncertain trailing JSONL evidence in {path}")
            if name == "attempts.jsonl":
                _validate_attempt_events(records, path)


def _validate_case_reconciliation(repeat_dir: Path) -> None:
    attempts, _ = _read_jsonl(repeat_dir / "attempts.jsonl")
    cases, _ = _read_jsonl(repeat_dir / "cases.jsonl")
    completed = {
        record.get("id") for record in cases
        if record.get("status") == "ok" and isinstance(record.get("id"), str)
    }
    for record in attempts:
        if record.get("event") != "finished":
            continue
        status = record.get("http_status")
        known_error = (
            isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599
        )
        if known_error:
            continue
        case_id = record.get("case_id")
        if case_id not in completed:
            raise LabError(
                f"Case {case_id!r} has a successful or uncertain HTTP response without a completed "
                "case checkpoint; reconcile its response in cases.jsonl before resuming",
                code="E_RESUME_RECONCILE",
            )


def _saved_result(
    config: LabConfig, repeat_dir: Path, previous: dict[str, Any] | None, *, completed: bool = False,
) -> dict[str, Any] | None:
    path = repeat_dir / "result.json"
    saved = _read_json(path, None)
    if not isinstance(saved, dict):
        if completed:
            raise LabError(f"Completed repeat has no saved result: {path}")
        return None
    if path.is_file() and (path.is_symlink() or not path.resolve().is_relative_to(repeat_dir.resolve())):
        raise LabError(f"Saved result path is unsafe: {path}")
    try:
        result = _validate_result(config, repeat_dir)
    except LabError:
        raise
    result_hash = _sha256(path) if path.is_file() else saved.get("result_sha256")
    expected_hash = (previous.get("result_sha256") if isinstance(previous, dict) else None) or saved.get("result_sha256")
    if expected_hash is not None and expected_hash != result_hash:
        raise LabError(f"Saved result hash changed: {path}")
    if completed and not isinstance(expected_hash, str):
        raise LabError(f"Completed repeat has no original result hash: {path}")
    if isinstance(previous, dict) and isinstance(previous.get("metrics"), dict):
        if previous["metrics"] != result.get("metrics"):
            raise LabError(f"Completed repeat metrics differ from saved result: {path}")
        previous_details = previous.get("details")
        if previous_details is not None and not isinstance(previous_details, dict):
            raise LabError(f"Completed repeat details are invalid: {path}")
        if isinstance(previous_details, dict) and "evidence_origin" in previous_details:
            previous_origin = previous_details["evidence_origin"]
            if previous_origin not in {"fresh", "offline_recompute", "historical", "imported"}:
                raise LabError(f"Completed repeat evidence origin is invalid: {path}")
            result_details = result.setdefault("details", {})
            result_origin = result_details.get("evidence_origin")
            if result_origin is not None and result_origin != previous_origin:
                raise LabError(f"Completed repeat evidence origin differs from saved result: {path}")
            result_details.setdefault("evidence_origin", previous_origin)
    result["duration_seconds"] = _duration_seconds(previous) if previous is not None else _duration_seconds(saved)
    result["result_sha256"] = result_hash
    result["usage"] = _merge_attempt_usage(repeat_dir, result.get("usage"))
    return result


def _set_repeat(manifest: dict[str, Any], index: int, result: dict[str, Any]) -> None:
    runs = manifest.setdefault("runs", [])
    if not isinstance(runs, list):
        raise LabError("Resume manifest runs must be an array")
    while len(runs) <= index:
        runs.append({})
    runs[index] = result


def _validate_snapshot(config: LabConfig, state: dict[str, Any], manifest: dict[str, Any]) -> str:
    from .integrity import skill_stamp

    run_id = str(manifest.get("id", ""))
    if run_id == "baseline" or run_id.startswith("baseline-"):
        kind = "baseline"
    elif run_id.startswith("trial-"):
        kind = "trial"
    else:
        raise LabError("Resume manifest has an unknown run type")
    declared_kind = manifest.get("type")
    if declared_kind is not None and declared_kind != kind:
        raise LabError("Resume manifest run type does not match its id")
    current_version, current_sha = skill_stamp()
    if manifest.get("skill_version") != current_version or manifest.get("skill_sha") != current_sha:
        raise LabError("Resume runtime build differs from the build that created this run")
    expected_inputs = {item["path"]: item["sha256"] for item in state.get("inputs", [])}
    if (manifest.get("config_sha256") != state.get("config_sha256")
            or manifest.get("input_hashes") != expected_inputs
            or manifest.get("evaluator") != list(config.evaluator)):
        raise LabError("Resume snapshot no longer matches the frozen configuration and inputs")
    if not isinstance(manifest.get("candidate_hashes"), dict):
        raise LabError("Resume manifest has no candidate snapshot fingerprints")
    if manifest.get("base_commit") != state.get("best_commit"):
        raise LabError("Resume base snapshot is no longer the current best candidate")
    if manifest.get("epoch") is not None and manifest.get("epoch") != state.get("current_epoch"):
        raise LabError("Resume epoch changed after the run was interrupted")
    from .state import _verify_migration_baseline_trust, _verify_passport_pins
    _verify_migration_baseline_trust(config, state)
    _verify_passport_pins(config, manifest)
    if kind == "trial":
        if (manifest.get("parent") != state.get("best_id")
                or state.get("baseline_id") is None
                or (config.baseline_validation is not None and state.get("baseline_verified") is not True)):
            raise LabError("Trial parent or verified baseline changed; refusing resume")
        if manifest.get("patch_available") is not True:
            raise LabError("Trial has no saved candidate patch")
    elif state.get("baseline_id") not in (None, manifest.get("id")):
        raise LabError("A different baseline is already active")
    return kind


def _restore_resume_candidate(config: LabConfig, state: dict[str, Any], manifest: dict[str, Any], kind: str) -> None:
    base_commit = manifest["base_commit"]
    candidate = config.state_dir / "candidate"
    if _git(candidate, "rev-parse", "HEAD") != base_commit:
        raise LabError("Live candidate Git HEAD changed; refusing to overwrite it")
    cached = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
        cwd=candidate, capture_output=True, check=False,
        **hidden_process_kwargs(),
    )
    if cached.returncode != 0:
        raise LabError("Live candidate has staged changes; refusing to overwrite it")
    is_base_snapshot = (
        not _git(candidate, "status", "--porcelain", "--untracked-files=all").strip()
        and not _git(candidate, "clean", "-ndx").strip()
    )
    is_run_snapshot = False
    try:
        _check_candidate_unchanged(config, manifest["candidate_hashes"])
        is_run_snapshot = True
    except LabError:
        pass
    if not is_base_snapshot and not is_run_snapshot:
        raise LabError("Live candidate differs from both saved resume snapshots")
    patch = None
    if kind == "trial":
        trial_dir = config.state_dir / "runs" / manifest["id"]
        patch = trial_dir / "change.patch"
        if patch.is_symlink() or not patch.is_file() or not patch.resolve().is_relative_to(trial_dir.resolve()):
            raise LabError("Saved trial patch is missing or unsafe")
    _restore_candidate(config, base_commit)
    if patch is not None:
        completed = subprocess.run(
            ["git", "apply", "--binary", str(patch.resolve())],
            cwd=config.state_dir / "candidate", capture_output=True, check=False,
            **hidden_process_kwargs(),
        )
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            _restore_candidate(config, base_commit)
            raise LabError(f"Saved trial patch could not be applied: {detail}")
    try:
        _check_candidate_unchanged(config, manifest["candidate_hashes"])
        _check_frozen(config, state)
    except LabError:
        _restore_candidate(config, base_commit)
        raise


def resume_run(config: LabConfig, run_id: str) -> dict[str, Any]:
    """Continue one incomplete baseline/trial under the caller's lab lock."""
    if not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id:
        raise LabError("Resume requires an exact run id")
    from .sqlite_store import EntityNotFoundError, SQLiteStore
    store = SQLiteStore()
    try:
        canonical_run_id = store.resolve_run_id(config.lab_id, run_id)
    except EntityNotFoundError as error:
        raise LabError(f"Unknown run id: {run_id}") from error
    manifest = store.load_manifest(config.lab_id, canonical_run_id)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("id"), str):
        raise LabError(f"Run manifest is unavailable: {run_id}")
    run_id = manifest["id"]
    state = load_state(config)
    manifests = _runs(config)
    matching = next((item for item in manifests if item.get("id") == run_id), None)
    if matching is None:
        raise LabError(f"Unknown run id: {run_id}")
    from .locking import _pid_alive
    from .worker import read_worker
    from .runtimeops import current_operation_id

    worker = read_worker(config.state_dir)
    if isinstance(worker, dict) and worker.get("status") == "running":
        try:
            worker_pid = int(worker.get("pid") or 0)
        except (TypeError, ValueError):
            worker_pid = 0
        if _pid_alive(worker_pid) and str(worker.get("operation_id") or "") != current_operation_id():
            raise LabError("A live background worker still owns this lab")
    if matching.get("status") == "finalizing":
        active = [item for item in manifests if item.get("status") in {"running", "finalizing"}]
        if len(active) != 1 or active[0].get("id") != run_id:
            raise LabError("Another run is active; refusing finalization recovery")
        kind = _validate_snapshot(config, state, matching)
        _check_external_frozen(config, state)
        _validate_resume_journals(config, run_id)
        if kind == "baseline":
            return _finalize_baseline(config, state, matching)
        best = next((item for item in manifests if item.get("id") == state.get("best_id")), None)
        if best is None:
            raise LabError("Current best manifest is unavailable for finalizing trial recovery")
        return _finalize_trial(config, state, matching, best)
    running = [item for item in manifests if item.get("status") == "running"]
    if running:
        if len(running) != 1 or running[0].get("id") != run_id:
            raise LabError("Another run is still marked running; recover it before resuming")
        recover_lab(config, restore_candidate=False)
        state = load_state(config)
        manifests = _runs(config)
        matching = next(item for item in manifests if item.get("id") == run_id)
    if matching.get("status") == "budget":
        raise LabError("Budget-exhausted runs cannot be resumed")
    repair_saved_result = (
        isinstance(matching.get("reason"), str)
        and (
            (matching.get("status") == "invalid"
             and matching["reason"].startswith("Invalid result.json:"))
            or (matching.get("status") == "incomparable"
                and matching["reason"].startswith("request_identity sha256 mismatch:"))
        )
    )
    if matching.get("status") not in _RESUMABLE_STATUSES and not repair_saved_result:
        raise LabError(f"Run status {matching.get('status')} is not resumable")
    kind = _validate_snapshot(config, state, matching)
    _check_external_frozen(config, state)
    _validate_resume_journals(config, run_id)
    runs = matching.get("runs", [])
    if not isinstance(runs, list) or len(runs) > config.repeats:
        raise LabError("Resume manifest repeat history is invalid")
    if any(not isinstance(item, dict) for item in runs):
        raise LabError("Resume manifest contains an invalid repeat record")
    best = next((item for item in manifests if item.get("id") == state.get("best_id")), None)
    if kind == "trial" and best is None:
        raise LabError("Current best manifest is unavailable")

    saved: dict[int, dict[str, Any]] = {}
    missing_repeats: list[int] = []
    needs_evaluator = False
    for index in range(config.repeats):
        previous = runs[index] if index < len(runs) and isinstance(runs[index], dict) else None
        completed = isinstance(previous, dict) and "metrics" in previous
        if completed and not isinstance(previous.get("metrics"), dict):
            raise LabError(f"Repeat {index + 1} has invalid completed metrics")
        value = _saved_result(
            config, config.state_dir / "runs" / run_id / f"repeat_{index + 1:02d}", previous,
            completed=completed,
        )
        if value is not None:
            saved[index] = value
        else:
            needs_evaluator = True
            missing_repeats.append(index)
    if needs_evaluator:
        if repair_saved_result:
            raise LabError("Invalid runs can only be recovered from already-saved valid results; refusing evaluator rerun")
        capabilities = getattr(config, "evaluator_capabilities", {}) or {}
        if capabilities.get("case_resume") is not True or capabilities.get("offline_verified") is not True:
            raise LabError("Missing repeats require verified offline case_resume evaluator capability")
        for index in missing_repeats:
            _validate_case_reconciliation(
                config.state_dir / "runs" / run_id / f"repeat_{index + 1:02d}"
            )

    resume_id = uuid.uuid4().hex
    _restore_resume_candidate(config, state, matching, kind)
    matching["resume_id"] = resume_id
    matching["resume_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    matching["resume_from_status"] = matching.get("status")
    matching["status"] = "running"
    matching.pop("reason", None)
    _save_manifest(config, matching)

    for index in range(config.repeats):
        previous = matching["runs"][index] if index < len(matching["runs"]) and isinstance(matching["runs"][index], dict) else None
        repeat_dir = config.state_dir / "runs" / run_id / f"repeat_{index + 1:02d}"
        result = saved.get(index)
        status = "ok"
        reason = ""
        if result is None:
            previous_duration = _duration_seconds(previous)
            try:
                timeout = min(config.timeout_seconds, _remaining(config))
            except LabError as error:
                status, result, reason = "budget", {"duration_seconds": 0.0}, str(error)
            else:
                status, result, reason = _evaluate_once(
                    config, repeat_dir, timeout, resume_id=resume_id,
                )
            result = dict(result)
            result["duration_seconds"] = previous_duration + _duration_seconds(result)
        if status != "ok":
            _set_repeat(matching, index, result)
            matching.update(status=status, reason=reason)
            _save_manifest(config, matching)
            _restore_candidate(config, state["best_commit"])
            if kind == "baseline":
                raise LabError(f"Baseline resume failed: {reason}")
            return _finalize_trial(config, state, matching, best)

        _set_repeat(matching, index, result)
        _save_manifest(config, matching)
        try:
            _check_candidate_unchanged(config, matching["candidate_hashes"])
            _check_frozen(config, state)
        except LabError as error:
            matching.update(status="invalid", reason=str(error))
            _save_manifest(config, matching)
            _restore_candidate(config, state["best_commit"])
            if kind == "baseline":
                raise LabError(f"Baseline resume failed: {error}") from error
            return _finalize_trial(config, state, matching, best)
        if kind == "trial" and index == 0:
            rejection = _first_repeat_rejection(config, best["runs"], result)
            if rejection:
                matching.update(status="rejected", reason=rejection)
                if config.repeat_policy != "complete_all":
                    break

    if kind == "baseline":
        return _finalize_baseline(config, state, matching)
    return _finalize_trial(config, state, matching, best)
