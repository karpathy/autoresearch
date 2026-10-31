"""Background trial runner: python -m autoresearch_runtime.worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
import uuid

from .config import load_config
from .errors import LabError
from .harness_policy import recheck_operation
from .locking import lab_lock
from .runtime_env import hardened_env
from .store import _write_json


def _operation_heartbeat(store: Any, operation_id: str, stop: threading.Event) -> None:
    while not stop.wait(5):
        try:
            operation = store.get_operation(operation_id)
            if operation.get("status") not in {"running", "finalizing"} or operation.get("pid") != os.getpid():
                return
            store.heartbeat_operation(operation_id, os.getpid())
        except Exception:
            return


def _dispatch_operation(config: Any, command: str, params: dict[str, Any]) -> Any:
    if command == "baseline":
        from .baseline import run_baseline
        return run_baseline(config, supersede=bool(params.get("supersede")), reason=params.get("reason"))
    if command == "trial":
        from .trial import run_trial
        return run_trial(config, params.get("hypothesis"))
    if command == "resume":
        from .resume import resume_run
        run_id = params.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("resume operation requires params.run_id")
        return resume_run(config, run_id)
    if command == "recover":
        from .trial import recover_lab
        return {"interrupted": recover_lab(config, restore_candidate=params.get("restore_candidate", True) is not False)}
    if command == "epoch":
        from .measurement import open_contract_epoch
        spec = params.get("spec", {})
        reason = params.get("reason")
        return open_contract_epoch(config.path, spec, reason=reason)
    if command == "probe":
        from .probe import run_probe
        env = params.get("env")
        if not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                            for key, value in env.items()):
            raise ValueError("probe operation requires an object of string environment values")
        return run_probe(
            config, params.get("commit"), env, params.get("group"), params.get("label"),
            params.get("repeats", config.repeats),
        )
    if command == "replay":
        from .replay import run_replay
        trial_id = params.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("replay operation requires params.trial_id")
        return run_replay(config, trial_id)
    raise ValueError(f"Unsupported operation command: {command}")


def _validate_worker_policy(config: Any, command: str, params: dict[str, Any]) -> dict[str, Any]:
    """Revalidate the saved host attestation and keep it out of executor arguments."""
    executor_params = dict(params)
    snapshot = executor_params.pop("harness_policy", None)
    recheck_operation(config, command, snapshot)
    return executor_params


def run_operation_worker(operation_id: str) -> Any:
    """Claim and execute one durable operation; uncertain failures require explicit recovery."""
    canonical_id = str(uuid.UUID(str(operation_id)))
    from .runtimeops import (
        OperationCancellationRequested, _recover_cancelled_lab,
        wait_for_job_worker_only, _worker_identity_matches,
        _worker_job_handle_matches_current_process,
        reset_current_operation, safe_operation_context, set_current_operation,
    )
    from .sqlite_store import SQLiteStore

    store = SQLiteStore()
    operation = store.get_operation(canonical_id)
    lab = store.get_lab(operation["lab_id"])
    config = load_config(lab["config_path"], allow_pending=True)
    deadline = time.monotonic() + 60.0
    while True:
        lock = lab_lock(config.state_dir)
        try:
            lock.__enter__()
            break
        except LabError as error:
            if error.code != "E_LOCKED":
                raise
            waiting = store.get_operation(canonical_id)
            if waiting.get("status") != "queued":
                return None
            owner_pid = int(waiting.get("pid") or 0)
            if owner_pid not in {0, os.getpid()}:
                raise LabError("Queued operation is assigned to another worker", code="E_WORKER_OWNER")
            if time.monotonic() >= deadline:
                raise LabError("Worker timed out waiting for its launcher to release the lab lock", code="E_LOCKED")
            time.sleep(0.05)
    try:
        operation = store.get_operation(canonical_id)
        if operation.get("status") != "queued":
            return None
        progress = operation.get("progress")
        progress = progress if isinstance(progress, dict) else {}
        cancel_requested = progress.get("cancel_requested") is True
        # Built-in Windows workers must prove durable identity and handle membership
        # before claiming work. Same-process embedding launchers are not process trees.
        if os.name == "nt" and operation.get("pid") != os.getpid() and not cancel_requested:
            identity = progress.get("worker_identity")
            job_handle = progress.get("worker_job_handle")
            valid = (
                isinstance(identity, dict)
                and _worker_identity_matches(os.getpid(), identity)
                and _worker_job_handle_matches_current_process(job_handle)
            )
            if not valid:
                store.transition_operation(
                    canonical_id, "queued", "failed",
                    error={
                        "code": "worker_identity_unavailable",
                        "message": "Windows worker could not verify its operation Job Object handle",
                    },
                )
                return None
        if not store.claim_operation(canonical_id, os.getpid()):
            return None
        operation = store.get_operation(canonical_id)
        token = set_current_operation(canonical_id)
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=_operation_heartbeat, args=(store, canonical_id, stop), daemon=True,
        )
        heartbeat.start()
        dispatch_started = False
        try:
            payload = operation.get("payload") or {}
            if payload.get("command") != operation.get("kind") or not isinstance(payload.get("params"), dict):
                raise ValueError("Stored operation payload is invalid")
            params = dict(payload["params"])
            stored_context = params.pop("operation_context", None)
            if stored_context is not None and stored_context != safe_operation_context(config):
                raise LabError("Stored operation configuration no longer matches this lab", code="E_PROVIDER_CONFIG")
            current = store.get_operation(canonical_id)
            progress = current.get("progress")
            if isinstance(progress, dict) and progress.get("cancel_requested") is True:
                raise OperationCancellationRequested()
            params = _validate_worker_policy(config, operation["kind"], params)
            dispatch_started = True
            result = _dispatch_operation(config, operation["kind"], params)
            current = store.get_operation(canonical_id)
            if current["status"] == "running":
                store.transition_operation(canonical_id, "running", "finalizing")
                current = store.get_operation(canonical_id)
            if current["status"] == "finalizing":
                if os.name == "nt":
                    progress = current.get("progress")
                    job_handle = progress.get("worker_job_handle") if isinstance(progress, dict) else None
                    if isinstance(job_handle, int) and not isinstance(job_handle, bool) and job_handle > 0:
                        # Unknown counts and live evaluator descendants keep the worker
                        # under the lab lock. A durable cancel terminates the whole job.
                        wait_for_job_worker_only(job_handle)
                store.transition_operation(canonical_id, "finalizing", "succeeded", result=result if isinstance(result, dict) else {"value": result})
            return result
        except OperationCancellationRequested:
            try:
                _recover_cancelled_lab(config, operation["kind"], canonical_id)
                current = store.get_operation(canonical_id)
                if current["status"] in {"running", "finalizing"}:
                    store.transition_operation(
                        canonical_id, current["status"], "failed",
                        error={"code": "operation_cancelled", "message": "Operation cancelled by request"},
                    )
                return None
            except Exception as error:
                current = store.get_operation(canonical_id)
                if current["status"] in {"running", "finalizing"}:
                    store.transition_operation(
                        canonical_id, current["status"], "recovery_required",
                        error={"code": "cancellation_recovery_failed", "message": str(error)[:400]},
                    )
                raise
        except Exception as error:
            current = store.get_operation(canonical_id)
            if current["status"] in {"running", "finalizing"}:
                target_status = "recovery_required" if dispatch_started else "failed"
                error_code = (
                    getattr(error, "code", "operation_outcome_uncertain")
                    if dispatch_started and getattr(error, "code", None) in {
                        "worker_tree_unavailable", "worker_tree_not_empty",
                    }
                    else "operation_outcome_uncertain" if dispatch_started else (
                        getattr(error, "code", "operation_validation_failed") or "operation_validation_failed"
                    )
                )
                store.transition_operation(
                    canonical_id, current["status"], target_status,
                    error={"code": error_code, "message": str(error)[:400]},
                )
            raise
        finally:
            stop.set()
            heartbeat.join(timeout=1)
            reset_current_operation(token)
    finally:
        lock.__exit__(None, None, None)


def _worker_path(state_dir: Path) -> Path:
    return state_dir / "worker.json"


def _write_worker(state_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(_worker_path(state_dir), payload)


def read_worker(state_dir: Path) -> dict[str, Any] | None:
    path = _worker_path(state_dir)
    from .store import _read_json
    from .store import _sqlite_document
    if _sqlite_document(path) is not None:
        payload = _read_json(path, None)
        return payload if isinstance(payload, dict) else None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _heartbeat_loop(state_dir: Path, stop: threading.Event) -> None:
    while not stop.wait(5):
        payload = read_worker(state_dir)
        if payload is None:
            return
        payload["heartbeat"] = time.time()
        payload["pid"] = os.getpid()
        try:
            _write_worker(state_dir, payload)
        except OSError:
            return


def run_worker(config_path: Path, hypothesis_path: Path) -> dict[str, Any]:
    config = load_config(config_path, allow_pending=False)
    if config.storage_mode != "sqlite":
        raise LabError(
            "Legacy trial workers are disabled; legacy labs are read-only",
            code="E_LEGACY_READ_ONLY",
            hint=(
                "Export a legacy lab with `--config <legacy-lab.json> storage export --output archive.zip`, "
                "then start fresh with `prepare` and a new AUTORESEARCH_HOME."
            ),
        )
    raise LabError(
        "Standalone trial workers are disabled; submit SQLite runs through the durable operation service",
        code="E_DURABLE_OPERATION_REQUIRED",
        hint="Run `python scripts/lab.py trial start --confirmed` with an approved harness profile.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one AutoResearch trial in the background")
    parser.add_argument("--operation-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hypothesis", type=Path)
    args = parser.parse_args(argv)
    os.environ.update(hardened_env())
    if args.operation_id:
        if args.config is not None or args.hypothesis is not None:
            parser.error("--operation-id cannot be combined with --config or --hypothesis")
        try:
            run_operation_worker(args.operation_id)
        except Exception as error:
            raise SystemExit(1) from error
        return 0
    if args.config is None or args.hypothesis is None:
        parser.error("--config and --hypothesis are required for a legacy trial worker")
    try:
        run_worker(args.config.resolve(), args.hypothesis.resolve())
    except Exception as error:
        if getattr(error, "code", None) not in {
            "E_LEGACY_READ_ONLY", "E_DURABLE_OPERATION_REQUIRED",
        }:
            try:
                config = load_config(args.config.resolve(), allow_pending=False)
                _write_worker(config.state_dir, {
                    "pid": os.getpid(),
                    "heartbeat": time.time(),
                    "hypothesis": str(args.hypothesis.resolve()),
                    "status": "failed",
                    "error": str(error),
                })
            except Exception:
                pass
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
