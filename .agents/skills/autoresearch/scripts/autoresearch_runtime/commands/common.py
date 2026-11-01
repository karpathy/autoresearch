"""Shared lab resolution for CLI subcommands."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..config import load_config
from ..errors import LabError
from ..resolve import resolve_config_ref
from ..runtime_env import hardened_env, hidden_process_kwargs
from ..sqlite_store import SQLiteStore, StoreBusyError
from ..state import load_state


def _reap_worker(
    process: subprocess.Popen[Any], identity_resource: Any = None, worker_identity: dict[str, Any] | None = None,
) -> None:
    """Keep the Popen handle alive until the independent worker exits."""
    try:
        process.wait()
    except Exception:
        # The durable operation row remains the recovery source if waiting fails.
        return
    from ..runtimeops import (
        _terminate_windows_job_handle, _windows_job_active_processes,
        close_worker_identity_resource, wait_for_job_empty,
    )
    if isinstance(worker_identity, dict) and worker_identity.get("kind") == "windows_job":
        if not identity_resource:
            return
        try:
            # The worker root has exited; any remaining evaluator process is orphaned.
            # Keep this handle alive while terminating and proving the job is empty.
            active_processes = _windows_job_active_processes(job_handle=identity_resource)
            if active_processes != 0:
                # The root is gone, so every remaining member is an orphan from this
                # operation. Unknown counts are never treated as an empty Job.
                if not _terminate_windows_job_handle(identity_resource):
                    return
            wait_for_job_empty(identity_resource)
        except Exception:
            # Unknown job state is safer as a retained handle than as a false cleanup.
            return
    close_worker_identity_resource(identity_resource)


def normalize_trial_hypothesis(value: Any) -> dict[str, Any]:
    """Apply the shared bounded hypothesis contract used by CLI and HTTP API."""
    if not isinstance(value, dict):
        raise ValueError("trial hypothesis must be an object")
    # `delegation` was accepted by the file-backed trial contract but is not
    # part of the experiment record; retain CLI compatibility and discard it.
    allowed = {
        "title", "observation", "expected_effect", "evidence",
        "falsification", "rationale", "target_files", "delegation",
    }
    if set(value) - allowed:
        raise ValueError("Unsupported hypothesis field")

    def text(item: Any, field: str, *, limit: int = 4000) -> str:
        if not isinstance(item, str) or not item.strip() or len(item) > limit:
            raise ValueError(f"{field} must be a non-empty string no longer than {limit} characters")
        return item

    result = {key: text(value.get(key), key) for key in ("title", "observation", "expected_effect")}
    if "falsification" in value and value["falsification"] is not None:
        result["falsification"] = text(value["falsification"], "falsification")
    if "rationale" in value and value["rationale"] is not None:
        result["rationale"] = text(value["rationale"], "rationale")
    if "target_files" in value and value["target_files"] is not None:
        tf = value["target_files"]
        if isinstance(tf, list) and all(isinstance(item, str) and item.strip() for item in tf):
            result["target_files"] = [text(item, "target_files item", limit=500) for item in tf[:50]]
        else:
            raise ValueError("target_files must be a list of non-empty strings")
    if "evidence" in value:
        evidence = value["evidence"]
        if isinstance(evidence, str):
            result["evidence"] = text(evidence, "evidence")
        elif isinstance(evidence, list) and 1 <= len(evidence) <= 20:
            result["evidence"] = [text(item, "evidence item", limit=1000) for item in evidence]
        else:
            raise ValueError("evidence must be a non-empty string or a list of up to 20 strings")
    return result


def resolve_config(
    args: Any,
    labs_root: Path | None,
    *,
    pending_ok: bool,
) -> Any:
    lab = getattr(args, "lab", None)
    config_ref = resolve_config_ref(
        config=getattr(args, "config", None),
        lab=lab,
        project=getattr(args, "project", None),
        labs_root=labs_root,
    )
    config = load_config(config_ref, allow_pending=pending_ok)
    if pending_ok and not load_state(config).get("setup_pending"):
        config = load_config(config_ref)
    return config


def explicit_confirmation(args: Any) -> dict[str, bool]:
    """Pass only an explicit CLI true value to the shared durable policy gate."""
    return {"confirmed": True} if getattr(args, "confirmed", None) is True else {}


def spawn_operation_worker(operation_id: uuid.UUID | str) -> dict[str, Any]:
    """Start a worker and capture its identity before releasing the launcher lock."""
    try:
        canonical_id = str(uuid.UUID(str(operation_id)))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("operation_id must be a UUID") from error
    scripts = Path(__file__).resolve().parents[2]
    env = hardened_env(os.environ.copy())
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(scripts), env.get("PYTHONPATH", "")) if item
    )
    command = [sys.executable, "-m", "autoresearch_runtime.worker", "--operation-id", canonical_id]
    options: dict[str, Any] = {"cwd": str(scripts), "env": env}
    if os.name == "nt":
        options.update(hidden_process_kwargs())
        # DETACHED_PROCESS avoids a console host becoming a permanent Job member.
        # The Job Object now owns cancellation, so a console process group is unnecessary.
        options["creationflags"] |= 0x00000008
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **options,
    )
    from .. import runtimeops
    identity, identity_resource = runtimeops.capture_worker_identity(process, canonical_id)
    if (identity is None or
            (identity.get("kind") == "windows_job" and identity_resource is None)):
        # The launcher still owns the lab lock, so the child cannot dispatch before it is stopped.
        try:
            process.kill()
        finally:
            process.wait()
            runtimeops.close_worker_identity_resource(identity_resource)
        raise RuntimeError("Could not safely bind the worker process identity")

    worker_job_handle = None
    if os.name == "nt":
        worker_job_handle = runtimeops.duplicate_job_handle_to_process(identity_resource, process.pid)
        if worker_job_handle is None:
            try:
                process.kill()
            finally:
                process.wait()
                runtimeops.close_worker_identity_resource(identity_resource)
            raise RuntimeError("Could not safely retain the worker Job Object handle")

    result: dict[str, Any] = {"pid": int(process.pid), "worker_identity": identity}
    if worker_job_handle is not None:
        result["worker_job_handle"] = worker_job_handle

    def reap() -> None:
        threading.Thread(
            target=_reap_worker, args=(process, identity_resource, identity),
            name=f"autoresearch-worker-reaper-{process.pid}", daemon=True,
        ).start()

    def abort() -> None:
        if os.name == "nt":
            try:
                process.kill()
            finally:
                process.wait()
                runtimeops.close_worker_identity_resource(identity_resource)
        else:
            # The caller still holds the lab lock; this exact child cannot have dispatched.
            process.kill()

    result["abort"] = abort
    if os.name == "nt":
        def resume() -> None:
            # The worker is already running but cannot pass the lab lock until its
            # PID, identity, and duplicated Job handle are durable.
            reap()

        result["resume"] = resume
    else:
        reap()
    return result


def enqueue_and_start_operation(
    lab_id: uuid.UUID | str,
    command: str,
    params: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    """Reserve durably and launch the shared worker under the lab lock."""
    if command not in {"baseline", "trial", "resume", "recover", "epoch", "probe", "replay"}:
        raise LabError(f"Unsupported operation: {command}", code="E_USAGE_COMMAND")
    payload = {"command": command, "params": params}
    from ..runtimeops import reserve_and_launch_operation

    return reserve_and_launch_operation(
        str(lab_id), command, idempotency_key, payload, spawn_operation_worker
    )


def wait_for_operation(operation_id: uuid.UUID | str, *, poll_seconds: float = 0.2) -> dict[str, Any]:
    """Wait for a durable CLI operation and return its recorded result projection."""
    store = SQLiteStore()
    while True:
        try:
            operation = store.get_operation(operation_id)
        except StoreBusyError:
            time.sleep(max(0.01, poll_seconds))
            continue
        status = operation.get("status")
        if status == "succeeded":
            result = operation.get("result")
            return result if isinstance(result, dict) else {"value": result}
        if status == "failed":
            raise LabError("Operation failed; inspect its durable job record", code="E_WORKER_FAILED")
        if status == "recovery_required":
            raise LabError(
                "Operation stopped with an uncertain outcome; run explicit recover before retrying",
                code="E_WORKER_DIED",
                hint=f"Job {operation_id}",
            )
        time.sleep(max(0.01, poll_seconds))


def run_durable_operation(
    config: Any,
    command: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Launch a DB-backed operation and preserve the synchronous CLI result shape."""
    operation = enqueue_and_start_operation(
        config.lab_id, command, params, f"cli-{command}-{uuid.uuid4()}"
    )
    return wait_for_operation(operation["operation_id"])
