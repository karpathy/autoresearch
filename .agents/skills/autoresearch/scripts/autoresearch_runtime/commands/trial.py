"""`trial` subcommand."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ..errors import LabError
from ..locking import _pid_alive, lab_lock
from ..runtime_env import hardened_env, hidden_process_kwargs
from ..state import _runs
from ..sqlite_store import EntityNotFoundError, SQLiteStore
from ..step import resolve_hypothesis_args
from ..trial import recover_lab, run_trial
from ..worker import _write_worker, read_worker
from .common import (
    enqueue_and_start_operation,
    explicit_confirmation,
    normalize_trial_hypothesis,
    require_sqlite_mutation,
    resolve_config,
    run_durable_operation,
)


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _dead_worker(worker: Any, config: Any) -> bool:
    """A worker.json that still says running, while its process is gone."""
    if not isinstance(worker, dict) or worker.get("status") != "running":
        return False
    return not _pid_alive(int(worker.get("pid") or 0))


def _worker_died(config: Any) -> LabError:
    require_sqlite_mutation(config, "trial wait recovery")
    running = any(item.get("status") == "running" for item in _runs(config))
    recovered = _recover_if_needed(config) if running else []
    if not running:
        _write_worker(config.state_dir, {"pid": 0, "heartbeat": time.time(), "status": "failed",
                                         "error": "Background trial stopped before it finished"})
    return LabError(
        "Background trial stopped before it finished",
        code="E_WORKER_DIED",
        hint="The interrupted run was recovered" if recovered else "Run trial start again",
    )


def _load_hypothesis(source: Path | str | dict[str, Any]) -> dict[str, Any]:
    try:
        if isinstance(source, dict):
            return dict(source)
        if isinstance(source, (str, Path)) and str(source).strip().startswith("{"):
            return json.loads(str(source))
        return json.loads(Path(source).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise LabError(f"Invalid hypothesis file: {error}") from error


def _sqlite_context(config: Any) -> tuple[SQLiteStore, uuid.UUID] | None:
    store = SQLiteStore()
    if not store.is_initialized():
        return None
    try:
        return store, store.resolve_lab_id(config.path)
    except EntityNotFoundError:
        return None


def _wait_for_sqlite_trial(store: SQLiteStore, lab_id: uuid.UUID, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        operation = store.load_worker(lab_id)
        if operation is None or operation.get("kind") != "trial":
            return {"pending": True}
        status = operation.get("status")
        if status == "succeeded":
            result = operation.get("result")
            if isinstance(result, dict):
                return result.get("manifest") if isinstance(result.get("manifest"), dict) else result
            return {"operation_id": str(operation["operation_id"]), "status": status}
        if status == "failed":
            raise LabError("Background trial failed; inspect its recorded job result", code="E_WORKER_FAILED")
        if status == "recovery_required":
            raise LabError("Background trial stopped before recording a result", code="E_WORKER_DIED",
                           hint="Run explicit recovery before retrying")
        if time.monotonic() >= deadline:
            return {"pending": True, "operation_id": str(operation["operation_id"]), "status": status}
        time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))
def _recover_if_needed(config: Any) -> list[str]:
    worker = read_worker(config.state_dir)
    worker_pid = int((worker or {}).get("pid") or 0)
    running = any(item.get("status") == "running" for item in _runs(config))
    if running and not _pid_alive(worker_pid):
        with lab_lock(config.state_dir):
            return recover_lab(config)
    return []


def _spawn_worker(config_path: Path, hypothesis_path: Path, state_dir: Path) -> int:
    env = hardened_env(os.environ.copy())
    scripts = _scripts_dir()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(scripts), env.get("PYTHONPATH")]))
    command = [
        sys.executable, "-m", "autoresearch_runtime.worker",
        "--config", str(config_path.resolve()),
        "--hypothesis", str(hypothesis_path.resolve()),
    ]
    kwargs: dict[str, Any] = {"cwd": str(scripts), "env": env}
    if os.name == "nt":
        kwargs.update(hidden_process_kwargs())
        kwargs["creationflags"] |= subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, **kwargs)
    _write_worker(state_dir, {
        "pid": process.pid,
        "heartbeat": time.time(),
        "hypothesis": str(hypothesis_path.resolve()),
        "status": "running",
    })
    return process.pid


def _add_hypothesis_arguments(parser: Any) -> None:
    parser.add_argument("--hypothesis", type=Path, default=None, help="Path to hypothesis JSON file")
    parser.add_argument("--hypothesis-json", type=str, default=None, help="Inline JSON string for the hypothesis (avoids writing a temporary file)")
    parser.add_argument("--title", type=str, default=None, help="Inline hypothesis title")
    parser.add_argument("--observation", type=str, default=None, help="Inline hypothesis observation")
    parser.add_argument("--expected-effect", type=str, default=None, help="Inline hypothesis expected effect")
    parser.add_argument("--rationale", type=str, default=None, help="Optional inline hypothesis rationale")
    parser.add_argument("--falsification", type=str, default=None, help="Optional inline hypothesis falsification criterion")
    parser.add_argument("--target-files", nargs="*", default=None, help="Optional target files for inline hypothesis")


def register(subparsers: Any) -> None:
    trial = subparsers.add_parser("trial")
    _add_hypothesis_arguments(trial)
    sub = trial.add_subparsers(dest="trial_command")
    start = sub.add_parser("start")
    _add_hypothesis_arguments(start)
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout", type=float, default=30.0)
    sub.add_parser("status")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=False)
    sqlite_context = _sqlite_context(config)
    action = getattr(args, "trial_command", None)
    if action == "status":
        if sqlite_context is not None:
            store, lab_id = sqlite_context
            worker = store.load_worker(lab_id)
            return {"worker": worker, "running_manifests": [
                item["id"] for item in _runs(config) if item.get("status") == "running"
            ]}
        worker = read_worker(config.state_dir)
        return {"worker": worker, "running_manifests": [
            item["id"] for item in _runs(config) if item.get("status") == "running"
        ]}
    if action == "wait":
        if sqlite_context is not None:
            store, lab_id = sqlite_context
            return _wait_for_sqlite_trial(store, lab_id, float(args.timeout))
        deadline = time.monotonic() + max(0.0, float(args.timeout))
        while time.monotonic() < deadline:
            worker = read_worker(config.state_dir)
            if isinstance(worker, dict) and worker.get("status") == "finished" and worker.get("manifest"):
                return worker["manifest"]
            if isinstance(worker, dict) and worker.get("status") == "failed":
                raise LabError(str(worker.get("error") or "Background trial failed"))
            if _dead_worker(worker, config):
                raise _worker_died(config)
            time.sleep(0.2)
        worker = read_worker(config.state_dir)
        if isinstance(worker, dict) and worker.get("status") == "failed":
            raise LabError(str(worker.get("error") or "Background trial failed"))
        if _dead_worker(worker, config):
            raise _worker_died(config)
        return {"pending": True}

    if action in {None, "start"}:
        require_sqlite_mutation(config, "trial start")
        hypothesis_input = resolve_hypothesis_args(args)
        if hypothesis_input is None:
            raise LabError(
                "A hypothesis is required (--hypothesis, --hypothesis-json, or --title/--observation/--expected-effect)",
                code="E_HYPOTHESIS_REQUIRED",
            )
        raw_hypothesis = _load_hypothesis(hypothesis_input)

    if action == "start":
        hypothesis = raw_hypothesis
        if sqlite_context is not None:
            allowed_durable = {
                "title", "observation", "expected_effect", "evidence",
                "falsification", "rationale", "target_files", "delegation",
            }
            durable_hypothesis = {k: v for k, v in hypothesis.items() if k in allowed_durable}
            try:
                hypothesis = normalize_trial_hypothesis(durable_hypothesis)
            except ValueError as error:
                raise LabError(str(error), code="E_CONTRACT_HYPOTHESIS") from error
            store, lab_id = sqlite_context
            operation = enqueue_and_start_operation(
                lab_id, "trial", {"hypothesis": hypothesis, **explicit_confirmation(args)},
                f"cli-trial-{uuid.uuid4()}"
            )
            return {
                "started": operation.get("status") in {"queued", "running", "finalizing"},
                "operation_id": str(operation["operation_id"]),
                "status": operation.get("status"),
            }
        recovered = _recover_if_needed(config)
        with lab_lock(config.state_dir):
            worker = read_worker(config.state_dir)
            worker_alive = isinstance(worker, dict) and worker.get("status") == "running" and _pid_alive(int(worker.get("pid") or 0))
            if worker_alive or any(item.get("status") == "running" for item in _runs(config)):
                raise LabError(
                    "A trial is already running",
                    code="E_LOCKED",
                    hint="Use trial wait or trial status",
                )
            if isinstance(hypothesis_input, Path) and hypothesis_input.exists():
                spawn_path = hypothesis_input.resolve()
            else:
                pending_file = config.state_dir / "pending_hypothesis.json"
                pending_file.write_text(json.dumps(raw_hypothesis), encoding="utf-8")
                spawn_path = pending_file.resolve()
            pid = _spawn_worker(config.path, spawn_path, config.state_dir)
        return {"started": True, "pid": pid, "recovered": recovered}

    hypothesis = raw_hypothesis
    if sqlite_context is not None:
        if hypothesis is None:
            raise LabError("A SQLite-backed trial requires an explicit hypothesis", code="E_CONTRACT_HYPOTHESIS")
        allowed_durable = {
            "title", "observation", "expected_effect", "evidence",
            "falsification", "rationale", "target_files", "delegation",
        }
        durable_hypothesis = {k: v for k, v in hypothesis.items() if k in allowed_durable}
        try:
            hypothesis = normalize_trial_hypothesis(durable_hypothesis)
        except ValueError as error:
            raise LabError(str(error), code="E_CONTRACT_HYPOTHESIS") from error
        return run_durable_operation(
            config, "trial", {"hypothesis": hypothesis, **explicit_confirmation(args)}
        )
    with lab_lock(config.state_dir):
        return run_trial(config, hypothesis)
