"""`trial` subcommand."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from ..errors import LabError
from ..sqlite_store import SQLiteStore
from ..state import _runs
from ..step import resolve_hypothesis_args
from .common import (
    enqueue_and_start_operation,
    explicit_confirmation,
    normalize_trial_hypothesis,
    resolve_config,
    run_durable_operation,
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
    store = SQLiteStore()
    lab_id = uuid.UUID(str(config.lab_id))
    action = getattr(args, "trial_command", None)
    if action == "status":
        worker = store.load_worker(lab_id)
        return {"worker": worker, "running_manifests": [
            item["id"] for item in _runs(config) if item.get("status") == "running"
        ]}
    if action == "wait":
        return _wait_for_sqlite_trial(store, lab_id, float(args.timeout))

    hypothesis_input = resolve_hypothesis_args(args)
    if hypothesis_input is None:
        raise LabError(
            "A hypothesis is required (--hypothesis, --hypothesis-json, or --title/--observation/--expected-effect)",
            code="E_HYPOTHESIS_REQUIRED",
        )
    allowed_durable = {
        "title", "observation", "expected_effect", "evidence",
        "falsification", "rationale", "target_files", "delegation",
    }
    raw_hypothesis = _load_hypothesis(hypothesis_input)
    durable_hypothesis = {k: v for k, v in raw_hypothesis.items() if k in allowed_durable}
    try:
        hypothesis = normalize_trial_hypothesis(durable_hypothesis)
    except ValueError as error:
        raise LabError(str(error), code="E_CONTRACT_HYPOTHESIS") from error
    if action == "start":
        operation = enqueue_and_start_operation(
            lab_id, "trial", {"hypothesis": hypothesis, **explicit_confirmation(args)},
            f"cli-trial-{uuid.uuid4()}"
        )
        return {
            "started": operation.get("status") in {"queued", "running", "finalizing"},
            "operation_id": str(operation["operation_id"]),
            "status": operation.get("status"),
        }
    return run_durable_operation(
        config, "trial", {"hypothesis": hypothesis, **explicit_confirmation(args)}
    )
