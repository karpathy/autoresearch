"""Reusable host profile and durable-worker helpers for SQLite integration tests."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from autoresearch_runtime.harness_policy import contract_fingerprint
from autoresearch_runtime.runtimeops import reserve_and_launch_operation
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.worker import run_operation_worker


@contextmanager
def approved_offline_profile(config: Any) -> Iterator[Path]:
    """Provide a scoped, host-side offline attestation bound to this lab contract."""
    with tempfile.TemporaryDirectory(prefix="autoresearch-host-profile-") as directory:
        profile_path = Path(directory) / "harness-profile.json"
        profile_path.write_text(json.dumps({
            "schema_version": 1,
            "harness": {"id": "autoresearch-test-harness", "version": "1"},
            "approval": {
                "approved": True,
                "contract_fingerprint": contract_fingerprint(config),
            },
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": {"mode": "disabled"},
            "spend": None,
        }), encoding="utf-8")
        resolved = profile_path.resolve()
        for protected in (getattr(config, "source_root", None), getattr(config, "state_dir", None)):
            if protected is not None and resolved.is_relative_to(Path(protected).resolve()):
                raise AssertionError(f"Host profile is inside protected path: {protected}")

        with patch.dict(os.environ, {"AUTORESEARCH_HARNESS_PROFILE": str(profile_path)}):
            yield profile_path


def run_durable_operation(config: Any, command: str, params: dict[str, Any]) -> dict[str, Any]:
    """Reserve and execute through the real SQLite worker in same-process host mode."""
    with approved_offline_profile(config):
        payload = {
            "command": command,
            "params": {**params, "confirmed": True},
        }
        operation = reserve_and_launch_operation(
            config,
            command,
            f"test-{command}-{uuid.uuid4()}",
            payload,
            # Use the supported same-process embedding launcher; the real worker
            # still claims the queued row and rechecks the saved policy snapshot.
            lambda _operation_id: os.getpid(),
        )
        result = run_operation_worker(str(operation["operation_id"]))
        stored = SQLiteStore().get_operation(operation["operation_id"])
        if stored.get("status") != "succeeded":
            raise AssertionError(f"Durable {command} operation did not succeed: {stored!r}")
        if not isinstance(result, dict):
            raise AssertionError(f"Durable {command} operation returned a non-object result: {result!r}")
        return result


def run_baseline(config: Any, *, supersede: bool = False, reason: str = "") -> dict[str, Any]:
    params: dict[str, Any] = {}
    if supersede:
        params["supersede"] = True
    if reason:
        params["reason"] = reason
    return run_durable_operation(config, "baseline", params)


def run_trial(config: Any, hypothesis: dict[str, Any]) -> dict[str, Any]:
    return run_durable_operation(config, "trial", {"hypothesis": hypothesis})


def run_probe(
    config: Any,
    commit: str,
    env: dict[str, str],
    group: str,
    label: str,
    repeats: int,
) -> dict[str, Any]:
    return run_durable_operation(config, "probe", {
        "commit": commit,
        "env": env,
        "group": group,
        "label": label,
        "repeats": repeats,
    })


def run_replay(config: Any, trial_id: str) -> dict[str, Any]:
    return run_durable_operation(config, "replay", {"trial_id": trial_id})
