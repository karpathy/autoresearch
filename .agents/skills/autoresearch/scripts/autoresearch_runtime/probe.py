"""Isolated probe runs at arbitrary commits (parallel-safe ids)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LabConfig
from .errors import LabError
from .evaluator import _evaluate_once
from .gitutil import _git
from .measurement import next_sequence_id
from .replay import _tree_hashes
from .store import _write_json


def _next_probe_id(config: LabConfig) -> str:
    from .sqlite_store import SQLiteStore

    store = SQLiteStore()
    numbers: list[int] = []
    offset = 0
    while True:
        page = store.list_runs(config.lab_id, limit=100, offset=offset)
        numbers.extend(int(item["alias"][6:]) for item in page["items"]
                       if item.get("kind") == "probe" and item.get("alias", "").startswith("probe-")
                       and item["alias"][6:].isdigit())
        offset += len(page["items"])
        if offset >= page["total"] or not page["items"]:
            break
    return f"probe-{max(numbers, default=0) + 1:04d}"


def _probe_checkout(config: LabConfig, commit: str, checkout: Path) -> None:
    live = config.state_dir / "candidate"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    # Check if commit exists in live candidate; if missing, attempt auto-fetch from source_root
    try:
        _git(live, "cat-file", "-e", f"{commit}^{{commit}}")
    except Exception:
        source_root = config.source_root
        fetched = False
        if source_root.is_dir():
            try:
                _git(live, "fetch", "--quiet", "--no-tags", str(source_root), commit)
                fetched = True
            except Exception:
                pass
        if not fetched:
            try:
                _git(live, "cat-file", "-e", f"{commit}^{{commit}}")
            except Exception as err:
                raise LabError(
                    f"Commit '{commit}' not found in candidate repo ({live}) or source_root ({source_root}). "
                    f"Please fetch the commit into '{live}' before running probe."
                ) from err

    _git(checkout.parent, "clone", "-q",
         "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
         str(live), checkout.name)
    _git(checkout, "config", "core.autocrlf", "false")
    _git(checkout, "config", "core.safecrlf", "false")
    _git(checkout, "checkout", "-q", "--detach", commit)


def run_probe(
    config: LabConfig,
    commit: str,
    env: dict[str, str] | None,
    group: str,
    label: str,
    repeats: int,
) -> dict[str, Any]:
    from .runtimeops import require_durable_operation_context
    require_durable_operation_context(config, "probe")
    if not isinstance(commit, str) or not commit.strip():
        raise LabError("Probe needs a commit")
    if not isinstance(group, str) or not group.strip() or not isinstance(label, str) or not label.strip():
        raise LabError("Probe needs a non-empty group and label")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise LabError("Probe repeats must be a positive integer")
    live = (config.state_dir / "candidate").resolve()
    requested = commit.strip()
    if requested == "best":
        from .state import load_state

        commit = str(load_state(config)["best_commit"])
    elif requested == "baseline":
        commit = _git(live, "rev-list", "--max-parents=0", "HEAD").splitlines()[0]
    head_before = _git(live, "rev-parse", "HEAD")
    worktree_before = _git(live, "status", "--porcelain")
    probes_root = config.state_dir / "probes"
    probe_id = _next_probe_id(config)
    probe_dir = probes_root / probe_id
    checkout = probe_dir / "checkout"
    run_dir = config.state_dir / "runs" / probe_id
    summary: dict[str, Any] = {
        "schema": 1,
        "id": probe_id,
        "type": "probe",
        "group": group.strip(),
        "label": label.strip(),
        "commit": commit.strip(),
        "commit_ref": requested,
        "status": "running",
        "adoptable": False,
        "repeats": repeats,
        "runs": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    from .sqlite_store import SQLiteStore

    store = SQLiteStore()
    state = __import__("autoresearch_runtime.state", fromlist=["load_state"]).load_state(config)
    run_uuid = store.create_run(config.lab_id, "probe", probe_id,
                                int(state.get("current_epoch") or 1),
                                {"candidate_commit": state.get("best_commit")})
    summary["run_id"] = str(run_uuid)
    summary["epoch"] = int(state.get("current_epoch") or 1)
    _write_json(run_dir / "manifest.json", summary)
    try:
        _probe_checkout(config, commit.strip(), checkout)
        expected = _tree_hashes(checkout)
        runs: list[dict[str, Any]] = []
        for index in range(repeats):
            repeat_dir = run_dir / f"repeat_{index + 1:02d}"
            status, result, reason = _evaluate_once(
                config, repeat_dir, config.timeout_seconds, candidate=checkout, extra_env=env,
            )
            if status != "ok":
                summary.update(status=status, reason=reason)
                _write_json(run_dir / "manifest.json", summary)
                return summary
            if _tree_hashes(checkout) != expected:
                summary.update(status="invalid", reason="checkout changed during evaluation")
                _write_json(run_dir / "manifest.json", summary)
                return summary
            runs.append(result)
        summary.update(status="complete", runs=runs)
        metrics = [run.get("metrics", {}) for run in runs if isinstance(run.get("metrics"), dict)]
        if metrics and config.primary:
            values = [float(item[config.primary]) for item in metrics if config.primary in item]
            if values:
                summary["primary"] = config.primary
                summary["direction"] = config.direction
                summary["worst"] = min(values) if config.direction == "max" else max(values)
                summary["mean"] = sum(values) / len(values)
                summary["spread"] = max(values) - min(values)
        _write_json(run_dir / "manifest.json", summary)
        return summary
    finally:
        if _git(live, "rev-parse", "HEAD") != head_before or _git(live, "status", "--porcelain") != worktree_before:
            raise LabError("Probe touched the live candidate")


def adopt_probe(config: LabConfig, probe_id_or_commit: str) -> dict[str, Any]:
    """Probe evidence cannot replace a snapshot verified by trial gates."""
    raise LabError(
        "Probe adoption cannot establish a verified baseline or replace best",
        code="E_CONTRACT_PROBE_ADOPTION",
        hint="Apply the proposed change to candidate, then run trial --hypothesis <file> against the verified baseline",
    )

