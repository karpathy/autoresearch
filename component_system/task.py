"""Shared queue and JSON state helpers for the component-system web app."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

COMPONENT_SYSTEM_ROOT = Path(__file__).resolve().parent
HISTORY_ROOT = COMPONENT_SYSTEM_ROOT / "history"
QUEUE_ROOT = HISTORY_ROOT / "queue"
STATE_ROOT = HISTORY_ROOT / "state"
SEEDS_ROOT = STATE_ROOT / "seeds"
RUNS_ROOT = STATE_ROOT / "runs"
EVENTS_ROOT = STATE_ROOT / "events"
BASELINE_BRANCHES_PATH = COMPONENT_SYSTEM_ROOT / "baseline_branches.json"
BASELINE_METRICS_PATH = COMPONENT_SYSTEM_ROOT / "baseline_metrics.json"
WORKTREE_ROOT = HISTORY_ROOT / "worktrees"
LOG_ROOT = HISTORY_ROOT / "logs"

STAGE_DIRS = {
    "p": QUEUE_ROOT / "p",
    "dca": QUEUE_ROOT / "dca",
    "direct": QUEUE_ROOT / "direct",
}
IN_PROGRESS_DIR = QUEUE_ROOT / "in_progress"
DONE_DIR = QUEUE_ROOT / "done"
ERROR_DIR = QUEUE_ROOT / "error"
DAEMON_HEARTBEAT_PATH = STATE_ROOT / "daemon_heartbeat.json"
DAEMON_HEARTBEAT_STALE_SECONDS = 5

def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def daemon_heartbeat() -> None:
    """Write the daemon heartbeat file (call from the daemon process)."""
    ensure_queue_layout()
    _write_json(
        DAEMON_HEARTBEAT_PATH,
        {"timestamp": now_ts(), "pid": os.getpid()},
    )


def get_daemon_status() -> str:
    """Return 'running' if the daemon heartbeat is recent, else 'stopped'."""
    if not DAEMON_HEARTBEAT_PATH.exists():
        return "stopped"
    try:
        data = _read_json(DAEMON_HEARTBEAT_PATH, {})
        ts = data.get("timestamp")
        if ts is None:
            return "stopped"
        if (now_ts() - float(ts)) <= DAEMON_HEARTBEAT_STALE_SECONDS:
            return "running"
    except Exception:
        pass
    return "stopped"


def ensure_queue_layout() -> None:
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    for d in STAGE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    IN_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    SEEDS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    EVENTS_ROOT.mkdir(parents=True, exist_ok=True)
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    # Auto-create baseline JSON files if missing (like results.tsv for recording run data)
    if not BASELINE_METRICS_PATH.exists():
        _write_json(BASELINE_METRICS_PATH, {})
    if not BASELINE_BRANCHES_PATH.exists():
        _write_json(BASELINE_BRANCHES_PATH, {})


def new_task_id(prefix: str | None = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    task_id = f"{ts}-{short}"
    return f"{prefix}-{task_id}" if prefix else task_id


def new_seed_id(prefix: str = "seed") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def new_run_id(stage: str) -> str:
    return new_task_id(stage)


def write_task(stage: str, payload: dict[str, Any], task_id: str | None = None) -> Path:
    ensure_queue_layout()
    if stage not in STAGE_DIRS:
        raise KeyError(f"Unknown stage {stage!r}")
    tid = task_id or new_task_id(stage)
    path = STAGE_DIRS[stage] / f"{tid}.json"
    payload_with_meta = {"task_id": tid, "stage": stage, "created_at": now_ts(), **payload}
    return _write_json(path, payload_with_meta)


def read_task(path: Path) -> dict[str, Any]:
    return _read_json(path, {})


def move_to_done(path: Path) -> Path:
    ensure_queue_layout()
    dest = DONE_DIR / path.name
    if not path.exists():
        raise FileNotFoundError(
            f"Task file already moved: {path}; possible duplicate daemon or double completion."
        )
    if dest.exists():
        dest.unlink()
    path.rename(dest)
    return dest


def move_to_error(path: Path) -> Path:
    ensure_queue_layout()
    dest = ERROR_DIR / path.name
    if dest.exists():
        dest.unlink()
    path.rename(dest)
    return dest


def list_pending(stage: str) -> list[Path]:
    ensure_queue_layout()
    if stage not in STAGE_DIRS:
        raise KeyError(f"Unknown stage {stage!r}")
    return sorted(STAGE_DIRS[stage].glob("*.json"))


def _is_aux_dca_task(payload: dict[str, Any]) -> bool:
    return payload.get("metrics_recovery") is True or payload.get("merge_resolution") is True


def claim_pending(
    stage: str,
    lane: Literal["any", "gpu", "aux"] = "any",
    eligible_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> Path | None:
    """Atomically claim the oldest pending task for a stage/lane. If eligible_fn is set, only claim tasks for which it returns True (avoids P/DCA races)."""
    ensure_queue_layout()
    if stage not in STAGE_DIRS:
        raise KeyError(f"Unknown stage {stage!r}")
    if lane not in {"any", "gpu", "aux"}:
        raise KeyError(f"Unknown lane {lane!r}")
    for path in sorted(STAGE_DIRS[stage].glob("*.json")):
        payload = _read_json(path, {})
        if eligible_fn is not None and not eligible_fn(payload):
            continue
        if stage == "dca" and lane != "any":
            is_aux = _is_aux_dca_task(payload)
            if lane == "aux" and not is_aux:
                continue
            if lane == "gpu" and is_aux:
                continue
        claimed_path = IN_PROGRESS_DIR / path.name
        try:
            path.rename(claimed_path)
            return claimed_path
        except FileNotFoundError:
            continue
        except OSError:
            # Another worker likely claimed the task first.
            continue
    return None


def restore_in_progress_tasks() -> dict[str, int]:
    """Move stranded in-progress tasks back to their stage queue."""
    ensure_queue_layout()
    restored = {stage: 0 for stage in STAGE_DIRS}
    for path in sorted(IN_PROGRESS_DIR.glob("*.json")):
        payload = _read_json(path, {})
        stage = payload.get("stage")
        if stage not in STAGE_DIRS:
            continue
        dest = STAGE_DIRS[stage] / path.name
        if dest.exists():
            dest.unlink()
        path.rename(dest)
        restored[stage] += 1
    return restored


def seed_path(seed_id: str) -> Path:
    return SEEDS_ROOT / f"{seed_id}.json"


def run_path(run_id: str) -> Path:
    return RUNS_ROOT / f"{run_id}.json"


def event_path(seed_id: str) -> Path:
    return EVENTS_ROOT / f"{seed_id}.json"


def save_seed(seed: dict[str, Any]) -> Path:
    seed_id = seed["seed_id"]
    return _write_json(seed_path(seed_id), seed)


def load_seed(seed_id: str) -> dict[str, Any]:
    return _read_json(seed_path(seed_id), {})


def list_seeds() -> list[dict[str, Any]]:
    ensure_queue_layout()
    seeds = [_read_json(path, {}) for path in SEEDS_ROOT.glob("*.json")]
    return sorted(seeds, key=lambda item: item.get("updated_at", item.get("created_at", 0)), reverse=True)


def save_run(run: dict[str, Any]) -> Path:
    return _write_json(run_path(run["run_id"]), run)


def load_run(run_id: str) -> dict[str, Any]:
    return _read_json(run_path(run_id), {})


def list_runs(seed_id: str | None = None) -> list[dict[str, Any]]:
    ensure_queue_layout()
    runs = [_read_json(path, {}) for path in RUNS_ROOT.glob("*.json")]
    if seed_id is not None:
        runs = [run for run in runs if run.get("seed_id") == seed_id]
    return sorted(runs, key=lambda item: item.get("updated_at", item.get("created_at", 0)), reverse=True)


def append_event(seed_id: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_queue_layout()
    payload = _read_json(event_path(seed_id), [])
    payload.append({"created_at": now_ts(), "created_at_human": now_iso(), **event})
    _write_json(event_path(seed_id), payload)
    return payload


def load_events(seed_id: str) -> list[dict[str, Any]]:
    return _read_json(event_path(seed_id), [])


def delete_seed(seed_id: str) -> None:
    for path in (seed_path(seed_id), event_path(seed_id)):
        if path.exists():
            path.unlink()
    for run in list_runs(seed_id):
        path = run_path(run["run_id"])
        if path.exists():
            path.unlink()


def load_baseline_branch_map() -> dict[str, str]:
    """Load seed_id -> baseline_branch mapping (for agent lookup and workflow)."""
    ensure_queue_layout()
    return _read_json(BASELINE_BRANCHES_PATH, {})


def save_baseline_branch_map(mapping: dict[str, str]) -> None:
    """Persist seed_id -> baseline_branch mapping."""
    ensure_queue_layout()
    _write_json(BASELINE_BRANCHES_PATH, mapping)


def load_baseline_metrics() -> dict[str, dict[str, Any]]:
    """Load baseline_branch -> { last_val_bpb, promoted_branch, promoted_at, promoted_idea, commit_sha }."""
    ensure_queue_layout()
    return _read_json(BASELINE_METRICS_PATH, {})


def save_baseline_metrics(metrics_by_branch: dict[str, dict[str, Any]]) -> None:
    """Persist per-branch baseline metrics."""
    ensure_queue_layout()
    _write_json(BASELINE_METRICS_PATH, metrics_by_branch)


def reset_worktree(path: str | Path) -> None:
    worktree = Path(path)
    if worktree.exists():
        shutil.rmtree(worktree)
