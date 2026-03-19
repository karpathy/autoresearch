"""Seed -> PD -> CA daemon for the pdca-system web app."""
from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdca_system.config import (
    TARGET_METRIC_KEY,
    TARGET_METRIC_LABEL,
    TARGET_METRIC_LOWER_IS_BETTER,
)
from pdca_system.services.workflow import (
    BASELINE_SEED_ID,
    DuplicateRunStartError,
    SyncResolutionQueued,
    WorkflowService,
)
from pdca_system.task import (
    BASELINE_BRANCHES_PATH,
    BASELINE_METRICS_PATH,
    DEFAULT_DAEMON_CONFIG,
    IN_PROGRESS_DIR,
    list_runs,
    PDCA_SYSTEM_ROOT,
    claim_pending,
    DAEMON_HEARTBEAT_PATH,
    daemon_heartbeat,
    ensure_queue_layout,
    get_daemon_config,
    load_run,
    load_seed,
    LOG_ROOT,
    move_to_done,
    move_to_error,
    now_ts,
    read_task,
    restore_in_progress_tasks,
    save_run,
)

PROJECT_ROOT = PDCA_SYSTEM_ROOT.parent
LOG_DIR = LOG_ROOT
RESULTS_TSV = PROJECT_ROOT / "results.tsv"
PROGRESS_PNG = PROJECT_ROOT / "progress.png"

POLL_INTERVAL = 10.0
_shutdown = False
WORKFLOW = WorkflowService()
# Set at startup when tasks are restored; printed once when a worker actually resumes (so log order is natural).
_restored_summary: str | None = None
_restored_summary_lock = threading.Lock()

STAGE_DOCS = {
    "pd": ["PDCA-Plan-Do.md"],
    "ca": ["PDCA-Check-Action.md"],
}

STAGE_SPECS = (
    ("pd", "any", 3, "pdca-pd"),
    ("ca", "gpu", 1, "pdca-ca-gpu"),
    ("ca", "aux", 1, "pdca-ca-aux"),
    ("direct", "any", 1, "pdca-direct"),
)

AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude": {"cmd": ["claude", "-p", "--verbose"], "via": "stdin"},
    "codex": {"cmd": ["codex", "exec", "-a", "never", "--sandbox", "workspace-write"], "via": "arg"},
    "gemini": {"cmd": ["gemini", "--yolo", "-p"], "via": "arg"},
    "opencode": {"cmd": ["opencode", "run"], "via": "arg"},
    "kimi": {"cmd": ["kimi", "--yolo", "-p"], "via": "arg"},
}

# Probe commands used at daemon startup to detect which CLIs are installed and responsive.
# `--version` is preferred; fallback probes keep detection robust across heterogeneous CLIs.
AGENT_VERSION_PROBES: dict[str, list[list[str]]] = {
    "claude": [["--version"]],
    "codex": [["--version"], ["-v"]],
    "gemini": [["--version"], ["-v"]],
    "opencode": [["--version"], ["-v"], ["version"]],
    "kimi": [["--version"], ["-v"], ["version"]],
}

AGENT_DETECT_TIMEOUT_SECONDS = 8.0
_AGENT_HEALTH_LOCK = threading.Lock()
_AGENT_RUNTIME_STATE: dict[str, Any] = {
    "available": [],
    "unavailable": {},
    "active": None,
    "nonzero_streak": {},
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


AGENT_DOWNGRADE_NONZERO_STREAK = _env_int("PDCA_AGENT_DOWNGRADE_AFTER", 5)
# Runs shorter than this are treated as abnormal and count toward agent downgrade streak.
# Keep below typical small-dataset run times (e.g. 20–30s) so valid short runs are not flagged.
AGENT_MIN_RUNTIME_SECONDS = _env_int("PDCA_AGENT_MIN_RUNTIME_SECONDS", 15)

# Stuck-check: run backup agent to confirm previous agent was stuck before switching.
# Default; env PDCA_STUCK_CHECK_TIMEOUT overrides; else value from history/state/daemon_config.json.
STUCK_CHECK_LOG_TAIL_LINES = 100


def _stuck_check_timeout_seconds() -> int:
    """Stuck-check timeout: env override, else config from history folder, else 120."""
    env_val = os.environ.get("PDCA_STUCK_CHECK_TIMEOUT")
    if env_val is not None:
        try:
            v = int(env_val)
            if v > 0:
                return v
        except ValueError:
            pass
    return get_daemon_config().get("stuck_check_timeout_seconds", 120)


def _run_agent_probe(cmd: list[str], timeout: float = AGENT_DETECT_TIMEOUT_SECONDS) -> tuple[bool, str]:
    run_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "check": False,
    }
    if sys.platform == "win32":
        run_kwargs["env"] = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    try:
        proc = subprocess.run(cmd, **run_kwargs)
    except (FileNotFoundError, OSError) as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired:
        return False, f"probe timeout after {timeout:.0f}s"
    output = _combined_output(proc.stdout, proc.stderr).strip()
    if proc.returncode == 0:
        return True, output.splitlines()[0] if output else "ok"
    return False, output.splitlines()[-1] if output else f"exit {proc.returncode}"


def _detect_available_agents(preferred_agent: str) -> tuple[list[str], dict[str, str]]:
    available: list[str] = []
    unavailable: dict[str, str] = {}

    # Keep operator preference first when present, then probe all known backends.
    ordered = [preferred_agent, *[name for name in AGENT_CONFIGS if name != preferred_agent]]
    for agent_name in ordered:
        config = AGENT_CONFIGS.get(agent_name)
        if config is None:
            unavailable[agent_name] = "unknown agent config"
            continue
        binary = config["cmd"][0]
        binary_path = shutil.which(binary)
        if binary_path is None:
            unavailable[agent_name] = f"binary {binary!r} not found on PATH"
            continue

        probes = AGENT_VERSION_PROBES.get(agent_name, [["--version"], ["-v"]])
        passed = False
        last_reason = ""
        for probe_args in probes:
            ok, reason = _run_agent_probe([binary, *probe_args])
            if ok:
                passed = True
                break
            last_reason = reason
        if passed:
            available.append(agent_name)
        else:
            unavailable[agent_name] = (
                f"version probe failed ({binary_path}): {last_reason or 'unknown error'}"
            )
    return available, unavailable


def _initialize_agent_runtime_state() -> None:
    preferred_agent = os.environ.get("PDCA_AGENT", "claude")
    available, unavailable = _detect_available_agents(preferred_agent)
    if not available:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        raise RuntimeError(
            "No supported code agents detected. Install at least one supported CLI "
            f"or adjust PATH. Detection details: {reasons}"
        )

    with _AGENT_HEALTH_LOCK:
        _AGENT_RUNTIME_STATE["available"] = list(available)
        _AGENT_RUNTIME_STATE["unavailable"] = dict(unavailable)
        _AGENT_RUNTIME_STATE["active"] = available[0]
        _AGENT_RUNTIME_STATE["nonzero_streak"] = {name: 0 for name in available}


def _active_agent_name() -> str:
    with _AGENT_HEALTH_LOCK:
        available = list(_AGENT_RUNTIME_STATE.get("available", []))
        active = _AGENT_RUNTIME_STATE.get("active")
        if not available:
            raise RuntimeError("No available code agent backend.")
        if active not in available:
            active = available[0]
            _AGENT_RUNTIME_STATE["active"] = active
        return str(active)


def _mark_agent_unavailable(agent_name: str, reason: str) -> None:
    with _AGENT_HEALTH_LOCK:
        available = list(_AGENT_RUNTIME_STATE.get("available", []))
        if agent_name in available:
            available.remove(agent_name)
        _AGENT_RUNTIME_STATE["available"] = available
        unavailable = dict(_AGENT_RUNTIME_STATE.get("unavailable", {}))
        unavailable[agent_name] = reason
        _AGENT_RUNTIME_STATE["unavailable"] = unavailable
        streak = dict(_AGENT_RUNTIME_STATE.get("nonzero_streak", {}))
        streak.pop(agent_name, None)
        _AGENT_RUNTIME_STATE["nonzero_streak"] = streak
        if _AGENT_RUNTIME_STATE.get("active") == agent_name:
            _AGENT_RUNTIME_STATE["active"] = available[0] if available else None


def _rotate_active_agent(reason: str) -> None:
    with _AGENT_HEALTH_LOCK:
        available = list(_AGENT_RUNTIME_STATE.get("available", []))
        if len(available) <= 1:
            return
        current = _AGENT_RUNTIME_STATE.get("active")
        try:
            idx = available.index(current)
        except ValueError:
            idx = -1
        next_agent = available[(idx + 1) % len(available)]
        if next_agent != current:
            _AGENT_RUNTIME_STATE["active"] = next_agent
            print(f"[daemon] agent downgrade: {current} -> {next_agent} ({reason})")


def _record_agent_exit(agent_name: str, exit_code: int) -> bool:
    """Update non-zero streak for agent. Returns True if caller should run stuck-check before switching (streak >= threshold and at least 2 agents). If only one agent, always returns False (continue on same agent)."""
    with _AGENT_HEALTH_LOCK:
        streak = dict(_AGENT_RUNTIME_STATE.get("nonzero_streak", {}))
        if agent_name not in streak:
            return False
        if exit_code == 0:
            streak[agent_name] = 0
        else:
            streak[agent_name] = int(streak.get(agent_name, 0)) + 1
        _AGENT_RUNTIME_STATE["nonzero_streak"] = streak
        current_streak = streak[agent_name]
        available = list(_AGENT_RUNTIME_STATE.get("available", []))
    if exit_code == 0 or current_streak < AGENT_DOWNGRADE_NONZERO_STREAK:
        return False
    if len(available) <= 1:
        return False
    return True


def _build_stuck_check_prompt(
    prev_agent: str,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> str:
    """Build prompt for backup agent: read tail of previous agent logs and ask to report if previous agent was stuck."""
    def tail(path: Path | None, n: int) -> str:
        if path is None or not path.exists():
            return "(no log file)"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:]) if lines else "(empty)"
        except OSError:
            return "(read error)"

    stdout_tail = tail(stdout_path, STUCK_CHECK_LOG_TAIL_LINES)
    stderr_tail = tail(stderr_path, STUCK_CHECK_LOG_TAIL_LINES)
    return (
        "You are the backup code agent. The previous agent run used the agent named "
        f"{prev_agent!r}. It exited with a non-zero or error state. Below are the last "
        f"{STUCK_CHECK_LOG_TAIL_LINES} lines of its stdout and stderr.\n\n"
        "--- STDOUT (tail) ---\n"
        f"{stdout_tail}\n\n"
        "--- STDERR (tail) ---\n"
        f"{stderr_tail}\n\n"
        "Decide whether the previous agent was stuck (e.g. quota, rate limit, auth failure, unrecoverable error). "
        f"Write your conclusion to the file named {SUMMARY_FILENAME} in your current working directory. "
        "Use this exact JSON shape and no other output:\n"
        '{"previous_agent_stuck": true or false, "reason": "brief explanation", "checks": ["stuck_check"]}\n'
        "Do not run any other tasks. Only write this file."
    )


def _run_stuck_check(
    candidate_agent: str,
    prompt: str,
    run_id: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Run candidate agent with stuck-check prompt; return (success, summary_dict). success=True when a valid summary with previous_agent_stuck key exists (even if process exited non-zero, e.g. stderr encoding)."""
    _invoke_agent(
        prompt,
        "stuck_check",
        run_id,
        worktree_path=None,
        agent_override=candidate_agent,
        timeout_override=_stuck_check_timeout_seconds(),
    )
    summary_path = Path(PROJECT_ROOT) / SUMMARY_FILENAME
    if not summary_path.exists():
        return False, None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, None
    if "previous_agent_stuck" not in summary:
        return False, None
    return True, summary


def _maybe_rotate_after_stuck_check(
    prev_agent: str,
    run_id: str,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> None:
    """If backup agent confirms previous agent was stuck, switch; else reset streak and continue on same agent."""
    with _AGENT_HEALTH_LOCK:
        available = list(_AGENT_RUNTIME_STATE.get("available", []))
    if len(available) <= 1:
        return
    try:
        idx = available.index(prev_agent)
    except ValueError:
        return
    candidate = available[(idx + 1) % len(available)]
    prompt = _build_stuck_check_prompt(prev_agent, stdout_path, stderr_path)
    stuck_check_run_id = f"{run_id}.stuck_check"
    print(f"[daemon] running stuck-check with {candidate!r} (run_id={stuck_check_run_id})")
    ok, summary = _run_stuck_check(candidate, prompt, stuck_check_run_id)
    if not ok or not summary:
        print("[daemon] stuck-check failed or no summary; not switching agent")
        with _AGENT_HEALTH_LOCK:
            streak = dict(_AGENT_RUNTIME_STATE.get("nonzero_streak", {}))
            if prev_agent in streak:
                streak[prev_agent] = 0
                _AGENT_RUNTIME_STATE["nonzero_streak"] = streak
        return
    stuck = summary.get("previous_agent_stuck") is True
    reason = summary.get("reason", "")
    if stuck:
        _rotate_active_agent(f"stuck check confirmed: {reason!s}")
    else:
        print(f"[daemon] stuck-check says previous agent not stuck ({reason!s}); not switching")
        with _AGENT_HEALTH_LOCK:
            streak = dict(_AGENT_RUNTIME_STATE.get("nonzero_streak", {}))
            if prev_agent in streak:
                streak[prev_agent] = 0
                _AGENT_RUNTIME_STATE["nonzero_streak"] = streak


def _signal_handler(_sig: int, _frame: Any) -> None:
    global _shutdown
    _shutdown = True
    print("\n[daemon] shutdown requested")


def _get_timeout(stage: str) -> int:
    env_val = os.environ.get(f"PDCA_TIMEOUT_{stage.upper()}")
    if env_val is not None:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    timeouts = get_daemon_config().get("default_timeouts", DEFAULT_DAEMON_CONFIG["default_timeouts"])
    return int(timeouts.get(stage, DEFAULT_DAEMON_CONFIG["default_timeouts"].get(stage, 900)))


def _build_log_paths(run_id: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{run_id}.stdout.log"
    stderr_path = LOG_DIR / f"{run_id}.stderr.log"
    return stdout_path, stderr_path


# Fixed filename in agent cwd; daemon reads from agent_cwd / SUMMARY_FILENAME.
SUMMARY_FILENAME = "autoresearch_summary.json"


def _summary_json_path_in_cwd(worktree_path: str | None) -> Path:
    """Path where the agent must write the run summary: fixed filename in agent cwd (worktree root or project root)."""
    cwd = _agent_cwd(worktree_path)
    return Path(cwd) / SUMMARY_FILENAME


def _write_prompt_file(run_id: str, prompt: str) -> Path:
    """Save the agent prompt to a file for debugging. Returns the path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = LOG_DIR / f"{run_id}.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def _build_resume_prompt(run_id: str, stage: str) -> str:
    """Build a resume prompt from the original prompt plus saved stdout/stderr so the agent can continue from where it left off after daemon restart."""
    def _read_log(path: Path, fallback: str = "") -> str:
        if not path or not path.exists():
            return fallback
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return fallback

    def _strip_daemon_log_header(text: str) -> str:
        """Remove the daemon-written header lines (stage/agent/timestamp and blank line) from a log so the resume block shows mainly agent output."""
        if not text:
            return text
        lines = text.splitlines()
        while lines and (
            lines[0].startswith("stage:") or lines[0].startswith("agent:") or lines[0].startswith("timestamp:") or not lines[0].strip()
        ):
            lines.pop(0)
        return "\n".join(lines).strip()

    prompt_path = LOG_DIR / f"{run_id}.prompt.txt"
    stdout_path = LOG_DIR / f"{run_id}.stdout.log"
    stderr_path = LOG_DIR / f"{run_id}.stderr.log"

    original_prompt = _read_log(prompt_path, "")
    stdout_content = _strip_daemon_log_header(_read_log(stdout_path, "(no stdout saved)"))
    stderr_content = _strip_daemon_log_header(_read_log(stderr_path, "(no stderr saved)"))

    resume_block = (
        "\n\n--- RESUMING (daemon was stopped; output so far) ---\n\n"
        "The previous run was interrupted when the daemon stopped. Below is the original prompt and the stdout/stderr captured before the interruption.\n\n"
        "Continue from where you left off. Complete the task as described in the original prompt and write the same summary JSON output as required (e.g. "
        f"file named {SUMMARY_FILENAME} in your current working directory). Do not repeat work already reflected in the output below.\n\n"
        "--- Stdout so far ---\n"
        f"{stdout_content}\n\n"
        "--- Stderr so far ---\n"
        f"{stderr_content}\n\n"
        "--- End of previous output; continue from here ---\n\n"
    )

    if not original_prompt:
        return (
            f"You are resuming an interrupted {stage.upper()} run (run_id={run_id}). "
            "The original prompt was not saved. Use the stdout/stderr below to infer what was in progress and complete the task. "
            f"Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory when done.\n\n"
            f"{resume_block}"
        )
    return original_prompt + resume_block


def _ca_command_guidance() -> tuple[str, str, str]:
    """Return (python_exe, note, runner_label) for the CA prompt. Only the Python executable is given; the project's canonical script (e.g. train.py) is defined in protocol/docs."""
    python_exe = sys.executable
    if " " in python_exe:
        python_exe_quoted = f'"{python_exe}"'
    else:
        python_exe_quoted = python_exe
    return (
        python_exe_quoted,
        "Daemon is running with this Python; use it in the worktree so the same environment is used.",
        sys.executable,
    )


def _build_direct_code_prompt(prompt: str) -> str:
    return (
        "You are running as a direct code agent from the project root of this repository.\n"
        "Execute the user's request directly in the current working tree.\n"
        "Do not switch into seed worktrees for this task.\n\n"
        "User request:\n"
        f"{prompt.strip()}\n"
    )


def _stream_pipe_to_file(pipe: Any, handle: Any, chunks: list[str]) -> None:
    try:
        while True:
            piece = pipe.readline()
            if not piece:
                break
            chunks.append(piece)
            handle.write(piece)
            handle.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _combined_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


# Agent-specific: when exit_code is 0, treat as failure for downgrade if output matches these.
# Sources (web-confirmed):
#   opencode: user run stderr "Error: You've reached your usage limit..."; GitHub anomalyco/opencode #3525, #877 (quota/usage limit).
#   claude:   GitHub anthropics/claude-code #27336 "API Error: Rate limit reached".
#   codex:    GitHub openai/codex #12114 "Invalid Responses API request" (OpenRouter/custom providers).
#   gemini:   GitHub google-gemini/gemini-cli #6125 "Exiting with code 0 on API Error"; headless schema "error"/"ApiError".
#   kimi:     GitHub MoonshotAI/kimi-cli #583, openclaw #41158 (rate limit / quota).
def _opencode_exit0_is_failure(stderr: str, stdout: str) -> bool:
    # OpenCode (with Kimi/Claude/etc.) can exit 0 while stderr has "Error: You've reached your usage limit..."
    s = (stderr or "").strip()
    if not s:
        return False
    lower = s.lower()
    if "error:" not in lower:
        return False
    return (
        "usage limit" in lower
        or "quota" in lower
        or "billing cycle" in lower
        or "upgrade to get more" in lower
        or "rate limit" in lower
    )


def _claude_exit0_is_failure(stderr: str, stdout: str) -> bool:
    # Claude CLI: "API Error: Rate limit reached" and similar in stderr
    s = ((stderr or "") + "\n" + (stdout or "")).lower()
    return "api error" in s and ("rate limit" in s or "rate limit reached" in s)


def _codex_exit0_is_failure(stderr: str, stdout: str) -> bool:
    # Codex: "Invalid Responses API request" when using OpenRouter/custom providers
    s = ((stderr or "") + "\n" + (stdout or "")).lower()
    return "invalid responses api" in s or "invalid response" in s


def _gemini_exit0_is_failure(stderr: str, stdout: str) -> bool:
    # Gemini headless: sometimes exits 0 on API error (issue #6125); stderr "Error when talking to Gemini API" or JSON "error"/ApiError
    s = ((stderr or "") + "\n" + (stdout or "")).lower()
    if "error when talking to gemini api" in s:
        return True
    return '"error"' in s and ("apierror" in s or "autherror" in s or "auth error" in s)


def _kimi_exit0_is_failure(stderr: str, stdout: str) -> bool:
    # Kimi native CLI: less documented; check for explicit error line + quota/limit
    s = (stderr or "").strip().lower()
    if "error" not in s:
        return False
    return "quota" in s or "usage limit" in s or "rate limit" in s


AGENT_EXIT0_FAILURE_CHECKS: dict[str, Any] = {
    "opencode": _opencode_exit0_is_failure,
    "claude": _claude_exit0_is_failure,
    "codex": _codex_exit0_is_failure,
    "gemini": _gemini_exit0_is_failure,
    "kimi": _kimi_exit0_is_failure,
}


def _effective_exit_for_downgrade(
    agent_name: str, exit_code: int, stderr: str, stdout: str
) -> int:
    """Return non-zero when we should treat this run as a failure for agent-rotation.
    Uses agent-specific checks for exit-0-but-error-output (e.g. opencode quota)."""
    if exit_code != 0:
        return exit_code
    check = AGENT_EXIT0_FAILURE_CHECKS.get(agent_name)
    if not check or not check(stderr or "", stdout or ""):
        return 0
    return 1


def _agent_failure_reason(exit_code: int, stdout: str, stderr: str) -> str:
    combined = _combined_output(stdout, stderr)
    if "timeout after " in combined:
        return combined.strip().splitlines()[-1]
    if exit_code == -1:
        if combined.strip():
            return combined.strip().splitlines()[-1]
        return "Agent execution failed before completion. See stdout/stderr logs for details."
    return f"Agent exited with code {exit_code}. See stdout/stderr logs for details."


def _should_salvage_completed_ca(
    stage: str, exit_code: int, run_id: str, worktree_path: str | None
) -> bool:
    """Accept a CA run when the summary file exists in agent cwd and contains the target metric despite agent exit code."""
    if stage != "ca" or exit_code == 0:
        return False
    path = _summary_json_path_in_cwd(worktree_path)
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        return summary.get("metrics", {}).get(TARGET_METRIC_KEY) is not None
    except (json.JSONDecodeError, OSError):
        return False


def _agent_cwd(worktree_path: str | None) -> str:
    """Resolve cwd for the agent: seed worktree when provided and present, else project root."""
    if not worktree_path:
        return str(PROJECT_ROOT)
    path = Path(worktree_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    return str(resolved) if resolved.is_dir() else str(PROJECT_ROOT)


def _resolve_worktree_path(worktree_path: str | None) -> Path | None:
    """Resolve worktree path to absolute Path, or None if invalid/missing."""
    if not worktree_path:
        return None
    path = Path(worktree_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    return resolved if resolved.is_dir() else None


def _sync_results_tsv_into_worktree(worktree_path: str | None) -> None:
    """Copy the latest root results.tsv into the seed worktree if it exists. Non-fatal on failure."""
    resolved = _resolve_worktree_path(worktree_path)
    if resolved is None or not RESULTS_TSV.exists():
        return
    dest = resolved / "results.tsv"
    try:
        shutil.copy2(RESULTS_TSV, dest)
    except OSError as err:
        print(f"[P] could not copy results.tsv into worktree: {err}", file=sys.stderr)


def _sync_baseline_json_into_worktree(worktree_path: str | None) -> None:
    """Copy baseline_metrics.json and baseline_branches.json from project pdca_system into the worktree.
    Worktrees check out from baseline-branch; without this sync the agent would see stale or missing baseline data."""
    resolved = _resolve_worktree_path(worktree_path)
    if resolved is None:
        return
    dest_dir = resolved / "pdca_system"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_path, name in [
        (BASELINE_METRICS_PATH, "baseline_metrics.json"),
        (BASELINE_BRANCHES_PATH, "baseline_branches.json"),
    ]:
        if not src_path.exists():
            continue
        dest = dest_dir / name
        try:
            shutil.copy2(src_path, dest)
        except OSError as err:
            print(f"[P] could not copy {name} into worktree: {err}", file=sys.stderr)


def _sync_worktree_context(worktree_path: str | None) -> None:
    """Sync all workflow-managed live data into the worktree so the agent sees current state.
    Call before invoking the agent when cwd is a worktree (PD or CA)."""
    _sync_results_tsv_into_worktree(worktree_path)
    _sync_baseline_json_into_worktree(worktree_path)


def _invoke_agent(
    prompt: str,
    stage: str,
    run_id: str,
    worktree_path: str | None = None,
    agent_override: str | None = None,
    timeout_override: int | None = None,
) -> tuple[str, int, str, str, Path | None, Path | None]:
    attempted_agents: set[str] = set()
    while True:
        if agent_override is not None:
            agent_name = agent_override
            if agent_name in attempted_agents:
                return (
                    agent_name,
                    -1,
                    "",
                    "Agent failed to launch (override). See daemon logs for details.",
                    None,
                    None,
                )
            attempted_agents.add(agent_name)
        else:
            agent_name = _active_agent_name()
            if agent_name in attempted_agents:
                return (
                    agent_name,
                    -1,
                    "",
                    "All detected agent backends failed to launch. See daemon logs for details.",
                    None,
                    None,
                )
            attempted_agents.add(agent_name)
        config = AGENT_CONFIGS.get(agent_name)
        if config is None:
            if agent_override is None:
                _mark_agent_unavailable(agent_name, "missing config entry")
            else:
                return (agent_name, -1, "", f"Unknown agent {agent_name!r}", None, None)
            continue

        cmd = list(config["cmd"])
        timeout = timeout_override if timeout_override is not None else _get_timeout(stage)
        cwd = _agent_cwd(worktree_path)
        # PYTHONUNBUFFERED=1 so child Python (e.g. uv run train.py) flushes stdout
        # immediately instead of block-buffering when stdout is a pipe; otherwise
        # stdout log only appears in one shot after the task finishes.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if sys.platform == "win32":
            # Force Python-based agents (e.g. kimi) to use UTF-8 stdio so box-drawing
            # and other Unicode don't trigger 'gbk' encode errors in Git Bash / CJK locales.
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
        if agent_name == "opencode":
            project_root_glob = str(PROJECT_ROOT.resolve().as_posix()) + "/**"
            existing = {}
            try:
                if os.environ.get("OPENCODE_PERMISSION"):
                    existing = json.loads(os.environ["OPENCODE_PERMISSION"])
            except (json.JSONDecodeError, KeyError):
                pass
            ext_dir = dict(existing.get("external_directory", {}))
            ext_dir[project_root_glob] = "allow"
            env["OPENCODE_PERMISSION"] = json.dumps({"external_directory": ext_dir})
        popen_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if config["via"] == "stdin":
            popen_kwargs["stdin"] = subprocess.PIPE
        else:
            # Use DEVNULL so the agent never reads from parent's stdin (avoids EBADF under nohup/redirects).
            popen_kwargs["stdin"] = subprocess.DEVNULL
            cmd.append(prompt)

        print(f"[{stage.upper()}] invoking {agent_name} (timeout={timeout}s)")
        stdout_path, stderr_path = _build_log_paths(run_id)
        try:
            process = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError:
            if agent_override is None:
                _mark_agent_unavailable(agent_name, "binary not found during launch")
                continue
            return (agent_name, -1, "", f"{agent_name!r} binary not found", None, None)
        except OSError as exc:
            if agent_override is None:
                _mark_agent_unavailable(agent_name, f"launch error: {exc}")
                continue
            return (agent_name, -1, "", str(exc), None, None)

        if config["via"] == "stdin" and process.stdin is not None:
            process.stdin.write(prompt)
            process.stdin.close()

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_handle:
            stdout_handle.write(f"stage:     {stage.upper()}\nagent:     {agent_name}\n")
            stdout_handle.write(f"timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n\n")
            stdout_handle.flush()
            stderr_handle.write(f"stage:     {stage.upper()}\nagent:     {agent_name}\n")
            stderr_handle.write(f"timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n\n")
            stderr_handle.flush()

            stdout_thread = threading.Thread(
                target=_stream_pipe_to_file,
                args=(process.stdout, stdout_handle, stdout_chunks),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_stream_pipe_to_file,
                args=(process.stderr, stderr_handle, stderr_chunks),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            timed_out = False
            start_time = time.monotonic()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()

            stdout_thread.join()
            stderr_thread.join()
            elapsed_seconds = time.monotonic() - start_time

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        if timed_out:
            timeout_message = f"timeout after {timeout}s"
            if stderr:
                stderr = f"{stderr}\n{timeout_message}"
            else:
                stderr = timeout_message
            if agent_override is None:
                if _record_agent_exit(agent_name, -1):
                    _maybe_rotate_after_stuck_check(agent_name, run_id, stdout_path, stderr_path)
            return agent_name, -1, stdout, stderr, stdout_path, stderr_path

        ret = int(process.returncode)
        if agent_override is None:
            effective = _effective_exit_for_downgrade(agent_name, ret, stderr, stdout)
            if elapsed_seconds < AGENT_MIN_RUNTIME_SECONDS:
                effective = effective or 1
            if _record_agent_exit(agent_name, effective):
                _maybe_rotate_after_stuck_check(agent_name, run_id, stdout_path, stderr_path)
        return agent_name, ret, stdout, stderr, stdout_path, stderr_path


def _build_metrics_recovery_prompt(task: dict[str, Any]) -> str:
    """Lightweight prompt for metrics-recovery CA: no protocol/docs, just task, log paths, report shape."""
    task_json = json.dumps(task, indent=2)
    source_run_id = task.get("source_run_id", "unknown")
    stdout_log = task.get("source_stdout_log_path", "missing")
    stderr_log = task.get("source_stderr_log_path", "missing")
    report_json = json.dumps({
        "checks": ["log_metrics_recovery"],
        "notes": "Recovered metrics from saved logs.",
        "completed_at": "YYYY-MM-DD HH:MM:SS",
        "commit_sha": "",
        "metrics": {
            TARGET_METRIC_KEY: 1.24,
            "training_seconds": 300.1,
            "total_seconds": 360.4,
            "startup_seconds": 25.8,
            "peak_vram_mb": 11967.8,
            "mfu_percent": 2.15,
            "total_tokens_M": 140.5,
            "num_steps": 268,
            "num_params_M": 11.5,
            "depth": 4,
        },
    }, indent=2)
    return (
        "METRICS RECOVERY (focused task). Do not read protocol or stage docs.\n\n"
        "Task (inline):\n"
        f"{task_json}\n\n"
        "Do not rerun training. Do not edit code. Do not create a commit.\n\n"
        f"Inspect logs for source run {source_run_id!r}:\n"
        f"  stdout: {stdout_log}\n"
        f"  stderr: {stderr_log}\n\n"
        f"Recover canonical metrics from those logs if present (the metrics object must include the target metric key {TARGET_METRIC_KEY!r}). "
        "If unrecoverable, use empty \"metrics\": {} and explain in notes.\n\n"
        f"Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory (cwd root). "
        "Do not print this JSON to stdout or stderr. Use this shape (reference):\n\n"
        f"{report_json}\n"
    )


def _build_sync_resolution_prompt(task: dict[str, Any]) -> str:
    """Prompt for sync-resolution: merge baseline into seed in the seed worktree, resolve conflicts, commit."""
    baseline_branch = task.get("baseline_branch", "master")
    seed_id = task.get("seed_id", "")
    return (
        "SYNC RESOLUTION (merge baseline into seed). You are in the seed worktree; the current branch is the seed branch.\n\n"
        "The run could not sync this worktree with the baseline branch because the merge had conflicts.\n\n"
        "Steps:\n"
        f"1. Merge the baseline branch into the current branch: git merge {baseline_branch!r}\n"
        "2. Resolve any conflicts, then commit the merge (e.g. git add . && git commit -m 'Merge baseline into seed').\n"
        "3. Do not run the training script (train.py).\n"
        f"4. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory (cwd root). "
        "Do not print this JSON to stdout or stderr. Use this shape (reference):\n"
        '{"checks":["sync_resolution"],"notes":"Merged baseline into seed; conflicts resolved.","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"<git rev-parse --short HEAD>","metrics":{}}\n'
    )


def _build_merge_resolution_prompt(task: dict[str, Any]) -> str:
    """Lightweight prompt for merge-resolution CA: no protocol/docs, just commit, merge, report."""
    task_json = json.dumps(task, indent=2)
    target_branch = task.get("baseline_branch", "master")  # branch we want to merge into (e.g. master)
    worktree_path = task.get("worktree_path") or ""
    seed_id = task.get("seed_id", "")
    last_metrics = task.get("last_metrics") or {}
    last_summary = task.get("last_summary") or {}
    notes = last_summary.get("notes", "Merge resolution: committed and merged into baseline.")
    completed_at = last_summary.get("completed_at", "YYYY-MM-DD HH:MM:SS")
    report_json = json.dumps({
        "checks": ["merge_resolution"],
        "notes": notes,
        "completed_at": completed_at,
        "commit_sha": "",
        "metrics": last_metrics,
    }, indent=2)

    if seed_id == BASELINE_SEED_ID:
        # We are resolving the merge of __baseline__ INTO target_branch (e.g. master).
        # git merge X = merge X into current branch; so we need to be on target_branch, then git merge __baseline__.
        cwd_note = (
            "Your working directory is the project root (main repo). "
            "Do NOT run the merge from the __baseline__ worktree: that would merge the wrong way.\n\n"
        )
        steps = (
            "Steps:\n"
            f"1. Find where {target_branch!r} is checked out: run git worktree list and identify the path whose branch is {target_branch!r} (often the main repo).\n"
            f"2. cd to that directory, then run: git merge {BASELINE_SEED_ID!r}. Resolve any conflicts, then commit the merge.\n"
            f"   Correct example (merge __baseline__ into {target_branch}):\n"
            f"     git worktree list\n"
            f"     cd <path-with-{target_branch}>   # e.g. main repo\n"
            f"     git merge {BASELINE_SEED_ID!r}\n"
            "   Wrong (do not do this): cd to the __baseline__ worktree and run git merge master — that merges master into __baseline__.\n"
            "3. Do not run the training script (train.py); the experiment already completed and metrics exist.\n"
            f"4. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory (same metrics as the previous run; metrics must include the target metric key {TARGET_METRIC_KEY!r}). Include the current commit SHA (after committing the merge). Do not print this JSON to stdout or stderr.\n\n"
        )
    else:
        # Normal seed: we need to merge the SEED branch INTO the baseline branch (so baseline gets the seed's changes).
        # Do NOT merge baseline into seed — that is the wrong direction. Work in the project root on the baseline branch.
        if worktree_path:
            cwd_note = (
                "Your working directory is the project root. "
                f"The seed worktree is at {worktree_path!r} (use it only to commit any pending changes on the seed branch).\n\n"
            )
        else:
            cwd_note = (
                "Your working directory is the project root. "
                f"The seed worktree is at pdca_system/history/worktrees/{seed_id!r} (use it only to commit any pending changes).\n\n"
            )
        steps = (
            "Steps:\n"
            "1. Commit any uncommitted changes in the seed worktree so the seed branch is complete.\n"
            f"2. In the project root (main repo): checkout the baseline branch, then merge the seed branch into it:\n"
            f"   git checkout {target_branch!r}\n"
            f"   git merge {seed_id!r}\n"
            "   Resolve any conflicts, then commit the merge. The result must be: the baseline branch contains the seed's changes (merge direction: seed → baseline).\n"
            "3. Do not run the training script (train.py); the experiment already completed and metrics exist.\n"
            f"4. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory (same metrics as the previous run; metrics must include the target metric key {TARGET_METRIC_KEY!r}). Use the merge commit SHA from the baseline branch (after the merge, from project root: git rev-parse HEAD). Do not print this JSON to stdout or stderr.\n\n"
        )

    return (
        "MERGE RESOLUTION (focused task). Do not read protocol or stage docs.\n\n"
        "Task (inline):\n"
        f"{task_json}\n\n"
        f"{cwd_note}"
        f"{steps}"
        "Use this shape (reference) for the summary JSON:\n\n"
        f"{report_json}\n"
    )


def _build_prompt(stage: str, task: dict[str, Any], task_path: Path) -> str:
    """Build the agent prompt for a stage. Prompt types (by weight):
    - PD: full header (protocol, stage doc, baseline files, task) + PD workflow. Heavy.
    - CA metrics_recovery: lightweight; task + log paths, report shape (no protocol/docs). Light.
    - CA merge_resolution: lightweight; task + commit, merge, report (no protocol/docs). Light.
    - CA baseline_measurement: full header + baseline retry/OOM/commit/run. Heavy.
    - CA normal: full header + adapt/run/commit/report. Heavy.
    """
    task_json = json.dumps(task, indent=2)
    rel_task = task_path.relative_to(PROJECT_ROOT).as_posix()
    worktree_path = task.get("worktree_path", "pdca_system/history/worktrees")
    agent_cwd = _agent_cwd(worktree_path)
    worktree_dir = Path(agent_cwd)

    # Worktree runs must stay entirely within the copied seed workspace to avoid external_directory requests.
    in_worktree = worktree_dir.resolve() != PROJECT_ROOT.resolve()
    if in_worktree:
        docs = "\n".join(f"  - pdca_system/{doc}" for doc in STAGE_DOCS[stage])
        task_block = (
            "Task content (provided inline; do not look up any external task file):\n"
            f"{task_json}\n\n"
        )
        worktree_note = (
            "Your working directory is the assigned workflow worktree (your current directory).\n"
            "All required file context is already copied into this worktree under pdca_system/.\n"
            "Use only paths relative to your current working directory. "
            "Do not request access to absolute paths, parent-directory paths, or files outside the worktree.\n"
        )
        scope_note = "Do not edit files outside the worktree unless the prompt explicitly requires it.\n\n"
    else:
        docs = "\n".join(f"  - pdca_system/{doc}" for doc in STAGE_DOCS[stage])
        task_path_rel = f"  - {rel_task}"
        task_block = f"Task file:\n{task_path_rel}\n\nTask content:\n{task_json}\n\n"
        worktree_note = "Your working directory is the project root.\n"
        scope_note = "Do not edit files outside your current directory (project root) unless the prompt explicitly requires it.\n\n"

    required_context = (
        "Required context (read first; paths relative to your cwd):\n"
        f"  - pdca_system/protocol.md\n"
        f"{docs}\n"
    )
    baseline_files_note = (
        "Baseline reference files (workflow-managed; read-only):\n"
        "  - pdca_system/baseline_branches.json (per-branch baseline mapping)\n"
        "  - pdca_system/baseline_metrics.json (baseline run metrics)\n"
        "The workflow writes these; only read them for context.\n\n"
    )
    header = (
        "You are working on the autoresearch pdca-system workflow.\n\n"
        f"{required_context}\n"
        f"{baseline_files_note}"
        f"{task_block}"
        f"{worktree_note}"
        f"{scope_note}"
    )

    if stage == "pd":
        if in_worktree:
            p_workflow = (
                "Workflow:\n"
                "1. Refine the seed prompt into a concrete implementation idea.\n"
                "2. Implement the first generated version of that idea in the provided worktree.\n"
                "3. Create a git commit in the seed branch (current branch in the worktree).\n"
                f"4. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory. "
                "Do not print this JSON to stdout or stderr. Use this shape (reference): "
                '{"idea":"...","target_component":"model|optimizer|trainer","description":"...","source_refs":["..."],"commit_sha":"...","completed_at":"YYYY-MM-DD HH:MM:SS"}\n'
                "One branch per seed: you are already on the seed branch in the worktree.\n"
                "Do not merge branches; only the CA stage may trigger a merge into baseline.\n"
            )
        else:
            p_workflow = (
                "Workflow:\n"
                "1. Refine the seed prompt into a concrete implementation idea.\n"
                "2. Implement the first generated version of that idea in the current directory (project root).\n"
                "3. Create a git commit on the current branch.\n"
                f"4. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory. "
                "Do not print this JSON to stdout or stderr. Use this shape (reference): "
                '{"idea":"...","target_component":"model|optimizer|trainer","description":"...","source_refs":["..."],"commit_sha":"...","completed_at":"YYYY-MM-DD HH:MM:SS"}\n'
                "One branch per seed: you are in the project root; use the current branch for your commit.\n"
                "Do not merge branches; only the Check-Action stage may trigger a merge into baseline.\n"
            )
        return header + (
            "You are working on the Plan-Do stage.\n\n"
            f"The Check-Action stage will run your code and require the target metric key {TARGET_METRIC_KEY!r} in the summary JSON file it writes.\n\n"
            "Read results.tsv first (avoid idea duplication)\n"
            "Before choosing a hypothesis, read `results.tsv` in your cwd if it exists. "
            "Use it to avoid proposing ideas already tried or discarded; only repeat an idea if you have a clear new angle (e.g. different implementation or target component). "
            "See pdca_system/PDCA-Plan-Do.md for full guidance.\n\n"
            f"{p_workflow}"
        )
    if stage == "ca":
        sync_resolution = task.get("sync_resolution") is True
        merge_resolution = task.get("merge_resolution") is True
        metrics_recovery = task.get("metrics_recovery") is True
        if sync_resolution:
            return _build_sync_resolution_prompt(task)
        if merge_resolution:
            return _build_merge_resolution_prompt(task)
        if metrics_recovery:
            return _build_metrics_recovery_prompt(task)
        python_exe, ca_note, runner_label = _ca_command_guidance()
        baseline_measurement = task.get("seed_id") == "__baseline__"
        conflict_block = ""
        if baseline_measurement:
            return header + conflict_block + (
                "BASELINE MEASUREMENT: establish the first reference metrics in the dedicated baseline worktree.\n"
                "You must retry until the run completes successfully and you can report real metrics. Do not report empty metrics and stop.\n"
                "If training fails with CUDA out of memory (OOM): the default batch size is for H100. Reduce DEVICE_BATCH_SIZE (and if needed TOTAL_BATCH_SIZE) in train.py so training fits in available VRAM, then rerun until the baseline run completes. Only trivial execution fixes (e.g. batch size) are allowed; do not change model architecture or training logic.\n"
                "If you modified any files (e.g. batch size for OOM), you must commit those changes on the baseline branch before reporting. An uncommitted worktree causes the follow-up merge to fail.\n"
                f"Use this Python executable for the canonical run: `{runner_label}` ({ca_note}). Run the project's canonical command (see protocol; e.g. train.py or the script your project uses) with it, e.g. `{python_exe} <script> > training.log 2>&1`. Set your command/tool timeout to at least {_get_timeout('ca')} seconds. After the run, inspect training.log to confirm completion and recover or verify metrics.\n"
                f"Write the final result JSON to the file named {SUMMARY_FILENAME} in your current working directory once training has completed successfully. The metrics object must include the target metric key {TARGET_METRIC_KEY!r}. Include the current commit SHA in the summary (commit any changes first). Do not print this JSON to stdout or stderr. Use this shape (reference): "
                f'{{"checks":["baseline_measurement"],"notes":"Measured the current baseline in the dedicated baseline worktree.","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"...","metrics":{{"{TARGET_METRIC_KEY}":1.239972,"training_seconds":300.1,"total_seconds":360.4,"startup_seconds":25.8,"peak_vram_mb":11967.8,"mfu_percent":2.15,"total_tokens_M":140.5,"num_steps":268,"num_params_M":11.5,"depth":4}}}}\n'
                "If after all retries (including batch size reduction) metrics are still unavailable, only then write the same object with an empty metrics object and explain in notes.\n"
            )
        return header + conflict_block + (
            "You are working on the Check-Action stage.\n"
            "Do not put forward new ideas or optimize for better metrics. Your only goal is to make the Plan-Do-stage code run and report the result. "
            '"Adapt or fix" means: fix bugs, import/runtime errors, OOM (e.g. reduce batch size), and config/path issues only. '
            "Do not change model architecture, optimizer logic, hyperparameters, or training logic to improve results. "
            "The task \"prompt\" is for context only; do not treat it as a goal to achieve in this stage.\n\n"
            "Workflow:\n"
            "1. Adapt or fix the generated code in the seed worktree until it runs.\n"
            f"2. Use this Python executable for the canonical run: `{runner_label}` ({ca_note}). Run the project's canonical command (see protocol; e.g. train.py or the script your project uses) with it, e.g. `{python_exe} <script> > training.log 2>&1` (or `... 2>&1 | tee training.log` to also see output). Set your command/tool timeout to at least {_get_timeout('ca')} seconds. After the run, inspect training.log to confirm completion and recover or verify metrics.\n"
            "3. If it fails for a simple reason, fix and rerun.\n"
            "4. Create a git commit in the seed branch for your changes.\n"
            f"5. Write the final result JSON to the file named {SUMMARY_FILENAME} in your current working directory. The metrics object must include the target metric key {TARGET_METRIC_KEY!r}. Include the current commit SHA in the summary. Do not print this JSON to stdout or stderr. Use this shape (reference) and include numeric metric values when available: "
            f'{{"checks":["entrypoint"],"notes":"what you adapted or fixed","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"...","metrics":{{"{TARGET_METRIC_KEY}":1.239972,"training_seconds":300.1,"total_seconds":360.4,"startup_seconds":25.8,"peak_vram_mb":11967.8,"mfu_percent":2.15,"total_tokens_M":140.5,"num_steps":268,"num_params_M":11.5,"depth":4}}}}\n'
            "If metrics are unavailable, still write the same object with an empty metrics object.\n"
            "Do not edit baseline_branches.json or baseline_metrics.json (workflow writes them; read only). Do not merge branches yourself; the system will evaluate and promote if appropriate.\n"
        )
    raise ValueError(f"Unknown stage: {stage}")


def _sanitize_tsv_field(value: str) -> str:
    """Replace tabs and newlines so TSV columns don't break."""
    if value is None:
        return ""
    s = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return s.strip()


def _append_results_tsv(
    seed_id: str,
    run_metrics: dict[str, Any],
    signal: str,
    idea: str,
    description: str,
) -> None:
    status = "KEEP" if signal == "positive_signal" else "DISCARD"
    target_metric_val = run_metrics.get(TARGET_METRIC_KEY, "")
    peak_vram_mb = run_metrics.get("peak_vram_mb", 0)
    memory_gb = round(float(peak_vram_mb) / 1024, 2) if peak_vram_mb else ""
    write_header = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a", encoding="utf-8") as handle:
        if write_header:
            handle.write(f"commit\t{TARGET_METRIC_KEY}\tmemory_gb\tstatus\tidea\tdescription\n")
        handle.write(
            f"{seed_id}\t{target_metric_val}\t{memory_gb}\t{status}\t{_sanitize_tsv_field(idea)}\t{_sanitize_tsv_field(description)}\n"
        )


def _regenerate_progress_png() -> None:
    if not RESULTS_TSV.exists():
        return

    try:
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        if TARGET_METRIC_KEY not in df.columns:
            return
        df[TARGET_METRIC_KEY] = pd.to_numeric(df[TARGET_METRIC_KEY], errors="coerce")
        df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
        df["status"] = df["status"].str.strip().str.upper()
        valid = df[df[TARGET_METRIC_KEY].notna()].copy().reset_index(drop=True)
        if valid.empty:
            return

        col = TARGET_METRIC_KEY
        baseline_val = float(valid.loc[0, col])
        kept = valid[valid["status"] == "KEEP"]
        best = (
            float(kept[col].min() if TARGET_METRIC_LOWER_IS_BETTER else kept[col].max())
            if not kept.empty
            else baseline_val
        )
        margin = (abs(baseline_val - best) * 0.15) if baseline_val != best else 0.005
        # Only plot points in the "interesting" region (near or better than baseline), like analysis.ipynb
        tol = max(margin * 0.33, 1e-9)
        if TARGET_METRIC_LOWER_IS_BETTER:
            in_range = valid[valid[col] <= baseline_val + tol]
        else:
            in_range = valid[valid[col] >= baseline_val - tol]

        # Labels: use "idea" or "description", truncate for display (match notebook 45 chars)
        if "idea" in valid.columns:
            valid["_label"] = valid["idea"].fillna("").astype(str).str.strip()
        elif "description" in valid.columns:
            valid["_label"] = valid["description"].fillna("").astype(str).str.strip()
        else:
            valid["_label"] = ""
        valid["_label"] = valid["_label"].str[:45]

        disc = in_range[in_range["status"] == "DISCARD"]
        kept_v = in_range[in_range["status"] == "KEEP"]
        cum_series = kept[col].cummin() if TARGET_METRIC_LOWER_IS_BETTER else kept[col].cummax()

        fig, ax = plt.subplots(figsize=(16, 8))
        # Discarded as faint background (match analysis.ipynb)
        ax.scatter(
            disc.index, disc[col],
            c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded",
        )
        # Kept as prominent green dots with black edge (match analysis.ipynb)
        if not kept_v.empty:
            ax.scatter(
                kept_v.index, kept_v[col],
                c="#2ecc71", s=50, zorder=4, label="Kept",
                edgecolors="black", linewidths=0.5,
            )
        # Running best step line: always from all kept experiments
        if not kept.empty:
            ax.step(
                kept.index, cum_series, where="post",
                color="#27ae60", linewidth=2, alpha=0.7, zorder=3, label="Running best",
            )
        # Label each kept experiment with its description (match notebook: only kept, rotation 30)
        for idx, row in kept.iterrows():
            label = (row["_label"] or "").strip()
            if len(label) >= 45:
                label = label[:42] + "..."
            if not label:
                continue
            ax.annotate(
                label,
                (idx, row[col]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8.0,
                color="#1a7a3a",
                alpha=0.9,
                rotation=30,
                ha="left",
                va="bottom",
            )

        n_total = len(valid)
        n_kept = len(kept)
        direction = " (lower is better)" if TARGET_METRIC_LOWER_IS_BETTER else " (higher is better)"
        ax.set_xlabel("Experiment #", fontsize=12)
        ax.set_ylabel(f"{TARGET_METRIC_LABEL}{direction}", fontsize=12)
        ax.set_title(
            f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
            fontsize=14,
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

        if TARGET_METRIC_LOWER_IS_BETTER:
            ax.set_ylim(best - margin, baseline_val + margin)
        else:
            ax.set_ylim(baseline_val - margin, best + margin)
        plt.tight_layout()
        plt.savefig(PROGRESS_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        traceback.print_exc()


def _worker(stage: str, lane: str = "any") -> None:
    worker_name = stage.upper() if lane == "any" else f"{stage.upper()}-{lane.upper()}"
    print(f"[daemon] worker-{worker_name} started")
    def eligible(payload: dict) -> bool:
        return bool(WORKFLOW.is_seed_eligible_for_stage(payload.get("seed_id"), stage))

    while not _shutdown:
        task_path = claim_pending(stage, lane=lane, eligible_fn=eligible)
        if task_path is None:
            time.sleep(POLL_INTERVAL)
            continue

        try:
            task = read_task(task_path)
            seed_id = task.get("seed_id")
            run_id = task.get("run_id")
            if not seed_id or not run_id:
                move_to_error(task_path)
                continue

            run_dict = load_run(run_id)
            run_status = run_dict.get("status")
            is_resume = str(run_status) == "running" if run_status is not None else False

            started_seed = None
            worktree_path = task.get("worktree_path")

            if is_resume:
                # Run was interrupted (daemon stopped); resume with original prompt + saved stdout/stderr
                global _restored_summary
                with _restored_summary_lock:
                    msg = _restored_summary
                    if msg is not None:
                        _restored_summary = None
                        print(msg)
                if stage == "direct":
                    pass  # worktree_path stays None
                else:
                    seed_dict = load_seed(seed_id)
                    worktree_path = seed_dict.get("worktree_path") or worktree_path
                    if (
                        stage == "ca"
                        and task.get("metrics_recovery") is not True
                        and task.get("merge_resolution") is not True
                        and task.get("sync_resolution") is not True
                    ):
                        started_seed = WORKFLOW.ensure_seed_worktree_ready(seed_id)
                        if started_seed is not None and started_seed.worktree_path is not None:
                            worktree_path = started_seed.worktree_path
                print(f"[{stage.upper()}] resuming {task['task_id']} for {seed_id} (run was interrupted)")
            else:
                if stage == "direct":
                    started_seed, _ = WORKFLOW.mark_direct_code_run_started(seed_id, run_id)
                else:
                    started_seed, _ = WORKFLOW.mark_run_started(seed_id, run_id)
                    if (
                        stage == "ca"
                        and task.get("metrics_recovery") is not True
                    ):
                        started_seed = WORKFLOW.ensure_seed_worktree_ready(seed_id)
                if started_seed is not None and started_seed.worktree_path is not None:
                    worktree_path = started_seed.worktree_path
                print(f"[{stage.upper()}] picked up {task['task_id']} for {seed_id}")

            # Merge-resolution and metrics_recovery CA run from project root; sync_resolution runs in seed worktree
            if stage == "ca" and (
                task.get("merge_resolution") is True or task.get("metrics_recovery") is True
            ) and task.get("sync_resolution") is not True:
                worktree_path = None

            if worktree_path:
                _sync_worktree_context(worktree_path)

            if is_resume:
                prompt = _build_resume_prompt(run_id, stage)
            elif stage == "direct":
                prompt = _build_direct_code_prompt(task["prompt"])
            else:
                prompt = _build_prompt(stage, task, task_path)
            prompt_path = _write_prompt_file(run_id, prompt)
            prompt_path_str = str(prompt_path)
            # Remove any existing summary file in agent cwd before PD so this run's result is not polluted by history.
            if stage == "pd":
                summary_path_pre = _summary_json_path_in_cwd(worktree_path)
                if summary_path_pre.exists():
                    summary_path_pre.unlink()
            # Set agent_type before run (if task halts or unexpected quit, we still have which agent was attempted).
            agent_before = _active_agent_name()
            WORKFLOW.set_run_agent_type(seed_id, run_id, agent_before)
            used_agent, exit_code, stdout, stderr, stdout_log_path, stderr_log_path = _invoke_agent(
                prompt, stage, run_id, worktree_path=worktree_path
            )
            # Set again after run (resume may have changed agent; this is the agent that actually ran).
            WORKFLOW.set_run_agent_type(seed_id, run_id, used_agent)

            salvaged_ca = _should_salvage_completed_ca(stage, exit_code, run_id, worktree_path)
            summary_path = _summary_json_path_in_cwd(worktree_path)
            task_completed = False
            if exit_code == 0 or salvaged_ca:
                if stage == "pd":
                    if not summary_path.exists():
                        if not _shutdown:
                            WORKFLOW.mark_run_failed(
                                seed_id,
                                run_id,
                                f"Summary file not found. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory.",
                                task_path=task_path,
                                prompt_path=prompt_path_str,
                                log_path=str(stdout_log_path) if stdout_log_path else None,
                                stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                            )
                            move_to_error(task_path)
                        else:
                            print(f"[{stage.upper()}] task {task['task_id']} interrupted by shutdown (will resume on restart)")
                    else:
                        WORKFLOW.finish_pd_run(
                            seed_id,
                            run_id,
                            str(summary_path),
                            log_path=str(stdout_log_path) if stdout_log_path else None,
                            stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                            prompt_path=prompt_path_str,
                        )
                        task_completed = True
                elif stage == "direct":
                    WORKFLOW.finish_direct_code_run(
                        seed_id,
                        run_id,
                        stdout,
                        stderr=stderr,
                        log_path=str(stdout_log_path) if stdout_log_path else None,
                        stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                        prompt_path=prompt_path_str,
                    )
                    task_completed = True
                else:
                    if task.get("sync_resolution") is True:
                        if not summary_path.exists():
                            if not _shutdown:
                                WORKFLOW.mark_run_failed(
                                    seed_id,
                                    run_id,
                                    f"Summary file not found. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory.",
                                    task_path=task_path,
                                    prompt_path=prompt_path_str,
                                    log_path=str(stdout_log_path) if stdout_log_path else None,
                                    stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                                )
                                move_to_error(task_path)
                            else:
                                print(f"[{stage.upper()}] task {task['task_id']} interrupted by shutdown (will resume on restart)")
                        else:
                            WORKFLOW.finish_sync_resolution(seed_id, run_id)
                            task_completed = True
                    else:
                        if not summary_path.exists():
                            if not _shutdown:
                                WORKFLOW.mark_run_failed(
                                    seed_id,
                                    run_id,
                                    f"Summary file not found. Write the summary JSON to the file named {SUMMARY_FILENAME} in your current working directory.",
                                    task_path=task_path,
                                    prompt_path=prompt_path_str,
                                    log_path=str(stdout_log_path) if stdout_log_path else None,
                                    stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                                )
                                move_to_error(task_path)
                            else:
                                print(f"[{stage.upper()}] task {task['task_id']} interrupted by shutdown (will resume on restart)")
                        else:
                            run = WORKFLOW.finish_ca_run(
                                seed_id,
                                run_id,
                                str(summary_path),
                                log_path=str(stdout_log_path) if stdout_log_path else None,
                                stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                                prompt_path=prompt_path_str,
                                metrics_recovery=task.get("metrics_recovery") is True,
                                merge_resolution=task.get("merge_resolution") is True,
                            )
                            task_completed = True
                            if not run.summary.get("metrics_recovery_queued"):
                                idea = run.summary.get("idea") or run.summary.get("notes") or ""
                                description = run.summary.get("description") or ""
                                _append_results_tsv(
                                    seed_id, run.metrics, run.signal or "error", idea, description
                                )
                                _regenerate_progress_png()
                            if salvaged_ca:
                                WORKFLOW.seed_repo.append_event(
                                    seed_id,
                                    "ca.salvaged",
                                    f"CA output contained final metrics, so the run was accepted despite agent exit code {exit_code}.",
                                    run_id=run_id,
                                )
            if task_completed:
                move_to_done(task_path)
                print(f"[{stage.upper()}] task {task['task_id']} done")
            else:
                if _shutdown:
                    # Leave run as "running" and task in in_progress so restart can resume.
                    print(f"[{stage.upper()}] task {task['task_id']} interrupted by shutdown (will resume on restart)")
                else:
                    if stage == "direct":
                        WORKFLOW.mark_direct_code_run_failed(
                            seed_id,
                            run_id,
                            f"[agent={used_agent}] {_agent_failure_reason(exit_code, stdout, stderr)}",
                            task_path=task_path,
                            prompt_path=prompt_path_str,
                            log_path=str(stdout_log_path) if stdout_log_path else None,
                            stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                        )
                    else:
                        WORKFLOW.mark_run_failed(
                            seed_id,
                            run_id,
                            f"[agent={used_agent}] {_agent_failure_reason(exit_code, stdout, stderr)}",
                            task_path=task_path,
                            prompt_path=prompt_path_str,
                        )
                    print(f"[{stage.upper()}] task {task['task_id']} failed")
        except SyncResolutionQueued:
            # Sync with baseline failed; sync-resolution CA was queued. Move PD task to error so we don't retry it.
            if task_path.exists():
                move_to_error(task_path)
            continue
        except DuplicateRunStartError:
            # Run was already started and is not "running" (edge case: e.g. run status was reset). Move to error to avoid double run.
            if task_path.exists():
                move_to_error(task_path)
            continue
        except Exception as exc:
            traceback.print_exc()
            if not task_path.exists():
                continue
            if _shutdown:
                # Leave run as "running" and task in in_progress so restart can resume.
                continue
            try:
                task = read_task(task_path)
                seed_id = task.get("seed_id")
                run_id = task.get("run_id")
                if not seed_id or not run_id:
                    continue
                prompt_path_str = None
                if run_id:
                    p_path = LOG_DIR / f"{run_id}.prompt.txt"
                    if p_path.exists():
                        prompt_path_str = str(p_path)
                if stage == "direct":
                    WORKFLOW.mark_direct_code_run_failed(
                        seed_id,
                        run_id,
                        str(exc),
                        task_path=task_path,
                        prompt_path=prompt_path_str,
                    )
                else:
                    WORKFLOW.mark_run_failed(
                        seed_id,
                        run_id,
                        str(exc),
                        task_path=task_path,
                        prompt_path=prompt_path_str,
                    )
            except Exception:
                traceback.print_exc()

    print(f"[daemon] worker-{worker_name} stopped")


def _mark_orphan_running_runs_failed(restored_run_ids: list[str]) -> None:
    """Mark any runs still 'running' that are not in restored_run_ids as failed (orphaned by a previous daemon stop)."""
    restored_set = set(restored_run_ids)
    for run_dict in list_runs(seed_id=None):
        if run_dict.get("status") != "running":
            continue
        if run_dict.get("run_id") in restored_set:
            continue
        seed_id = run_dict.get("seed_id")
        run_id = run_dict.get("run_id")
        if not seed_id or not run_id:
            continue
        try:
            WORKFLOW.mark_run_failed(
                seed_id,
                run_id,
                "Run was interrupted by daemon stop (no task in queue to resume).",
            )
        except Exception:
            traceback.print_exc()


def main() -> None:
    global _shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    ensure_queue_layout()
    _initialize_agent_runtime_state()
    restored, restored_run_ids = restore_in_progress_tasks()
    total_restored = sum(restored.values())
    global _restored_summary
    if total_restored:
        _restored_summary = (
            f"[daemon] in_progress restore: {total_restored} task(s) from {IN_PROGRESS_DIR!s} -> stage queues "
            f"(pd={restored['pd']}, ca={restored['ca']}, direct={restored['direct']}); run_ids={restored_run_ids!r}"
        )
    else:
        _restored_summary = None
    restored_set = set(restored_run_ids)
    if total_restored:
        # Mark restored runs as "running" so workers treat them as resume (they may have been set to "failed" on stop/start).
        for run_id in restored_run_ids:
            run_dict = load_run(run_id)
            if run_dict and run_dict.get("status") != "succeeded":
                run_dict["status"] = "running"
                run_dict["updated_at"] = now_ts()
                save_run(run_dict)
    _mark_orphan_running_runs_failed(restored_run_ids)
    daemon_heartbeat()
    with _AGENT_HEALTH_LOCK:
        active = _AGENT_RUNTIME_STATE.get("active")
        available = list(_AGENT_RUNTIME_STATE.get("available", []))
        unavailable = dict(_AGENT_RUNTIME_STATE.get("unavailable", {}))
    print(
        "[daemon] starting pdca-system daemon - "
        f"agent={active}, available={available}, workers=PD/CA-GPU/CA-AUX/DIRECT"
    )
    if unavailable:
        for name, reason in unavailable.items():
            print(f"[daemon] agent unavailable: {name} ({reason})")

    pools: list[ThreadPoolExecutor] = []
    for stage, lane, worker_count, prefix in STAGE_SPECS:
        pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=prefix)
        pools.append(pool)
        for _ in range(worker_count):
            pool.submit(_worker, stage, lane)

    last_heartbeat = time.monotonic()
    try:
        while not _shutdown:
            time.sleep(1.0)
            if not _shutdown and (time.monotonic() - last_heartbeat) >= 5.0:
                daemon_heartbeat()
                last_heartbeat = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown = True
        if DAEMON_HEARTBEAT_PATH.exists():
            try:
                DAEMON_HEARTBEAT_PATH.unlink()
            except OSError:
                pass
        for pool in pools:
            pool.shutdown(wait=True)

    print("[daemon] all workers stopped")


if __name__ == "__main__":
    main()
