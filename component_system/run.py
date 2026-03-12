"""Seed -> P -> DCA daemon for the component-system web app."""
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

from component_system.domain.models import StageName
from component_system.services.workflow import BASELINE_SEED_ID, WorkflowService
from component_system.task import (
    BASELINE_BRANCHES_PATH,
    BASELINE_METRICS_PATH,
    COMPONENT_SYSTEM_ROOT,
    claim_pending,
    DAEMON_HEARTBEAT_PATH,
    daemon_heartbeat,
    ensure_queue_layout,
    LOG_ROOT,
    move_to_done,
    move_to_error,
    read_task,
    restore_in_progress_tasks,
)

PROJECT_ROOT = COMPONENT_SYSTEM_ROOT.parent
LOG_DIR = LOG_ROOT
RESULTS_TSV = PROJECT_ROOT / "results.tsv"
PROGRESS_PNG = PROJECT_ROOT / "progress.png"

POLL_INTERVAL = 10.0
_shutdown = False
WORKFLOW = WorkflowService()

DEFAULT_TIMEOUTS = {"p": 900, "dca": 3600, "direct": 3600}

# Canonical DCA entrypoint run: require ≥600s so training can complete. Agent must set command/tool timeout ≥ this.
DCA_CANONICAL_RUN_TIMEOUT_SECONDS = 600

STAGE_DOCS = {
    "p": ["PDCA-PLAN.md"],
    "dca": ["PDCA-DO-CHECK-ACTION.md"],
}

AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude": {"cmd": ["claude", "-p", "--verbose"], "via": "stdin"},
    "codex": {"cmd": ["codex", "exec", "-a", "never", "--sandbox", "workspace-write"], "via": "arg"},
    "opencode": {"cmd": ["opencode", "run"], "via": "arg"},
}


def _signal_handler(_sig: int, _frame: Any) -> None:
    global _shutdown
    _shutdown = True
    print("\n[daemon] shutdown requested")


def _get_timeout(stage: str) -> int:
    return int(os.environ.get(f"PDCA_TIMEOUT_{stage.upper()}", DEFAULT_TIMEOUTS.get(stage, 600)))


def _build_log_paths(run_id: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{run_id}.stdout.log"
    stderr_path = LOG_DIR / f"{run_id}.stderr.log"
    return stdout_path, stderr_path


def _write_prompt_file(run_id: str, prompt: str) -> Path:
    """Save the agent prompt to a file for debugging. Returns the path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = LOG_DIR / f"{run_id}.prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def _is_root_venv_active() -> bool:
    expected = (PROJECT_ROOT / ".venv").resolve()
    active = os.environ.get("VIRTUAL_ENV")
    if not active:
        return False
    try:
        return Path(active).resolve() == expected
    except OSError:
        return False


def _dca_command_guidance() -> tuple[str, str]:
    timeout_prefix = f"timeout {DCA_CANONICAL_RUN_TIMEOUT_SECONDS}"
    if _is_root_venv_active():
        return (
            f"{timeout_prefix} uv run --active component_system/entrypoint.py",
            "Root .venv is active; use --active to reuse it from the worktree.",
        )
    return (
        f"{timeout_prefix} uv run component_system/entrypoint.py",
        "No active root .venv detected; fallback avoids --active so uv can run normally.",
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
            piece = pipe.read(16)
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


def _agent_failure_reason(exit_code: int, stdout: str, stderr: str) -> str:
    combined = _combined_output(stdout, stderr)
    if "timeout after " in combined:
        return combined.strip().splitlines()[-1]
    if exit_code == -1:
        if combined.strip():
            return combined.strip().splitlines()[-1]
        return "Agent execution failed before completion. See stdout/stderr logs for details."
    return f"Agent exited with code {exit_code}. See stdout/stderr logs for details."


def _should_salvage_completed_dca(stage: str, exit_code: int, output_text: str) -> bool:
    """Accept a DCA run when canonical metrics were printed despite agent exit issues."""
    if stage != "dca" or exit_code == 0:
        return False
    summary = WORKFLOW.extract_summary(output_text, StageName.dca) or {}
    metrics = WORKFLOW.extract_dca_metrics(output_text, summary)
    return metrics.get("val_bpb") is not None


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
    """Copy baseline_metrics.json and baseline_branches.json from project component_system into the worktree.
    Worktrees check out from baseline-branch; without this sync the agent would see stale or missing baseline data."""
    resolved = _resolve_worktree_path(worktree_path)
    if resolved is None:
        return
    dest_dir = resolved / "component_system"
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
    Call before invoking the agent when cwd is a worktree (P or DCA)."""
    _sync_results_tsv_into_worktree(worktree_path)
    _sync_baseline_json_into_worktree(worktree_path)


def _invoke_agent(
    prompt: str, stage: str, run_id: str, worktree_path: str | None = None
) -> tuple[int, str, str, Path | None, Path | None]:
    agent_name = os.environ.get("PDCA_AGENT", "claude")
    config = AGENT_CONFIGS.get(agent_name)
    if config is None:
        raise ValueError(f"Unknown PDCA_AGENT={agent_name!r}. Supported: {', '.join(AGENT_CONFIGS)}")

    cmd = list(config["cmd"])
    timeout = _get_timeout(stage)
    cwd = _agent_cwd(worktree_path)
    # PYTHONUNBUFFERED=1 so child Python (e.g. uv run entrypoint.py) flushes stdout
    # immediately instead of block-buffering when stdout is a pipe; otherwise
    # stdout log only appears in one shot after the task finishes.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
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
        msg = f"{agent_name!r} binary not found. Install it or set PDCA_AGENT to a different backend."
        return -1, "", msg, None, None

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
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()

        stdout_thread.join()
        stderr_thread.join()

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    if timed_out:
        timeout_message = f"timeout after {timeout}s"
        if stderr:
            stderr = f"{stderr}\n{timeout_message}"
        else:
            stderr = timeout_message
        return -1, stdout, stderr, stdout_path, stderr_path

    return process.returncode, stdout, stderr, stdout_path, stderr_path


def _build_merge_resolution_prompt(task: dict[str, Any]) -> str:
    """Lightweight prompt for merge-resolution DCA: no protocol/docs, just commit, merge, report."""
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
            "3. Do not run the training entrypoint; the experiment already completed and metrics exist.\n"
            "4. Print the DCA summary block below (same metrics as the previous run). Include the current commit SHA (after committing the merge) in the DCA summary JSON.\n\n"
        )
    else:
        # Normal seed: merge the baseline branch (__baseline__) INTO the seed worktree so the seed is up to date.
        if worktree_path:
            cwd_note = (
                "Your working directory is the project root. "
                f"The seed worktree is at {worktree_path!r}; run git commands from that directory (e.g. cd there first).\n\n"
            )
        else:
            cwd_note = (
                "Your working directory is the project root. "
                f"The seed worktree is at component_system/history/worktrees/{seed_id!r}; run git commands from that directory for the merge.\n\n"
            )
        steps = (
            "Steps:\n"
            "1. Commit any uncommitted changes in the seed worktree (e.g. batch-size or other fixes).\n"
            f"2. In the seed worktree, merge the baseline branch into the current branch: git merge {BASELINE_SEED_ID!r}. Resolve any conflicts, then commit the merge.\n"
            "3. Do not run the training entrypoint; the experiment already completed and metrics exist.\n"
            "4. Print the DCA summary block below (same metrics as the previous run). Include the current commit SHA (after committing the merge) in the DCA summary JSON.\n\n"
        )

    return (
        "MERGE RESOLUTION (focused task). Do not read protocol or stage docs.\n\n"
        "Task (inline):\n"
        f"{task_json}\n\n"
        f"{cwd_note}"
        f"{steps}"
        "AUTORESEARCH_DCA_SUMMARY_BEGIN\n"
        f"{report_json}\n"
        "AUTORESEARCH_DCA_SUMMARY_END\n"
    )


def _build_prompt(stage: str, task: dict[str, Any], task_path: Path) -> str:
    """Build the agent prompt for a stage. Prompt types (by weight):
    - P: full header (protocol, stage doc, baseline files, task) + P workflow. Heavy.
    - DCA metrics_recovery: full header + log-recovery instructions. Heavy.
    - DCA merge_resolution: lightweight; task + commit, merge, report (no protocol/docs). Light.
    - DCA baseline_measurement: full header + baseline retry/OOM/commit/run. Heavy.
    - DCA normal: full header + adapt/run/commit/report. Heavy.
    """
    task_json = json.dumps(task, indent=2)
    rel_task = task_path.relative_to(PROJECT_ROOT).as_posix()
    worktree_path = task.get("worktree_path", "component_system/history/worktrees")
    agent_cwd = _agent_cwd(worktree_path)
    worktree_dir = Path(agent_cwd)

    # Worktree runs must stay entirely within the copied seed workspace to avoid external_directory requests.
    if worktree_dir.resolve() != PROJECT_ROOT.resolve():
        context_protocol = "  - component_system/protocol.md"
        docs = "\n".join(f"  - component_system/{doc}" for doc in STAGE_DOCS[stage])
        task_block = (
            "Task content (provided inline; do not look up any external task file):\n"
            f"{task_json}\n\n"
        )
        worktree_note = (
            "Your working directory is the assigned workflow worktree (your current directory).\n"
            "All required file context is already copied into this worktree under component_system/.\n"
            "Use only paths relative to your current working directory. "
            "Do not request access to absolute paths, parent-directory paths, or files outside the worktree.\n"
        )
    else:
        context_protocol = "  - component_system/protocol.md"
        docs = "\n".join(f"  - component_system/{doc}" for doc in STAGE_DOCS[stage])
        task_path_rel = f"  - {rel_task}"
        task_block = f"Task file:\n{task_path_rel}\n\nTask content:\n{task_json}\n\n"
        worktree_note = "Your working directory is the project root.\n"

    required_context = (
        "Required context (read first; paths relative to your cwd):\n"
        f"  - component_system/protocol.md\n"
        f"{docs}\n"
    )
    baseline_files_note = (
        "Baseline reference files (workflow-managed; read-only):\n"
        "  - component_system/baseline_branches.json (per-branch baseline mapping)\n"
        "  - component_system/baseline_metrics.json (baseline run metrics)\n"
        "The workflow writes these; only read them for context.\n\n"
    )
    header = (
        "You are working on the autoresearch component-system workflow.\n\n"
        f"{required_context}\n"
        f"{baseline_files_note}"
        f"{task_block}"
        f"{worktree_note}"
        "Do not edit files outside the worktree unless the prompt explicitly requires it.\n\n"
    )

    if stage == "p":
        return header + (
            "You are the P stage.\n\n"
            "## Read results.tsv first (avoid idea duplication)\n"
            "Before choosing a hypothesis, read `results.tsv` in your cwd if it exists. "
            "Use it to avoid proposing ideas already tried or discarded; only repeat an idea if you have a clear new angle (e.g. different implementation or target component). "
            "See component_system/PDCA-PLAN.md for full guidance.\n\n"
            "Workflow:\n"
            "1. Refine the seed prompt into a concrete implementation idea.\n"
            "2. Implement the first generated version of that idea in the provided worktree.\n"
            "3. Create a git commit in the seed branch (current branch in the worktree).\n"
            "4. Print a JSON summary between these exact markers:\n"
            "AUTORESEARCH_P_SUMMARY_BEGIN\n"
            '{"idea":"...","target_component":"model|optimizer|trainer","description":"...","source_refs":["..."],"commit_sha":"...","completed_at":"YYYY-MM-DD HH:MM:SS"}\n'
            "AUTORESEARCH_P_SUMMARY_END\n"
            "One branch per seed: you are already on the seed branch in the worktree.\n"
            "Do not merge branches; only the DCA stage may trigger a merge into baseline.\n"
        )
    if stage == "dca":
        merge_resolution = task.get("merge_resolution") is True
        metrics_recovery = task.get("metrics_recovery") is True
        if merge_resolution:
            return _build_merge_resolution_prompt(task)
        dca_cmd, dca_note = _dca_command_guidance()
        baseline_measurement = task.get("seed_id") == "__baseline__"
        conflict_block = ""
        if metrics_recovery:
            source_run_id = task.get("source_run_id", "unknown")
            stdout_log = task.get("source_stdout_log_path", "missing")
            stderr_log = task.get("source_stderr_log_path", "missing")
            return header + (
                "METRICS RECOVERY: The previous DCA run completed, but the runner could not confirm metrics from its final report.\n"
                "Do not rerun training. Do not edit code. Do not create a commit.\n"
                f"Inspect the saved logs for source run {source_run_id!r}:\n"
                f"- stdout log: {stdout_log}\n"
                f"- stderr log: {stderr_log}\n"
                "Recover the canonical metrics from those logs if they are present, then print the final JSON summary.\n"
                "Use this exact shape:\n"
                "AUTORESEARCH_DCA_SUMMARY_BEGIN\n"
                '{"checks":["log_metrics_recovery"],"notes":"Recovered metrics from saved logs.","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"","metrics":{"val_bpb":1.239972,"training_seconds":300.1,"total_seconds":360.4,"startup_seconds":25.8,"peak_vram_mb":11967.8,"mfu_percent":2.15,"total_tokens_M":140.5,"num_steps":268,"num_params_M":11.5,"depth":4}}\n'
                "AUTORESEARCH_DCA_SUMMARY_END\n"
                "If you still cannot recover metrics, print the same object with an empty metrics object and explain why in notes.\n"
            )
        if baseline_measurement:
            return header + conflict_block + (
                "BASELINE MEASUREMENT: establish the first reference metrics in the dedicated baseline worktree.\n"
                "You must retry until the run completes successfully and you can report real metrics. Do not report empty metrics and stop.\n"
                "If training fails with CUDA out of memory (OOM): the default batch size is for H100. Reduce device_batch_size (and if needed total_batch_size) in component_system/components/trainer.py (TrainingSettings) so training fits in available VRAM, then rerun until the baseline run completes. Only trivial execution fixes (e.g. batch size) are allowed; do not change model architecture or training logic.\n"
                "If you modified any files (e.g. batch size for OOM), you must commit those changes on the baseline branch before reporting. An uncommitted worktree causes the follow-up merge to fail.\n"
                f"Run the canonical command (≥{DCA_CANONICAL_RUN_TIMEOUT_SECONDS}s): {dca_cmd}\n"
                f"({dca_note}) When you invoke this command, set your command/tool timeout to at least {DCA_CANONICAL_RUN_TIMEOUT_SECONDS} seconds so the process is not killed early.\n"
                "Report the final result in JSON between these exact markers once training has completed successfully. Include the current commit SHA in the summary (commit any changes first).\n"
                "AUTORESEARCH_DCA_SUMMARY_BEGIN\n"
                '{"checks":["baseline_measurement"],"notes":"Measured the current baseline in the dedicated baseline worktree.","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"...","metrics":{"val_bpb":1.239972,"training_seconds":300.1,"total_seconds":360.4,"startup_seconds":25.8,"peak_vram_mb":11967.8,"mfu_percent":2.15,"total_tokens_M":140.5,"num_steps":268,"num_params_M":11.5,"depth":4}}\n'
                "AUTORESEARCH_DCA_SUMMARY_END\n"
                "If after all retries (including batch size reduction) metrics are still unavailable, only then print the same object with an empty metrics object and explain in notes.\n"
            )
        return header + conflict_block + (
            "You are the DCA stage.\n"
            "Do not put forward new ideas or optimize for better metrics. Your only goal is to make the P-stage code run and report the result. "
            '"Adapt or fix" means: fix bugs, import/runtime errors, OOM (e.g. reduce batch size), and config/path issues only. '
            "Do not change model architecture, optimizer logic, hyperparameters, or training logic to improve results. "
            "The task \"prompt\" is for context only; do not treat it as a goal to achieve in this stage.\n\n"
            "Workflow:\n"
            "1. Adapt or fix the generated code in the seed worktree until it runs.\n"
            f"2. Run the canonical command (≥{DCA_CANONICAL_RUN_TIMEOUT_SECONDS}s): {dca_cmd}\n"
            f"   ({dca_note}) When you invoke this command, set your command/tool timeout to at least {DCA_CANONICAL_RUN_TIMEOUT_SECONDS} seconds so the process is not killed early.\n"
            "3. If it fails for a simple reason, fix and rerun.\n"
            "4. Create a git commit in the seed branch for your changes.\n"
            "5. Report the final result in JSON between these exact markers. Include the current commit SHA in the summary.\n"
            "   Use this exact shape and include numeric metric values when available:\n"
            "AUTORESEARCH_DCA_SUMMARY_BEGIN\n"
            '{"checks":["entrypoint"],"notes":"what you adapted or fixed","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"...","metrics":{"val_bpb":1.239972,"training_seconds":300.1,"total_seconds":360.4,"startup_seconds":25.8,"peak_vram_mb":11967.8,"mfu_percent":2.15,"total_tokens_M":140.5,"num_steps":268,"num_params_M":11.5,"depth":4}}\n'
            "AUTORESEARCH_DCA_SUMMARY_END\n"
            "   Do not omit the markers. Prefer this exact JSON report over prose. If metrics are unavailable,\n"
            "   still print the same object with an empty metrics object.\n"
            "Do not edit baseline_branches.json or baseline_metrics.json (workflow writes them; read only). Do not merge branches yourself; the system will evaluate and promote if appropriate.\n"
        )
    raise ValueError(f"Unknown stage: {stage}")


def _append_results_tsv(seed_id: str, run_metrics: dict[str, Any], signal: str, description: str) -> None:
    status = "KEEP" if signal == "positive_signal" else "DISCARD"
    val_bpb = run_metrics.get("val_bpb", "")
    peak_vram_mb = run_metrics.get("peak_vram_mb", 0)
    memory_gb = round(float(peak_vram_mb) / 1024, 2) if peak_vram_mb else ""
    write_header = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a", encoding="utf-8") as handle:
        if write_header:
            handle.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
        handle.write(f"{seed_id}\t{val_bpb}\t{memory_gb}\t{status}\t{description}\n")


def _regenerate_progress_png() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    if not RESULTS_TSV.exists():
        return

    try:
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
        df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
        df["status"] = df["status"].str.strip().str.upper()
        valid = df[df["val_bpb"].notna()].copy().reset_index(drop=True)
        if valid.empty:
            return

        baseline_bpb = valid.loc[0, "val_bpb"]
        kept = valid[valid["status"] == "KEEP"]
        best = float(kept["val_bpb"].min()) if not kept.empty else float(baseline_bpb)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(valid.index, valid["val_bpb"], c="#94a3b8", s=18, alpha=0.6, label="Runs")
        if not kept.empty:
            ax.scatter(kept.index, kept["val_bpb"], c="#38bdf8", s=42, label="Promoted")
            ax.step(kept.index, kept["val_bpb"].cummin(), where="post", color="#0ea5e9", linewidth=2)
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("Validation BPB (lower is better)")
        ax.set_title("Component System Progress")
        margin = (baseline_bpb - best) * 0.15 if baseline_bpb != best else 0.005
        ax.set_ylim(best - margin, float(baseline_bpb) + margin)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
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
            started_seed = None
            if stage == "direct":
                started_seed, _ = WORKFLOW.mark_direct_code_run_started(seed_id, run_id)
            else:
                started_seed, _ = WORKFLOW.mark_run_started(seed_id, run_id)
                if (
                    stage == "dca"
                    and task.get("metrics_recovery") is not True
                ):
                    started_seed = WORKFLOW.ensure_seed_worktree_ready(seed_id)
            print(f"[{stage.upper()}] picked up {task['task_id']} for {seed_id}")

            worktree_path = task.get("worktree_path")
            if started_seed is not None and started_seed.worktree_path is not None:
                worktree_path = started_seed.worktree_path
            # Merge-resolution DCA runs from project root so the agent can operate on repo and worktrees
            if stage == "dca" and (
                task.get("merge_resolution") is True or task.get("metrics_recovery") is True
            ):
                worktree_path = None

            if worktree_path:
                _sync_worktree_context(worktree_path)

            if stage == "direct":
                prompt = _build_direct_code_prompt(task["prompt"])
            else:
                prompt = _build_prompt(stage, task, task_path)
            prompt_path = _write_prompt_file(run_id, prompt)
            prompt_path_str = str(prompt_path)
            exit_code, stdout, stderr, stdout_log_path, stderr_log_path = _invoke_agent(
                prompt, stage, run_id, worktree_path=worktree_path
            )

            combined_output = _combined_output(stdout, stderr)
            salvaged_dca = _should_salvage_completed_dca(stage, exit_code, combined_output)
            if exit_code == 0 or salvaged_dca:
                if stage == "p":
                    WORKFLOW.finish_p_run(
                        seed_id,
                        run_id,
                        stdout,
                        str(stdout_log_path) if stdout_log_path else None,
                        str(stderr_log_path) if stderr_log_path else None,
                        prompt_path_str,
                    )
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
                else:
                    run = WORKFLOW.finish_dca_run(
                        seed_id,
                        run_id,
                        stdout,
                        stderr=stderr,
                        log_path=str(stdout_log_path) if stdout_log_path else None,
                        stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                        prompt_path=prompt_path_str,
                        metrics_recovery=task.get("metrics_recovery") is True,
                        merge_resolution=task.get("merge_resolution") is True,
                    )
                    if not run.summary.get("metrics_recovery_queued"):
                        description = run.summary.get("notes") or run.summary.get("idea") or seed_id
                        _append_results_tsv(seed_id, run.metrics, run.signal or "error", str(description))
                        _regenerate_progress_png()
                    if salvaged_dca:
                        WORKFLOW.seed_repo.append_event(
                            seed_id,
                            "dca.salvaged",
                            f"DCA output contained final metrics, so the run was accepted despite agent exit code {exit_code}.",
                            run_id=run_id,
                        )
                move_to_done(task_path)
                print(f"[{stage.upper()}] task {task['task_id']} done")
            else:
                if stage == "direct":
                    WORKFLOW.mark_direct_code_run_failed(
                        seed_id,
                        run_id,
                        _agent_failure_reason(exit_code, stdout, stderr),
                        task_path=task_path,
                        prompt_path=prompt_path_str,
                        log_path=str(stdout_log_path) if stdout_log_path else None,
                        stderr_log_path=str(stderr_log_path) if stderr_log_path else None,
                    )
                else:
                    WORKFLOW.mark_run_failed(
                        seed_id,
                        run_id,
                        _agent_failure_reason(exit_code, stdout, stderr),
                        task_path=task_path, prompt_path=prompt_path_str,
                    )
                print(f"[{stage.upper()}] task {task['task_id']} failed")
        except Exception as exc:
            traceback.print_exc()
            if not task_path.exists():
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
                        seed_id, run_id, str(exc),
                        task_path=task_path, prompt_path=prompt_path_str,
                    )
            except Exception:
                traceback.print_exc()

    print(f"[daemon] worker-{worker_name} stopped")


def main() -> None:
    global _shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    ensure_queue_layout()
    restored = restore_in_progress_tasks()
    total_restored = sum(restored.values())
    if total_restored:
        print(
            "[daemon] restored in_progress tasks "
            f"(p={restored['p']}, dca={restored['dca']}, direct={restored['direct']})"
        )
    daemon_heartbeat()
    agent = os.environ.get("PDCA_AGENT", "claude")
    print(f"[daemon] starting component-system daemon — agent={agent}, workers=P/DCA-GPU/DCA-AUX/DIRECT")

    pools: list[ThreadPoolExecutor] = []
    stage_specs = (
        ("p", "any", 2, "pdca-p"),
        ("dca", "gpu", 1, "pdca-dca-gpu"),
        ("dca", "aux", 1, "pdca-dca-aux"),
        ("direct", "any", 1, "pdca-direct"),
    )
    for stage, lane, worker_count, prefix in stage_specs:
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
