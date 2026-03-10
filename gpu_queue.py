#!/usr/bin/env python3
"""
File-based GPU job queue for autoresearch.
Enables multiple agents to share a single GPU via priority-ordered time-sharing.

Usage:
    python gpu_queue.py worker                          # start queue worker (one per GPU)
    python gpu_queue.py submit --agent-id X --scale quick --command "uv run train.py"
    python gpu_queue.py status                          # show queue state
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from config import get_queue_dir, SCALE_PRIORITY

QUEUE_DIR = get_queue_dir()
JOBS_FILE = os.path.join(QUEUE_DIR, "jobs.jsonl")
ACTIVE_FILE = os.path.join(QUEUE_DIR, "active.json")
WORKER_PID_FILE = os.path.join(QUEUE_DIR, "worker.pid")


def _ensure_queue_dir():
    os.makedirs(QUEUE_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jobs() -> list[dict]:
    """Read all jobs from the jobs file."""
    if not os.path.exists(JOBS_FILE):
        return []
    jobs = []
    with open(JOBS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return jobs


def _write_jobs(jobs: list[dict]):
    """Rewrite the entire jobs file (used for status updates)."""
    _ensure_queue_dir()
    fd = os.open(JOBS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = "".join(json.dumps(j, separators=(",", ":")) + "\n" for j in jobs)
        os.write(fd, data.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _append_job(job: dict):
    """Append a single job with locking."""
    _ensure_queue_dir()
    line = json.dumps(job, separators=(",", ":")) + "\n"
    fd = os.open(JOBS_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit_job(agent_id: str, worktree_path: str, command: str,
               scale: str, env_overrides: dict = None) -> str:
    """Submit a job to the queue. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]
    job = {
        "job_id": job_id,
        "agent_id": agent_id,
        "worktree_path": worktree_path,
        "command": command,
        "scale": scale,
        "priority": SCALE_PRIORITY.get(scale, 2),
        "env": env_overrides or {},
        "submitted_at": _now_iso(),
        "status": "pending",
        "exit_code": None,
        "started_at": None,
        "finished_at": None,
    }
    _append_job(job)
    return job_id


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------

def wait_for_job(job_id: str, timeout: int = 3600, poll_interval: float = 2.0) -> int:
    """Block until job completes. Returns exit code."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = _read_jobs()
        for job in jobs:
            if job["job_id"] == job_id:
                if job["status"] in ("done", "failed"):
                    return job.get("exit_code", -1)
        time.sleep(poll_interval)
    print(f"Timeout waiting for job {job_id}")
    return -1


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _get_next_job() -> tuple:
    """Get highest-priority pending job. Returns (job, all_jobs)."""
    jobs = _read_jobs()
    pending = [(i, j) for i, j in enumerate(jobs) if j["status"] == "pending"]
    if not pending:
        return None, jobs
    # Sort by priority (lower = higher priority), then by submission time
    pending.sort(key=lambda x: (x[1]["priority"], x[1]["submitted_at"]))
    idx, job = pending[0]
    return job, jobs


def _update_job_status(job_id: str, updates: dict):
    """Update a specific job's fields."""
    jobs = _read_jobs()
    for job in jobs:
        if job["job_id"] == job_id:
            job.update(updates)
            break
    _write_jobs(jobs)


def run_queue_worker():
    """Main queue worker loop. Owns the GPU, runs jobs sequentially."""
    _ensure_queue_dir()

    # Write PID file
    with open(WORKER_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    print(f"[queue-worker] Started (pid={os.getpid()})")
    print(f"[queue-worker] Queue dir: {QUEUE_DIR}")

    # Handle graceful shutdown
    running = True
    current_proc = None

    def shutdown(signum, frame):
        nonlocal running
        print(f"\n[queue-worker] Shutting down...")
        running = False
        if current_proc and current_proc.poll() is None:
            current_proc.terminate()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while running:
        job, all_jobs = _get_next_job()
        if job is None:
            time.sleep(2)
            continue

        job_id = job["job_id"]
        agent_id = job["agent_id"]
        scale = job["scale"]
        command = job["command"]
        worktree = job["worktree_path"]

        print(f"[queue-worker] Running job {job_id} ({agent_id}, {scale}): {command}")

        # Mark as running
        _update_job_status(job_id, {
            "status": "running",
            "started_at": _now_iso(),
        })

        # Write active file
        with open(ACTIVE_FILE, "w") as f:
            json.dump({"job_id": job_id, "agent_id": agent_id, "scale": scale,
                       "started_at": _now_iso()}, f)

        # Build environment
        env = os.environ.copy()
        env.update(job.get("env", {}))

        # Run the command
        log_path = os.path.join(worktree, "run.log")
        try:
            with open(log_path, "w") as logf:
                current_proc = subprocess.Popen(
                    command.split(),
                    env=env, cwd=worktree,
                    stdout=logf, stderr=subprocess.STDOUT,
                )
                exit_code = current_proc.wait()
                current_proc = None
        except Exception as e:
            print(f"[queue-worker] Error running job {job_id}: {e}")
            exit_code = -1

        # Update job status
        status = "done" if exit_code == 0 else "failed"
        _update_job_status(job_id, {
            "status": status,
            "exit_code": exit_code,
            "finished_at": _now_iso(),
        })

        # Clear active file
        if os.path.exists(ACTIVE_FILE):
            os.remove(ACTIVE_FILE)

        print(f"[queue-worker] Job {job_id} {status} (exit={exit_code})")

    # Cleanup
    if os.path.exists(WORKER_PID_FILE):
        os.remove(WORKER_PID_FILE)
    print("[queue-worker] Stopped")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status():
    """Print queue status."""
    jobs = _read_jobs()

    # Active job
    if os.path.exists(ACTIVE_FILE):
        with open(ACTIVE_FILE, "r") as f:
            active = json.load(f)
        print(f"ACTIVE: job={active['job_id']} agent={active['agent_id']} scale={active['scale']}")
    else:
        print("ACTIVE: (none)")

    # Worker status
    if os.path.exists(WORKER_PID_FILE):
        with open(WORKER_PID_FILE, "r") as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            print(f"WORKER: running (pid={pid})")
        except ProcessLookupError:
            print(f"WORKER: dead (stale pid={pid})")
    else:
        print("WORKER: not running")

    # Pending jobs
    pending = [j for j in jobs if j["status"] == "pending"]
    running = [j for j in jobs if j["status"] == "running"]
    done = [j for j in jobs if j["status"] in ("done", "failed")]

    print(f"\nPending: {len(pending)}, Running: {len(running)}, Done: {len(done)}")

    if pending:
        print("\nPENDING:")
        pending.sort(key=lambda x: (x["priority"], x["submitted_at"]))
        for j in pending:
            print(f"  {j['job_id']}  {j['agent_id']:>16}  {j['scale']:>8}  {j['command']}")

    if done:
        print(f"\nRECENT (last 5):")
        for j in done[-5:]:
            status = "OK" if j["status"] == "done" else "FAIL"
            print(f"  {j['job_id']}  {j['agent_id']:>16}  {j['scale']:>8}  {status}  exit={j.get('exit_code', '?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU job queue for autoresearch")
    sub = parser.add_subparsers(dest="cmd")

    # worker
    sub.add_parser("worker", help="Start queue worker")

    # submit
    submit_p = sub.add_parser("submit", help="Submit a job")
    submit_p.add_argument("--agent-id", required=True)
    submit_p.add_argument("--scale", default="standard")
    submit_p.add_argument("--command", required=True)
    submit_p.add_argument("--worktree", default=os.getcwd())
    submit_p.add_argument("--wait", action="store_true", default=True)
    submit_p.add_argument("--timeout", type=int, default=3600)

    # status
    sub.add_parser("status", help="Show queue status")

    args = parser.parse_args()

    if args.cmd == "worker":
        run_queue_worker()
    elif args.cmd == "submit":
        job_id = submit_job(
            agent_id=args.agent_id,
            worktree_path=args.worktree,
            command=args.command,
            scale=args.scale,
        )
        print(f"Submitted job {job_id}")
        if args.wait:
            exit_code = wait_for_job(job_id, timeout=args.timeout)
            sys.exit(exit_code)
    elif args.cmd == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
