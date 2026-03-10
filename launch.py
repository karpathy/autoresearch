#!/usr/bin/env python3
"""
Multi-agent launcher for autoresearch.
Sets up worktrees, generates agent instructions, and launches Claude Code instances.

Usage:
    python launch.py --tag mar10 --preset solo
    python launch.py --tag mar10 --preset balanced --single-gpu
    python launch.py --tag mar10 --agents explorer:2,optimizer,director --tmux
    python launch.py --tag mar10 --preset full --single-gpu --tmux
    python launch.py cleanup --tag mar10
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKTREES_DIR = os.path.join(REPO_ROOT, "worktrees")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
QUEUE_DIR = os.path.join(REPO_ROOT, "queue")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
PROGRAMS_DIR = os.path.join(REPO_ROOT, "programs")

ROLES = {
    "explorer": {
        "description": "Tries bold architectural changes and novel ideas",
        "program": "explorer.md",
    },
    "optimizer": {
        "description": "Fine-tunes hyperparameters methodically",
        "program": "optimizer.md",
    },
    "analyst": {
        "description": "Analyzes results, identifies patterns, writes lessons",
        "program": "analyst.md",
    },
    "reviewer": {
        "description": "Validates promising results at longer scales",
        "program": "reviewer.md",
    },
    "director": {
        "description": "Coordinates research agenda, merges best results",
        "program": "director.md",
    },
}

PRESETS = {
    "solo": ["explorer"],
    "duo": ["explorer", "optimizer"],
    "balanced": ["explorer", "optimizer", "analyst"],
    "full": ["explorer", "optimizer", "analyst", "reviewer", "director"],
}


def parse_agents_spec(spec: str) -> list[tuple[str, int]]:
    """Parse 'explorer:2,optimizer,director' into [('explorer', 2), ('optimizer', 1), ('director', 1)]."""
    agents = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            role, count = part.split(":")
            agents.append((role.strip(), int(count)))
        else:
            agents.append((part, 1))
    for role, _ in agents:
        if role not in ROLES:
            print(f"Error: unknown role '{role}'. Available: {', '.join(ROLES.keys())}")
            sys.exit(1)
    return agents


def expand_agents(agents_spec: list[tuple[str, int]]) -> list[tuple[str, str]]:
    """Expand [(role, count)] into [(role, agent_id)] pairs."""
    result = []
    for role, count in agents_spec:
        for i in range(count):
            agent_id = f"{role}-{i}"
            result.append((role, agent_id))
    return result


def setup_shared_dirs():
    """Create shared directories."""
    for d in [RESULTS_DIR, QUEUE_DIR, CHECKPOINTS_DIR, WORKTREES_DIR]:
        os.makedirs(d, exist_ok=True)


def create_worktree(tag: str, agent_id: str) -> str:
    """Create a git worktree for an agent. Returns the worktree path."""
    worktree_name = f"{tag}-{agent_id}"
    worktree_path = os.path.join(WORKTREES_DIR, worktree_name)
    branch_name = f"autoresearch/{worktree_name}"

    if os.path.exists(worktree_path):
        print(f"  Worktree already exists: {worktree_path}")
        return worktree_path

    # Create worktree with a new branch
    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, worktree_path, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Branch might already exist, try without -b
        result = subprocess.run(
            ["git", "worktree", "add", worktree_path, branch_name],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Error creating worktree: {result.stderr}")
            sys.exit(1)

    # Create symlinks for shared directories
    for dirname, target in [
        ("results", RESULTS_DIR),
        ("queue", QUEUE_DIR),
        ("checkpoints", CHECKPOINTS_DIR),
    ]:
        link_path = os.path.join(worktree_path, dirname)
        if os.path.exists(link_path) or os.path.islink(link_path):
            if os.path.islink(link_path):
                os.unlink(link_path)
            elif os.path.isdir(link_path):
                shutil.rmtree(link_path)
        os.symlink(target, link_path)

    print(f"  Created worktree: {worktree_path} (branch: {branch_name})")
    return worktree_path


def generate_claude_md(worktree_path: str, role: str, agent_id: str,
                       tag: str, single_gpu: bool) -> None:
    """Write CLAUDE.md into the worktree with role-specific instructions."""
    program_path = os.path.join(PROGRAMS_DIR, ROLES[role]["program"])
    with open(program_path, "r") as f:
        program_content = f.read()

    use_queue_note = ""
    if single_gpu:
        use_queue_note = """
## GPU Queue Mode

This session is running in **single-GPU mode** with a shared GPU queue.
Instead of running experiments directly, use the `--use-queue` flag:

```
python run_experiment.py --scale quick --description "..." --agent-id {agent_id} --agent-role {role} --use-queue
```

This submits your job to the GPU queue. Probe jobs (30s) get priority so you get fast
memory feedback. While waiting, you can plan your next experiment.
""".format(agent_id=agent_id, role=role)

    claude_md = f"""# Agent: {agent_id}

- **Tag**: {tag}
- **Role**: {role}
- **Agent ID**: {agent_id}
- **Description**: {ROLES[role]['description']}

## Quick Reference

```bash
# Read research briefing (do this before every experiment)
python run_experiment.py --briefing

# Run an experiment
python run_experiment.py --scale quick --description "what you tried" --agent-id {agent_id} --agent-role {role}

# Log a lesson
python run_experiment.py --lesson <category> <confidence> "lesson text"

# Check GPU queue status (if single-gpu mode)
python gpu_queue.py status
```
{use_queue_note}
## Files You Can Modify

- `train.py` — The main training script. Architecture, optimizer, hyperparameters.

## Files You Should NOT Modify

- `prepare.py` — Data prep, tokenizer, evaluation (the ground truth metric).
- `config.py`, `knowledge.py`, `checkpoint.py`, `run_experiment.py`, `gpu_queue.py` — Infrastructure.
- Other agents' worktrees.

## Getting Started

1. Read this file (you're doing it now)
2. Read `python run_experiment.py --briefing` for current research state
3. Read `train.py` to understand the current model
4. Begin your experiment loop as described in your role instructions below

---

{program_content}
"""
    claude_md_path = os.path.join(worktree_path, "CLAUDE.md")
    with open(claude_md_path, "w") as f:
        f.write(claude_md)
    print(f"  Generated CLAUDE.md for {agent_id}")


def launch_agent_tmux(worktree_path: str, agent_id: str, session_name: str) -> None:
    """Launch a Claude Code agent in a tmux pane."""
    cmd = (
        f'cd "{worktree_path}" && '
        f'claude -p "Read CLAUDE.md and begin your research loop. You are {agent_id}." '
        f'--dangerously-skip-permissions '
        f'--model opus '
        f'2>&1 | tee agent.log'
    )

    # Check if tmux session exists
    check = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
    )
    if check.returncode != 0:
        # Create new session with first agent
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "-n", agent_id, cmd],
        )
    else:
        # Add new window
        subprocess.run(
            ["tmux", "new-window", "-t", session_name, "-n", agent_id, cmd],
        )
    print(f"  Launched {agent_id} in tmux session '{session_name}'")


def launch_agent_background(worktree_path: str, agent_id: str) -> subprocess.Popen:
    """Launch a Claude Code agent as a background process."""
    log_path = os.path.join(worktree_path, "agent.log")
    cmd = [
        "claude", "-p",
        f"Read CLAUDE.md and begin your research loop. You are {agent_id}.",
        "--dangerously-skip-permissions",
        "--model", "opus",
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=worktree_path,
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    print(f"  Launched {agent_id} (pid={proc.pid}, log={log_path})")
    return proc


def launch_queue_worker() -> subprocess.Popen:
    """Start the GPU queue worker as a background process."""
    log_path = os.path.join(QUEUE_DIR, "worker.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO_ROOT, "gpu_queue.py"), "worker"],
        cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
    )
    print(f"  Queue worker started (pid={proc.pid}, log={log_path})")
    return proc


def cleanup_worktrees(tag: str):
    """Remove worktrees and branches for a tag."""
    if not os.path.exists(WORKTREES_DIR):
        print("No worktrees directory found.")
        return

    for entry in os.listdir(WORKTREES_DIR):
        if entry.startswith(f"{tag}-"):
            worktree_path = os.path.join(WORKTREES_DIR, entry)
            branch_name = f"autoresearch/{entry}"
            print(f"  Removing worktree: {worktree_path}")
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=REPO_ROOT, capture_output=True,
            )
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=REPO_ROOT, capture_output=True,
            )

    print("Cleanup complete.")


def monitor_loop(processes: list[tuple[str, subprocess.Popen]], queue_worker: subprocess.Popen = None):
    """Monitor running agents, handle Ctrl+C gracefully."""
    print("\n" + "=" * 60)
    print("All agents launched. Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    def shutdown(signum, frame):
        print("\nShutting down agents...")
        for agent_id, proc in processes:
            if proc.poll() is None:
                print(f"  Stopping {agent_id} (pid={proc.pid})")
                proc.terminate()
        if queue_worker and queue_worker.poll() is None:
            print("  Stopping queue worker")
            queue_worker.terminate()
        # Wait for processes to finish
        for _, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if queue_worker:
            try:
                queue_worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                queue_worker.kill()
        print("All agents stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(30)
        alive = sum(1 for _, p in processes if p.poll() is None)
        total = len(processes)
        if alive == 0:
            print("All agents have exited.")
            break
        # Check for crashed agents
        for agent_id, proc in processes:
            if proc.poll() is not None and proc.returncode != 0:
                print(f"  WARNING: {agent_id} exited with code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Launch multi-agent autoresearch")
    sub = parser.add_subparsers(dest="cmd")

    # launch (default)
    launch_p = sub.add_parser("launch", help="Launch agents")
    launch_p.add_argument("--tag", required=True, help="Run tag (e.g., mar10)")
    launch_p.add_argument("--agents", default=None, help="Agent spec (e.g., explorer:2,optimizer)")
    launch_p.add_argument("--preset", default=None, choices=PRESETS.keys(), help="Agent preset")
    launch_p.add_argument("--single-gpu", action="store_true", help="Enable GPU queue for time-sharing")
    launch_p.add_argument("--tmux", action="store_true", help="Launch in tmux panes")
    launch_p.add_argument("--model", default="opus", help="Claude model (default: opus)")

    # cleanup
    cleanup_p = sub.add_parser("cleanup", help="Remove worktrees for a tag")
    cleanup_p.add_argument("--tag", required=True, help="Run tag to clean up")

    # status
    sub.add_parser("status", help="Show running agents and queue status")

    args = parser.parse_args()

    # Default to launch if no subcommand
    if args.cmd is None:
        # Re-parse with launch as default
        if "--tag" in sys.argv:
            args.cmd = "launch"
            args = launch_p.parse_args(sys.argv[1:])
        else:
            parser.print_help()
            return

    if args.cmd == "cleanup":
        cleanup_worktrees(args.tag)
        return

    if args.cmd == "status":
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "gpu_queue.py"), "status"])
        return

    # Launch mode
    tag = args.tag
    single_gpu = args.single_gpu
    use_tmux = args.tmux

    # Parse agent spec
    if args.preset:
        agents_spec = [(role, 1) for role in PRESETS[args.preset]]
    elif args.agents:
        agents_spec = parse_agents_spec(args.agents)
    else:
        print("Error: specify --preset or --agents")
        sys.exit(1)

    agents = expand_agents(agents_spec)

    print(f"\n{'=' * 60}")
    print(f"  AUTORESEARCH v2 — Multi-Agent Launch")
    print(f"{'=' * 60}")
    print(f"  Tag:        {tag}")
    print(f"  Agents:     {', '.join(aid for _, aid in agents)}")
    print(f"  Single GPU: {single_gpu}")
    print(f"  TMux:       {use_tmux}")
    print(f"{'=' * 60}\n")

    # Step 1: Create shared directories
    print("[1/5] Setting up shared directories...")
    setup_shared_dirs()

    # Step 2: Create worktrees
    print("[2/5] Creating worktrees...")
    worktree_paths = {}
    for role, agent_id in agents:
        path = create_worktree(tag, agent_id)
        worktree_paths[agent_id] = path

    # Step 3: Generate CLAUDE.md for each agent
    print("[3/5] Generating agent instructions...")
    for role, agent_id in agents:
        generate_claude_md(worktree_paths[agent_id], role, agent_id, tag, single_gpu)

    # Step 4: Start queue worker if single-gpu
    queue_worker = None
    if single_gpu:
        print("[4/5] Starting GPU queue worker...")
        queue_worker = launch_queue_worker()
    else:
        print("[4/5] Skipping queue worker (multi-GPU mode)")

    # Step 5: Launch agents
    print("[5/5] Launching agents...")
    tmux_session = f"autoresearch-{tag}"

    if use_tmux:
        for role, agent_id in agents:
            launch_agent_tmux(worktree_paths[agent_id], agent_id, tmux_session)
        print(f"\nAgents running in tmux session: {tmux_session}")
        print(f"  Attach with: tmux attach -t {tmux_session}")
        print(f"  Cleanup with: python launch.py cleanup --tag {tag}")
    else:
        processes = []
        for role, agent_id in agents:
            proc = launch_agent_background(worktree_paths[agent_id], agent_id)
            processes.append((agent_id, proc))
        monitor_loop(processes, queue_worker)


if __name__ == "__main__":
    main()
