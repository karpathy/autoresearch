"""
Multi-GPU launcher for autoresearch experiments.

Creates isolated git worktrees so multiple agents (Claude Code, Codex, etc.)
can run experiments in parallel, each on its own GPU and branch.

Usage:
    uv run swarm.py --tag mar10 --gpus 0,1,2,3
    uv run swarm.py --tag mar10 --gpus 0,1 --baseline   # also run baseline
    uv run swarm.py --cleanup                            # remove all worktrees
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKTREE_DIR = os.path.join(BASE_DIR, "worktrees")


def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, **kwargs)


def setup_worktrees(tag, gpu_ids):
    os.makedirs(WORKTREE_DIR, exist_ok=True)
    agents = []
    for i, gpu in enumerate(gpu_ids):
        branch = f"autoresearch/{tag}/agent-{i}"
        agent_dir = os.path.join(WORKTREE_DIR, f"agent-{i}")
        if os.path.isdir(agent_dir):
            print(f"  agent-{i}: reusing existing worktree")
        else:
            result = run(["git", "worktree", "add", "-b", branch, agent_dir, "HEAD"])
            if result.returncode != 0:
                print(f"  agent-{i}: failed — {result.stderr.strip()}")
                continue
            print(f"  agent-{i}: created on branch {branch}")
        agents.append({"id": i, "gpu": gpu, "dir": agent_dir, "branch": branch})
    return agents


def run_baselines(agents):
    """Run baseline train.py on all agents in parallel, one per GPU."""
    procs = []
    for a in agents:
        log_path = os.path.join(a["dir"], "run.log")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a["gpu"])}
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, os.path.join(a["dir"], "train.py")],
            cwd=a["dir"], env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs.append((a, proc, log_f))
        print(f"  agent-{a['id']} (GPU {a['gpu']}): running...")

    print(f"\nWaiting for {len(procs)} baseline runs (~5 min)...\n")
    for a, proc, log_f in procs:
        proc.wait()
        log_f.close()
        log_path = os.path.join(a["dir"], "run.log")
        bpb = "N/A"
        try:
            with open(log_path) as f:
                for line in f:
                    if line.startswith("val_bpb:"):
                        bpb = line.split(":")[1].strip()
        except FileNotFoundError:
            pass
        status = "OK" if proc.returncode == 0 else "FAILED"
        print(f"  agent-{a['id']} (GPU {a['gpu']}): {status}, val_bpb={bpb}")


def cleanup():
    result = run(["git", "worktree", "list", "--porcelain"])
    for line in result.stdout.splitlines():
        if line.startswith("worktree ") and "/worktrees/agent-" in line:
            path = line.split("worktree ", 1)[1]
            run(["git", "worktree", "remove", "--force", path])
            print(f"  removed {path}")
    if os.path.isdir(WORKTREE_DIR):
        try:
            os.rmdir(WORKTREE_DIR)
        except OSError:
            pass
    print("Cleanup done.")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU launcher for autoresearch")
    parser.add_argument("--tag", help="Experiment tag (e.g. mar10)")
    parser.add_argument("--gpus", help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--baseline", action="store_true", help="Run baseline on all agents")
    parser.add_argument("--cleanup", action="store_true", help="Remove all worktrees")
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return

    if not args.tag or not args.gpus:
        parser.error("--tag and --gpus are required (or use --cleanup)")

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n = len(gpu_ids)

    print(f"Setting up {n} agent worktrees (tag={args.tag}):\n")
    agents = setup_worktrees(args.tag, gpu_ids)

    if not agents:
        print("No agents created.")
        return

    if args.baseline:
        print(f"\nRunning baselines:\n")
        run_baselines(agents)

    print(f"\n{'='*60}")
    print(f"Ready! Open your agent in each worktree:\n")
    for a in agents:
        print(f"  GPU {a['gpu']}: cd {a['dir']}")
    print(f"\nPrompt each agent:")
    print(f'  "Set CUDA_VISIBLE_DEVICES={agents[0]["gpu"]}. Read program.md and start."')
    print(f"\nCleanup later: uv run swarm.py --cleanup")


if __name__ == "__main__":
    main()
