#!/usr/bin/env python3
"""Reset local autoresearch history/runtime artifacts.

Actions:
1) Checkout main branch (configurable)
2) Remove all extra git worktrees
3) Delete all local branches except main
4) Clear component_system runtime state/history folders
5) Remove .pytest_cache, __pycache__, and results.tsv

With --preserve-seeds SEED_IDS: keep everything for those seeds (state, events, runs,
queue tasks, worktrees, branches, logs, baseline mappings); remove only other seeds' data.
SEED_IDS can be comma-separated, e.g. --preserve-seeds seed-a,seed-b,seed-c.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_git(args: list[str], cwd: Path, dry_run: bool = False) -> list[str]:
    cmd = ["git", *args]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return []
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def is_broken_worktree_remove_error(error: RuntimeError) -> bool:
    msg = str(error)
    return (
        "worktree remove --force" in msg
        and "validation failed, cannot remove working tree" in msg
        and ".git' does not exist" in msg
    )


def remove_children(path: Path, dry_run: bool = False) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if dry_run:
            print(f"[dry-run] remove {child}")
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def remove_pycache_dirs(repo_root: Path, dry_run: bool = False) -> None:
    for pycache in repo_root.rglob("__pycache__"):
        parts = set(pycache.parts)
        if ".venv" in parts or ".git" in parts:
            continue
        if pycache.is_dir():
            if dry_run:
                print(f"[dry-run] remove {pycache}")
            else:
                shutil.rmtree(pycache, ignore_errors=True)


def _gather_preserved_seed_info(
    repo_root: Path, seed_ids: list[str]
) -> tuple[set[str], set[str]]:
    """Return (preserved_run_ids, baseline_branches). Exits if any seed not found."""
    comp = repo_root / "component_system"
    history = comp / "history"
    state = history / "state"
    seeds_dir = state / "seeds"
    runs_dir = state / "runs"
    preserved_ids = set(seed_ids)
    baseline_branches: set[str] = set()
    run_ids: set[str] = set()

    for seed_id in seed_ids:
        seed_file = seeds_dir / f"{seed_id}.json"
        if not seed_file.exists():
            raise SystemExit(f"Seed not found: {seed_id} (no {seed_file})")
        seed_data = _read_json(seed_file, {})
        if isinstance(seed_data, dict):
            bl = seed_data.get("baseline_branch")
            if isinstance(bl, str):
                baseline_branches.add(bl)

    for path in runs_dir.glob("*.json"):
        data = _read_json(path, {})
        if isinstance(data, dict) and data.get("seed_id") in preserved_ids:
            rid = data.get("run_id")
            if isinstance(rid, str):
                run_ids.add(rid)
    return run_ids, baseline_branches


def _clean_state_preserving_seeds(
    repo_root: Path,
    preserved_seed_ids: set[str],
    preserved_run_ids: set[str],
    dry_run: bool,
) -> None:
    comp = repo_root / "component_system"
    history = comp / "history"
    state = history / "state"
    seeds_dir = state / "seeds"
    events_dir = state / "events"
    runs_dir = state / "runs"

    for path in seeds_dir.glob("*.json"):
        if path.stem not in preserved_seed_ids:
            if dry_run:
                print(f"[dry-run] remove {path}")
            else:
                path.unlink(missing_ok=True)

    for path in events_dir.glob("*.json"):
        if path.stem not in preserved_seed_ids:
            if dry_run:
                print(f"[dry-run] remove {path}")
            else:
                path.unlink(missing_ok=True)

    for path in runs_dir.glob("*.json"):
        data = _read_json(path, {})
        rid = data.get("run_id") if isinstance(data, dict) else None
        if rid not in preserved_run_ids:
            if dry_run:
                print(f"[dry-run] remove {path}")
            else:
                path.unlink(missing_ok=True)


def _clean_queue_preserving_seeds(
    repo_root: Path, preserved_seed_ids: set[str], dry_run: bool
) -> None:
    history = repo_root / "component_system" / "history" / "queue"
    stage_dirs = [
        history / "p",
        history / "dca",
        history / "direct",
        history / "in_progress",
        history / "done",
        history / "error",
    ]
    for stage_dir in stage_dirs:
        if not stage_dir.exists():
            continue
        for path in stage_dir.glob("*.json"):
            data = _read_json(path, {})
            task_seed = data.get("seed_id") if isinstance(data, dict) else None
            if task_seed not in preserved_seed_ids:
                if dry_run:
                    print(f"[dry-run] remove {path}")
                else:
                    path.unlink(missing_ok=True)


def _clean_worktrees_preserving_seeds(
    repo_root: Path,
    preserved_seed_ids: set[str],
    dry_run: bool,
) -> None:
    worktrees_dir = repo_root / "component_system" / "history" / "worktrees"
    if not worktrees_dir.exists():
        return
    keep_names = preserved_seed_ids | {"baseline"}
    for child in worktrees_dir.iterdir():
        if child.is_dir() and child.name not in keep_names:
            if dry_run:
                print(f"[dry-run] remove {child}")
            else:
                shutil.rmtree(child, ignore_errors=True)


def _clean_logs_preserving_seed(
    repo_root: Path,
    preserved_run_ids: set[str],
    dry_run: bool,
) -> None:
    logs_dir = repo_root / "component_system" / "history" / "logs"
    if not logs_dir.exists():
        return
    for path in logs_dir.iterdir():
        if not path.is_file():
            continue
        # logs: {run_id}.stdout.log, {run_id}.stderr.log, {run_id}.prompt.txt
        run_id = path.stem
        if path.suffix in (".log", ".txt"):
            run_id = run_id.rsplit(".", 1)[0] if "." in run_id else run_id
        if run_id not in preserved_run_ids:
            if dry_run:
                print(f"[dry-run] remove {path}")
            else:
                path.unlink(missing_ok=True)


def _filter_baseline_jsons_preserving_seeds(
    repo_root: Path,
    preserved_seed_ids: set[str],
    baseline_branches: set[str],
    dry_run: bool,
) -> None:
    comp = repo_root / "component_system"
    branches_path = comp / "baseline_branches.json"
    metrics_path = comp / "baseline_metrics.json"

    if branches_path.exists():
        data = _read_json(branches_path, {})
        if isinstance(data, dict):
            new_data = {k: v for k, v in data.items() if k in preserved_seed_ids}
            if dry_run:
                print(f"[dry-run] write {branches_path} (keep {preserved_seed_ids})")
            else:
                _write_json(branches_path, new_data)

    if metrics_path.exists():
        data = _read_json(metrics_path, {})
        if isinstance(data, dict):
            keep_branches = preserved_seed_ids | baseline_branches
            new_data = {k: v for k, v in data.items() if k in keep_branches}
            if dry_run:
                print(f"[dry-run] write {metrics_path} (keep branches {keep_branches})")
            else:
                _write_json(metrics_path, new_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean local branches/worktrees and runtime history.")
    parser.add_argument("--main-branch", default="master", help="Branch to keep. Default: master")
    parser.add_argument(
        "--preserve-seeds",
        metavar="SEED_IDS",
        help="Comma-separated seed IDs to keep (e.g. seed-a,seed-b). Keep their state, events, runs, queue, worktrees, branches, logs, baseline mappings; remove only other seeds.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing anything")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    print(f"Repository: {repo_root}")

    raw_preserve = getattr(args, "preserve_seeds", None)
    preserve_seeds: list[str] = (
        [s.strip() for s in raw_preserve.split(",") if s.strip()] if raw_preserve else []
    )
    preserved_run_ids: set[str] = set()
    baseline_branches: set[str] = set()
    preserved_seed_ids: set[str] = set()
    if preserve_seeds:
        preserved_seed_ids = set(preserve_seeds)
        print(f"Preserving everything for seeds: {', '.join(sorted(preserved_seed_ids))}")
        preserved_run_ids, baseline_branches = _gather_preserved_seed_info(repo_root, preserve_seeds)
        print(f"  runs to keep: {len(preserved_run_ids)}")

    print("Verifying git repository...")
    run_git(["rev-parse", "--is-inside-work-tree"], cwd=repo_root, dry_run=args.dry_run)

    print(f"Checking out '{args.main_branch}'...")
    run_git(["checkout", args.main_branch], cwd=repo_root, dry_run=args.dry_run)

    print("Removing extra worktrees...")
    run_git(["worktree", "prune"], cwd=repo_root, dry_run=args.dry_run)
    wt_lines = run_git(["worktree", "list", "--porcelain"], cwd=repo_root, dry_run=args.dry_run)
    worktrees: list[Path] = []
    for line in wt_lines:
        if line.startswith("worktree "):
            worktrees.append(Path(line[len("worktree ") :]).resolve())

    branches_to_keep = {args.main_branch} | preserved_seed_ids
    worktree_keep_names = preserved_seed_ids | {"baseline"} if preserved_seed_ids else set()

    for wt in worktrees:
        if wt == repo_root:
            continue
        if worktree_keep_names and wt.name in worktree_keep_names:
            print(f"  - keeping worktree {wt} (preserved: {wt.name})")
            continue
        print(f"  - removing worktree {wt}")
        try:
            run_git(["worktree", "remove", "--force", str(wt)], cwd=repo_root, dry_run=args.dry_run)
        except RuntimeError as error:
            if not is_broken_worktree_remove_error(error):
                raise
            print(f"    ! stale/broken worktree metadata detected, deleting directory: {wt}")
            if args.dry_run:
                print(f"[dry-run] remove {wt}")
            else:
                shutil.rmtree(wt, ignore_errors=True)
    run_git(["worktree", "prune"], cwd=repo_root, dry_run=args.dry_run)

    print(f"Deleting local branches except {sorted(branches_to_keep)}...")
    branches = run_git(
        ["for-each-ref", "--format=%(refname:short)", "refs/heads"],
        cwd=repo_root,
        dry_run=args.dry_run,
    )
    for branch in branches:
        if branch not in branches_to_keep:
            print(f"  - deleting branch {branch}")
            run_git(["branch", "-D", branch], cwd=repo_root, dry_run=args.dry_run)

    history_root = repo_root / "component_system" / "history"
    if preserved_seed_ids:
        print("Clearing component-system state (keeping preserved seeds)...")
        _clean_state_preserving_seeds(
            repo_root, preserved_seed_ids, preserved_run_ids, args.dry_run
        )
        print("Clearing queue (keeping tasks for preserved seeds)...")
        _clean_queue_preserving_seeds(repo_root, preserved_seed_ids, args.dry_run)
        print("Clearing worktrees (keeping preserved seeds + baseline)...")
        _clean_worktrees_preserving_seeds(repo_root, preserved_seed_ids, args.dry_run)
        print("Clearing logs (keeping logs for preserved seed runs)...")
        _clean_logs_preserving_seed(repo_root, preserved_run_ids, args.dry_run)
        print("Filtering baseline_branches.json and baseline_metrics.json...")
        _filter_baseline_jsons_preserving_seeds(
            repo_root, preserved_seed_ids, baseline_branches, args.dry_run
        )
    else:
        print("Clearing component-system runtime/history artifacts...")
        for name in ("state", "queue", "worktrees", "logs"):
            remove_children(history_root / name, dry_run=args.dry_run)

    pytest_cache = repo_root / ".pytest_cache"
    if pytest_cache.exists():
        if args.dry_run:
            print(f"[dry-run] remove {pytest_cache}")
        else:
            shutil.rmtree(pytest_cache, ignore_errors=True)

    results_tsv = repo_root / "results.tsv"
    if results_tsv.exists():
        if args.dry_run:
            print(f"[dry-run] remove {results_tsv}")
        else:
            results_tsv.unlink(missing_ok=True)

    print("Removing __pycache__ directories...")
    remove_pycache_dirs(repo_root, dry_run=args.dry_run)

    print("Done.")
    print("Remaining branches:")
    for branch in run_git(["branch", "--format=%(refname:short)"], cwd=repo_root, dry_run=args.dry_run):
        print(f"  {branch}")


if __name__ == "__main__":
    main()
