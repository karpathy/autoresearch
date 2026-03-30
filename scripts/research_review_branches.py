from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from research_pr_plan import (
    REPO_ROOT,
    BranchCommit,
    build_plan_bundles,
    collect_branch_commits,
    normalize_prefixes,
    ref_exists,
    resolve_base_ref,
    resolve_selected_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local review branches from planned PR bundles.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo to inspect.")
    parser.add_argument("--base", help="Base ref to compare against. Defaults to origin/master or origin/main.")
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Path prefix to ignore when computing dependencies. Repeatable.",
    )
    parser.add_argument("--hash", action="append", default=[], help="Commit hash or unique prefix to include.")
    parser.add_argument("--all", action="store_true", help="Use every branch-only commit as the explicit selection.")
    parser.add_argument(
        "--mode",
        choices=("consolidated", "stacked", "individual"),
        default="consolidated",
        help="Bundle strategy to materialize as branches.",
    )
    parser.add_argument(
        "--prefix",
        help="Branch name prefix. Defaults to review/<repo-name>.",
    )
    parser.add_argument("--yes", action="store_true", help="Actually create the branches.")
    parser.add_argument("--force", action="store_true", help="Delete and recreate existing review branches.")
    return parser.parse_args()


def git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


def git_stdout(args: list[str], *, cwd: Path) -> str:
    return git(args, cwd=cwd).stdout.strip()


def slugify(value: str) -> str:
    value = value.lower()
    parts = []
    for char in value:
        if char.isalnum():
            parts.append(char)
        elif parts and parts[-1] != "-":
            parts.append("-")
    return "".join(parts).strip("-") or "bundle"


def derive_prefix(repo_root: Path, raw_prefix: str | None) -> str:
    if raw_prefix:
        return raw_prefix.rstrip("/")
    return f"review/{repo_root.name}"


def branch_name_for_bundle(prefix: str, mode: str, index: int, bundle: dict[str, object]) -> str:
    selected_commits: list[BranchCommit] = bundle["selected"]  # type: ignore[assignment]
    if mode == "consolidated":
        return f"{prefix}/consolidated"
    if selected_commits:
        subject_slug = slugify(selected_commits[-1].subject)[:48]
        short_hash = selected_commits[-1].short_hash
    else:
        subject_slug = "bundle"
        short_hash = f"{index:02d}"
    if mode == "stacked":
        return f"{prefix}/stack-{index:02d}-{short_hash}-{subject_slug}"
    return f"{prefix}/pr-{index:02d}-{short_hash}-{subject_slug}"


def render_branch_plan(
    *,
    repo_root: Path,
    base_ref: str,
    mode: str,
    bundles: list[dict[str, object]],
    branch_names: list[str],
) -> str:
    lines = [f"Repo: {repo_root}", f"Base: {base_ref}", f"Mode: {mode}", ""]
    for index, bundle in enumerate(bundles, start=1):
        bundle_commits: list[BranchCommit] = bundle["commits"]  # type: ignore[assignment]
        selected_commits: list[BranchCommit] = bundle["selected"]  # type: ignore[assignment]
        auto_commits: list[BranchCommit] = bundle["auto_included"]  # type: ignore[assignment]
        target = base_ref if mode != "stacked" or index == 1 else branch_names[index - 2]
        lines.append(f"{branch_names[index - 1]} -> target {target}")
        lines.append(
            f"  selected: {', '.join(commit.short_hash for commit in selected_commits) if selected_commits else 'none'}"
        )
        lines.append(
            f"  auto-included: {', '.join(commit.short_hash for commit in auto_commits) if auto_commits else 'none'}"
        )
        lines.append(
            f"  cherry-picks: {', '.join(commit.short_hash for commit in bundle_commits) if bundle_commits else 'none'}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def ensure_branch_target(branch_name: str, *, repo_root: Path, force: bool) -> None:
    if not ref_exists(branch_name, cwd=repo_root):
        return
    if not force:
        raise RuntimeError(f"Branch already exists: {branch_name}")
    git(["branch", "-D", branch_name], cwd=repo_root)


def create_branch_from_bundle(
    *,
    repo_root: Path,
    base_ref: str,
    branch_name: str,
    bundle_commits: list[BranchCommit],
) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="review-branch-"))
    try:
        git(["worktree", "add", "--quiet", "-b", branch_name, str(temp_dir), base_ref], cwd=repo_root)
        try:
            if bundle_commits:
                git(["cherry-pick", *[commit.commit_hash for commit in bundle_commits]], cwd=temp_dir)
        except Exception:
            git(["cherry-pick", "--abort"], cwd=temp_dir, check=False)
            raise
        finally:
            git(["worktree", "remove", "--force", str(temp_dir)], cwd=repo_root, check=False)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    exclude_prefixes = normalize_prefixes(args.exclude_prefix)
    base_ref = resolve_base_ref(args.base, cwd=repo_root)
    commits = collect_branch_commits(cwd=repo_root, base_ref=base_ref, exclude_prefixes=exclude_prefixes)
    selected_hashes = resolve_selected_hashes(args.hash, commits, select_all=args.all)
    bundles, _required_hashes = build_plan_bundles(commits, selected_hashes, mode=args.mode)
    prefix = derive_prefix(repo_root, args.prefix)
    branch_names = [branch_name_for_bundle(prefix, args.mode, index, bundle) for index, bundle in enumerate(bundles, start=1)]

    print(render_branch_plan(repo_root=repo_root, base_ref=base_ref, mode=args.mode, bundles=bundles, branch_names=branch_names))
    if not args.yes:
        print("\nDry run only. Re-run with --yes to create the review branches.")
        return 0

    for branch_name in branch_names:
        ensure_branch_target(branch_name, repo_root=repo_root, force=args.force)

    parent_ref = base_ref
    for branch_name, bundle in zip(branch_names, bundles):
        bundle_commits: list[BranchCommit] = bundle["commits"]  # type: ignore[assignment]
        create_branch_from_bundle(repo_root=repo_root, base_ref=parent_ref, branch_name=branch_name, bundle_commits=bundle_commits)
        if args.mode == "stacked":
            parent_ref = branch_name

    print("")
    print("Created review branches:")
    for branch_name in branch_names:
        print(f"- {branch_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
