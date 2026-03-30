from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CANDIDATES = ("origin/master", "origin/main", "master", "main")
DEFAULT_EXCLUDE_PREFIXES = ("results/", "queue/", ".history/", ".vscode/")


@dataclass(frozen=True)
class BranchCommit:
    index: int
    commit_hash: str
    short_hash: str
    subject: str
    files: tuple[str, ...]
    filtered_files: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan reviewable PR bundles from branch-only commits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list", help="List commits on the current branch not present in the base ref.")
    list_cmd.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo to inspect.")
    list_cmd.add_argument("--base", help="Base ref to compare against. Defaults to origin/master or origin/main.")
    list_cmd.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Path prefix to ignore when computing dependencies. Repeatable.",
    )
    list_cmd.add_argument("--show-files", action="store_true", help="Print filtered file lists for each commit.")

    plan_cmd = subparsers.add_parser("plan", help="Build consolidated, stacked, or individual PR bundles.")
    plan_cmd.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo to inspect.")
    plan_cmd.add_argument("--base", help="Base ref to compare against. Defaults to origin/master or origin/main.")
    plan_cmd.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Path prefix to ignore when computing dependencies. Repeatable.",
    )
    plan_cmd.add_argument("--hash", action="append", default=[], help="Commit hash or unique prefix to include.")
    plan_cmd.add_argument("--all", action="store_true", help="Use every branch-only commit as the explicit selection.")
    plan_cmd.add_argument(
        "--mode",
        choices=("consolidated", "stacked", "individual"),
        default="consolidated",
        help="Bundle strategy for the selected commits.",
    )

    return parser.parse_args()


def git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def normalize_prefixes(raw_prefixes: list[str]) -> tuple[str, ...]:
    prefixes = []
    for prefix in [*DEFAULT_EXCLUDE_PREFIXES, *raw_prefixes]:
        normalized = prefix.replace("\\", "/").strip()
        if normalized and not normalized.endswith("/"):
            normalized += "/"
        if normalized and normalized not in prefixes:
            prefixes.append(normalized)
    return tuple(prefixes)


def ref_exists(ref: str, *, cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def resolve_base_ref(provided: str | None, *, cwd: Path) -> str:
    if provided:
        if not ref_exists(provided, cwd=cwd):
            raise RuntimeError(f"Base ref not found: {provided}")
        return provided
    for candidate in DEFAULT_BASE_CANDIDATES:
        if ref_exists(candidate, cwd=cwd):
            return candidate
    raise RuntimeError("Could not resolve a default base ref. Pass --base explicitly.")


def filter_files(files: list[str], *, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    kept: list[str] = []
    for file_path in files:
        normalized = file_path.replace("\\", "/")
        if any(normalized.startswith(prefix) for prefix in prefixes):
            continue
        kept.append(normalized)
    return tuple(kept)


def collect_branch_commits(*, cwd: Path, base_ref: str, exclude_prefixes: tuple[str, ...]) -> list[BranchCommit]:
    rev_list = git(["rev-list", "--reverse", "--topo-order", f"{base_ref}..HEAD"], cwd=cwd)
    if not rev_list:
        return []

    commit_hashes = [line.strip() for line in rev_list.splitlines() if line.strip()]
    commits: list[BranchCommit] = []
    for index, commit_hash in enumerate(commit_hashes, start=1):
        subject = git(["show", "-s", "--format=%s", commit_hash], cwd=cwd)
        diff_tree = git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash], cwd=cwd)
        files = [line.strip() for line in diff_tree.splitlines() if line.strip()]
        commits.append(
            BranchCommit(
                index=index,
                commit_hash=commit_hash,
                short_hash=commit_hash[:7],
                subject=subject,
                files=tuple(files),
                filtered_files=filter_files(files, prefixes=exclude_prefixes),
            )
        )
    return commits


def resolve_selected_hashes(raw_hashes: list[str], commits: list[BranchCommit], *, select_all: bool) -> list[str]:
    if select_all:
        return [commit.commit_hash for commit in commits]

    if not raw_hashes:
        raise RuntimeError("Provide --hash at least once, or pass --all.")

    resolved: list[str] = []
    for raw_hash in raw_hashes:
        matches = [commit.commit_hash for commit in commits if commit.commit_hash.startswith(raw_hash)]
        if not matches:
            raise RuntimeError(f"No branch commit matches selection: {raw_hash}")
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous commit prefix: {raw_hash}")
        if matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def build_dependency_resolver(commits: list[BranchCommit]):
    index_by_hash = {commit.commit_hash: commit.index - 1 for commit in commits}

    @lru_cache(maxsize=None)
    def dependency_closure(commit_hash: str) -> tuple[str, ...]:
        commit = commits[index_by_hash[commit_hash]]
        required = {commit.commit_hash}
        commit_files = set(commit.filtered_files)
        if not commit_files:
            return tuple(sorted(required, key=lambda value: index_by_hash[value]))
        for earlier in commits[: commit.index - 1]:
            if not earlier.filtered_files:
                continue
            if commit_files.intersection(earlier.filtered_files):
                required.update(dependency_closure(earlier.commit_hash))
        return tuple(sorted(required, key=lambda value: index_by_hash[value]))

    return dependency_closure


def describe_commit(commit: BranchCommit) -> str:
    if commit.filtered_files:
        return f"{commit.short_hash}  {commit.subject}  [{len(commit.filtered_files)} tracked file(s)]"
    if commit.files:
        return f"{commit.short_hash}  {commit.subject}  [metadata-only after exclusions]"
    return f"{commit.short_hash}  {commit.subject}  [no files]"


def render_commit_list(commits: list[BranchCommit], *, base_ref: str, repo_root: Path, show_files: bool) -> str:
    lines = [f"Repo: {repo_root}", f"Base: {base_ref}", f"Branch-only commits: {len(commits)}", ""]
    if not commits:
        lines.append("No commits ahead of the base ref.")
        return "\n".join(lines)

    for commit in commits:
        lines.append(f"{commit.index:02d}. {describe_commit(commit)}")
        if show_files:
            files = ", ".join(commit.filtered_files or commit.files or ("(none)",))
            lines.append(f"    files: {files}")
    return "\n".join(lines)


def chronological(commits: list[BranchCommit], commit_hashes: set[str]) -> list[BranchCommit]:
    return [commit for commit in commits if commit.commit_hash in commit_hashes]


def build_plan_bundles(
    commits: list[BranchCommit],
    selected_hashes: list[str],
    *,
    mode: str,
) -> tuple[list[dict[str, object]], set[str]]:
    dependency_closure = build_dependency_resolver(commits)
    selected_set = set(selected_hashes)

    if mode == "consolidated":
        required: set[str] = set()
        for commit_hash in selected_hashes:
            required.update(dependency_closure(commit_hash))
        bundle_commits = chronological(commits, required)
        return (
            [
                {
                    "name": "PR #1",
                    "mode": mode,
                    "commits": bundle_commits,
                    "selected": [commit for commit in bundle_commits if commit.commit_hash in selected_set],
                    "auto_included": [commit for commit in bundle_commits if commit.commit_hash not in selected_set],
                }
            ],
            required,
        )

    if mode == "individual":
        bundles = []
        required_union: set[str] = set()
        for index, commit_hash in enumerate(selected_hashes, start=1):
            required = set(dependency_closure(commit_hash))
            required_union.update(required)
            bundle_commits = chronological(commits, required)
            bundles.append(
                {
                    "name": f"PR #{index}",
                    "mode": mode,
                    "commits": bundle_commits,
                    "selected": [commit for commit in bundle_commits if commit.commit_hash == commit_hash],
                    "auto_included": [commit for commit in bundle_commits if commit.commit_hash != commit_hash],
                }
            )
        return bundles, required_union

    if mode == "stacked":
        bundles = []
        emitted: set[str] = set()
        required_union: set[str] = set()
        for index, commit_hash in enumerate(selected_hashes, start=1):
            required = set(dependency_closure(commit_hash))
            required_union.update(required)
            bundle_hashes = required - emitted
            bundle_commits = chronological(commits, bundle_hashes)
            bundles.append(
                {
                    "name": f"PR #{index}",
                    "mode": mode,
                    "commits": bundle_commits,
                    "selected": [commit for commit in bundle_commits if commit.commit_hash == commit_hash],
                    "auto_included": [commit for commit in bundle_commits if commit.commit_hash != commit_hash],
                }
            )
            emitted.update(bundle_hashes)
        return bundles, required_union

    raise RuntimeError(f"Unsupported mode: {mode}")


def render_plan(
    *,
    repo_root: Path,
    base_ref: str,
    mode: str,
    commits: list[BranchCommit],
    selected_hashes: list[str],
    bundles: list[dict[str, object]],
    required_hashes: set[str],
) -> str:
    commit_by_hash = {commit.commit_hash: commit for commit in commits}
    selected_labels = [commit_by_hash[commit_hash].short_hash for commit_hash in selected_hashes]
    auto_labels = [commit.short_hash for commit in chronological(commits, required_hashes - set(selected_hashes))]

    lines = [
        f"Repo: {repo_root}",
        f"Base: {base_ref}",
        f"Mode: {mode}",
        f"Explicit selection: {', '.join(selected_labels) if selected_labels else 'none'}",
        f"Auto-included dependencies: {', '.join(auto_labels) if auto_labels else 'none'}",
        "",
    ]

    if not bundles:
        lines.append("No bundle plan produced.")
        return "\n".join(lines)

    for bundle in bundles:
        bundle_commits: list[BranchCommit] = bundle["commits"]  # type: ignore[assignment]
        selected_commits: list[BranchCommit] = bundle["selected"]  # type: ignore[assignment]
        auto_commits: list[BranchCommit] = bundle["auto_included"]  # type: ignore[assignment]
        lines.append(f"{bundle['name']} ({len(bundle_commits)} commit(s))")
        lines.append(
            f"  selected: {', '.join(commit.short_hash for commit in selected_commits) if selected_commits else 'none'}"
        )
        lines.append(
            f"  auto-included: {', '.join(commit.short_hash for commit in auto_commits) if auto_commits else 'none'}"
        )
        for commit in bundle_commits:
            marker = "*" if commit in auto_commits else "-"
            lines.append(f"  {marker} {describe_commit(commit)}")
        lines.append("")

    lines.append("Legend: '*' = auto-included dependency based on filtered file overlap.")
    return "\n".join(lines).rstrip()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    exclude_prefixes = normalize_prefixes(args.exclude_prefix)
    base_ref = resolve_base_ref(args.base, cwd=repo_root)
    commits = collect_branch_commits(cwd=repo_root, base_ref=base_ref, exclude_prefixes=exclude_prefixes)

    if args.command == "list":
        print(render_commit_list(commits, base_ref=base_ref, repo_root=repo_root, show_files=args.show_files))
        return 0

    if args.command == "plan":
        selected_hashes = resolve_selected_hashes(args.hash, commits, select_all=args.all)
        bundles, required_hashes = build_plan_bundles(commits, selected_hashes, mode=args.mode)
        print(
            render_plan(
                repo_root=repo_root,
                base_ref=base_ref,
                mode=args.mode,
                commits=commits,
                selected_hashes=selected_hashes,
                bundles=bundles,
                required_hashes=required_hashes,
            )
        )
        return 0

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
