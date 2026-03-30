from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCLUDE_PREFIXES = ("results/", "queue/", ".history/", ".vscode/")
COAUTHOR_TRAILER = "Co-authored-by: Codex <noreply@openai.com>"
GROUP_PRIORITY = {
    "scripts": 10,
    "specs": 20,
    "job_templates": 30,
    "tests": 40,
    "docs": 90,
}


@dataclass(frozen=True)
class WorktreeChange:
    status: str
    path: str


@dataclass(frozen=True)
class SnapshotGroup:
    key: str
    title: str
    commit_message: str
    changes: tuple[WorktreeChange, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or create reviewable snapshot commits from a dirty worktree.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_cmd = subparsers.add_parser("plan", help="Show grouped snapshot commits without mutating the repo.")
    plan_cmd.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo to inspect.")
    plan_cmd.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Path prefix to ignore. Repeatable.",
    )

    commit_cmd = subparsers.add_parser("commit", help="Create grouped snapshot commits from the current worktree.")
    commit_cmd.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repo to inspect.")
    commit_cmd.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Path prefix to ignore. Repeatable.",
    )
    commit_cmd.add_argument("--group", action="append", default=[], help="Only commit the named group(s). Repeatable.")
    commit_cmd.add_argument("--yes", action="store_true", help="Actually create the commits.")

    return parser.parse_args()


def run_git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
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


def normalize_prefixes(raw_prefixes: list[str]) -> tuple[str, ...]:
    prefixes: list[str] = []
    for prefix in [*DEFAULT_EXCLUDE_PREFIXES, *raw_prefixes]:
        normalized = prefix.replace("\\", "/").strip()
        if normalized and not normalized.endswith("/"):
            normalized += "/"
        if normalized and normalized not in prefixes:
            prefixes.append(normalized)
    return tuple(prefixes)


def collect_worktree_changes(*, cwd: Path, exclude_prefixes: tuple[str, ...]) -> list[WorktreeChange]:
    result = run_git(["status", "--short", "--untracked-files=all"], cwd=cwd)
    changes: list[WorktreeChange] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        status = raw_line[:2]
        path = raw_line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        normalized = path.replace("\\", "/")
        if any(normalized.startswith(prefix) for prefix in exclude_prefixes):
            continue
        changes.append(WorktreeChange(status=status, path=normalized))
    return changes


def classify_group(path: str) -> tuple[str, str, str]:
    if "/" not in path:
        if path.endswith((".md", ".txt", ".rst")):
            return ("docs", "Docs & Guidance", "docs: snapshot guidance updates")
        return ("root", "Root Files", "chore(root): snapshot root file updates")

    head = path.split("/", 1)[0]
    if head == "scripts":
        return ("scripts", "Scripts", "feat(scripts): snapshot automation updates")
    if head == "specs":
        return ("specs", "Research Specs", "feat(specs): snapshot research spec updates")
    if head == "job_templates":
        return ("job_templates", "Job Templates", "chore(job-templates): snapshot queue template updates")
    if head == "tests":
        return ("tests", "Tests", "test: snapshot test updates")
    if head in {"docs", "doc"}:
        return ("docs", "Docs & Guidance", "docs: snapshot documentation updates")
    return (head, head.replace("_", " ").title(), f"chore({head}): snapshot {head} updates")


def group_changes(changes: list[WorktreeChange]) -> list[SnapshotGroup]:
    grouped: dict[str, list[WorktreeChange]] = {}
    titles: dict[str, str] = {}
    messages: dict[str, str] = {}
    for change in changes:
        key, title, message = classify_group(change.path)
        grouped.setdefault(key, []).append(change)
        titles[key] = title
        messages[key] = message

    groups = [
        SnapshotGroup(
            key=key,
            title=titles[key],
            commit_message=messages[key],
            changes=tuple(sorted(grouped[key], key=lambda item: item.path)),
        )
        for key in grouped
    ]
    groups.sort(key=lambda group: (GROUP_PRIORITY.get(group.key, 50), group.key))
    return groups


def render_plan(*, repo_root: Path, groups: list[SnapshotGroup], exclude_prefixes: tuple[str, ...]) -> str:
    lines = [f"Repo: {repo_root}", f"Excluded prefixes: {', '.join(exclude_prefixes)}", ""]
    if not groups:
        lines.append("No non-excluded worktree changes found.")
        return "\n".join(lines)

    lines.append(f"Snapshot groups: {len(groups)}")
    lines.append("")
    for index, group in enumerate(groups, start=1):
        lines.append(f"{index:02d}. {group.key} - {group.title}")
        lines.append(f"    commit: {group.commit_message}")
        for change in group.changes:
            lines.append(f"    {change.status} {change.path}")
        lines.append("")
    return "\n".join(lines).rstrip()


def ensure_no_staged_changes(*, cwd: Path) -> None:
    result = run_git(["diff", "--cached", "--quiet"], cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError("Refusing to snapshot while the index already has staged changes.")


def select_groups(groups: list[SnapshotGroup], raw_names: list[str]) -> list[SnapshotGroup]:
    if not raw_names:
        return groups
    wanted = {name.strip() for name in raw_names if name.strip()}
    selected = [group for group in groups if group.key in wanted]
    missing = sorted(wanted - {group.key for group in selected})
    if missing:
        raise RuntimeError(f"Unknown group name(s): {', '.join(missing)}")
    return selected


def create_snapshot_commits(*, cwd: Path, groups: list[SnapshotGroup]) -> list[SnapshotGroup]:
    committed_groups: list[SnapshotGroup] = []
    for group in groups:
        run_git(["add", "--", *[change.path for change in group.changes]], cwd=cwd)
        staged = run_git(["diff", "--cached", "--name-only"], cwd=cwd)
        if not staged.stdout.strip():
            continue
        commit_message = f"{group.commit_message}\n\n{COAUTHOR_TRAILER}"
        run_git(["commit", "-m", commit_message], cwd=cwd)
        committed_groups.append(group)
    return committed_groups


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    exclude_prefixes = normalize_prefixes(args.exclude_prefix)
    changes = collect_worktree_changes(cwd=repo_root, exclude_prefixes=exclude_prefixes)
    groups = group_changes(changes)

    if args.command == "plan":
        print(render_plan(repo_root=repo_root, groups=groups, exclude_prefixes=exclude_prefixes))
        return 0

    if args.command == "commit":
        groups = select_groups(groups, args.group)
        print(render_plan(repo_root=repo_root, groups=groups, exclude_prefixes=exclude_prefixes))
        if not groups:
            return 0
        if not args.yes:
            print("\nDry run only. Re-run with --yes to create the snapshot commits.")
            return 0
        ensure_no_staged_changes(cwd=repo_root)
        committed_groups = create_snapshot_commits(cwd=repo_root, groups=groups)
        print("")
        if committed_groups:
            print("Committed snapshot groups:")
            for group in committed_groups:
                print(f"- {group.key}: {group.commit_message}")
        else:
            print("No commits were created.")
        return 0

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
