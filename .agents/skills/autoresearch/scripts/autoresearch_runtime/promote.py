"""Deliver an accepted candidate without touching the user's current branch."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .errors import LabError
from .gitutil import _git, _git_bytes, source_git
from .paths import resolve_home
from .runtime_env import hidden_process_kwargs
from .state import load_state
from .store import _sha256

# Worktrees write to the shared .git/config, so line endings are set per command instead.
_WORKTREE_GIT_CONFIG: tuple[str, ...] = ("-c", "core.autocrlf=false")
_CONFLICT_HINT = "Use --to-branch NAME to apply the patch in a separate worktree and resolve the conflict there"
_RESTORE_HINT = "Check these paths with git status and restore them from HEAD by hand"


def _repo_root(state: dict[str, Any], source: Path) -> Path:
    toplevel = state.get("source_toplevel")
    if isinstance(toplevel, str) and toplevel:
        return Path(toplevel).resolve()
    return source.resolve()


def _apply_directory(state: dict[str, Any]) -> str:
    prefix = state.get("project_prefix") or ""
    return prefix.strip("/")


def _changed_paths(candidate: Path, base: str, best: str) -> set[str]:
    """Project-relative paths that base-to-best touches. Renames are split, so both names are included."""
    output = _git_bytes(candidate, "diff", "--name-only", "-z", "--no-renames", base, best)
    return {os.fsdecode(item) for item in output.split(b"\0") if item}


def _repo_paths(targets: set[str], prefix: str) -> list[str]:
    """Map project-relative paths to paths relative to the repository root."""
    return [f"{prefix}/{name}".strip("/") if prefix else name for name in sorted(targets)]


def _dirty_paths(source: Path, paths: set[str], prefix: str) -> list[str]:
    """Target paths that have local changes, as repository-relative names. Git errors propagate, so the check fails closed."""
    output = source_git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True)
    fields = output.split(b"\0")
    dirty: set[str] = set()
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if len(entry) < 4:
            continue
        dirty.add(os.fsdecode(entry[3:]))
        if b"R" in entry[:2] or b"C" in entry[:2]:
            # A rename or copy is followed by its source path as a separate field.
            if index < len(fields):
                dirty.add(os.fsdecode(fields[index]))
            index += 1
    return [name for name in _repo_paths(paths, prefix) if name in dirty]


def _run_apply(
    workdir: Path,
    patch: Path,
    *,
    check: bool = False,
    three_way: bool = False,
    directory: str = "",
    git_config: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[bytes]:
    args = ["git", *git_config, "apply", "--binary", "--whitespace=nowarn", "--ignore-space-change"]
    if check:
        args.append("--check")
    if three_way:
        args.append("--3way")
    if directory:
        args.extend(["--directory", directory])
    args.append(str(patch))
    return subprocess.run(args, cwd=workdir, capture_output=True, check=False, **hidden_process_kwargs())


def _conflicted_paths(workdir: Path) -> list[str]:
    try:
        output = _git(workdir, "diff", "--name-only", "-z", "--diff-filter=U")
    except LabError:
        return []
    return [item for item in output.split("\0") if item]


def _ensure_drift(state: dict[str, Any], repo: Path, *, allow_drift: bool) -> None:
    head = state.get("source_head")
    if not head:
        return
    current = _git(repo, "rev-parse", "HEAD")
    if current != head and not allow_drift:
        raise LabError(
            "Source HEAD moved since the snapshot",
            code="E_DRIFT",
            hint="Re-run with --allow-drift to apply onto the new HEAD",
        )


def _verify_promotable(config: Any, state: dict[str, Any], targets: set[str]) -> None:
    """Refuse a promotion that would touch tampered inputs or non-editable files. Runs before the repository changes."""
    for item in state.get("inputs", []):
        inp_file = Path(item["path"])
        if not inp_file.is_file() or _sha256(inp_file) != item["sha256"]:
            raise LabError(f"Input file changed: {inp_file}", code="input_tampered")

    candidate = config.state_dir / "candidate"
    editable_files = getattr(config, "editable_files", None)
    editable_set = set(editable_files) if editable_files is not None else None

    for item in state.get("source", []):
        rel = item["path"]
        if editable_set is not None and rel not in editable_set:
            cand_frozen = candidate / rel
            if not cand_frozen.is_file() or _sha256(cand_frozen) != item["sha256"]:
                raise LabError(f"Protected frozen file was changed: {rel}", code="frozen_tampered")

    if editable_set is not None:
        for name in sorted(targets):
            if name not in editable_set:
                raise LabError(f"Attempted to promote non-editable file: {name}", code="frozen_tampered")


def _verify_prefix(source: Path, state: dict[str, Any]) -> None:
    """Refuse when the stored project prefix does not match where the source sits in its repository.

    Labs prepared before the toplevel fix stored an empty prefix for an unborn repository. Applying from the
    wrong directory would report success without changing the project, so such labs must be re-prepared.
    """
    live = str(source_git(source, "rev-parse", "--show-prefix")).strip("/")
    stored = _apply_directory(state)
    if live != stored:
        raise LabError(
            f"Stored project prefix '{stored}' does not match the source location '{live}'",
            code="E_PREFIX_MISMATCH",
            hint="Re-prepare the lab so the stored project prefix matches the source",
        )


def export_best(config: Any) -> dict[str, Any]:
    """Write the baseline-to-best patch. The source project is not modified."""
    state = load_state(config)
    candidate = config.state_dir / "candidate"
    root = _git(candidate, "rev-list", "--max-parents=0", "HEAD").splitlines()[0]
    best = state.get("best_commit") or root
    patch = _git_bytes(candidate, "diff", "--binary", root, best)
    destination = config.state_dir / "promotion" / "best.patch"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(patch)
    prefix = _apply_directory(state)
    report: dict[str, Any] = {
        "patch": str(destination),
        "bytes": len(patch),
        "from": root,
        "to": best,
        "project_prefix": prefix,
        "paths": sorted(_changed_paths(candidate, root, best)),
        "apply_check": None,
    }
    if not patch:
        report["apply_check"] = "skipped"
        return report
    repo = _repo_root(state, config.source_root.resolve())
    try:
        _git(repo, "rev-parse", "--git-dir")
    except LabError:
        report["apply_check"] = "skipped"
        return report
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".patch") as handle:
        handle.write(patch)
        temp_path = Path(handle.name)
    try:
        args = ["apply", "--check", "--binary", "--whitespace=nowarn"]
        if prefix:
            args.extend(["--directory", prefix])
        args.append(str(temp_path))
        try:
            source_git(repo, *args)
            report["apply_check"] = "ok"
        except LabError as error:
            report["apply_check"] = str(error)
    finally:
        temp_path.unlink(missing_ok=True)
    return report


def promote_to_branch(config: Any, branch: str, *, allow_drift: bool = False) -> dict[str, Any]:
    """Create a new branch in a worktree outside the project and commit the patch there."""
    if not branch or any(part in branch for part in (" ", "..", "/")):
        raise LabError("Branch name must be a single path segment", code="E_BRANCH")
    state = load_state(config)
    exported = export_best(config)
    source = config.source_root.resolve()
    repo = _repo_root(state, source)
    _ensure_drift(state, repo, allow_drift=allow_drift)
    targets = set(exported["paths"])
    _verify_promotable(config, state, targets)
    head = state.get("source_head") or _git(repo, "rev-parse", "HEAD")
    work = (resolve_home() / "promotions" / branch).resolve()
    work.parent.mkdir(parents=True, exist_ok=True)
    if work.exists():
        raise LabError(f"Promotion worktree already exists: {work}")
    _git(repo, *_WORKTREE_GIT_CONFIG, "worktree", "add", "-b", branch, str(work), head)
    result: dict[str, Any] = {
        "branch": branch,
        "worktree": str(work),
        "patch": exported["patch"],
        "project_prefix": exported.get("project_prefix", ""),
        "conflicts": [],
        "applied": False,
    }
    if not exported["bytes"]:
        result["applied"] = True
        return result
    patch_path = Path(exported["patch"])
    completed = _run_apply(
        work, patch_path, three_way=True, directory=_apply_directory(state), git_config=_WORKTREE_GIT_CONFIG,
    )
    conflicts = _conflicted_paths(work)
    if completed.returncode or conflicts:
        result["conflicts"] = conflicts or sorted(targets)
        result["apply_error"] = completed.stderr.decode("utf-8", errors="replace").strip()
        return result
    _git(work, *_WORKTREE_GIT_CONFIG, "add", "-A")
    _git(
        work,
        *_WORKTREE_GIT_CONFIG,
        "-c",
        "user.name=AutoResearch",
        "-c",
        "user.email=lab@local",
        "commit",
        "-q",
        "-m",
        f"autoresearch: {state.get('best_id') or 'best'}",
    )
    result["applied"] = True
    return result


def promote_in_place(config: Any, *, allow_drift: bool = False) -> dict[str, Any]:
    """Apply the best patch to the source worktree. Changes are staged but never committed."""
    state = load_state(config)
    exported = export_best(config)
    source = config.source_root.resolve()
    repo = _repo_root(state, source)
    _ensure_drift(state, repo, allow_drift=allow_drift)
    _verify_prefix(source, state)
    targets = set(exported["paths"])
    _verify_promotable(config, state, targets)
    prefix = _apply_directory(state)
    dirty = _dirty_paths(source, targets, prefix)
    if dirty:
        raise LabError(
            f"Source worktree has local changes in: {', '.join(dirty)}",
            code="E_DIRTY",
            hint="Commit or stash changes before --in-place, or use --to-branch",
        )
    if not exported["bytes"]:
        return {"applied": True, "patch": exported["patch"], "project_prefix": prefix, "conflicts": []}
    patch_path = Path(exported["patch"])
    apply_root = repo if prefix else source
    relative = _repo_paths(targets, prefix)
    # A rejected check writes nothing, so it can fail without any restore.
    checked = _run_apply(apply_root, patch_path, check=True, three_way=True, directory=prefix)
    if checked.returncode:
        detail = checked.stderr.decode("utf-8", errors="replace").strip()
        raise LabError(
            f"In-place apply conflicts in: {', '.join(sorted(targets))}: {detail}",
            code="E_CONFLICT",
            hint=_CONFLICT_HINT,
        )
    # --check --3way can pass for a patch that conflicts, so the real apply decides. Snapshot first, so a failure restores exact bytes.
    snapshot = _snapshot_targets(apply_root, relative)
    completed = _run_apply(apply_root, patch_path, three_way=True, directory=prefix)
    touched = set(relative)
    conflicts = [name for name in _conflicted_paths(apply_root) if name in touched]
    if completed.returncode or conflicts:
        names = conflicts or relative
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        _restore_applied_paths(apply_root, relative, snapshot)
        raise LabError(
            f"In-place apply conflicts in: {', '.join(names)}: {detail}",
            code="E_CONFLICT",
            hint=_CONFLICT_HINT,
        )
    return {"applied": True, "patch": exported["patch"], "project_prefix": prefix, "conflicts": []}


def _restore_git(workdir: Path, relative: list[str], command: str, *flags: str) -> bytes:
    completed = subprocess.run(
        ["git", "--literal-pathspecs", command, *flags, "--", *relative],
        cwd=workdir, capture_output=True, check=False, **hidden_process_kwargs(),
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LabError(
            f"Could not restore paths after a failed in-place apply: {', '.join(relative)}: {detail}",
            code="E_RESTORE",
            hint=_RESTORE_HINT,
        )
    return completed.stdout


def _entry_kind(path: Path) -> tuple[Any, ...]:
    """Describe what is at path, with the bytes and mode needed to put a regular file back exactly."""
    if os.path.islink(path):
        return ("link", os.readlink(path))
    if os.path.isdir(path):
        return ("dir",)
    if os.path.isfile(path):
        return ("file", path.read_bytes(), stat.S_IMODE(os.stat(path).st_mode))
    if os.path.lexists(path):
        raise LabError(f"Unsupported file type at {path}", code="E_CONFLICT", hint=_CONFLICT_HINT)
    return ("absent",)


def _current_kind(path: Path) -> str | None:
    if os.path.islink(path):
        return "link"
    if os.path.isdir(path):
        return "dir"
    if os.path.lexists(path):
        return "file"
    return None


def _by_depth(name: str) -> tuple[int, str]:
    return (name.count("/"), name)


def _snapshot_targets(workdir: Path, relative: list[str]) -> dict[str, tuple[Any, ...]]:
    """Record every target and each of its ancestor directories, so a failed apply can be undone exactly."""
    names: set[str] = set()
    for name in relative:
        parts = name.split("/")
        for depth in range(1, len(parts) + 1):
            names.add("/".join(parts[:depth]))
    snapshot: dict[str, tuple[Any, ...]] = {}
    for name in sorted(names, key=_by_depth):
        try:
            snapshot[name] = _entry_kind(workdir / name)
        except OSError as error:
            # Raised before the real apply, so the user's files have not changed yet.
            raise LabError(
                f"Could not read {name} before the in-place apply: {error}",
                code="E_SOURCE_READ",
                hint="Check that the file is readable, then run promote again",
            ) from error
    return snapshot


def _remove(path: Path, kind: str) -> None:
    if kind == "dir":
        shutil.rmtree(path)
    else:
        path.unlink()


def _restore_applied_paths(workdir: Path, relative: list[str], snapshot: dict[str, tuple[Any, ...]]) -> None:
    """Undo a failed in-place apply: every snapshotted path goes back to its recorded state, byte for byte.

    Removal is limited to entries that were absent, or of another kind, before the apply. Those can only hold
    content the patch created, so a path the user already had is never deleted, only rewritten from the snapshot.
    """
    if not relative:
        return
    # The index reset is attempted first, but its failure must not stop the working tree from being restored.
    index_error: LabError | None = None
    try:
        _restore_git(workdir, relative, "reset", "-q")
    except LabError as error:
        index_error = error
    names = sorted(snapshot, key=_by_depth)
    # Every entry is attempted. One stuck path must not leave the others patched, so failures are collected and named.
    failures: dict[str, str] = {}
    for name in reversed(names):
        entry = snapshot[name]
        path = workdir / name
        try:
            current = _current_kind(path)
            if current is None or entry[0] == current:
                continue
            _remove(path, current)
        except OSError as error:
            failures[name] = str(error)
    for name in names:
        entry = snapshot[name]
        if entry[0] == "absent":
            continue
        try:
            _write_entry(workdir / name, entry)
        except OSError as error:
            failures[name] = str(error)
    if index_error is not None or failures:
        details = []
        if index_error is not None:
            details.append(f"index reset failed: {index_error}")
        details.extend(f"{name}: {message}" for name, message in sorted(failures.items()))
        raise LabError(
            f"Restore after a failed in-place apply is incomplete: {'; '.join(details)}",
            code="E_RESTORE",
            hint=_RESTORE_HINT,
        )


def _write_entry(path: Path, entry: tuple[Any, ...]) -> None:
    """Put one snapshot entry back. A regular file whose bytes already match is not rewritten, so a read-only file the apply never touched is fine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if entry[0] == "dir":
        path.mkdir(exist_ok=True)
    elif entry[0] == "file":
        if os.path.isfile(path) and not os.path.islink(path) and path.read_bytes() == entry[1]:
            if stat.S_IMODE(os.stat(path).st_mode) != entry[2]:
                os.chmod(path, entry[2])
            return
        path.write_bytes(entry[1])
        os.chmod(path, entry[2])
    else:
        current = _current_kind(path)
        if current == "link" and os.readlink(path) == entry[1]:
            return
        if current is not None:
            _remove(path, current)
        os.symlink(entry[1], path)
