"""Git helpers for candidate repos and read-only source inspection."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .errors import LabError
from .runtime_env import hidden_process_kwargs

_SOURCE_READ_ONLY = frozenset({"rev-parse", "ls-files", "status"})


def _git(directory: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=directory, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
        **hidden_process_kwargs(),
    )
    if completed.returncode:
        raise LabError(f"git {' '.join(args)}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _git_bytes(directory: Path, *args: str) -> bytes:
    """Read git stdout without a text decode, so CR bytes survive on Windows."""
    completed = subprocess.run(
        ["git", *args], cwd=directory, capture_output=True, check=False,
        **hidden_process_kwargs(),
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LabError(f"git {' '.join(args)}: {detail}")
    return completed.stdout


def source_git(project: Path, *args: str, binary: bool = False) -> str | bytes:
    """Read-only git against a user project. It does not take the repository lock."""
    if not args:
        raise LabError("Refusing to run git on the source project")
    command = args[0]
    if command == "apply":
        if "--check" not in args:
            raise LabError("Refusing to run git apply on the source project without --check")
    elif command not in _SOURCE_READ_ONLY:
        raise LabError(f"Refusing to run git {command} on the source project")
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args], cwd=project, env=environment,
        capture_output=True, check=False,
        **hidden_process_kwargs(),
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LabError(f"git {' '.join(args)}: {detail}")
    if binary:
        return completed.stdout
    text = completed.stdout.decode("utf-8", errors="replace")
    return text.rstrip("\r\n")


def _restore_candidate(config, commit: str) -> None:
    from .config import LabConfig

    candidate = config.resolve() if isinstance(config, Path) else (config.state_dir / "candidate").resolve()
    if not isinstance(config, Path) and candidate.parent != config.state_dir.resolve():
        raise LabError("Candidate path left the state directory")
    _git(candidate, "reset", "--hard", commit)
    _git(candidate, "clean", "-fd")
