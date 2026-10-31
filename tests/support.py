"""Helpers for labs that must not touch the project under test."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from autoresearch_runtime.runtime_env import hidden_process_kwargs

# CLI tests must not start persistent background dashboard processes by default.
# Lifecycle tests explicitly clear this flag when they exercise auto-start.
os.environ.setdefault("AUTORESEARCH_NO_DASHBOARD", "1")


def tree_digest(root: Path) -> dict[str, str]:
    """Hash every file under root, including git metadata."""
    digest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256()
        file_hash.update(path.read_bytes())
        digest[relative] = file_hash.hexdigest()
    return digest


def git_project(root: Path, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (files or {"value.txt": "1"}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, **hidden_process_kwargs())
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, **hidden_process_kwargs())
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "-m", "initial"],
        cwd=root, check=True,
        **hidden_process_kwargs(),
    )
    return root


def use_home(path: Path) -> str | None:
    previous = os.environ.get("AUTORESEARCH_HOME")
    os.environ["AUTORESEARCH_HOME"] = str(path)
    return previous
