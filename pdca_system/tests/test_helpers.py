"""Shared test fixtures and helpers for pdca_system tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pdca_system.services.workflow import GitCommandError, GitService


def write_summary_file(summary: dict) -> str:
    """Write a summary dict to a temp file and return its path (for finish_pd_run/finish_ca_run tests)."""
    import os

    fd, path = tempfile.mkstemp(suffix=".summary.json")
    try:
        Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path
    finally:
        os.close(fd)


class RecordingGitService(GitService):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[tuple[tuple[str, ...], object | None]] = []

    def current_head(self) -> str:
        return "HEAD"

    def ensure_branch(self, branch: str, start_point: str) -> None:
        self.commands.append((("ensure_branch", branch, start_point), None))

    def _run_git(self, *args: str, cwd=None) -> str:
        self.commands.append((args, cwd))
        if args[:2] == ("rev-parse", "--short"):
            return "deadbee"
        return ""


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return None


class MergeFailingGitService(RecordingGitService):
    def promote_seed_branch(self, seed, target_branch=None):
        raise GitCommandError("simulated baseline merge conflict")


class NoOpPromoteGitService(RecordingGitService):
    """RecordingGitService that no-ops promote so tests don't depend on worktree/merge."""

    def promote_seed_branch(self, seed, target_branch=None):
        return "deadbee"


class NoBaselineMetricsStub:
    """Stub so ensure_baseline_result() does not return early; baseline CA gets queued and worktree created."""

    def get_all(self):
        return {}

    def get_for_branch(self, branch):
        return None

    def append_baseline_run(self, branch, target_metric_value):
        pass

    def append_promotion_for_branch(self, branch, record):
        pass
