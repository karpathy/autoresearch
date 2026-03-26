from __future__ import annotations

import unittest
from pathlib import Path

from typing import Any

from pdca_system.domain.models import SeedRecord, SeedStatus
from pdca_system.services.workflow import BASELINE_SEED_ID, GitCommandError, GitService, WorkflowService
from pdca_system.task import WORKTREE_ROOT


class NoBaselineMetricsStub:
    """Stub so ensure_baseline_result() does not return early; baseline CA gets queued and worktree created."""

    def get_all(self) -> dict[str, Any]:
        return {}

    def get_for_branch(self, branch: str) -> None:
        return None

    def append_baseline_run(self, branch: str, target_metric_value: float) -> None:
        pass

    def append_promotion_for_branch(self, branch: str, record: dict[str, Any]) -> None:
        pass


class RecordingGitService(GitService):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[tuple[tuple[str, ...], Path | None]] = []

    def current_head(self) -> str:
        return "HEAD"

    def ensure_branch(self, branch: str, start_point: str) -> None:
        self.commands.append((("ensure_branch", branch, start_point), None))

    def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        self.commands.append((args, cwd))
        if args[:2] == ("rev-parse", "--short"):
            return "deadbee"
        return ""


class AlreadyCheckedOutGitService(RecordingGitService):
    def __init__(self) -> None:
        super().__init__()
        self._failed_once = False

    def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        self.commands.append((args, cwd))
        if args[:3] == ("worktree", "add", "-B") and not self._failed_once:
            self._failed_once = True
            raise GitCommandError(
                "fatal: '__baseline__' is already checked out at '/autodl-fs/data/autoresearch/pdca_system/worktrees/__baseline__'"
            )
        if args[:2] == ("rev-parse", "--short"):
            return "deadbee"
        return ""


class GitServiceWorktreeTests(unittest.TestCase):
    def test_ensure_seed_worktrees_creates_worktree_on_seed_branch(self) -> None:
        """One branch per seed (SSOT): worktree is created on branch = seed_id."""
        service = RecordingGitService()
        seed = SeedRecord(
            seed_id="seed-abc123",
            prompt="test",
            created_at=0.0,
            updated_at=0.0,
            baseline_branch="master",
        )

        updated = service.ensure_seed_worktrees(seed)

        self.assertIsNotNone(updated.worktree_path)
        self.assertTrue(updated.worktree_path.endswith(str(Path("worktrees") / seed.seed_id)))

        add_calls = [cmd for cmd, _ in service.commands if cmd[:2] == ("worktree", "add")]
        self.assertEqual(len(add_calls), 1)
        # -B <branch> <path> <start>: branch name must be seed_id
        self.assertIn(seed.seed_id, add_calls[0], "worktree add should use seed_id as branch name")

    def test_promote_seed_branch_forces_baseline_worktree_checkout(self) -> None:
        service = RecordingGitService()
        seed = SeedRecord(
            seed_id="seed-abc123",
            prompt="test",
            created_at=0.0,
            updated_at=0.0,
            baseline_branch="master",
        )

        service.promote_seed_branch(seed)

        add_calls = [cmd for cmd, _ in service.commands if cmd[:2] == ("worktree", "add")]
        self.assertEqual(len(add_calls), 1)
        self.assertIn("--force", add_calls[0])

    def test_worktree_is_created_when_plan_do_run_starts(self) -> None:
        git_service = RecordingGitService()
        service = WorkflowService(git_service=git_service)
        service.metrics_repo.append_baseline_run("master", 1.23)
        seed = service.create_seed("defer worktree until pd starts", baseline_branch="master")

        service.queue_pd(seed.seed_id)
        queued_seed = service.require_seed(seed.seed_id)

        self.assertEqual(queued_seed.status, SeedStatus.queued)
        self.assertFalse(any(cmd[:2] == ("worktree", "add") for cmd, _ in git_service.commands))

        service.mark_run_started(seed.seed_id, queued_seed.latest_run_id)

        add_calls = [cmd for cmd, _ in git_service.commands if cmd[:2] == ("worktree", "add")]
        self.assertEqual(len(add_calls), 1)
        self.assertIn(seed.seed_id, add_calls[0])

    def test_ensure_seed_worktree_ready_recreates_missing_worktree_before_ca(self) -> None:
        git_service = RecordingGitService()
        service = WorkflowService(git_service=git_service)
        service.metrics_repo.append_baseline_run("master", 1.23)
        seed = service.create_seed("allow ca after baseline", baseline_branch="master")
        seed.status = SeedStatus.generated
        service.seed_repo.save(seed)
        service.queue_ca(seed.seed_id)

        self.assertFalse(any(cmd[:2] == ("worktree", "add") for cmd, _ in git_service.commands))

        updated = service.ensure_seed_worktree_ready(seed.seed_id)

        self.assertIsNotNone(updated.worktree_path)
        self.assertTrue(updated.worktree_path.endswith(str(Path("worktrees") / seed.seed_id)))
        add_calls = [cmd for cmd, _ in git_service.commands if cmd[:2] == ("worktree", "add")]
        self.assertEqual(len(add_calls), 1)
        self.assertIn(seed.seed_id, add_calls[0])

    def test_existing_worktree_is_reused_when_plan_do_run_starts(self) -> None:
        git_service = RecordingGitService()
        service = WorkflowService(git_service=git_service)
        service.metrics_repo.append_baseline_run("master", 1.23)
        seed = service.create_seed("reuse existing worktree", baseline_branch="master")
        worktree_dir = WORKTREE_ROOT / seed.seed_id
        worktree_dir.mkdir(parents=True, exist_ok=True)
        try:
            service.queue_pd(seed.seed_id)
            queued_seed = service.require_seed(seed.seed_id)

            service.mark_run_started(seed.seed_id, queued_seed.latest_run_id)

            add_calls = [cmd for cmd, _ in git_service.commands if cmd[:2] == ("worktree", "add")]
            self.assertEqual(add_calls, [])
            updated_seed = service.require_seed(seed.seed_id)
            self.assertEqual(updated_seed.worktree_path, str(worktree_dir))
        finally:
            if worktree_dir.exists():
                worktree_dir.rmdir()

    def test_baseline_worktree_is_created_when_baseline_ca_starts(self) -> None:
        git_service = RecordingGitService()
        service = WorkflowService(git_service=git_service, metrics_repo=NoBaselineMetricsStub())
        service.ensure_baseline_result("master")
        baseline_seed = service.require_seed(BASELINE_SEED_ID)
        service.mark_run_started(BASELINE_SEED_ID, baseline_seed.latest_run_id)
        baseline_seed = service.ensure_seed_worktree_ready(BASELINE_SEED_ID)

        self.assertIsNotNone(baseline_seed.worktree_path)
        self.assertTrue(baseline_seed.worktree_path.endswith(str(Path("worktrees") / BASELINE_SEED_ID)))
        add_calls = [cmd for cmd, _ in git_service.commands if cmd[:2] == ("worktree", "add")]
        self.assertGreaterEqual(len(add_calls), 1, "worktree add may run in ensure_baseline_result and again when CA starts")
        self.assertIn(BASELINE_SEED_ID, add_calls[0])

    def test_ensure_baseline_result_accepts_baseline_branch_ensures_worktree_then_queues_ca(self) -> None:
        """ensure_baseline_result(baseline_branch) ensures worktree from that branch first, then queues CA (no workaround)."""
        git_service = RecordingGitService()
        service = WorkflowService(git_service=git_service, metrics_repo=NoBaselineMetricsStub())
        service.ensure_baseline_result("master")

        baseline_seed = service.require_seed(BASELINE_SEED_ID)
        self.assertEqual(baseline_seed.baseline_branch, "master")
        self.assertIsNotNone(baseline_seed.worktree_path, "worktree should be ensured before CA is queued")
        self.assertIsNotNone(baseline_seed.latest_run_id, "CA should be queued after worktree is ready")
        add_calls = [c for c, _ in git_service.commands if c[:3] == ("worktree", "add", "-B")]
        if add_calls:
            self.assertIn(BASELINE_SEED_ID, add_calls[0])
            self.assertEqual(add_calls[0][-1], "master", "worktree add should use baseline_branch as start point")

    def test_ensure_seed_worktrees_recovers_stale_baseline_checkout_conflict(self) -> None:
        git_service = AlreadyCheckedOutGitService()
        baseline_seed = SeedRecord(
            seed_id=BASELINE_SEED_ID,
            prompt="baseline",
            created_at=0.0,
            updated_at=0.0,
            baseline_branch="master",
            status=SeedStatus.generated,
        )

        updated = git_service.ensure_seed_worktrees(baseline_seed)

        self.assertTrue(updated.worktree_path.endswith(str(Path("worktrees") / BASELINE_SEED_ID)))
        add_calls = [cmd for cmd, _ in git_service.commands if cmd[:2] == ("worktree", "add")]
        self.assertEqual(len(add_calls), 2, "should retry worktree add after recovery")
        prune_calls = [cmd for cmd, _ in git_service.commands if cmd[:2] == ("worktree", "prune")]
        self.assertTrue(prune_calls, "should prune stale worktree registrations before retry")


if __name__ == "__main__":
    unittest.main()
