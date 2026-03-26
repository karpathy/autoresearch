from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pdca_system.config import (
    DEFAULT_BASELINE_BRANCH,
    PROMOTION_THRESHOLD,
    TARGET_METRIC_KEY,
    TARGET_METRIC_LABEL,
    TARGET_METRIC_LOWER_IS_BETTER,
    best_target_metric_key,
    former_target_metric_key,
)
from pdca_system.domain.models import (
    DashboardColumn,
    DashboardViewModel,
    PlanIdea,
    RunStatus,
    SeedRecord,
    SeedStatus,
    StageName,
    StageRun,
)
from pdca_system.repositories.state import (
    BaselineBranchMapRepository,
    BaselineMetricsRepository,
    RunRepository,
    SeedRepository,
)
from pdca_system.logging_utils import get_logger
from pdca_system.task import (
    IN_PROGRESS_DIR,
    PDCA_SYSTEM_ROOT,
    STAGE_DIRS,
    WORKTREE_ROOT,
    get_daemon_config,
    get_daemon_status,
    move_to_error,
    now_ts,
    new_run_id,
    new_seed_id,
    read_task,
    write_task,
)


def _pd_summary_for_ca_task(summary: dict[str, Any], pd_run_id: str) -> dict[str, Any]:
    """Compact Plan-Do outcome for Check-Action task JSON and prompts."""
    refs = summary.get("source_refs")
    if not isinstance(refs, list):
        refs = []
    return {
        "idea": summary.get("idea"),
        "target_component": summary.get("target_component", "model"),
        "description": summary.get("description", ""),
        "source_refs": refs,
        "commit_sha": summary.get("commit_sha"),
        "completed_at": summary.get("completed_at"),
        "pd_run_id": pd_run_id,
    }


BASELINE_SEED_ID = "__baseline__"
LOGGER = get_logger(__name__)

# Short display labels for timeline (kind -> one-line text). Events not in this map use message as-is (truncated if long).
TIMELINE_SHORT_MESSAGES = {
    "seed.created": "Seed created",
    "seed.updated": "Seed updated",
    "seed.worktree_ready": "Worktree ready",
    "seed.reconciled": "Seed state reconciled",
    "ralph.enabled": "Ralph loop enabled",
    "ralph.disabled": "Ralph loop disabled",
    "ralph.requeued": "Ralph loop queued next Plan-Do",
    "ralph.requeue_failed": "Ralph loop could not queue next Plan-Do",
    "ralph.max_reached": "Ralph loop max iterations reached",
    "ralph.worktree_restored": "Ralph: worktree restored before Plan-Do",
    "ralph.worktree_restore_failed": "Ralph: worktree restore failed",
    "ralph.start_failed": "Ralph loop could not queue initial Plan-Do",
    "pd.queued": "Plan-Do queued",
    "pd.started": "Plan-Do started",
    "pd.completed": "Plan-Do completed",
    "pd.failed": "Plan-Do failed",
    "ca.queued": "Check-Action queued",
    "ca.merge_resolution_queued": "Check-Action (merge resolution) queued",
    "ca.started": "Check-Action started",
    "ca.completed": "Check-Action completed",
    "ca.merge_resolution_completed": "Check-Action (merge resolution) completed",
    "ca.merge_failed": "Merge into baseline failed",
    "pd.sync_resolution_queued": "Sync failed; merge resolution queued",
    "pd.sync_resolution_done": "Sync resolution done; Plan-Do re-queued",
    "pd.released": "Baseline ready; Plan-Do queued",
    "pd.waiting_for_baseline": "Waiting for baseline",
    "baseline.queued": "Baseline Check-Action queued",
    "ca.failed": "Check-Action failed",
    "ca.metrics_recovery_queued": "Check-Action metrics recovery queued",
    "ca.salvaged": "Check-Action run salvaged (metrics accepted)",
    "direct_code.queued": "Direct code agent queued",
    "direct_code.started": "Direct code agent started",
    "direct_code.completed": "Direct code agent completed",
    "direct_code.failed": "Direct code failed",
}


def _timeline_display_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return events in reverse order (newest first), deduplicated by (kind, message), with concise display text."""
    if not events:
        return []
    reversed_list = list(reversed(events))
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for e in reversed_list:
        kind = e.get("kind", "")
        message = e.get("message", "")
        key = (kind, message)
        if key in seen:
            continue
        seen.add(key)
        display = TIMELINE_SHORT_MESSAGES.get(kind)
        if display is not None:
            # Keep commit_sha / target_branch in a short suffix when present
            parts = [display]
            if e.get("commit_sha"):
                parts.append(f"commit: {e.get('commit_sha', '')[:7]}")
            if e.get("target_branch"):
                parts.append(f"→ {e.get('target_branch')}")
            display = " · ".join(parts)
        else:
            display = message if len(message) <= 80 else message[:77] + "..."
        out.append({**e, "display_message": display})
    return out


class GitCommandError(RuntimeError):
    pass


class SyncResolutionQueued(RuntimeError):
    """Raised when PD run cannot start because worktree sync with baseline failed; a sync-resolution CA task was queued."""


class DuplicateRunStartError(RuntimeError):
    """Raised when mark_run_started is called for a run that was already started (e.g. restored in-progress task)."""


class GitService:
    def __init__(self) -> None:
        pass

    def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            LOGGER.error("git executable is unavailable on PATH")
            raise GitCommandError("Git is not installed or not available on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            LOGGER.warning("git command failed: git %s | %s", " ".join(args), stderr or "unknown git error")
            raise GitCommandError(stderr or f"git {' '.join(args)} failed") from exc
        return result.stdout.strip()

    def repo_root(self) -> Path:
        return Path(self._run_git("rev-parse", "--show-toplevel"))

    def current_head(self) -> str:
        return self._run_git("rev-parse", "HEAD")

    def branch_exists(self, branch: str) -> bool:
        try:
            self._run_git("rev-parse", "--verify", branch)
            return True
        except GitCommandError:
            return False

    def ensure_branch(self, branch: str, start_point: str) -> None:
        if not self.branch_exists(branch):
            self._run_git("branch", branch, start_point)

    def list_branches(self) -> list[str]:
        output = self._run_git("branch", "--format=%(refname:short)")
        branches = [line.strip() for line in output.splitlines() if line.strip()]
        if not branches:
            # Unborn repositories can have HEAD pointing to a branch name even before first commit.
            try:
                head_branch = self._run_git("symbolic-ref", "--short", "HEAD").strip()
                if head_branch:
                    branches.append(head_branch)
            except GitCommandError:
                pass
        return sorted(set(branches))

    @staticmethod
    def is_seed_specific_branch(branch: str) -> bool:
        """True if this branch is the single working branch for a seed (seed_id), not a baseline choice."""
        if branch == BASELINE_SEED_ID:
            return True
        # One branch per seed: seed- + 6 hex chars, e.g. seed-e57b95
        if branch.startswith("seed-") and len(branch) == 11 and all(
            c in "abcdef0123456789" for c in branch[5:]
        ):
            return True
        return False

    def setup_error(self) -> str | None:
        try:
            self.repo_root()
            return None
        except GitCommandError as exc:
            return str(exc)

    def setup_error_for_branches(self, baseline_branch: str) -> str | None:
        try:
            root = self.repo_root()
            if not baseline_branch:
                return "Please select a baseline branch."
            if not self.branch_exists(baseline_branch):
                return (
                    f"Git repo found at {root}, but branch {baseline_branch!r} does not exist yet. "
                    "Select an existing baseline branch."
                )
            return None
        except GitCommandError as exc:
            return str(exc)

    def ensure_seed_worktrees(self, seed: SeedRecord) -> SeedRecord:
        """Ensure the seed worktree exists on the single branch for this seed: seed_id (SSOT)."""
        repo_head = self.current_head()
        self.ensure_branch(seed.baseline_branch, repo_head)

        seed_worktree = WORKTREE_ROOT / seed.seed_id
        if seed_worktree.exists():
            seed.worktree_path = str(seed_worktree)
            return seed
        # One branch per seed: branch name = seed_id, created from baseline.
        try:
            self._run_git("worktree", "add", "-B", seed.seed_id, str(seed_worktree), seed.baseline_branch)
        except GitCommandError as exc:
            # Recover from stale git worktree metadata like:
            # "__baseline__ is already checked out at /old/path/__baseline__"
            if not self._recover_checked_out_worktree_conflict(
                seed.seed_id, seed_worktree, seed.baseline_branch, str(exc)
            ):
                raise

        seed.worktree_path = str(seed_worktree)
        return seed

    @staticmethod
    def _extract_checked_out_path(error: str) -> Path | None:
        # git message example: fatal: '__baseline__' is already checked out at '/path'
        match = re.search(r"already checked out at ['\"]([^'\"]+)['\"]", error)
        if not match:
            return None
        return Path(match.group(1))

    def _recover_checked_out_worktree_conflict(
        self, branch: str, target_worktree: Path, start_point: str, error: str
    ) -> bool:
        if "already checked out at" not in error:
            return False
        # First, prune stale registrations from missing worktrees.
        try:
            self._run_git("worktree", "prune")
        except GitCommandError:
            pass
        conflict_path = self._extract_checked_out_path(error)
        if conflict_path is not None:
            # Force-remove the conflicting worktree from registry (same path after hard reset or different path).
            try:
                self._run_git("worktree", "remove", "--force", str(conflict_path))
            except GitCommandError:
                pass
            try:
                self._run_git("worktree", "prune")
            except GitCommandError:
                pass
        self._run_git("worktree", "add", "-B", branch, str(target_worktree), start_point)
        return True

    def commit_sha(self, ref: str) -> str:
        return self._run_git("rev-parse", "--short", ref)

    def _current_branch(self, cwd: Path | None = None) -> str | None:
        """Return the current branch name, or None if detached HEAD."""
        try:
            branch = self._run_git("branch", "--show-current", cwd=cwd)
            return branch.strip() or None
        except GitCommandError:
            return None

    def head_sha_at(self, cwd: Path) -> str:
        """Return the short commit SHA of HEAD in the given worktree directory."""
        return self._run_git("rev-parse", "--short", "HEAD", cwd=cwd)

    def reset_seed_branch_to(self, seed: SeedRecord, ref: str) -> None:
        """Reset the seed worktree's branch to the given ref (e.g. commit before Plan-Do).
        No-op for baseline seed or when worktree is missing."""
        if seed.seed_id == BASELINE_SEED_ID:
            return
        if not seed.worktree_path:
            return
        worktree_path = Path(seed.worktree_path)
        if not worktree_path.is_dir():
            return
        self._run_git("reset", "--hard", ref, cwd=worktree_path)

    def sync_seed_worktree_with_baseline(self, seed: SeedRecord) -> None:
        """Merge the baseline branch into the seed branch in the seed worktree.
        Call before each P run so the worktree has the latest baseline."""
        if seed.seed_id == BASELINE_SEED_ID:
            return
        if not seed.worktree_path:
            return
        worktree_path = Path(seed.worktree_path)
        if not worktree_path.is_dir():
            return
        self._run_git("merge", "--no-edit", seed.baseline_branch, cwd=worktree_path)

    def promote_seed_branch(
        self, seed: SeedRecord, target_branch: str | None = None
    ) -> str:
        """Merge the seed's branch (seed_id) into the target branch. Only CA (Check-Action) may call this; Plan-Do must never merge.
        If target_branch is None, use seed.baseline_branch (e.g. for normal seed promotion). For __baseline__ completion,
        pass the first user seed's selected branch so the merge goes there instead of a fixed config value.
        When the target branch is already checked out (e.g. master in the main repo), we merge in place to avoid
        'cannot force update the branch used by worktree' from creating a second worktree on the same branch."""
        merge_into = target_branch if target_branch is not None else seed.baseline_branch
        repo_root = self.repo_root()
        current = self._current_branch(cwd=repo_root)

        def do_merge(cwd: Path | None) -> None:
            self._run_git("merge", "--no-edit", seed.seed_id, cwd=cwd)

        def merge_already_up_to_date(cwd: Path | None) -> bool:
            try:
                self._run_git(
                    "merge-base", "--is-ancestor", seed.seed_id, "HEAD", cwd=cwd
                )
                return True
            except GitCommandError:
                return False

        if current == merge_into:
            # Target branch is already checked out (e.g. main repo on master). Merge in place.
            try:
                do_merge(cwd=repo_root)
            except GitCommandError as merge_err:
                if merge_already_up_to_date(cwd=repo_root):
                    return self.commit_sha(merge_into)
                raise merge_err
            return self.commit_sha(merge_into)

        # Target is not current branch: use a temporary worktree with a temp branch so we don't
        # try to check out the same branch in two worktrees (Git forbids that).
        baseline_worktree = WORKTREE_ROOT / "baseline"
        temp_branch = f"__promote_{merge_into}__"
        if baseline_worktree.exists():
            try:
                self._run_git("worktree", "remove", "--force", str(baseline_worktree))
            except GitCommandError:
                pass
            if baseline_worktree.exists():
                shutil.rmtree(baseline_worktree, ignore_errors=True)
        self._run_git(
            "worktree",
            "add",
            "--force",
            "-B",
            temp_branch,
            str(baseline_worktree),
            merge_into,
            cwd=repo_root,
        )
        try:
            try:
                do_merge(cwd=baseline_worktree)
            except GitCommandError as merge_err:
                if merge_already_up_to_date(cwd=baseline_worktree):
                    result_sha = self._run_git("rev-parse", "HEAD", cwd=baseline_worktree)
                    self._run_git("branch", "-f", merge_into, result_sha, cwd=repo_root)
                    return self.commit_sha(merge_into)
                raise merge_err
            result_sha = self._run_git("rev-parse", "HEAD", cwd=baseline_worktree)
            self._run_git("branch", "-f", merge_into, result_sha, cwd=repo_root)
            return self.commit_sha(merge_into)
        finally:
            try:
                self._run_git("worktree", "remove", "--force", str(baseline_worktree))
            except GitCommandError:
                pass
            try:
                self._run_git("branch", "-D", temp_branch)
            except GitCommandError:
                pass


class WorkflowService:
    def __init__(
        self,
        seed_repo: SeedRepository | None = None,
        run_repo: RunRepository | None = None,
        branch_map_repo: BaselineBranchMapRepository | None = None,
        metrics_repo: BaselineMetricsRepository | None = None,
        git_service: GitService | None = None,
    ) -> None:
        self.seed_repo = seed_repo or SeedRepository()
        self.run_repo = run_repo or RunRepository()
        self.branch_map_repo = branch_map_repo or BaselineBranchMapRepository()
        self.metrics_repo = metrics_repo or BaselineMetricsRepository()
        self.git_service = git_service or GitService()

    @staticmethod
    def _seed_worktree_path(seed_id: str) -> str:
        return str(WORKTREE_ROOT / seed_id)

    @staticmethod
    def _baseline_worktree_path() -> str:
        return str(WORKTREE_ROOT / BASELINE_SEED_ID)

    def _normalize_seed_runtime_state(self, seed: SeedRecord) -> SeedRecord:
        """Ensure baseline seed worktree_path matches the canonical path."""
        if seed.seed_id != BASELINE_SEED_ID:
            return seed
        expected_worktree = self._baseline_worktree_path()
        if seed.worktree_path == expected_worktree:
            return seed
        seed.worktree_path = expected_worktree
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        return seed

    def ensure_seed_worktree_ready(self, seed_id: str) -> SeedRecord:
        """Ensure the runtime seed worktree exists; recreate only when missing."""
        seed = self.require_seed(seed_id)
        if seed.seed_id == BASELINE_SEED_ID:
            expected_worktree = self._baseline_worktree_path()
            if Path(expected_worktree).is_dir():
                if seed.worktree_path != expected_worktree:
                    seed.worktree_path = expected_worktree
                    seed.updated_at = now_ts()
                    self.seed_repo.save(seed)
                return seed
            seed = self.git_service.ensure_seed_worktrees(seed)
            seed.updated_at = now_ts()
            self.seed_repo.save(seed)
            commit_sha = ""
            try:
                commit_sha = self.git_service.commit_sha(seed.baseline_branch)
            except GitCommandError:
                pass
            self.seed_repo.append_event(
                seed.seed_id,
                "seed.worktree_ready",
                "Recreated missing baseline worktree before the run started.",
                commit_sha=commit_sha or None,
            )
            return seed
        expected_worktree = self._seed_worktree_path(seed.seed_id)
        if Path(expected_worktree).is_dir():
            if seed.worktree_path != expected_worktree:
                seed.worktree_path = expected_worktree
                seed.updated_at = now_ts()
                self.seed_repo.save(seed)
            return seed
        seed = self.git_service.ensure_seed_worktrees(seed)
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        commit_sha = ""
        try:
            commit_sha = self.git_service.commit_sha(seed.seed_id)
        except GitCommandError:
            pass
        self.seed_repo.append_event(
            seed.seed_id,
            "seed.worktree_ready",
            "Recreated missing seed worktree before the run started.",
            commit_sha=commit_sha or None,
        )
        return seed

    def _preferred_baseline_branch(self) -> str:
        setup_error = self.git_service.setup_error()
        if setup_error is not None:
            return DEFAULT_BASELINE_BRANCH
        try:
            branches = [
                branch
                for branch in self.git_service.list_branches()
                if not self.git_service.is_seed_specific_branch(branch)
            ]
        except GitCommandError:
            return DEFAULT_BASELINE_BRANCH
        if branches and DEFAULT_BASELINE_BRANCH in branches:
            return DEFAULT_BASELINE_BRANCH
        return branches[0] if branches else DEFAULT_BASELINE_BRANCH

    def _first_user_seed_baseline_branch(self) -> str | None:
        """Return the baseline_branch of the earliest-created user seed (excluding __baseline__), or None."""
        user_seeds = [s for s in self.seed_repo.list() if s.seed_id != BASELINE_SEED_ID]
        if not user_seeds:
            return None
        first = min(user_seeds, key=lambda s: s.created_at)
        return first.baseline_branch or None

    def _enqueue_plan_do_run(self, seed: SeedRecord, event_kind: str = "pd.queued", event_message: str = "Queued Plan-Do stage for the seed.") -> StageRun:
        run = StageRun(
            run_id=new_run_id("pd"),
            seed_id=seed.seed_id,
            stage=StageName.pd,
            status=RunStatus.queued,
            task_id=new_run_id("task-pd"),
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        seed.status = SeedStatus.queued
        seed.updated_at = now_ts()
        seed.latest_run_id = run.run_id
        seed.last_error = None
        self.seed_repo.save(seed)
        self.run_repo.save(run)
        self.seed_repo.append_event(seed.seed_id, event_kind, event_message)
        write_task(
            "pd",
            {
                "seed_id": seed.seed_id,
                "run_id": run.run_id,
                "prompt": seed.prompt,
                "worktree_path": seed.worktree_path,
            },
            task_id=run.task_id,
        )
        return run

    def _release_seeds_waiting_for_baseline(self, branch: str) -> None:
        """Release seeds that were waiting for baseline result on the given branch."""
        branch_metrics = self.metrics_repo.get_for_branch(branch)
        if not branch_metrics or branch_metrics.get(best_target_metric_key()) is None:
            return
        waiting_seeds = sorted(self.seed_repo.list(), key=lambda item: item.created_at)
        for seed in waiting_seeds:
            if seed.seed_id == BASELINE_SEED_ID:
                continue
            if seed.baseline_branch != branch:
                continue
            if seed.status is not SeedStatus.queued or seed.latest_run_id is not None:
                continue
            self._enqueue_plan_do_run(
                seed,
                event_kind="pd.released",
                event_message="Baseline is ready; queued Plan-Do stage for the waiting seed.",
            )

    def _commit_sha_for_branch(self, branch: str) -> str:
        """Return current commit SHA for branch, or 'unknown' if unavailable (ensures baseline_metrics never has null commit_sha)."""
        try:
            sha = self.git_service.commit_sha(branch)
            return sha if (isinstance(sha, str) and sha.strip()) else "unknown"
        except GitCommandError:
            return "unknown"

    @staticmethod
    def _status_from_ca_signal(signal: str) -> SeedStatus:
        """Centralized mapping from CA signal to terminal seed status."""
        if signal == "positive_signal":
            return SeedStatus.promoted
        if signal == "error":
            return SeedStatus.failed
        return SeedStatus.passed

    def _reconcile_seed_status_signal(self, seed: SeedRecord) -> bool:
        """
        Auto-heal known inconsistent terminal combinations from historical data.

        Returns True when the seed was updated and persisted.
        """
        if seed.status is SeedStatus.passed and seed.latest_signal == "error":
            seed.status = SeedStatus.failed
            seed.updated_at = now_ts()
            self.seed_repo.save(seed)
            self.seed_repo.append_event(
                seed.seed_id,
                "seed.reconciled",
                "Reconciled inconsistent terminal state (passed + error) to failed.",
            )
            return True
        return False

    def create_seed(
        self,
        prompt: str,
        baseline_branch: str | None = None,
        ralph_loop_enabled: bool = False,
    ) -> SeedRecord:
        seed_id = new_seed_id()
        selected_baseline = (baseline_branch or DEFAULT_BASELINE_BRANCH).strip()
        seed = SeedRecord(
            seed_id=seed_id,
            prompt=prompt.strip(),
            status=SeedStatus.draft,
            created_at=now_ts(),
            updated_at=now_ts(),
            baseline_branch=selected_baseline,
            worktree_path=self._seed_worktree_path(seed_id),
            ralph_loop_enabled=ralph_loop_enabled,
        )
        self.seed_repo.save(seed)
        self.branch_map_repo.set_branch_for_seed(seed_id, selected_baseline)
        try:
            pass  # branch seed_id is created when Plan-Do is queued (ensure_seed_worktrees)
        except GitCommandError:
            # Keep seed creation non-blocking; branch creation will be retried at P queue time.
            pass
        self.seed_repo.append_event(seed.seed_id, "seed.created", "Seed created from prompt.")
        if ralph_loop_enabled:
            self.seed_repo.append_event(
                seed.seed_id,
                "ralph.enabled",
                "Ralph loop enabled; Plan-Do will auto-requeue after each Check-Action completion.",
            )
        LOGGER.info("created seed %s on baseline branch %s", seed.seed_id, selected_baseline)
        return seed

    def create_direct_code_seed(self, prompt: str) -> tuple[SeedRecord, StageRun]:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise RuntimeError("Prompt cannot be empty.")
        baseline_branch = self._preferred_baseline_branch()
        seed_id = new_seed_id("direct")
        now = now_ts()
        run = StageRun(
            run_id=new_run_id("direct"),
            seed_id=seed_id,
            stage=StageName.direct,
            status=RunStatus.queued,
            task_id=new_run_id("task-direct"),
            created_at=now,
            updated_at=now,
        )
        seed = SeedRecord(
            seed_id=seed_id,
            prompt=cleaned_prompt,
            status=SeedStatus.adapting,
            created_at=now,
            updated_at=now,
            baseline_branch=baseline_branch,
            worktree_path=str(PDCA_SYSTEM_ROOT.parent),
            latest_run_id=run.run_id,
            plan=PlanIdea(
                title="Direct code agent",
                target_component="project_root",
                description="Direct code agent run requested from the dashboard and executed from the project root.",
            ),
        )
        self.seed_repo.save(seed)
        self.branch_map_repo.set_branch_for_seed(seed_id, baseline_branch)
        self.run_repo.save(run)
        self.seed_repo.append_event(seed.seed_id, "seed.created", "Seed created from direct code agent prompt.")
        self.seed_repo.append_event(
            seed.seed_id,
            "direct_code.queued",
            "Queued direct code agent run from the project root.",
            run_id=run.run_id,
        )
        write_task(
            "direct",
            {
                "seed_id": seed.seed_id,
                "run_id": run.run_id,
                "prompt": seed.prompt,
                "worktree_path": None,
            },
            task_id=run.task_id,
        )
        return seed, run

    def _get_or_create_baseline_seed(self, baseline_branch: str | None = None) -> SeedRecord:
        """Return the baseline seed for establishing initial target metric; create with baseline_branch if missing."""
        seed = self.seed_repo.get(BASELINE_SEED_ID)
        if seed is not None:
            return self._normalize_seed_runtime_state(seed)
        branch = baseline_branch if baseline_branch is not None else (
            self._first_user_seed_baseline_branch() or DEFAULT_BASELINE_BRANCH
        )
        seed = SeedRecord(
            seed_id=BASELINE_SEED_ID,
            prompt="Baseline measurement: run training on current code without changes.",
            status=SeedStatus.draft,
            created_at=now_ts(),
            updated_at=now_ts(),
            baseline_branch=branch,
            worktree_path=self._baseline_worktree_path(),
            ralph_loop_enabled=False,
        )
        self.seed_repo.save(seed)
        self.branch_map_repo.set_branch_for_seed(BASELINE_SEED_ID, branch)
        self.seed_repo.append_event(
            seed.seed_id,
            "seed.created",
            "Baseline seed created for initial measurement.",
        )
        return seed

    def ensure_baseline_result(self, baseline_branch: str) -> None:
        """
        If there is no baseline result (best target metric) for the given branch, ensure a baseline seed exists for that
        branch, ensure its worktree is checked out from baseline_branch, then queue CA to establish baseline.
        Idempotent; safe to call before queue_pd for any user seed. Call with seed.baseline_branch.
        """
        seed = self._get_or_create_baseline_seed(baseline_branch)
        branch_metrics = self.metrics_repo.get_for_branch(baseline_branch)
        if branch_metrics and branch_metrics.get(best_target_metric_key()) is not None:
            return
        if seed.baseline_branch != baseline_branch:
            return
        if seed.status in (SeedStatus.ca_queued, SeedStatus.adapting, SeedStatus.running):
            return
        if seed.status in (SeedStatus.passed, SeedStatus.failed, SeedStatus.promoted):
            branch_metrics = self.metrics_repo.get_for_branch(baseline_branch)
            if branch_metrics and branch_metrics.get(best_target_metric_key()) is not None:
                return
        setup_error = self.git_service.setup_error()
        if setup_error is not None:
            return
        try:
            self.git_service.ensure_branch(baseline_branch, self.git_service.current_head())
        except GitCommandError:
            return
        setup_error = self.git_service.setup_error_for_branches(baseline_branch)
        if setup_error is not None:
            return
        self.ensure_seed_worktree_ready(BASELINE_SEED_ID)
        seed = self.require_seed(BASELINE_SEED_ID)
        seed.status = SeedStatus.generated
        seed.plan = PlanIdea(title="Baseline", description="No changes; measure current baseline.")
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "baseline.queued",
            "Queued Check-Action to establish baseline result before first seed.",
        )
        self.queue_ca(seed.seed_id)

    def set_ralph_loop(self, seed_id: str, enabled: bool) -> SeedRecord:
        seed = self.require_seed(seed_id)
        if seed.ralph_loop_enabled == enabled:
            return seed
        seed.ralph_loop_enabled = enabled
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        if enabled:
            self.seed_repo.append_event(
                seed.seed_id,
                "ralph.enabled",
                "Ralph loop enabled; Plan-Do will auto-requeue after each Check-Action completion.",
            )
        else:
            self.seed_repo.append_event(seed.seed_id, "ralph.disabled", "Ralph loop disabled by user.")
        return seed

    def can_edit_seed_prompt(self, seed: SeedRecord) -> bool:
        return seed.status in {SeedStatus.draft, SeedStatus.queued}

    def update_seed_prompt(self, seed_id: str, prompt: str) -> SeedRecord:
        seed = self.require_seed(seed_id)
        if not self.can_edit_seed_prompt(seed):
            raise RuntimeError("Seed prompt can only be edited before Plan-Do starts.")
        updated_prompt = prompt.strip()
        if not updated_prompt:
            raise RuntimeError("Prompt cannot be empty.")
        if updated_prompt == seed.prompt:
            return seed
        seed.prompt = updated_prompt
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        self.seed_repo.append_event(seed.seed_id, "seed.updated", "Seed prompt was edited before execution.")
        return seed

    def queue_pd(self, seed_id: str) -> StageRun | None:
        seed = self.require_seed(seed_id)
        branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch) if seed_id != BASELINE_SEED_ID else None
        has_baseline = branch_metrics is not None and branch_metrics.get(best_target_metric_key()) is not None
        if seed_id != BASELINE_SEED_ID and not has_baseline:
            self.ensure_baseline_result(seed.baseline_branch)
            branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
            has_baseline = branch_metrics is not None and branch_metrics.get(best_target_metric_key()) is not None
            if not has_baseline:
                baseline_seed = self.seed_repo.get(BASELINE_SEED_ID)
                # Only wait for baseline when the baseline seed is for this branch (e.g. master).
                # For another branch (e.g. dev), no baseline run is queued for it, so allow planning;
                # the first CA completion on this branch will establish baseline metrics.
                if baseline_seed is not None and baseline_seed.baseline_branch == seed.baseline_branch:
                    if not (seed.status is SeedStatus.queued and seed.latest_run_id is None):
                        seed.status = SeedStatus.queued
                        seed.updated_at = now_ts()
                        seed.latest_run_id = None
                        seed.last_error = None
                        self.seed_repo.save(seed)
                        self.seed_repo.append_event(
                            seed.seed_id,
                            "pd.waiting_for_baseline",
                            "Baseline run is still in progress; Plan-Do will queue after baseline finishes.",
                        )
                    return None
                # Branch has no baseline and is not the baseline seed's branch: proceed with planning.
        setup_error = self.git_service.setup_error()
        if setup_error is not None:
            raise RuntimeError(setup_error)
        try:
            self.git_service.ensure_branch(seed.baseline_branch, self.git_service.current_head())
        except GitCommandError:
            pass
        setup_error = self.git_service.setup_error_for_branches(seed.baseline_branch)
        if setup_error is not None:
            raise RuntimeError(setup_error)
        return self._enqueue_plan_do_run(seed)

    def _pd_summary_from_seed(self, seed: SeedRecord) -> dict[str, Any] | None:
        """Build Plan-Do summary for CA when not passed explicitly (e.g. manual queue)."""
        if seed.seed_id == BASELINE_SEED_ID or seed.plan is None:
            return None
        plan = seed.plan
        if not (plan.title or plan.description or plan.commit_sha):
            return None
        out: dict[str, Any] = {
            "idea": plan.title,
            "target_component": plan.target_component,
            "description": plan.description,
            "source_refs": list(plan.source_refs) if plan.source_refs else [],
            "commit_sha": plan.commit_sha,
        }
        pd_runs = [
            r
            for r in self.run_repo.list(seed.seed_id)
            if r.stage == StageName.pd and r.status == RunStatus.succeeded
        ]
        if pd_runs:
            best = max(pd_runs, key=lambda r: r.updated_at)
            ca = best.summary.get("completed_at")
            if ca:
                out["completed_at"] = ca
            out["pd_run_id"] = best.run_id
        return out

    def queue_ca(
        self,
        seed_id: str,
        merge_resolution: bool = False,
        metrics_recovery: bool = False,
        source_run_id: str | None = None,
        source_stdout_log_path: str | None = None,
        source_stderr_log_path: str | None = None,
        last_metrics: dict[str, Any] | None = None,
        last_summary: dict[str, Any] | None = None,
        commit_sha_before_pd: str | None = None,
        pd_summary: dict[str, Any] | None = None,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        if not metrics_recovery and seed.status in {SeedStatus.draft, SeedStatus.queued, SeedStatus.planning}:
            raise RuntimeError("Run Plan-Do first. Check-Action is available after code is generated into the seed branch.")
        if not metrics_recovery:
            setup_error = self.git_service.setup_error_for_branches(seed.baseline_branch)
            if setup_error is not None:
                raise RuntimeError(setup_error)
        run = StageRun(
            run_id=new_run_id("ca"),
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.queued,
            task_id=new_run_id("task-ca"),
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        if seed.seed_id != BASELINE_SEED_ID:
            try:
                # Ref to restore worktree to on negative signal (commit before PD when from finish_pd_run, else baseline).
                run.summary["commit_sha_before_pd"] = (
                    commit_sha_before_pd
                    if commit_sha_before_pd is not None
                    else self.git_service.commit_sha(seed.baseline_branch)
                )
            except GitCommandError:
                pass
        seed.status = SeedStatus.ca_queued
        seed.updated_at = now_ts()
        seed.latest_run_id = run.run_id
        seed.last_error = None
        self.seed_repo.save(seed)
        self.run_repo.save(run)
        self.seed_repo.append_event(
            seed.seed_id,
            "ca.merge_resolution_queued" if merge_resolution else "ca.queued",
            "Queued Check-Action for merge conflict resolution."
            if merge_resolution
            else "Queued Check-Action for metrics recovery from saved logs."
            if metrics_recovery
            else "Queued Check-Action stage for the seed.",
        )
        payload = {
            "seed_id": seed.seed_id,
            "run_id": run.run_id,
            "prompt": seed.prompt,
            "worktree_path": seed.worktree_path,
            "merge_resolution": merge_resolution,
            "metrics_recovery": metrics_recovery,
        }
        if merge_resolution:
            payload["baseline_branch"] = seed.baseline_branch
            if last_metrics is not None:
                payload["last_metrics"] = last_metrics
            if last_summary is not None:
                payload["last_summary"] = last_summary
        if metrics_recovery:
            payload["source_run_id"] = source_run_id
            payload["source_stdout_log_path"] = source_stdout_log_path
            payload["source_stderr_log_path"] = source_stderr_log_path
            payload["worktree_path"] = None
        resolved_pd_summary = pd_summary if pd_summary is not None else self._pd_summary_from_seed(seed)
        # pd_summary lives on the queue JSON for tooling; daemon builds the agent prompt as a derived
        # view (labeled Plan-Do block + inline task JSON without repeating this key).
        if resolved_pd_summary:
            payload["pd_summary"] = resolved_pd_summary
        write_task("ca", payload, task_id=run.task_id)
        LOGGER.info(
            "queued CA run %s for seed %s (merge_resolution=%s, metrics_recovery=%s)",
            run.run_id,
            seed.seed_id,
            merge_resolution,
            metrics_recovery,
        )
        return run

    def terminate_non_completed_tasks(self, seed_id: str) -> int:
        """Mark every run for this seed that is not in the completed column (succeeded/failed) as failed and move any queued/in-progress task to error. Disables Ralph loop for this seed. Returns the number of runs terminated."""
        seed = self.require_seed(seed_id)
        runs = self.run_repo.list(seed_id)
        completed = {RunStatus.succeeded, RunStatus.failed}
        count = 0
        for run in runs:
            if run.status in completed:
                continue
            task_path: Path | None = None
            stage_dir = STAGE_DIRS.get(run.stage.value)
            if stage_dir is not None:
                candidate = stage_dir / f"{run.task_id}.json"
                if candidate.exists():
                    task_path = candidate
            if task_path is None:
                candidate = IN_PROGRESS_DIR / f"{run.task_id}.json"
                if candidate.exists():
                    task_path = candidate
            self.mark_run_failed(
                seed_id,
                run.run_id,
                "Terminated by user.",
                task_path=task_path,
            )
            count += 1
        if seed.ralph_loop_enabled:
            self.set_ralph_loop(seed_id, False)
        return count

    def queue_sync_resolution(self, seed_id: str) -> StageRun:
        """Queue a merge-resolution run to resolve 'merge baseline into seed' in the seed worktree (e.g. after sync failed before PD)."""
        seed = self.require_seed(seed_id)
        if seed.seed_id == BASELINE_SEED_ID:
            raise RuntimeError("Sync resolution is not used for the baseline seed.")
        setup_error = self.git_service.setup_error_for_branches(seed.baseline_branch)
        if setup_error is not None:
            raise RuntimeError(setup_error)
        run = StageRun(
            run_id=new_run_id("ca"),
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.queued,
            task_id=new_run_id("task-ca"),
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        seed.status = SeedStatus.ca_queued
        seed.updated_at = now_ts()
        seed.latest_run_id = run.run_id
        seed.last_error = None
        self.seed_repo.save(seed)
        self.run_repo.save(run)
        self.seed_repo.append_event(
            seed.seed_id,
            "pd.sync_resolution_queued",
            "Worktree sync with baseline failed; queued merge-resolution to resolve and re-run Plan-Do.",
        )
        payload = {
            "seed_id": seed.seed_id,
            "run_id": run.run_id,
            "prompt": seed.prompt,
            "worktree_path": seed.worktree_path,
            "baseline_branch": seed.baseline_branch,
            "sync_resolution": True,
        }
        write_task("ca", payload, task_id=run.task_id)
        return run

    def finish_sync_resolution(self, seed_id: str, run_id: str) -> None:
        """Mark sync-resolution run completed and re-queue Plan-Do for the seed."""
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        run.status = RunStatus.succeeded
        run.updated_at = now_ts()
        self.run_repo.save(run)
        seed.status = SeedStatus.queued
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "pd.sync_resolution_done",
            "Sync resolution completed; Plan-Do re-queued.",
            run_id=run_id,
        )
        self._enqueue_plan_do_run(
            seed,
            event_kind="pd.queued",
            event_message="Re-queued Plan-Do after sync resolution.",
        )

    def require_seed(self, seed_id: str) -> SeedRecord:
        seed = self.seed_repo.get(seed_id)
        if seed is None:
            raise KeyError(f"Unknown seed_id={seed_id}")
        return self._normalize_seed_runtime_state(seed)

    def require_run(self, run_id: str) -> StageRun:
        run = self.run_repo.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id={run_id}")
        return run

    def set_run_agent_type(self, seed_id: str, run_id: str, agent_type: str) -> StageRun:
        """Record which agent ran this run. Call right after invoking the agent, before finish/fail."""
        run = self.require_run(run_id)
        run.agent_type = agent_type
        if run.summary is None:
            run.summary = {}
        run.summary["agent_type"] = agent_type
        run.updated_at = now_ts()
        self.run_repo.save(run)
        return run

    def is_seed_eligible_for_stage(self, seed_id: str | None, stage: str) -> bool:
        """True if this seed is in a state that allows the given stage to run (used at claim time to avoid PD/CA races)."""
        if not seed_id:
            return False
        seed = self.seed_repo.get(seed_id)
        if seed is None:
            return False
        seed = self._normalize_seed_runtime_state(seed)
        if stage == "pd":
            return seed.status not in (SeedStatus.adapting, SeedStatus.running, SeedStatus.ca_queued)
        if stage == "ca":
            return seed.status is not SeedStatus.planning
        if stage == "direct":
            return True
        return False

    def mark_run_started(self, seed_id: str, run_id: str) -> tuple[SeedRecord, StageRun]:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        if run.status != RunStatus.queued:
            raise DuplicateRunStartError(
                f"Run {run_id} already started (status={run.status}); possible restored in-progress task."
            )
        run.status = RunStatus.running
        run.updated_at = now_ts()
        if run.stage is StageName.pd:
            setup_error = self.git_service.setup_error()
            if setup_error is not None:
                raise RuntimeError(setup_error)
            try:
                self.git_service.ensure_branch(seed.baseline_branch, self.git_service.current_head())
            except GitCommandError:
                pass
            setup_error = self.git_service.setup_error_for_branches(seed.baseline_branch)
            if setup_error is not None:
                raise RuntimeError(setup_error)
            seed = self.ensure_seed_worktree_ready(seed.seed_id)
            # Sync seed worktree with baseline branch before PD so Plan-Do runs from latest baseline.
            try:
                self.git_service.sync_seed_worktree_with_baseline(seed)
            except GitCommandError as sync_err:
                run.status = RunStatus.failed
                run.error = f"Worktree sync with baseline failed: {sync_err}"
                self.run_repo.save(run)
                self.queue_sync_resolution(seed.seed_id)
                raise SyncResolutionQueued(
                    f"Worktree sync with baseline failed: {sync_err}. Queued merge-resolution."
                ) from sync_err
            # Record baseline target metric at sync time for positive/negative/neutral judgement in CA.
            branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
            former = branch_metrics.get(best_target_metric_key()) if branch_metrics else None
            if run.summary is None:
                run.summary = {}
            run.summary[former_target_metric_key()] = former
            seed.former_target_metric_value = float(former) if former is not None else None
            if seed.worktree_path:
                worktree_path = Path(seed.worktree_path)
                if worktree_path.is_dir():
                    try:
                        run.summary["commit_sha_before_pd"] = self.git_service.head_sha_at(
                            worktree_path
                        )
                    except GitCommandError:
                        pass
            seed.status = SeedStatus.planning
            event_kind = "pd.started"
            event_message = "Plan-Do stage started in the candidate worktree."
        else:
            seed.status = SeedStatus.adapting
            event_kind = "ca.started"
            event_message = (
                "Baseline measurement started in the baseline worktree."
                if seed.seed_id == BASELINE_SEED_ID
                else "Check-Action stage started in the seed worktree."
            )
        seed.updated_at = now_ts()
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(seed.seed_id, event_kind, event_message, run_id=run_id)
        return seed, run

    def mark_direct_code_run_started(self, seed_id: str, run_id: str) -> tuple[SeedRecord, StageRun]:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        run.status = RunStatus.running
        run.updated_at = now_ts()
        seed.status = SeedStatus.adapting
        seed.updated_at = now_ts()
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "direct_code.started",
            "Direct code agent started from the project root.",
            run_id=run_id,
        )
        return seed, run

    def mark_direct_code_run_failed(
        self,
        seed_id: str,
        run_id: str,
        error: str,
        task_path: Path | None = None,
        prompt_path: str | None = None,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
    ) -> None:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        run.status = RunStatus.failed
        run.updated_at = now_ts()
        run.error = error
        if prompt_path is not None:
            run.prompt_path = prompt_path
        if log_path is not None:
            run.log_path = log_path
        if stderr_log_path is not None:
            run.stderr_log_path = stderr_log_path
        seed.status = SeedStatus.failed
        seed.updated_at = now_ts()
        seed.last_error = error
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(seed.seed_id, "direct_code.failed", error, run_id=run_id)
        if task_path is not None and task_path.exists():
            move_to_error(task_path)

    def _ralph_try_restore_worktree(self, seed: SeedRecord, ref: str | None) -> None:
        """Reset seed worktree to ref (e.g. commit before Plan-Do) and log result. No-op if ref missing or baseline seed."""
        if not ref or not str(ref).strip() or seed.seed_id == BASELINE_SEED_ID:
            return
        try:
            self.git_service.reset_seed_branch_to(seed, ref)
            self.seed_repo.append_event(
                seed.seed_id,
                "ralph.worktree_restored",
                "Restored seed worktree to commit before Plan-Do for next Plan-Do run.",
                commit_sha=ref,
            )
        except GitCommandError as exc:
            self.seed_repo.append_event(
                seed.seed_id,
                "ralph.worktree_restore_failed",
                f"Could not restore seed worktree to commit before Plan-Do: {exc}",
                commit_sha=ref,
            )

    def _ralph_should_requeue(self, seed_id: str) -> bool:
        """True if Ralph loop may requeue (max not reached). Config ralph_max_loop ≤0 = infinite."""
        cfg = get_daemon_config()
        max_loop = cfg.get("ralph_max_loop", 0)
        if max_loop <= 0:
            return True
        events = self.seed_repo.events(seed_id)
        requeued = sum(1 for e in events if e.get("kind") == "ralph.requeued")
        return requeued < max_loop

    def mark_run_failed(
        self,
        seed_id: str,
        run_id: str,
        error: str,
        task_path: Path | None = None,
        prompt_path: str | None = None,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
    ) -> None:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        task_payload: dict[str, Any] = {}
        if task_path is not None and task_path.exists():
            task_payload = read_task(task_path)
        run.status = RunStatus.failed
        run.updated_at = now_ts()
        run.error = error
        if prompt_path is not None:
            run.prompt_path = prompt_path
        if log_path is not None:
            run.log_path = log_path
        if stderr_log_path is not None:
            run.stderr_log_path = stderr_log_path
        seed.status = SeedStatus.failed
        seed.updated_at = now_ts()
        seed.last_error = error
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(seed.seed_id, f"{run.stage.value}.failed", error, run_id=run_id)
        LOGGER.error("%s run %s failed for seed %s: %s", run.stage.value.upper(), run_id, seed_id, error)
        if (
            run.stage is StageName.ca
            and seed.ralph_loop_enabled
            and seed.seed_id != BASELINE_SEED_ID
            and (task_payload.get("merge_resolution") is True or task_payload.get("metrics_recovery") is True)
        ):
            self._ralph_try_restore_worktree(seed, run.summary.get("commit_sha_before_pd"))
            if self._ralph_should_requeue(seed.seed_id):
                try:
                    self.queue_pd(seed.seed_id)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.requeued",
                        "Ralph loop queued the next Plan-Do run after failed Check-Action.",
                    )
                except (RuntimeError, GitCommandError) as exc:
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.requeue_failed",
                        f"Ralph loop could not queue the next Plan-Do run after failed Check-Action: {exc}",
                    )
            else:
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ralph.max_reached",
                    "Ralph loop max iterations reached; not queuing another Plan-Do run.",
                )
        if task_path is not None and task_path.exists():
            move_to_error(task_path)

    def finish_direct_code_run(
        self,
        seed_id: str,
        run_id: str,
        stdout: str,
        stderr: str | None = None,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
        prompt_path: str | None = None,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        run.status = RunStatus.succeeded
        run.updated_at = now_ts()
        run.log_path = log_path
        run.stderr_log_path = stderr_log_path
        run.prompt_path = prompt_path
        run.summary = {
            "mode": "direct_code_agent",
            "cwd": str(PDCA_SYSTEM_ROOT.parent),
            "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
            "stderr_bytes": len((stderr or "").encode("utf-8", errors="replace")),
        }
        run.signal = "direct_code_completed"
        seed.status = SeedStatus.passed
        seed.updated_at = now_ts()
        seed.latest_signal = run.signal
        seed.last_error = None
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "direct_code.completed",
            "Direct code agent completed from the project root.",
            run_id=run_id,
        )
        return run

    @staticmethod
    def load_summary(path: Path) -> dict[str, object]:
        """Load run summary JSON from file. Raises FileNotFoundError or json.JSONDecodeError on failure."""
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def finish_pd_run(
        self,
        seed_id: str,
        run_id: str,
        summary_path: str,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
        prompt_path: str | None = None,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        summary = self.load_summary(Path(summary_path))
        seed.plan = PlanIdea(
            title=summary.get("idea", "Generated plan"),
            target_component=summary.get("target_component", "model"),
            description=summary.get("description", ""),
            source_refs=summary.get("source_refs", []),
            commit_sha=summary.get("commit_sha"),
        )
        # Single branch per seed (SSOT): worktree is already on seed_id branch.
        commit_sha = self.git_service.commit_sha(seed.seed_id)
        run.status = RunStatus.succeeded
        run.updated_at = now_ts()
        run.log_path = log_path
        run.stderr_log_path = stderr_log_path
        run.prompt_path = prompt_path
        # Preserve run.summary fields set earlier (e.g. commit_sha_before_pd) when merging PD output.
        run.summary = run.summary | summary | {"commit_sha": commit_sha}
        seed.status = SeedStatus.generated
        seed.updated_at = now_ts()
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "pd.completed",
            "Plan-Do completed on seed branch.",
            commit_sha=commit_sha,
        )
        self.queue_ca(
            seed.seed_id,
            commit_sha_before_pd=run.summary.get("commit_sha_before_pd"),
            pd_summary=_pd_summary_for_ca_task(summary, run_id),
        )
        return run

    def finish_ca_run(
        self,
        seed_id: str,
        run_id: str,
        summary_path: str,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
        prompt_path: str | None = None,
        metrics_recovery: bool = False,
        merge_resolution: bool = False,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        summary = self.load_summary(Path(summary_path))
        metrics = self.extract_ca_metrics_from_summary(summary)
        ca_done_kind = "ca.merge_resolution_completed" if merge_resolution else "ca.completed"
        branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
        best_key = best_target_metric_key()
        best_target_metric = (
            float(branch_metrics[best_key])
            if branch_metrics and branch_metrics.get(best_key) is not None
            else None
        )
        # Use baseline at sync-before-P time (former_target_metric_value) when available; else branch best for baseline seed.
        baseline_for_signal = (
            seed.former_target_metric_value
            if (seed.former_target_metric_value is not None and seed.seed_id != BASELINE_SEED_ID)
            else best_target_metric
        )
        signal = self.evaluate_signal(metrics, baseline_for_signal, PROMOTION_THRESHOLD)
        commit_sha = summary.get("commit_sha")
        if not (isinstance(commit_sha, str) and commit_sha.strip()):
            try:
                commit_sha = self.git_service.commit_sha(seed.seed_id)
            except GitCommandError:
                commit_sha = ""
        run.status = RunStatus.succeeded
        run.updated_at = now_ts()
        run.log_path = log_path
        run.stderr_log_path = stderr_log_path
        run.prompt_path = prompt_path
        # Preserve runner-set keys (e.g. commit_sha_before_pd, former target metric) for restore and comparison.
        former_key = former_target_metric_key()
        preserved = {k: run.summary[k] for k in ("commit_sha_before_pd", former_key) if run.summary and k in run.summary}
        if seed.former_target_metric_value is not None and former_key not in preserved:
            preserved[former_key] = seed.former_target_metric_value
        run.summary = summary | {"commit_sha": commit_sha} | preserved
        # CA runs don't produce idea/description; fill from the latest PD run so TSV and UI show the experiment idea.
        latest_pd = next((r for r in self.run_repo.list(seed_id) if r.stage == StageName.pd), None)
        if latest_pd and latest_pd.summary:
            if run.summary.get("idea") is None and latest_pd.summary.get("idea") is not None:
                run.summary["idea"] = latest_pd.summary["idea"]
            if run.summary.get("description") is None and latest_pd.summary.get("description") is not None:
                run.summary["description"] = latest_pd.summary["description"]
        run.metrics = metrics
        run.signal = signal
        seed.updated_at = now_ts()
        if signal == "error" and not metrics_recovery:
            run.summary = run.summary | {"metrics_recovery_queued": True}
            self.run_repo.save(run)
            self.seed_repo.save(seed)
            self.seed_repo.append_event(
                seed.seed_id,
                "ca.metrics_recovery_queued",
                "Check-Action completed without recoverable metrics in the structured report; queued a follow-up Check-Action to inspect saved logs.",
                run_id=run_id,
            )
            self.queue_ca(
                seed.seed_id,
                metrics_recovery=True,
                source_run_id=run_id,
                source_stdout_log_path=log_path,
                source_stderr_log_path=stderr_log_path,
            )
            LOGGER.warning(
                "CA run %s for seed %s completed without recoverable metrics; queued metrics recovery",
                run_id,
                seed.seed_id,
            )
            if (
                seed.ralph_loop_enabled
                and seed.seed_id != BASELINE_SEED_ID
            ):
                self._ralph_try_restore_worktree(seed, run.summary.get("commit_sha_before_pd"))
                if self._ralph_should_requeue(seed.seed_id):
                    try:
                        self.queue_pd(seed.seed_id)
                        self.seed_repo.append_event(
                            seed.seed_id,
                            "ralph.requeued",
                            "Ralph loop queued the next Plan-Do run after Check-Action (no metrics).",
                        )
                    except (RuntimeError, GitCommandError) as exc:
                        self.seed_repo.append_event(
                            seed.seed_id,
                            "ralph.requeue_failed",
                            f"Ralph loop could not queue the next Plan-Do run after Check-Action (no metrics): {exc}",
                        )
                else:
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.max_reached",
                        "Ralph loop max iterations reached; not queuing another Plan-Do run.",
                    )
            return run
        seed.latest_metrics = metrics
        seed.latest_signal = signal
        terminal_status = self._status_from_ca_signal(signal)
        merge_commit_sha = None  # set when seed branch is successfully merged into baseline
        if seed.seed_id == BASELINE_SEED_ID and best_target_metric is None:
            if TARGET_METRIC_KEY not in metrics:
                seed.status = SeedStatus.failed
                event_message = (
                    "Baseline metrics recovery could not recover metrics; marked as failed."
                    if metrics_recovery
                    else "Baseline measurement completed without metrics; marked as failed."
                )
                self.run_repo.save(run)
                self.seed_repo.save(seed)
                self.seed_repo.append_event(
                    seed.seed_id,
                    ca_done_kind,
                    event_message,
                    signal=signal,
                    metrics=metrics,
                )
                LOGGER.info(
                    "CA run %s finished for seed %s with signal=%s status=%s",
                    run_id,
                    seed.seed_id,
                    signal,
                    seed.status.value,
                )
                return run
            target_branch = self._first_user_seed_baseline_branch() or seed.baseline_branch
            _idea = summary.get("idea") or summary.get("notes")
            if isinstance(_idea, str) and _idea.strip():
                baseline_promoted_idea = _idea[:80]
            elif _idea:
                baseline_promoted_idea = str(_idea)[:80]
            else:
                baseline_promoted_idea = "Initial baseline adaptation"
            # Only positive_signal is merged into the per-seed baseline branch; record baseline value otherwise.
            if signal != "positive_signal":
                self.metrics_repo.append_baseline_run(target_branch, metrics[TARGET_METRIC_KEY])
                seed.status = terminal_status
                self.run_repo.save(run)
                self.seed_repo.save(seed)
                self.seed_repo.append_event(
                    seed.seed_id,
                    ca_done_kind,
                    "Baseline measurement completed (no promotion); not merged into baseline branch.",
                    signal=signal,
                    metrics=metrics,
                )
                LOGGER.info(
                    "CA run %s finished for seed %s with signal=%s status=%s",
                    run_id,
                    seed.seed_id,
                    signal,
                    seed.status.value,
                )
                return run
            # Merge-resolution CA: agent already ran merge (or "Already up to date"). Treat as pass; do not run promote_seed_branch again.
            if merge_resolution:
                effective_sha = self._commit_sha_for_branch(target_branch)
                self.metrics_repo.append_promotion_for_branch(
                    target_branch,
                    {
                        TARGET_METRIC_KEY: metrics[TARGET_METRIC_KEY],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": baseline_promoted_idea,
                        "promoted_at": summary.get("completed_at"),
                        "commit_sha": effective_sha,
                    },
                )
                seed.status = SeedStatus.passed
                self.run_repo.save(run)
                self.seed_repo.save(seed)
                self.seed_repo.append_event(
                    seed.seed_id,
                    ca_done_kind,
                    f"Merge resolution Check-Action completed; __baseline__ merged or already up to date with {target_branch}.",
                    signal=signal,
                    metrics=metrics,
                    commit_sha=effective_sha,
                )
                self._release_seeds_waiting_for_baseline(target_branch)
                LOGGER.info(
                    "CA run %s finished for seed %s with signal=%s status=%s",
                    run_id,
                    seed.seed_id,
                    signal,
                    seed.status.value,
                )
                return run
            try:
                merge_commit_sha = self.git_service.promote_seed_branch(seed, target_branch=target_branch)
                effective_sha = (
                    merge_commit_sha
                    if (isinstance(merge_commit_sha, str) and merge_commit_sha.strip())
                    else self._commit_sha_for_branch(target_branch)
                )
                self.metrics_repo.append_promotion_for_branch(
                    target_branch,
                    {
                        TARGET_METRIC_KEY: metrics[TARGET_METRIC_KEY],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": baseline_promoted_idea,
                        "promoted_at": summary.get("completed_at"),
                        "commit_sha": effective_sha,
                    },
                )
                seed.status = SeedStatus.passed
                event_message = f"Baseline measurement completed and __baseline__ was merged into {target_branch}; waiting seeds can now start Plan-Do."
                self.run_repo.save(run)
                self.seed_repo.save(seed)
                self.seed_repo.append_event(
                    seed.seed_id,
                    ca_done_kind,
                    event_message,
                    signal=signal,
                    metrics=metrics,
                    commit_sha=merge_commit_sha,
                )
                self._release_seeds_waiting_for_baseline(target_branch)
                return run
            except GitCommandError as merge_err:
                tried_sha = commit_sha or ""
                try:
                    tried_sha = self.git_service.commit_sha(seed.seed_id)
                except GitCommandError:
                    pass
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ca.merge_failed",
                    f"Merge into baseline failed: {merge_err}. Queued a new Check-Action run to resolve conflicts.",
                    commit_sha=tried_sha or None,
                    target_branch=target_branch,
                )
                if not merge_resolution:
                    self.queue_ca(
                        seed.seed_id,
                        merge_resolution=True,
                        last_metrics=metrics,
                        last_summary=summary,
                    )
                    seed.status = SeedStatus.ca_queued
                    seed.updated_at = now_ts()
                    self.seed_repo.save(seed)
                    self.run_repo.save(run)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        ca_done_kind,
                        "Baseline measurement completed but merge failed; conflict-resolution Check-Action queued.",
                        signal=signal,
                        metrics=metrics,
                    )
                    return run
                effective_sha = self._commit_sha_for_branch(target_branch)
                self.metrics_repo.append_promotion_for_branch(
                    target_branch,
                    {
                        TARGET_METRIC_KEY: metrics[TARGET_METRIC_KEY],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": baseline_promoted_idea,
                        "promoted_at": summary.get("completed_at"),
                        "commit_sha": effective_sha,
                    },
                )
                seed.status = SeedStatus.passed
                self.seed_repo.save(seed)
                self.run_repo.save(run)
                self.seed_repo.append_event(
                    seed.seed_id,
                    ca_done_kind,
                    "Baseline measurement completed; merge into baseline branch failed again after resolution run (loop avoided). Baseline metrics recorded; manual merge may be needed.",
                    signal=signal,
                    metrics=metrics,
                )
                self._release_seeds_waiting_for_baseline(target_branch)
                return run
        if terminal_status is SeedStatus.promoted:
            # Merge seed into baseline first on positive signal; then update metrics/state.
            # Merge-resolution CA: agent already ran merge (or "Already up to date"). Treat as pass; do not run promote_seed_branch again.
            if merge_resolution:
                effective_sha = self._commit_sha_for_branch(seed.baseline_branch)
                self.metrics_repo.append_promotion_for_branch(
                    seed.baseline_branch,
                    {
                        TARGET_METRIC_KEY: metrics[TARGET_METRIC_KEY],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": seed.plan.title if seed.plan else seed.prompt[:80],
                        "promoted_at": summary.get("completed_at"),
                        "commit_sha": effective_sha,
                    },
                )
                seed.status = terminal_status
                event_message = "Merge resolution Check-Action completed; seed merged or already up to date with baseline."
            else:
                try:
                    merge_commit_sha = self.git_service.promote_seed_branch(seed)
                    effective_sha = (
                        merge_commit_sha
                        if (isinstance(merge_commit_sha, str) and merge_commit_sha.strip())
                        else self._commit_sha_for_branch(seed.baseline_branch)
                    )
                    self.metrics_repo.append_promotion_for_branch(
                        seed.baseline_branch,
                        {
                            TARGET_METRIC_KEY: metrics[TARGET_METRIC_KEY],
                            "promoted_branch": seed.seed_id,
                            "promoted_idea": seed.plan.title if seed.plan else seed.prompt[:80],
                            "promoted_at": summary.get("completed_at"),
                            "commit_sha": effective_sha,
                        },
                    )
                    seed.status = terminal_status
                    event_message = "Check-Action succeeded and seed branch was promoted into baseline."
                except GitCommandError as merge_err:
                    tried_sha = commit_sha or ""
                    try:
                        tried_sha = self.git_service.commit_sha(seed.seed_id)
                    except GitCommandError:
                        pass
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ca.merge_failed",
                        (
                            f"Merge into baseline failed: {merge_err}. Queued a new Check-Action run to resolve conflicts."
                            if not merge_resolution
                            else f"Merge into baseline failed again after a conflict-resolution Check-Action: {merge_err}. "
                            "Ralph can proceed with the next Plan-Do run."
                        ),
                        commit_sha=tried_sha or None,
                        target_branch=seed.baseline_branch,
                    )
                    if not merge_resolution:
                        self.queue_ca(
                            seed.seed_id,
                            merge_resolution=True,
                            last_metrics=metrics,
                            last_summary=summary,
                        )
                        seed.status = SeedStatus.ca_queued
                        seed.updated_at = now_ts()
                        self.seed_repo.save(seed)
                        self.run_repo.save(run)
                        self.seed_repo.append_event(
                            seed.seed_id,
                            ca_done_kind,
                            "Check-Action run completed but merge failed; conflict-resolution Check-Action queued.",
                            signal=signal,
                            metrics=metrics,
                        )
                        return run
                    # Resolution run also failed to merge; avoid infinite resolution loop and continue Ralph.
                    seed.status = SeedStatus.generated
                    seed.updated_at = now_ts()
                    self.seed_repo.save(seed)
                    self.run_repo.save(run)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        ca_done_kind,
                        "Conflict-resolution Check-Action completed but merge still failed; proceeding to next Plan-Do run.",
                        signal=signal,
                        metrics=metrics,
                    )
                    if seed.ralph_loop_enabled:
                        if self._ralph_should_requeue(seed.seed_id):
                            try:
                                self.queue_pd(seed.seed_id)
                                self.seed_repo.append_event(
                                    seed.seed_id,
                                    "ralph.requeued",
                                    "Ralph loop queued the next Plan-Do run after unresolved merge conflict.",
                                )
                            except (RuntimeError, GitCommandError) as exc:
                                self.seed_repo.append_event(
                                    seed.seed_id,
                                    "ralph.requeue_failed",
                                    f"Ralph loop could not queue the next Plan-Do run after unresolved merge conflict: {exc}",
                                )
                        else:
                            self.seed_repo.append_event(
                                seed.seed_id,
                                "ralph.max_reached",
                                "Ralph loop max iterations reached; not queuing another Plan-Do run.",
                            )
                    return run
        elif terminal_status is SeedStatus.failed:
            seed.status = terminal_status
            event_message = (
                "Check-Action metrics recovery could not recover metrics; marked as failed."
                if metrics_recovery
                else "Check-Action completed but metrics were missing; marked as failed."
            )
        else:
            seed.status = terminal_status
            event_message = "Check-Action completed without promotion."
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        event_commit_sha = merge_commit_sha if merge_commit_sha else run.summary.get("commit_sha")
        self.seed_repo.append_event(
            seed.seed_id,
            ca_done_kind,
            event_message,
            signal=signal,
            metrics=metrics,
            **({"commit_sha": event_commit_sha} if event_commit_sha else {}),
        )
        if (
            seed.ralph_loop_enabled
            and signal in ("negative_signal", "neutral", "error")
            and not merge_resolution
            and not metrics_recovery
            and seed.seed_id != BASELINE_SEED_ID
        ):
            self._ralph_try_restore_worktree(seed, run.summary.get("commit_sha_before_pd"))
        if seed.ralph_loop_enabled:
            if self._ralph_should_requeue(seed.seed_id):
                try:
                    self.queue_pd(seed.seed_id)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.requeued",
                        "Ralph loop queued the next Plan-Do run.",
                    )
                except (RuntimeError, GitCommandError) as exc:
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.requeue_failed",
                        f"Ralph loop could not queue the next Plan-Do run: {exc}",
                    )
            else:
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ralph.max_reached",
                    "Ralph loop max iterations reached; not queuing another Plan-Do run.",
                )
        LOGGER.info(
            "CA run %s finished for seed %s with signal=%s status=%s",
            run_id,
            seed.seed_id,
            signal,
            seed.status.value,
        )
        return run

    def build_dashboard(self, selected_seed_id: str | None = None) -> DashboardViewModel:
        seeds = self.seed_repo.list()
        selected_seed = self.seed_repo.get(selected_seed_id) if selected_seed_id else None
        baseline_metrics_by_branch = self.metrics_repo.get_all()
        available_branches: list[str] = []
        setup_error = self.git_service.setup_error()
        if setup_error is None:
            try:
                all_branches = self.git_service.list_branches()
                if not all_branches:
                    setup_error = "No local branches found yet. Create an initial commit/branch, then reload."
                else:
                    available_branches = [
                        b for b in all_branches
                        if not self.git_service.is_seed_specific_branch(b)
                    ]
                    # Use only branches that exist in the repo; do not add DEFAULT_BASELINE_BRANCH
                    # if it does not exist, so the dropdown never shows a non-existent branch.
            except GitCommandError as exc:
                setup_error = str(exc)
        # Default to first existing branch so the selected value is always valid.
        default_baseline_branch = (available_branches[0] if available_branches else DEFAULT_BASELINE_BRANCH) or "master"
        status_column_map = {
            SeedStatus.draft: "seedInbox",
            SeedStatus.queued: "seedInbox",
            SeedStatus.planning: "generated",
            SeedStatus.generated: "generated",
            SeedStatus.ca_queued: "generated",
            SeedStatus.adapting: "activeCa",
            SeedStatus.running: "activeCa",
            SeedStatus.passed: "completed",
            SeedStatus.failed: "completed",
            SeedStatus.promoted: "completed",
        }
        seeds_by_column: dict[str, list[SeedRecord]] = {
            "seedInbox": [],
            "generated": [],
            "activeCa": [],
            "completed": [],
        }
        for seed in seeds:
            self._reconcile_seed_status_signal(seed)
            column_id = status_column_map.get(seed.status, "seedInbox")
            seeds_by_column[column_id].append(seed)
        columns = [
            DashboardColumn(
                id="seedInbox",
                title="Seed",
                description="New prompts and queued planning work.",
                seeds=seeds_by_column["seedInbox"],
            ),
            DashboardColumn(
                id="generated",
                title="Plan-Do",
                description="Planning and generated code ready for Check-Action.",
                seeds=seeds_by_column["generated"],
            ),
            DashboardColumn(
                id="activeCa",
                title="Check-Action",
                description="Adapting, fixing, and running the seed run.",
                seeds=seeds_by_column["activeCa"],
            ),
            DashboardColumn(
                id="completed",
                title="Completed",
                description="Finished runs; promoted seeds merged into baseline.",
                seeds=seeds_by_column["completed"],
            ),
        ]
        return DashboardViewModel(
            setup_error=setup_error,
            baseline_metrics_by_branch=baseline_metrics_by_branch,
            default_baseline_branch=default_baseline_branch,
            available_branches=available_branches,
            seed_count=len(seeds),
            columns=columns,
            selected_seed=selected_seed,
            daemon_status=get_daemon_status(),
            target_metric_key=TARGET_METRIC_KEY,
            target_metric_label=TARGET_METRIC_LABEL,
            best_target_metric_key=best_target_metric_key(),
        )

    def seed_detail(self, seed_id: str) -> dict[str, object]:
        seed = self.require_seed(seed_id)
        expected_worktree = (
            self._baseline_worktree_path()
            if seed.seed_id == BASELINE_SEED_ID
            else self._seed_worktree_path(seed.seed_id)
        )
        needs_save = False
        if expected_worktree is not None and not seed.worktree_path:
            seed.worktree_path = expected_worktree
            needs_save = True
        if needs_save:
            seed.updated_at = now_ts()
            self.seed_repo.save(seed)
        self._reconcile_seed_status_signal(seed)
        raw_events = self.seed_repo.events(seed_id)
        runs = self.run_repo.list(seed_id)
        completed = {RunStatus.succeeded, RunStatus.failed}
        has_incomplete_runs = any(r.status not in completed for r in runs)
        seed_in_early_stage = seed.status in (SeedStatus.draft, SeedStatus.queued, SeedStatus.planning)
        has_non_completed_runs = has_incomplete_runs or seed_in_early_stage
        return {
            "seed": seed,
            "can_edit_prompt": self.can_edit_seed_prompt(seed),
            "runs": runs,
            "events": _timeline_display_events(raw_events),
            "baseline_metrics_for_branch": self.metrics_repo.get_for_branch(seed.baseline_branch),
            "setup_error": self.git_service.setup_error_for_branches(seed.baseline_branch),
            "has_non_completed_runs": has_non_completed_runs,
        }

    def seed_detail_versions(self, seed_id: str) -> dict[str, str]:
        """Return version fingerprints for runs and timeline so the client can skip refresh when unchanged."""
        self.require_seed(seed_id)
        runs = self.run_repo.list(seed_id)
        events = self.seed_repo.events(seed_id)
        runs_version = (
            ",".join(f"{r.run_id}:{r.status.value}:{r.updated_at}" for r in runs)
            if runs
            else "0"
        )
        timeline_version = (
            ",".join(str(e.get("created_at", "")) for e in events[-20:])
            if events
            else "0"
        )
        return {
            "runs_version": runs_version,
            "timeline_version": timeline_version,
        }

    def extract_ca_metrics_from_summary(
        self, summary: dict[str, object]
    ) -> dict[str, float | int]:
        """Extract typed metrics from summary['metrics']. Returns empty dict if missing or invalid."""
        summary_metrics = summary.get("metrics")
        if not isinstance(summary_metrics, dict):
            return {}
        parsed: dict[str, float | int] = {}
        int_keys = {"num_steps", "depth"}
        float_keys = {
            TARGET_METRIC_KEY,
            "training_seconds",
            "total_seconds",
            "startup_seconds",
            "peak_vram_mb",
            "mfu_percent",
            "total_tokens_M",
            "num_params_M",
        }
        for key in int_keys | float_keys:
            value = summary_metrics.get(key)
            if value is None:
                continue
            try:
                parsed[key] = int(value) if key in int_keys else float(value)
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def evaluate_signal(
        metrics: dict[str, float | int],
        baseline_target_metric: float | None,
        promotion_threshold: float = PROMOTION_THRESHOLD,
    ) -> str:
        current = metrics.get(TARGET_METRIC_KEY)
        if current is None:
            return "error"
        if baseline_target_metric is None:
            return "positive_signal"
        # Positive delta = improvement (lower is better: baseline - current; higher is better: current - baseline)
        delta = (
            float(baseline_target_metric) - float(current)
            if TARGET_METRIC_LOWER_IS_BETTER
            else float(current) - float(baseline_target_metric)
        )
        if delta >= promotion_threshold:
            return "positive_signal"
        if delta <= -promotion_threshold:
            return "negative_signal"
        return "neutral"


def default_workflow_service() -> WorkflowService:
    return WorkflowService()
