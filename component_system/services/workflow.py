from __future__ import annotations

import json
from typing import Any
import re
import subprocess
from pathlib import Path

from component_system.config import DEFAULT_BASELINE_BRANCH, PROMOTION_THRESHOLD
from component_system.domain.models import (
    DashboardColumn,
    DashboardViewModel,
    PlanIdea,
    RunStatus,
    SeedRecord,
    SeedStatus,
    StageName,
    StageRun,
)
from component_system.repositories.state import (
    BaselineBranchMapRepository,
    BaselineMetricsRepository,
    RunRepository,
    SeedRepository,
)
from component_system.task import (
    COMPONENT_SYSTEM_ROOT,
    WORKTREE_ROOT,
    get_daemon_status,
    move_to_error,
    now_ts,
    new_run_id,
    new_seed_id,
    read_task,
    write_task,
)

SUMMARY_MARKERS = {
    "p": ("AUTORESEARCH_P_SUMMARY_BEGIN", "AUTORESEARCH_P_SUMMARY_END"),
    "dca": ("AUTORESEARCH_DCA_SUMMARY_BEGIN", "AUTORESEARCH_DCA_SUMMARY_END"),
}

BASELINE_SEED_ID = "__baseline__"


class GitCommandError(RuntimeError):
    pass


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
            raise GitCommandError("Git is not installed or not available on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
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
        if branch.startswith("seed/"):
            return True  # legacy candidate branches, e.g. seed/seed-e57b95
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
        if conflict_path is not None and conflict_path != target_worktree:
            # If the conflicting worktree still exists, force-remove it from registry.
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

    def head_sha_at(self, cwd: Path) -> str:
        """Return the short commit SHA of HEAD in the given worktree directory."""
        return self._run_git("rev-parse", "--short", "HEAD", cwd=cwd)

    def reset_seed_branch_to(self, seed: SeedRecord, ref: str) -> None:
        """Reset the seed worktree's branch to the given ref (e.g. commit before P).
        No-op for baseline seed or when worktree is missing."""
        if seed.seed_id == BASELINE_SEED_ID:
            return
        if not seed.worktree_path:
            return
        worktree_path = Path(seed.worktree_path)
        if not worktree_path.is_dir():
            return
        self._run_git("reset", "--hard", ref, cwd=worktree_path)

    def promote_seed_branch(
        self, seed: SeedRecord, target_branch: str | None = None
    ) -> str:
        """Merge the seed's branch (seed_id) into the target branch. Only DCA Action may call this; Plan must never merge.
        If target_branch is None, use seed.baseline_branch (e.g. for normal seed promotion). For __baseline__ completion,
        pass the first user seed's selected branch so the merge goes there instead of a fixed config value."""
        merge_into = target_branch if target_branch is not None else seed.baseline_branch
        baseline_worktree = WORKTREE_ROOT / "baseline"
        if baseline_worktree.exists():
            try:
                self._run_git("worktree", "remove", "--force", str(baseline_worktree))
            except GitCommandError:
                pass
        self._run_git(
            "worktree",
            "add",
            "--force",
            "-B",
            merge_into,
            str(baseline_worktree),
            merge_into,
        )
        self._run_git("merge", "--no-edit", seed.seed_id, cwd=baseline_worktree)
        return self.commit_sha(merge_into)


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
        """Clean up legacy persisted seed state that no longer matches runtime rules."""
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

    def _enqueue_plan_run(self, seed: SeedRecord, event_kind: str = "p.queued", event_message: str = "Queued Plan stage for the seed.") -> StageRun:
        run = StageRun(
            run_id=new_run_id("p"),
            seed_id=seed.seed_id,
            stage=StageName.p,
            status=RunStatus.queued,
            task_id=new_run_id("task-p"),
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
            "p",
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
        if not branch_metrics or branch_metrics.get("last_val_bpb") is None:
            return
        waiting_seeds = sorted(self.seed_repo.list(), key=lambda item: item.created_at)
        for seed in waiting_seeds:
            if seed.seed_id == BASELINE_SEED_ID:
                continue
            if seed.baseline_branch != branch:
                continue
            if seed.status is not SeedStatus.queued or seed.latest_run_id is not None:
                continue
            self._enqueue_plan_run(
                seed,
                event_kind="p.released",
                event_message="Baseline is ready; queued Plan stage for the waiting seed.",
            )

    @staticmethod
    def _status_from_dca_signal(signal: str) -> SeedStatus:
        """Centralized mapping from DCA signal to terminal seed status."""
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
            pass  # branch seed_id is created when Plan is queued (ensure_seed_worktrees)
        except GitCommandError:
            # Keep seed creation non-blocking; branch creation will be retried at P queue time.
            pass
        self.seed_repo.append_event(seed.seed_id, "seed.created", "Seed created from prompt.")
        if ralph_loop_enabled:
            self.seed_repo.append_event(
                seed.seed_id,
                "ralph.enabled",
                "Ralph loop enabled; Plan will auto-requeue after each DCA completion.",
            )
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
            worktree_path=str(COMPONENT_SYSTEM_ROOT.parent),
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

    def _get_or_create_baseline_seed(self) -> SeedRecord:
        """Return the baseline seed used to establish initial val_bpb; create and persist it if missing."""
        seed = self.seed_repo.get(BASELINE_SEED_ID)
        if seed is not None:
            return self._normalize_seed_runtime_state(seed)
        branch = self._first_user_seed_baseline_branch() or DEFAULT_BASELINE_BRANCH
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

    def ensure_baseline_result(self) -> None:
        """
        If there is no baseline result (last_val_bpb) for the baseline seed's branch, ensure a baseline seed exists and
        queue its DCA so the first run establishes the baseline. Idempotent; safe to call
        before queue_p for any user seed.
        """
        seed = self._get_or_create_baseline_seed()
        branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
        if branch_metrics and branch_metrics.get("last_val_bpb") is not None:
            return
        if seed.status in (SeedStatus.dca_queued, SeedStatus.adapting, SeedStatus.running):
            return
        if seed.status in (SeedStatus.passed, SeedStatus.failed, SeedStatus.promoted):
            branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
            if branch_metrics and branch_metrics.get("last_val_bpb") is not None:
                return
        setup_error = self.git_service.setup_error()
        if setup_error is not None:
            return
        try:
            self.git_service.ensure_branch(seed.baseline_branch, self.git_service.current_head())
        except GitCommandError:
            return
        setup_error = self.git_service.setup_error_for_branches(seed.baseline_branch)
        if setup_error is not None:
            return
        seed.status = SeedStatus.generated
        seed.plan = PlanIdea(title="Baseline", description="No changes; measure current baseline.")
        seed.updated_at = now_ts()
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "baseline.queued",
            "Queued DCA to establish baseline result before first seed.",
        )
        self.queue_dca(seed.seed_id)

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
                "Ralph loop enabled; Plan will auto-requeue after each DCA completion.",
            )
        else:
            self.seed_repo.append_event(seed.seed_id, "ralph.disabled", "Ralph loop disabled by user.")
        return seed

    def can_edit_seed_prompt(self, seed: SeedRecord) -> bool:
        return seed.status in {SeedStatus.draft, SeedStatus.queued}

    def update_seed_prompt(self, seed_id: str, prompt: str) -> SeedRecord:
        seed = self.require_seed(seed_id)
        if not self.can_edit_seed_prompt(seed):
            raise RuntimeError("Seed prompt can only be edited before Plan starts.")
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

    def queue_p(self, seed_id: str) -> StageRun | None:
        seed = self.require_seed(seed_id)
        branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch) if seed_id != BASELINE_SEED_ID else None
        has_baseline = branch_metrics is not None and branch_metrics.get("last_val_bpb") is not None
        if seed_id != BASELINE_SEED_ID and not has_baseline:
            self.ensure_baseline_result()
            branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
            has_baseline = branch_metrics is not None and branch_metrics.get("last_val_bpb") is not None
            if not has_baseline:
                if not (seed.status is SeedStatus.queued and seed.latest_run_id is None):
                    seed.status = SeedStatus.queued
                    seed.updated_at = now_ts()
                    seed.latest_run_id = None
                    seed.last_error = None
                    self.seed_repo.save(seed)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "p.waiting_for_baseline",
                        "Baseline run is still in progress; Plan will queue after baseline finishes.",
                    )
                return None
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
        return self._enqueue_plan_run(seed)

    def queue_dca(
        self,
        seed_id: str,
        merge_resolution: bool = False,
        metrics_recovery: bool = False,
        source_run_id: str | None = None,
        source_stdout_log_path: str | None = None,
        source_stderr_log_path: str | None = None,
        last_metrics: dict[str, Any] | None = None,
        last_summary: dict[str, Any] | None = None,
        commit_sha_before_p: str | None = None,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        if not metrics_recovery and seed.status in {SeedStatus.draft, SeedStatus.queued, SeedStatus.planning}:
            raise RuntimeError("Run Plan first. Do-Check-Action is available after code is generated into the seed branch.")
        if not metrics_recovery:
            setup_error = self.git_service.setup_error_for_branches(seed.baseline_branch)
            if setup_error is not None:
                raise RuntimeError(setup_error)
        run = StageRun(
            run_id=new_run_id("dca"),
            seed_id=seed.seed_id,
            stage=StageName.dca,
            status=RunStatus.queued,
            task_id=new_run_id("task-dca"),
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        if seed.seed_id != BASELINE_SEED_ID:
            try:
                # Ref to restore worktree to on negative signal (commit before P when from finish_p_run, else baseline).
                run.summary["commit_sha_before_p"] = (
                    commit_sha_before_p
                    if commit_sha_before_p is not None
                    else self.git_service.commit_sha(seed.baseline_branch)
                )
            except GitCommandError:
                pass
        seed.status = SeedStatus.dca_queued
        seed.updated_at = now_ts()
        seed.latest_run_id = run.run_id
        seed.last_error = None
        self.seed_repo.save(seed)
        self.run_repo.save(run)
        self.seed_repo.append_event(
            seed.seed_id,
            "dca.queued",
            "Queued DCA for merge conflict resolution."
            if merge_resolution
            else "Queued DCA for metrics recovery from saved logs."
            if metrics_recovery
            else "Queued DCA stage for the seed.",
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
        write_task("dca", payload, task_id=run.task_id)
        return run

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

    def is_seed_eligible_for_stage(self, seed_id: str | None, stage: str) -> bool:
        """True if this seed is in a state that allows the given stage to run (used at claim time to avoid P/DCA races)."""
        if not seed_id:
            return False
        seed = self.seed_repo.get(seed_id)
        if seed is None:
            return False
        seed = self._normalize_seed_runtime_state(seed)
        if stage == "p":
            return seed.status not in (SeedStatus.adapting, SeedStatus.running, SeedStatus.dca_queued)
        if stage == "dca":
            return seed.status is not SeedStatus.planning
        if stage == "direct":
            return True
        return False

    def mark_run_started(self, seed_id: str, run_id: str) -> tuple[SeedRecord, StageRun]:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        run.status = RunStatus.running
        run.updated_at = now_ts()
        if run.stage is StageName.p:
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
            if seed.worktree_path:
                worktree_path = Path(seed.worktree_path)
                if worktree_path.is_dir():
                    try:
                        run.summary["commit_sha_before_p"] = self.git_service.head_sha_at(
                            worktree_path
                        )
                    except GitCommandError:
                        pass
            seed.status = SeedStatus.planning
            event_kind = "p.started"
            event_message = "Plan stage started in the candidate worktree."
        else:
            seed.status = SeedStatus.adapting
            event_kind = "dca.started"
            event_message = (
                "Baseline measurement started in the baseline worktree."
                if seed.seed_id == BASELINE_SEED_ID
                else "DCA stage started in the seed worktree."
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
        if (
            run.stage is StageName.dca
            and seed.ralph_loop_enabled
            and seed.seed_id != BASELINE_SEED_ID
            and task_payload.get("merge_resolution") is not True
            and task_payload.get("metrics_recovery") is not True
        ):
            try:
                self.queue_p(seed.seed_id)
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ralph.requeued",
                    "Ralph loop queued the next Plan run after failed DCA.",
                )
            except (RuntimeError, GitCommandError) as exc:
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ralph.requeue_failed",
                    f"Ralph loop could not queue the next Plan run after failed DCA: {exc}",
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
            "cwd": str(COMPONENT_SYSTEM_ROOT.parent),
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

    def finish_p_run(
        self,
        seed_id: str,
        run_id: str,
        stdout: str,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
        prompt_path: str | None = None,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        summary = self.extract_summary(stdout, StageName.p) or {}
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
        # Preserve run.summary fields set earlier (e.g. commit_sha_before_p) when merging P output.
        run.summary = run.summary | summary | {"commit_sha": commit_sha}
        seed.status = SeedStatus.generated
        seed.updated_at = now_ts()
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        self.seed_repo.append_event(
            seed.seed_id,
            "p.completed",
            "Plan completed on seed branch.",
            commit_sha=commit_sha,
        )
        self.queue_dca(
            seed.seed_id,
            commit_sha_before_p=run.summary.get("commit_sha_before_p"),
        )
        return run

    @staticmethod
    def combine_output(stdout: str, stderr: str | None = None) -> str:
        if stdout and stderr:
            return f"{stdout}\n{stderr}"
        return stdout or stderr or ""

    def finish_dca_run(
        self,
        seed_id: str,
        run_id: str,
        stdout: str,
        stderr: str | None = None,
        log_path: str | None = None,
        stderr_log_path: str | None = None,
        prompt_path: str | None = None,
        metrics_recovery: bool = False,
        merge_resolution: bool = False,
    ) -> StageRun:
        seed = self.require_seed(seed_id)
        run = self.require_run(run_id)
        branch_metrics = self.metrics_repo.get_for_branch(seed.baseline_branch)
        last_val_bpb = float(branch_metrics["last_val_bpb"]) if branch_metrics and branch_metrics.get("last_val_bpb") is not None else None
        output_text = self.combine_output(stdout, stderr)
        summary = self.extract_summary(output_text, StageName.dca) or {}
        metrics = self.extract_dca_metrics(output_text, summary)
        signal = self.evaluate_signal(metrics, last_val_bpb, PROMOTION_THRESHOLD)
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
        # Preserve runner-set keys (e.g. commit_sha_before_p) so negative-signal restore can run
        preserved = {k: run.summary[k] for k in ("commit_sha_before_p",) if run.summary and k in run.summary}
        run.summary = summary | {"commit_sha": commit_sha} | preserved
        run.metrics = metrics
        run.signal = signal
        seed.updated_at = now_ts()
        if signal == "error" and not metrics_recovery:
            run.summary = run.summary | {"metrics_recovery_queued": True}
            self.run_repo.save(run)
            self.seed_repo.save(seed)
            self.seed_repo.append_event(
                seed.seed_id,
                "dca.metrics_recovery_queued",
                "DCA completed without recoverable metrics in the structured report; queued a follow-up DCA to inspect saved logs.",
                run_id=run_id,
            )
            self.queue_dca(
                seed.seed_id,
                metrics_recovery=True,
                source_run_id=run_id,
                source_stdout_log_path=log_path,
                source_stderr_log_path=stderr_log_path,
            )
            return run
        seed.latest_metrics = metrics
        seed.latest_signal = signal
        terminal_status = self._status_from_dca_signal(signal)
        merge_commit_sha = None  # set when seed branch is successfully merged into baseline
        if seed.seed_id == BASELINE_SEED_ID and last_val_bpb is None:
            if "val_bpb" not in metrics:
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
                    "dca.completed",
                    event_message,
                    signal=signal,
                    metrics=metrics,
                )
                return run
            target_branch = self._first_user_seed_baseline_branch() or seed.baseline_branch
            # Only positive_signal is merged into the per-seed baseline branch; record baseline value otherwise.
            if signal != "positive_signal":
                self.metrics_repo.update_for_branch(
                    target_branch,
                    {"last_val_bpb": metrics["val_bpb"]},
                )
                seed.status = terminal_status
                self.run_repo.save(run)
                self.seed_repo.save(seed)
                self.seed_repo.append_event(
                    seed.seed_id,
                    "dca.completed",
                    "Baseline measurement completed (no promotion); not merged into baseline branch.",
                    signal=signal,
                    metrics=metrics,
                )
                return run
            try:
                merge_commit_sha = self.git_service.promote_seed_branch(seed, target_branch=target_branch)
                self.metrics_repo.update_for_branch(
                    target_branch,
                    {
                        "last_val_bpb": metrics["val_bpb"],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": "Initial baseline adaptation",
                        "promoted_at": summary.get("completed_at"),
                        "commit_sha": merge_commit_sha,
                    },
                )
                seed.status = SeedStatus.passed
                event_message = f"Baseline measurement completed and __baseline__ was merged into {target_branch}; waiting seeds can now start Plan."
                self.run_repo.save(run)
                self.seed_repo.save(seed)
                self.seed_repo.append_event(
                    seed.seed_id,
                    "dca.completed",
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
                    "dca.merge_failed",
                    f"Merge into baseline failed: {merge_err}. Queued a new DCA run to resolve conflicts.",
                    commit_sha=tried_sha or None,
                    target_branch=target_branch,
                )
                if not merge_resolution:
                    self.queue_dca(
                        seed.seed_id,
                        merge_resolution=True,
                        last_metrics=metrics,
                        last_summary=summary,
                    )
                    seed.status = SeedStatus.dca_queued
                    seed.updated_at = now_ts()
                    self.seed_repo.save(seed)
                    self.run_repo.save(run)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "dca.completed",
                        "Baseline measurement completed but merge failed; conflict-resolution DCA queued.",
                        signal=signal,
                        metrics=metrics,
                    )
                    return run
                self.metrics_repo.update_for_branch(
                    target_branch,
                    {
                        "last_val_bpb": metrics["val_bpb"],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": "Initial baseline adaptation",
                        "promoted_at": summary.get("completed_at"),
                    },
                )
                seed.status = SeedStatus.passed
                self.seed_repo.save(seed)
                self.run_repo.save(run)
                self.seed_repo.append_event(
                    seed.seed_id,
                    "dca.completed",
                    "Baseline measurement completed; merge into baseline branch failed again after resolution run (loop avoided). Baseline metrics recorded; manual merge may be needed.",
                    signal=signal,
                    metrics=metrics,
                )
                self._release_seeds_waiting_for_baseline(target_branch)
                return run
        if terminal_status is SeedStatus.promoted:
            try:
                merge_commit_sha = self.git_service.promote_seed_branch(seed)
                self.metrics_repo.update_for_branch(
                    seed.baseline_branch,
                    {
                        "last_val_bpb": metrics["val_bpb"],
                        "promoted_branch": seed.seed_id,
                        "promoted_idea": seed.plan.title if seed.plan else seed.prompt[:80],
                        "promoted_at": summary.get("completed_at"),
                        "commit_sha": merge_commit_sha,
                    },
                )
                seed.status = terminal_status
                event_message = "DCA succeeded and seed branch was promoted into baseline."
            except GitCommandError as merge_err:
                tried_sha = commit_sha or ""
                try:
                    tried_sha = self.git_service.commit_sha(seed.seed_id)
                except GitCommandError:
                    pass
                self.seed_repo.append_event(
                    seed.seed_id,
                    "dca.merge_failed",
                    (
                        f"Merge into baseline failed: {merge_err}. Queued a new DCA run to resolve conflicts."
                        if not merge_resolution
                        else f"Merge into baseline failed again after a conflict-resolution DCA: {merge_err}. "
                        "Ralph can proceed with the next Plan run."
                    ),
                    commit_sha=tried_sha or None,
                    target_branch=seed.baseline_branch,
                )
                if not merge_resolution:
                    self.queue_dca(
                        seed.seed_id,
                        merge_resolution=True,
                        last_metrics=metrics,
                        last_summary=summary,
                    )
                    seed.status = SeedStatus.dca_queued
                    seed.updated_at = now_ts()
                    self.seed_repo.save(seed)
                    self.run_repo.save(run)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "dca.completed",
                        "DCA run completed but merge failed; conflict-resolution DCA queued.",
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
                    "dca.completed",
                    "Conflict-resolution DCA completed but merge still failed; proceeding to next Plan run.",
                    signal=signal,
                    metrics=metrics,
                )
                if seed.ralph_loop_enabled:
                    try:
                        self.queue_p(seed.seed_id)
                        self.seed_repo.append_event(
                            seed.seed_id,
                            "ralph.requeued",
                            "Ralph loop queued the next Plan run after unresolved merge conflict.",
                        )
                    except (RuntimeError, GitCommandError) as exc:
                        self.seed_repo.append_event(
                            seed.seed_id,
                            "ralph.requeue_failed",
                            f"Ralph loop could not queue the next Plan run after unresolved merge conflict: {exc}",
                        )
                return run
        elif terminal_status is SeedStatus.failed:
            seed.status = terminal_status
            event_message = (
                "DCA metrics recovery could not recover metrics; marked as failed."
                if metrics_recovery
                else "DCA completed but metrics were missing; marked as failed."
            )
        else:
            seed.status = terminal_status
            event_message = "DCA completed without promotion."
        self.run_repo.save(run)
        self.seed_repo.save(seed)
        event_commit_sha = merge_commit_sha if merge_commit_sha else run.summary.get("commit_sha")
        self.seed_repo.append_event(
            seed.seed_id,
            "dca.completed",
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
            ref = run.summary.get("commit_sha_before_p")
            if ref:
                try:
                    self.git_service.reset_seed_branch_to(seed, ref)
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.worktree_restored",
                        "Restored seed worktree to commit before P for next Plan.",
                        commit_sha=ref,
                    )
                except GitCommandError as exc:
                    self.seed_repo.append_event(
                        seed.seed_id,
                        "ralph.worktree_restore_failed",
                        f"Could not restore seed worktree to commit before P: {exc}",
                        commit_sha=ref,
                    )
        if seed.ralph_loop_enabled:
            try:
                self.queue_p(seed.seed_id)
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ralph.requeued",
                    "Ralph loop queued the next Plan run.",
                )
            except (RuntimeError, GitCommandError) as exc:
                self.seed_repo.append_event(
                    seed.seed_id,
                    "ralph.requeue_failed",
                    f"Ralph loop could not queue the next Plan run: {exc}",
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
            SeedStatus.dca_queued: "generated",
            SeedStatus.adapting: "activeDca",
            SeedStatus.running: "activeDca",
            SeedStatus.passed: "completed",
            SeedStatus.failed: "completed",
            SeedStatus.promoted: "completed",
        }
        seeds_by_column: dict[str, list[SeedRecord]] = {
            "seedInbox": [],
            "generated": [],
            "activeDca": [],
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
                title="Plan",
                description="Planning and generated code ready for Do-Check-Action.",
                seeds=seeds_by_column["generated"],
            ),
            DashboardColumn(
                id="activeDca",
                title="Do-Check-Action",
                description="Adapting, fixing, and running the seed run.",
                seeds=seeds_by_column["activeDca"],
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
        return {
            "seed": seed,
            "can_edit_prompt": self.can_edit_seed_prompt(seed),
            "runs": self.run_repo.list(seed_id),
            "events": self.seed_repo.events(seed_id),
            "baseline_metrics_for_branch": self.metrics_repo.get_for_branch(seed.baseline_branch),
            "setup_error": self.git_service.setup_error_for_branches(seed.baseline_branch),
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

    def extract_summary(self, output_text: str, stage: StageName) -> dict[str, object] | None:
        start_marker, end_marker = SUMMARY_MARKERS[stage.value]
        pattern = rf"{start_marker}\s*(\{{.*?\}})\s*{end_marker}"
        match = re.search(pattern, output_text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"raw_summary": match.group(1)}

    def extract_metrics(self, output_text: str) -> dict[str, float | int]:
        patterns = {
            "val_bpb": r"^val_bpb:\s+([0-9.]+)",
            "training_seconds": r"^training_seconds:\s+([0-9.]+)",
            "total_seconds": r"^total_seconds:\s+([0-9.]+)",
            "startup_seconds": r"^startup_seconds:\s+([0-9.]+)",
            "peak_vram_mb": r"^peak_vram_mb:\s+([0-9.]+)",
            "mfu_percent": r"^mfu_percent:\s+([0-9.]+)",
            "total_tokens_M": r"^total_tokens_M:\s+([0-9.]+)",
            "num_steps": r"^num_steps:\s+([0-9]+)",
            "num_params_M": r"^num_params_M:\s+([0-9.]+)",
            "depth": r"^depth:\s+([0-9]+)",
        }
        metrics: dict[str, float | int] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, output_text, flags=re.MULTILINE)
            if not match:
                continue
            metrics[key] = int(match.group(1)) if key in {"num_steps", "depth"} else float(match.group(1))
        return metrics

    def extract_dca_metrics(
        self, output_text: str, summary: dict[str, object] | None = None
    ) -> dict[str, float | int]:
        if summary:
            summary_metrics = summary.get("metrics")
            if isinstance(summary_metrics, dict):
                parsed: dict[str, float | int] = {}
                int_keys = {"num_steps", "depth"}
                float_keys = {
                    "val_bpb",
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
                if parsed:
                    return parsed
        return self.extract_metrics(output_text)

    @staticmethod
    def evaluate_signal(
        metrics: dict[str, float | int],
        last_val_bpb: float | None,
        promotion_threshold: float = PROMOTION_THRESHOLD,
    ) -> str:
        val_bpb = metrics.get("val_bpb")
        if val_bpb is None:
            return "error"
        if last_val_bpb is None:
            return "positive_signal"
        delta = float(last_val_bpb) - float(val_bpb)
        if delta >= promotion_threshold:
            return "positive_signal"
        if delta <= -promotion_threshold:
            return "negative_signal"
        return "neutral"


def default_workflow_service() -> WorkflowService:
    return WorkflowService()
