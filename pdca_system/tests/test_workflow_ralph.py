"""Tests for WorkflowService Ralph loop: merge resolution, restore, requeue.

Ralph loop coverage (all paths where next Plan-Do is or is not queued):

  finish_ca_run (CA completes with summary):
    - positive_signal, merge success     → queue PD  (test_ralph_merge_outcomes_queue_policy)
    - positive_signal, merge fail        → queue resolution CA only (test_ralph_does_not_queue_plan_do_when_merge_resolution_ca_is_queued)
    - merge_resolution CA success        → queue PD  (test_ralph_conflict_resolution_ca_outcomes_queue_policy)
    - merge_resolution CA, merge fail again → queue PD (test_ralph_conflict_resolution_ca_outcomes_queue_policy)
    - negative_signal / neutral          → queue PD  (test_ralph_restore_on_negative/neutral_signal_resets_worktree)
    - error, no metrics (early return)   → queue metrics_recovery CA + queue PD (test_ralph_restore_on_error_signal_resets_worktree)
    - metrics_recovery CA success        → queue PD  (test_ralph_metrics_recovery_ca_success_queues_next_plan_do)
    - metrics_recovery CA, still no metric → queue PD (test_ralph_metrics_recovery_ca_still_no_metrics_queues_next_plan_do)

  mark_run_failed (CA run fails, no summary):
    - common CA fail                     → no PD     (test_mark_run_failed_does_not_requeue_for_common_ca_failure)
    - merge_resolution CA fail          → queue PD  (test_mark_run_failed_requeues_plan_do_for_ralph_ca_failure)
    - metrics_recovery CA fail          → queue PD  (test_ralph_restore_on_mark_run_failed_ca_resets_worktree)
"""
from __future__ import annotations

import tempfile
import unittest

from pdca_system.config import TARGET_METRIC_KEY, best_target_metric_key
from pdca_system.domain.models import RunStatus, SeedStatus, StageName, StageRun
from pdca_system.services.workflow import WorkflowService
from pdca_system.task import load_events, list_pending, read_task, write_task

from .test_helpers import MergeFailingGitService, NoOpPromoteGitService, RecordingGitService, write_summary_file


class WorkflowRalphTests(unittest.TestCase):
    def test_ralph_does_not_queue_plan_do_when_merge_resolution_ca_is_queued(self) -> None:
        class StubMetricsRepo:
            def get_all(self):
                return {"master": {best_target_metric_key(): 1.20, "history": []}}

            def get_for_branch(self, branch):
                return {best_target_metric_key(): 1.20, "history": []} if branch == "master" else None

            def append_baseline_run(self, branch, target_metric_value):
                pass

            def append_promotion_for_branch(self, branch, record):
                pass

        service = WorkflowService(git_service=MergeFailingGitService())
        service.metrics_repo = StubMetricsRepo()
        seed = service.create_seed("ralph merge conflict behavior", baseline_branch="master", ralph_loop_enabled=True)
        run = StageRun(
            run_id="ca-20990101-000041-deadbeef",
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.running,
            task_id="task-ca-20990101-000041-cafebabe",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        seed.status = SeedStatus.adapting
        seed.latest_run_id = run.run_id
        service.seed_repo.save(seed)

        # Positive signal (1.18 < 1.20 when lower is better) so promote is attempted; MergeFailingGitService triggers merge_resolution CA
        summary_path = write_summary_file({
            "checks": ["entrypoint"],
            "notes": "done",
            "completed_at": "2099-01-01 00:00:00",
            "metrics": {TARGET_METRIC_KEY: 1.18, "training_seconds": 300.1},
        })
        service.finish_ca_run(seed.seed_id, run.run_id, summary_path)

        merge_resolution_tasks = [
            read_task(path)
            for path in list_pending("ca")
            if read_task(path).get("seed_id") == seed.seed_id
            and read_task(path).get("merge_resolution") is True
        ]
        requeued_plan_do_tasks = [
            read_task(path)
            for path in list_pending("pd")
            if read_task(path).get("seed_id") == seed.seed_id
        ]
        self.assertTrue(merge_resolution_tasks)
        self.assertFalse(requeued_plan_do_tasks)

    def test_mark_run_failed_requeues_plan_do_for_ralph_ca_failure(self) -> None:
        service = WorkflowService()
        seed = service.create_seed("ralph requeue on ca failure", baseline_branch="master", ralph_loop_enabled=True)
        run = StageRun(
            run_id="ca-20990101-000042-deadbeef",
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.running,
            task_id="task-ca-20990101-000042-cafebabe",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        seed.status = SeedStatus.adapting
        seed.latest_run_id = run.run_id
        service.seed_repo.save(seed)
        # Ralph only requeues when merge_resolution or metrics_recovery CA fails (not common CA).
        task_path = write_task(
            "ca",
            {"seed_id": seed.seed_id, "run_id": run.run_id, "merge_resolution": True},
            task_id=run.task_id,
        )

        service.mark_run_failed(
            seed.seed_id, run.run_id, "agent failed unexpectedly", task_path=task_path
        )

        requeued_plan_do_tasks = [
            read_task(path)
            for path in list_pending("pd")
            if read_task(path).get("seed_id") == seed.seed_id
        ]
        self.assertTrue(requeued_plan_do_tasks)
        events = load_events(seed.seed_id)
        self.assertTrue(any(event.get("kind") == "ralph.requeued" for event in events))

    def test_mark_run_failed_does_not_requeue_for_common_ca_failure(self) -> None:
        """Ralph only requeues when merge_resolution or metrics_recovery CA fails; common CA failure does not."""
        service = WorkflowService()
        seed = service.create_seed(
            "ralph common ca failure", baseline_branch="master", ralph_loop_enabled=True
        )
        run = StageRun(
            run_id="ca-20990101-000043-deadbeef",
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.running,
            task_id="task-ca-20990101-000043-cafebabe",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        seed.status = SeedStatus.adapting
        seed.latest_run_id = run.run_id
        service.seed_repo.save(seed)
        # No task_path, or task without merge_resolution/metrics_recovery → no requeue
        service.mark_run_failed(seed.seed_id, run.run_id, "agent failed unexpectedly")

        requeued_plan_do_tasks = [
            path for path in list_pending("pd") if read_task(path).get("seed_id") == seed.seed_id
        ]
        self.assertFalse(requeued_plan_do_tasks)
        events = load_events(seed.seed_id)
        self.assertFalse(any(event.get("kind") == "ralph.requeued" for event in events))

    def test_ralph_merge_outcomes_queue_policy(self) -> None:
        scenarios = [
            {
                "name": "merge_success_queues_next_plan",
                "git_service": NoOpPromoteGitService(),
                "expect_plan_do": True,
                "expect_merge_resolution_ca": False,
            },
            {
                "name": "merge_failure_queues_resolution_ca_only",
                "git_service": MergeFailingGitService(),
                "expect_plan_do": False,
                "expect_merge_resolution_ca": True,
            },
        ]
        for case in scenarios:
            with self.subTest(case["name"]):
                class StubMetricsRepo:
                    def get_all(self):
                        return {"master": {best_target_metric_key(): 1.20, "history": []}}

                    def get_for_branch(self, branch):
                        return {best_target_metric_key(): 1.20, "history": []} if branch == "master" else None

                    def append_baseline_run(self, branch, target_metric_value):
                        pass

                    def append_promotion_for_branch(self, branch, record):
                        pass

                service = WorkflowService(git_service=case["git_service"])
                service.metrics_repo = StubMetricsRepo()
                seed = service.create_seed(
                    f"ralph merge policy {case['name']}",
                    baseline_branch="master",
                    ralph_loop_enabled=True,
                )
                run = StageRun(
                    run_id=f"ca-20990101-00005{1 if case['expect_plan_do'] else 2}-deadbeef",
                    seed_id=seed.seed_id,
                    stage=StageName.ca,
                    status=RunStatus.running,
                    task_id=f"task-ca-20990101-00005{1 if case['expect_plan_do'] else 2}-cafebabe",
                    created_at=0.0,
                    updated_at=0.0,
                )
                service.run_repo.save(run)
                seed.status = SeedStatus.adapting
                seed.latest_run_id = run.run_id
                service.seed_repo.save(seed)

                # Positive signal (1.18 < 1.20 when lower is better) so promotion is attempted
                summary_path = write_summary_file({
                    "checks": ["entrypoint"],
                    "notes": "done",
                    "completed_at": "2099-01-01 00:00:00",
                    "metrics": {TARGET_METRIC_KEY: 1.18, "training_seconds": 300.1},
                })
                service.finish_ca_run(seed.seed_id, run.run_id, summary_path)

                queued_plan_do_tasks = [
                    read_task(path)
                    for path in list_pending("pd")
                    if read_task(path).get("seed_id") == seed.seed_id
                ]
                queued_resolution_ca_tasks = [
                    read_task(path)
                    for path in list_pending("ca")
                    if read_task(path).get("seed_id") == seed.seed_id
                    and read_task(path).get("merge_resolution") is True
                ]
                events = load_events(seed.seed_id)
                ralph_requeued = any(event.get("kind") == "ralph.requeued" for event in events)

                self.assertEqual(bool(queued_plan_do_tasks), case["expect_plan_do"])
                self.assertEqual(bool(queued_resolution_ca_tasks), case["expect_merge_resolution_ca"])
                self.assertEqual(ralph_requeued, case["expect_plan_do"])

    def test_ralph_conflict_resolution_ca_outcomes_queue_policy(self) -> None:
        scenarios = [
            {
                "name": "resolution_pass_queues_next_plan",
                "git_service": NoOpPromoteGitService(),
                "expect_plan_do": True,
                "expect_next_resolution_ca": False,
            },
            {
                "name": "resolution_merge_fail_does_not_queue_plan",
                "git_service": MergeFailingGitService(),
                "expect_plan_do": True,
                "expect_next_resolution_ca": False,
            },
        ]
        for case in scenarios:
            with self.subTest(case["name"]):
                service = WorkflowService(git_service=case["git_service"])
                service.metrics_repo.append_baseline_run("master", 1.20)
                seed = service.create_seed(
                    f"ralph conflict resolution {case['name']}",
                    baseline_branch="master",
                    ralph_loop_enabled=True,
                )
                run = StageRun(
                    run_id=f"ca-20990101-00006{1 if case['expect_plan_do'] else 2}-deadbeef",
                    seed_id=seed.seed_id,
                    stage=StageName.ca,
                    status=RunStatus.running,
                    task_id=f"task-ca-20990101-00006{1 if case['expect_plan_do'] else 2}-cafebabe",
                    created_at=0.0,
                    updated_at=0.0,
                )
                service.run_repo.save(run)
                seed.status = SeedStatus.adapting
                seed.latest_run_id = run.run_id
                service.seed_repo.save(seed)

                summary_path = write_summary_file({
                    "checks": ["entrypoint"],
                    "notes": "done",
                    "completed_at": "2099-01-01 00:00:00",
                    "metrics": {TARGET_METRIC_KEY: 1.17, "training_seconds": 300.1},
                })
                service.finish_ca_run(
                    seed.seed_id,
                    run.run_id,
                    summary_path,
                    merge_resolution=True,
                )

                queued_plan_do_tasks = [
                    read_task(path)
                    for path in list_pending("pd")
                    if read_task(path).get("seed_id") == seed.seed_id
                ]
                queued_resolution_ca_tasks = [
                    read_task(path)
                    for path in list_pending("ca")
                    if read_task(path).get("seed_id") == seed.seed_id
                    and read_task(path).get("merge_resolution") is True
                ]
                events = load_events(seed.seed_id)
                ralph_requeued = any(event.get("kind") == "ralph.requeued" for event in events)

                self.assertEqual(bool(queued_plan_do_tasks), case["expect_plan_do"])
                self.assertEqual(bool(queued_resolution_ca_tasks), case["expect_next_resolution_ca"])
                self.assertEqual(ralph_requeued, case["expect_plan_do"])

    def test_ralph_restore_uses_commit_before_pd_not_baseline(self) -> None:
        git_service = RecordingGitService()
        service = WorkflowService(git_service=git_service)
        service.metrics_repo.append_baseline_run("master", 1.10)
        seed = service.create_seed("ralph restore ref", baseline_branch="master", ralph_loop_enabled=True)
        service.queue_pd(seed.seed_id)
        seed = service.require_seed(seed.seed_id)
        pd_run = service.require_run(seed.latest_run_id)
        self.assertEqual(pd_run.stage, StageName.pd)
        pd_run.summary["commit_sha_before_pd"] = "pre_pd_abc123"
        service.run_repo.save(pd_run)
        pd_summary = {
            "idea": "plan",
            "target_component": "model",
            "description": "desc",
            "source_refs": [],
            "commit_sha": "post_pd_sha",
            "completed_at": "2099-01-01 00:00:00",
        }
        service.finish_pd_run(seed.seed_id, pd_run.run_id, write_summary_file(pd_summary))
        seed = service.require_seed(seed.seed_id)
        ca_run = service.require_run(seed.latest_run_id)
        self.assertEqual(ca_run.stage, StageName.ca)
        self.assertEqual(ca_run.summary.get("commit_sha_before_pd"), "pre_pd_abc123")

    def test_ralph_restore_on_negative_signal_resets_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            git_service = RecordingGitService()
            service = WorkflowService(git_service=git_service)
            service.metrics_repo.append_baseline_run("master", 1.10)
            seed = service.create_seed(
                "ralph negative signal reset",
                baseline_branch="master",
                ralph_loop_enabled=True,
            )
            seed.worktree_path = tmpdir
            service.seed_repo.save(seed)
            service.queue_pd(seed.seed_id)
            seed = service.require_seed(seed.seed_id)
            pd_run = service.require_run(seed.latest_run_id)
            pd_run.summary["commit_sha_before_pd"] = "4387e4e"
            service.run_repo.save(pd_run)
            pd_summary = {
                "idea": "plan",
                "target_component": "model",
                "description": "desc",
                "source_refs": [],
                "commit_sha": "post_pd",
                "completed_at": "2099-01-01 00:00:00",
            }
            service.finish_pd_run(seed.seed_id, pd_run.run_id, write_summary_file(pd_summary))
            seed = service.require_seed(seed.seed_id)
            ca_run = service.require_run(seed.latest_run_id)
            ca_summary_negative = {
                "checks": ["entrypoint"],
                "notes": "done",
                "completed_at": "2099-01-01 00:00:00",
                "metrics": {TARGET_METRIC_KEY: 1.15, "training_seconds": 100},
            }
            service.finish_ca_run(seed.seed_id, ca_run.run_id, write_summary_file(ca_summary_negative))
            reset_calls = [c for c in git_service.commands if c[0][:2] == ("reset", "--hard")]
            self.assertEqual(len(reset_calls), 1)
            self.assertEqual(reset_calls[0][0][2], "4387e4e")
            # Ralph common tail: negative_signal → queue next Plan-Do
            pd_tasks = [read_task(p) for p in list_pending("pd") if read_task(p).get("seed_id") == seed.seed_id]
            self.assertTrue(pd_tasks, "Ralph should queue next Plan-Do after negative_signal CA")
            self.assertTrue(
                any(e.get("kind") == "ralph.requeued" for e in load_events(seed.seed_id)),
                "ralph.requeued event after negative_signal",
            )

    def test_ralph_restore_on_neutral_signal_resets_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            git_service = RecordingGitService()
            service = WorkflowService(git_service=git_service)
            service.metrics_repo.append_baseline_run("master", 1.10)
            seed = service.create_seed(
                "ralph neutral signal reset",
                baseline_branch="master",
                ralph_loop_enabled=True,
            )
            seed.worktree_path = tmpdir
            service.seed_repo.save(seed)
            service.queue_pd(seed.seed_id)
            seed = service.require_seed(seed.seed_id)
            pd_run = service.require_run(seed.latest_run_id)
            pd_run.summary["commit_sha_before_pd"] = "4387e4e"
            service.run_repo.save(pd_run)
            pd_summary = {
                "idea": "plan",
                "target_component": "model",
                "description": "desc",
                "source_refs": [],
                "commit_sha": "post_pd",
                "completed_at": "2099-01-01 00:00:00",
            }
            service.finish_pd_run(seed.seed_id, pd_run.run_id, write_summary_file(pd_summary))
            seed = service.require_seed(seed.seed_id)
            ca_run = service.require_run(seed.latest_run_id)
            ca_summary_neutral = {
                "checks": ["entrypoint"],
                "notes": "done",
                "completed_at": "2099-01-01 00:00:00",
                "metrics": {TARGET_METRIC_KEY: 1.10, "training_seconds": 100},
            }
            service.finish_ca_run(seed.seed_id, ca_run.run_id, write_summary_file(ca_summary_neutral))
            reset_calls = [c for c in git_service.commands if c[0][:2] == ("reset", "--hard")]
            self.assertEqual(len(reset_calls), 1)
            self.assertEqual(reset_calls[0][0][2], "4387e4e")
            # Ralph common tail: neutral → queue next Plan-Do
            pd_tasks = [read_task(p) for p in list_pending("pd") if read_task(p).get("seed_id") == seed.seed_id]
            self.assertTrue(pd_tasks, "Ralph should queue next Plan-Do after neutral CA")
            self.assertTrue(
                any(e.get("kind") == "ralph.requeued" for e in load_events(seed.seed_id)),
                "ralph.requeued event after neutral",
            )

    def test_ralph_restore_on_error_signal_resets_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            git_service = RecordingGitService()
            service = WorkflowService(git_service=git_service)
            service.metrics_repo.append_baseline_run("master", 1.10)
            seed = service.create_seed(
                "ralph error signal reset",
                baseline_branch="master",
                ralph_loop_enabled=True,
            )
            seed.worktree_path = tmpdir
            service.seed_repo.save(seed)
            run = StageRun(
                run_id="ca-20990101-000050-deadbeef",
                seed_id=seed.seed_id,
                stage=StageName.ca,
                status=RunStatus.running,
                task_id="task-ca-20990101-000050-cafebabe",
                created_at=0.0,
                updated_at=0.0,
                summary={"commit_sha_before_pd": "4387e4e"},
            )
            service.run_repo.save(run)
            seed.status = SeedStatus.adapting
            seed.latest_run_id = run.run_id
            service.seed_repo.save(seed)
            # No TARGET_METRIC_KEY in metrics → signal "error" → early return: queue metrics_recovery CA + Ralph queues next Plan-Do
            summary_path = write_summary_file({"checks": ["entrypoint"], "notes": "done", "metrics": {"training_seconds": 1.0}})
            service.finish_ca_run(seed.seed_id, run.run_id, summary_path)
            reset_calls = [c for c in git_service.commands if c[0][:2] == ("reset", "--hard")]
            self.assertEqual(len(reset_calls), 1)
            self.assertEqual(reset_calls[0][0][2], "4387e4e")
            # Early return path: metrics_recovery CA queued and Ralph queues next Plan-Do
            ca_recovery = [read_task(p) for p in list_pending("ca") if read_task(p).get("seed_id") == seed.seed_id and read_task(p).get("metrics_recovery") is True]
            self.assertTrue(ca_recovery, "metrics_recovery CA should be queued when CA completes with error (no metrics)")
            pd_tasks = [read_task(p) for p in list_pending("pd") if read_task(p).get("seed_id") == seed.seed_id]
            self.assertTrue(pd_tasks, "Ralph should queue next Plan-Do after error (no metrics) early return")
            self.assertTrue(
                any(e.get("kind") == "ralph.requeued" for e in load_events(seed.seed_id)),
                "ralph.requeued event after error no-metrics path",
            )

    def test_ralph_restore_on_mark_run_failed_ca_resets_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            git_service = RecordingGitService()
            service = WorkflowService(git_service=git_service)
            seed = service.create_seed(
                "ralph failed ca reset",
                baseline_branch="master",
                ralph_loop_enabled=True,
            )
            seed.worktree_path = tmpdir
            service.seed_repo.save(seed)
            run = StageRun(
                run_id="ca-20990101-000051-deadbeef",
                seed_id=seed.seed_id,
                stage=StageName.ca,
                status=RunStatus.running,
                task_id="task-ca-20990101-000051-cafebabe",
                created_at=0.0,
                updated_at=0.0,
                summary={"commit_sha_before_pd": "4387e4e"},
            )
            service.run_repo.save(run)
            seed.status = SeedStatus.adapting
            seed.latest_run_id = run.run_id
            service.seed_repo.save(seed)
            # Ralph restore/requeue only runs when merge_resolution or metrics_recovery CA fails.
            task_path = write_task(
                "ca",
                {"seed_id": seed.seed_id, "run_id": run.run_id, "metrics_recovery": True},
                task_id=run.task_id,
            )
            service.mark_run_failed(
                seed.seed_id, run.run_id, "agent failed unexpectedly", task_path=task_path
            )
            reset_calls = [c for c in git_service.commands if c[0][:2] == ("reset", "--hard")]
            self.assertEqual(len(reset_calls), 1)
            self.assertEqual(reset_calls[0][0][2], "4387e4e")
            # metrics_recovery CA fail → Ralph queues next Plan-Do
            pd_tasks = [read_task(p) for p in list_pending("pd") if read_task(p).get("seed_id") == seed.seed_id]
            self.assertTrue(pd_tasks, "Ralph should queue next Plan-Do after metrics_recovery CA failure")
            self.assertTrue(
                any(e.get("kind") == "ralph.requeued" for e in load_events(seed.seed_id)),
                "ralph.requeued event after metrics_recovery CA fail",
            )

    def test_ralph_metrics_recovery_ca_success_queues_next_plan_do(self) -> None:
        """When metrics_recovery CA completes successfully (with metrics), Ralph queues next Plan-Do via common tail."""
        service = WorkflowService(git_service=NoOpPromoteGitService())
        service.metrics_repo.append_baseline_run("master", 1.20)
        seed = service.create_seed(
            "ralph metrics recovery success",
            baseline_branch="master",
            ralph_loop_enabled=True,
        )
        run = StageRun(
            run_id="ca-20990101-000060-deadbeef",
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.running,
            task_id="task-ca-20990101-000060-cafebabe",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        seed.status = SeedStatus.adapting
        seed.latest_run_id = run.run_id
        service.seed_repo.save(seed)
        summary_path = write_summary_file({
            "checks": ["entrypoint"],
            "notes": "recovered",
            "completed_at": "2099-01-01 00:00:00",
            "metrics": {TARGET_METRIC_KEY: 1.18, "training_seconds": 100},
        })
        service.finish_ca_run(
            seed.seed_id,
            run.run_id,
            summary_path,
            metrics_recovery=True,
        )
        pd_tasks = [read_task(p) for p in list_pending("pd") if read_task(p).get("seed_id") == seed.seed_id]
        self.assertTrue(pd_tasks, "Ralph should queue next Plan-Do after metrics_recovery CA success")
        self.assertTrue(
            any(e.get("kind") == "ralph.requeued" for e in load_events(seed.seed_id)),
            "ralph.requeued event after metrics_recovery CA success",
        )

    def test_ralph_metrics_recovery_ca_still_no_metrics_queues_next_plan_do(self) -> None:
        """When metrics_recovery CA completes but still has no target metric, common tail still queues next Plan-Do."""
        service = WorkflowService(git_service=NoOpPromoteGitService())
        service.metrics_repo.append_baseline_run("master", 1.20)
        seed = service.create_seed(
            "ralph metrics recovery still failed",
            baseline_branch="master",
            ralph_loop_enabled=True,
        )
        run = StageRun(
            run_id="ca-20990101-000061-deadbeef",
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.running,
            task_id="task-ca-20990101-000061-cafebabe",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        seed.status = SeedStatus.adapting
        seed.latest_run_id = run.run_id
        service.seed_repo.save(seed)
        # metrics_recovery=True so we do NOT take early return; signal "error" → terminal_status failed → common tail
        summary_path = write_summary_file({
            "checks": ["entrypoint"],
            "notes": "no metrics",
            "completed_at": "2099-01-01 00:00:00",
            "metrics": {"training_seconds": 100},
        })
        service.finish_ca_run(
            seed.seed_id,
            run.run_id,
            summary_path,
            metrics_recovery=True,
        )
        pd_tasks = [read_task(p) for p in list_pending("pd") if read_task(p).get("seed_id") == seed.seed_id]
        self.assertTrue(pd_tasks, "Ralph should queue next Plan-Do after metrics_recovery CA with still no target metric")
        self.assertTrue(
            any(e.get("kind") == "ralph.requeued" for e in load_events(seed.seed_id)),
            "ralph.requeued event",
        )


if __name__ == "__main__":
    unittest.main()
