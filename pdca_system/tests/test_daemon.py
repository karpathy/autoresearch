"""Tests for daemon: prompt building, salvage/agent failure, claim_pending, restore_in_progress."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pdca_system.daemon as run_module
import pdca_system.task as task_module
from pdca_system.config import TARGET_METRIC_KEY
from pdca_system.services.workflow import BASELINE_SEED_ID
from pdca_system.task import (
    PDCA_SYSTEM_ROOT,
    claim_pending,
    read_task,
    restore_in_progress_tasks,
    write_task,
)

from pdca_system.daemon import _agent_failure_reason, _build_prompt, _should_salvage_completed_ca


class DaemonTests(unittest.TestCase):
    def test_should_salvage_completed_ca_only_when_metrics_exist(self) -> None:
        """Salvage is True only for CA with non-zero exit when summary file in cwd has target metric."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            summary_path = cwd / run_module.SUMMARY_FILENAME
            summary_path.write_text(
                json.dumps({"metrics": {TARGET_METRIC_KEY: 1.23}}, indent=2),
                encoding="utf-8",
            )
            self.assertTrue(
                _should_salvage_completed_ca("ca", 1, "run-1", str(cwd)),
                "CA exit 1 with summary file containing target metric should salvage",
            )
        with tempfile.TemporaryDirectory() as tmp:
            # No summary file
            self.assertFalse(
                _should_salvage_completed_ca("ca", 1, "run-1", str(tmp)),
                "No summary file should not salvage",
            )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / run_module.SUMMARY_FILENAME).write_text(
                json.dumps({"metrics": {"other": 1.0}}),
                encoding="utf-8",
            )
            self.assertFalse(
                _should_salvage_completed_ca("ca", 1, "run-1", str(tmp)),
                "Summary without target metric should not salvage",
            )
        self.assertFalse(_should_salvage_completed_ca("ca", 0, "run-1", None))
        self.assertFalse(_should_salvage_completed_ca("pd", 1, "run-1", None))

    def test_agent_failure_reason_uses_exit_state_not_stderr_body(self) -> None:
        self.assertEqual(
            _agent_failure_reason(2, "ok", "agent chatter on stderr"),
            "Agent exited with code 2. See stdout/stderr logs for details.",
        )
        self.assertEqual(
            _agent_failure_reason(-1, "", "timeout after 3600s"),
            "timeout after 3600s",
        )

    def test_build_prompt_references_baseline_path_without_inlining_json(self) -> None:
        prompt = _build_prompt(
            "ca",
            {
                "seed_id": BASELINE_SEED_ID,
                "run_id": "ca-20990101-000099-deadbeef",
                "prompt": "baseline run",
                "worktree_path": None,
                "merge_resolution": False,
                "metrics_recovery": False,
            },
            PDCA_SYSTEM_ROOT / "queue" / "ca" / "fake-task.json",
        )
        self.assertIn("pdca_system/baseline_branches.json", prompt)

    def test_build_prompt_ca_contains_fix_only_constraint(self) -> None:
        prompt = _build_prompt(
            "ca",
            {
                "seed_id": "seed-868a41",
                "run_id": "ca-20260311-200436-27af8a0d",
                "prompt": "Try your best to achieve new SOTA",
                "worktree_path": None,
                "merge_resolution": False,
                "metrics_recovery": False,
            },
            PDCA_SYSTEM_ROOT / "queue" / "ca" / "fake-task.json",
        )
        self.assertIn("Do not put forward new ideas or optimize for better metrics", prompt)
        self.assertIn("The task \"prompt\" is for context only", prompt)

    def test_daemon_submits_two_plan_do_workers(self) -> None:
        class ExecutorSpy:
            instances: list["ExecutorSpy"] = []

            def __init__(self, max_workers: int, thread_name_prefix: str) -> None:
                self.max_workers = max_workers
                self.thread_name_prefix = thread_name_prefix
                self.submit_calls: list[tuple[object, tuple[object, ...]]] = []
                ExecutorSpy.instances.append(self)

            def submit(self, fn, *args):
                self.submit_calls.append((fn, args))
                return None

            def shutdown(self, wait: bool = True) -> None:
                return None

        original_executor = run_module.ThreadPoolExecutor
        original_sleep = run_module.time.sleep
        original_signal = run_module.signal.signal
        original_ensure = run_module.ensure_queue_layout
        original_restore = run_module.restore_in_progress_tasks
        original_heartbeat = run_module.daemon_heartbeat

        def stop_after_first_sleep(_seconds: float) -> None:
            run_module._shutdown = True

        try:
            run_module.ThreadPoolExecutor = ExecutorSpy
            run_module.time.sleep = stop_after_first_sleep
            run_module.signal.signal = lambda *_args, **_kwargs: None
            run_module.ensure_queue_layout = lambda: None
            run_module.restore_in_progress_tasks = lambda: {"pd": 0, "ca": 0, "direct": 0}
            run_module.daemon_heartbeat = lambda: None
            run_module._shutdown = False
            run_module.main()
        finally:
            run_module.ThreadPoolExecutor = original_executor
            run_module.time.sleep = original_sleep
            run_module.signal.signal = original_signal
            run_module.ensure_queue_layout = original_ensure
            run_module.restore_in_progress_tasks = original_restore
            run_module.daemon_heartbeat = original_heartbeat
            run_module._shutdown = False

        by_prefix = {instance.thread_name_prefix: instance for instance in ExecutorSpy.instances}
        self.assertEqual(by_prefix["pdca-pd"].max_workers, 2)
        self.assertEqual(len(by_prefix["pdca-pd"].submit_calls), 2)
        self.assertEqual(len(by_prefix["pdca-ca-gpu"].submit_calls), 1)
        self.assertEqual(len(by_prefix["pdca-ca-aux"].submit_calls), 1)
        self.assertEqual(len(by_prefix["pdca-direct"].submit_calls), 1)

    def test_claim_pending_claims_each_task_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_root = root / "history"
            queue_root = history_root / "queue"
            state_root = history_root / "state"
            stage_dirs = {
                "pd": queue_root / "pd",
                "ca": queue_root / "ca",
                "direct": queue_root / "direct",
            }
            replacements = {
                "HISTORY_ROOT": history_root,
                "QUEUE_ROOT": queue_root,
                "STATE_ROOT": state_root,
                "SEEDS_ROOT": state_root / "seeds",
                "RUNS_ROOT": state_root / "runs",
                "EVENTS_ROOT": state_root / "events",
                "BASELINE_BRANCHES_PATH": root / "baseline_branches.json",
                "BASELINE_METRICS_PATH": root / "baseline_metrics.json",
                "WORKTREE_ROOT": history_root / "worktrees",
                "LOG_ROOT": history_root / "logs",
                "STAGE_DIRS": stage_dirs,
                "IN_PROGRESS_DIR": queue_root / "in_progress",
                "DONE_DIR": queue_root / "done",
                "ERROR_DIR": queue_root / "error",
                "DAEMON_HEARTBEAT_PATH": state_root / "daemon_heartbeat.json",
            }
            originals = {name: getattr(task_module, name) for name in replacements}
            try:
                for name, value in replacements.items():
                    setattr(task_module, name, value)
                task_module.ensure_queue_layout()
                write_task("pd", {"seed_id": "seed-a", "run_id": "pd-run-a"}, task_id="pd-task-a")
                write_task("pd", {"seed_id": "seed-b", "run_id": "pd-run-b"}, task_id="pd-task-b")

                claim_one = claim_pending("pd")
                claim_two = claim_pending("pd")
                claim_three = claim_pending("pd")
            finally:
                for name, value in originals.items():
                    setattr(task_module, name, value)

        self.assertIsNotNone(claim_one)
        self.assertIsNotNone(claim_two)
        self.assertNotEqual(claim_one.name, claim_two.name)
        self.assertIsNone(claim_three)

    def test_claim_pending_ca_lanes_split_gpu_and_aux_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_root = root / "history"
            queue_root = history_root / "queue"
            state_root = history_root / "state"
            stage_dirs = {
                "pd": queue_root / "pd",
                "ca": queue_root / "ca",
                "direct": queue_root / "direct",
            }
            replacements = {
                "HISTORY_ROOT": history_root,
                "QUEUE_ROOT": queue_root,
                "STATE_ROOT": state_root,
                "SEEDS_ROOT": state_root / "seeds",
                "RUNS_ROOT": state_root / "runs",
                "EVENTS_ROOT": state_root / "events",
                "BASELINE_BRANCHES_PATH": root / "baseline_branches.json",
                "BASELINE_METRICS_PATH": root / "baseline_metrics.json",
                "WORKTREE_ROOT": history_root / "worktrees",
                "LOG_ROOT": history_root / "logs",
                "STAGE_DIRS": stage_dirs,
                "IN_PROGRESS_DIR": queue_root / "in_progress",
                "DONE_DIR": queue_root / "done",
                "ERROR_DIR": queue_root / "error",
                "DAEMON_HEARTBEAT_PATH": state_root / "daemon_heartbeat.json",
            }
            originals = {name: getattr(task_module, name) for name in replacements}
            try:
                for name, value in replacements.items():
                    setattr(task_module, name, value)
                task_module.ensure_queue_layout()
                write_task("ca", {"seed_id": "seed-gpu", "run_id": "ca-gpu"}, task_id="task-ca-gpu")
                write_task(
                    "ca",
                    {"seed_id": "seed-aux", "run_id": "ca-aux", "merge_resolution": True},
                    task_id="task-ca-aux",
                )
                aux_claim = claim_pending("ca", lane="aux")
                gpu_claim = claim_pending("ca", lane="gpu")
                none_left = claim_pending("ca", lane="any")
                aux_payload = read_task(aux_claim) if aux_claim is not None else {}
                gpu_payload = read_task(gpu_claim) if gpu_claim is not None else {}
            finally:
                for name, value in originals.items():
                    setattr(task_module, name, value)

        self.assertIsNotNone(aux_claim)
        self.assertEqual(aux_payload.get("merge_resolution"), True)
        self.assertIsNotNone(gpu_claim)
        self.assertNotEqual(gpu_payload.get("merge_resolution"), True)
        self.assertIsNone(none_left)

    def test_restore_in_progress_tasks_requeues_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_root = root / "history"
            queue_root = history_root / "queue"
            state_root = history_root / "state"
            stage_dirs = {
                "pd": queue_root / "pd",
                "ca": queue_root / "ca",
                "direct": queue_root / "direct",
            }
            replacements = {
                "HISTORY_ROOT": history_root,
                "QUEUE_ROOT": queue_root,
                "STATE_ROOT": state_root,
                "SEEDS_ROOT": state_root / "seeds",
                "RUNS_ROOT": state_root / "runs",
                "EVENTS_ROOT": state_root / "events",
                "BASELINE_BRANCHES_PATH": root / "baseline_branches.json",
                "BASELINE_METRICS_PATH": root / "baseline_metrics.json",
                "WORKTREE_ROOT": history_root / "worktrees",
                "LOG_ROOT": history_root / "logs",
                "STAGE_DIRS": stage_dirs,
                "IN_PROGRESS_DIR": queue_root / "in_progress",
                "DONE_DIR": queue_root / "done",
                "ERROR_DIR": queue_root / "error",
                "DAEMON_HEARTBEAT_PATH": state_root / "daemon_heartbeat.json",
            }
            originals = {name: getattr(task_module, name) for name in replacements}
            try:
                for name, value in replacements.items():
                    setattr(task_module, name, value)
                task_module.ensure_queue_layout()
                write_task("pd", {"seed_id": "seed-a", "run_id": "pd-run-a"}, task_id="pd-task-a")
                write_task("ca", {"seed_id": "seed-b", "run_id": "ca-run-b"}, task_id="task-ca-b")
                write_task("direct", {"seed_id": "seed-c", "run_id": "direct-run-c"}, task_id="direct-task-c")
                claim_pending("pd")
                claim_pending("ca")
                claim_pending("direct")

                restored = restore_in_progress_tasks()
                p_count = len(list(stage_dirs["pd"].glob("*.json")))
                ca_count = len(list(stage_dirs["ca"].glob("*.json")))
                direct_count = len(list(stage_dirs["direct"].glob("*.json")))
                in_progress_count = len(list((queue_root / "in_progress").glob("*.json")))
            finally:
                for name, value in originals.items():
                    setattr(task_module, name, value)

        self.assertEqual(restored, {"pd": 1, "ca": 1, "direct": 1})
        self.assertEqual((p_count, ca_count, direct_count), (1, 1, 1))
        self.assertEqual(in_progress_count, 0)


if __name__ == "__main__":
    unittest.main()
