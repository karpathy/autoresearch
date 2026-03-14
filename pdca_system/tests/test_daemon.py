"""Tests for daemon: prompt building, salvage/agent failure, claim_pending, restore_in_progress, agent rotation."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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

from pdca_system.daemon import (
    AGENT_CONFIGS,
    AGENT_EXIT0_FAILURE_CHECKS,
    _agent_failure_reason,
    _build_prompt,
    _build_stuck_check_prompt,
    _effective_exit_for_downgrade,
    _invoke_agent,
    _maybe_rotate_after_stuck_check,
    _record_agent_exit,
    _run_stuck_check,
    _should_salvage_completed_ca,
)


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


class AgentRotationAndStuckCheckTests(unittest.TestCase):
    """Tests for agent detection, effective-exit-for-downgrade, stuck-check prompt, and rotation logic."""

    def setUp(self) -> None:
        self._saved_state = dict(run_module._AGENT_RUNTIME_STATE)
        self._saved_state.setdefault("available", [])
        self._saved_state.setdefault("unavailable", {})
        self._saved_state.setdefault("active", None)
        self._saved_state.setdefault("nonzero_streak", {})

    def tearDown(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = self._saved_state.get("available", [])
        run_module._AGENT_RUNTIME_STATE["unavailable"] = self._saved_state.get("unavailable", {})
        run_module._AGENT_RUNTIME_STATE["active"] = self._saved_state.get("active")
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = dict(self._saved_state.get("nonzero_streak", {}))

    def test_effective_exit_for_downgrade_nonzero_unchanged(self) -> None:
        self.assertEqual(_effective_exit_for_downgrade("claude", 1, "", ""), 1)
        self.assertEqual(_effective_exit_for_downgrade("opencode", -1, "timeout", ""), -1)

    def test_effective_exit_for_downgrade_opencode_exit0_quota_treated_as_failure(self) -> None:
        stderr = "Error: You've reached your usage limit for this billing cycle. Upgrade to get more."
        self.assertEqual(_effective_exit_for_downgrade("opencode", 0, stderr, ""), 1)

    def test_effective_exit_for_downgrade_opencode_exit0_no_quota_unchanged(self) -> None:
        self.assertEqual(_effective_exit_for_downgrade("opencode", 0, "normal output", ""), 0)
        self.assertEqual(_effective_exit_for_downgrade("opencode", 0, "Error: something else", ""), 0)

    def test_effective_exit_for_downgrade_claude_exit0_rate_limit_treated_as_failure(self) -> None:
        stderr = "API Error: Rate limit reached"
        self.assertEqual(_effective_exit_for_downgrade("claude", 0, stderr, ""), 1)

    def test_effective_exit_for_downgrade_claude_exit0_no_api_error_unchanged(self) -> None:
        self.assertEqual(_effective_exit_for_downgrade("claude", 0, "normal output", ""), 0)

    def test_effective_exit_for_downgrade_codex_exit0_invalid_responses_api_treated_as_failure(self) -> None:
        stderr = '{"error":{"message":"Invalid Responses API request"}}'
        self.assertEqual(_effective_exit_for_downgrade("codex", 0, stderr, ""), 1)

    def test_effective_exit_for_downgrade_gemini_exit0_error_when_talking_treated_as_failure(self) -> None:
        stderr = "Error when talking to Gemini API Full report at /tmp/x.json"
        self.assertEqual(_effective_exit_for_downgrade("gemini", 0, stderr, ""), 1)

    def test_opencode_exit0_is_failure_requires_error_and_quota(self) -> None:
        check = AGENT_EXIT0_FAILURE_CHECKS["opencode"]
        self.assertTrue(check("Error: You've reached your usage limit.", ""))
        self.assertTrue(check("Error: quota exceeded", ""))
        self.assertFalse(check("", ""))
        # Must have "error:" and a quota phrase; "usage limit" alone without "error:" is not enough
        self.assertFalse(check("The usage limit for this month is 100 (no error line).", ""))

    def test_claude_exit0_is_failure_requires_api_error_and_rate_limit(self) -> None:
        check = AGENT_EXIT0_FAILURE_CHECKS["claude"]
        self.assertTrue(check("API Error: Rate limit reached", ""))
        self.assertFalse(check("API Error: something else", ""))
        self.assertFalse(check("Rate limit", ""))

    def test_build_stuck_check_prompt_contains_agent_and_summary_filename(self) -> None:
        prompt = _build_stuck_check_prompt("opencode", None, None)
        self.assertIn("opencode", prompt)
        self.assertIn(run_module.SUMMARY_FILENAME, prompt)
        self.assertIn("previous_agent_stuck", prompt)
        self.assertIn("(no log file)", prompt)

    def test_build_stuck_check_prompt_includes_tail_of_logs(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write("line1\nline2\nline3\n")
            stderr_path = Path(f.name)
        try:
            prompt = _build_stuck_check_prompt("claude", None, stderr_path)
            self.assertIn("line3", prompt)
            self.assertIn("line2", prompt)
        finally:
            stderr_path.unlink(missing_ok=True)

    def test_record_agent_exit_returns_false_when_single_agent(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude"]
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 5}
        run_module._AGENT_RUNTIME_STATE["active"] = "claude"
        self.assertFalse(_record_agent_exit("claude", 1))
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["nonzero_streak"]["claude"], 6)

    def test_record_agent_exit_returns_true_when_streak_reached_and_two_agents(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude", "opencode"]
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 4, "opencode": 0}
        run_module._AGENT_RUNTIME_STATE["active"] = "claude"
        # threshold is 5: one more failure -> streak 5, should return True (run stuck-check)
        self.assertTrue(_record_agent_exit("claude", 1))
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["nonzero_streak"]["claude"], 5)

    def test_record_agent_exit_resets_streak_on_success(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude", "opencode"]
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 3, "opencode": 0}
        _record_agent_exit("claude", 0)
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["nonzero_streak"]["claude"], 0)

    def test_record_agent_exit_returns_false_for_unknown_agent(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude"]
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 0}
        self.assertFalse(_record_agent_exit("unknown_agent", 1))

    def test_maybe_rotate_after_stuck_check_no_switch_when_single_agent(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude"]
        run_module._AGENT_RUNTIME_STATE["active"] = "claude"
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 5}
        with patch.object(run_module, "_run_stuck_check") as mock_run:
            _maybe_rotate_after_stuck_check("claude", "run-1", None, None)
            mock_run.assert_not_called()
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["active"], "claude")

    def test_maybe_rotate_after_stuck_check_switches_when_stuck_confirmed(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude", "opencode"]
        run_module._AGENT_RUNTIME_STATE["active"] = "claude"
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 5, "opencode": 0}
        with patch.object(run_module, "_run_stuck_check", return_value=(True, {"previous_agent_stuck": True, "reason": "quota"})):
            _maybe_rotate_after_stuck_check("claude", "run-1", None, None)
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["active"], "opencode")

    def test_maybe_rotate_after_stuck_check_no_switch_when_not_stuck_resets_streak(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude", "opencode"]
        run_module._AGENT_RUNTIME_STATE["active"] = "claude"
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 5, "opencode": 0}
        with patch.object(run_module, "_run_stuck_check", return_value=(True, {"previous_agent_stuck": False, "reason": "transient"})):
            _maybe_rotate_after_stuck_check("claude", "run-1", None, None)
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["active"], "claude")
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["nonzero_streak"]["claude"], 0)

    def test_maybe_rotate_after_stuck_check_no_switch_when_stuck_check_fails_resets_streak(self) -> None:
        run_module._AGENT_RUNTIME_STATE["available"] = ["claude", "opencode"]
        run_module._AGENT_RUNTIME_STATE["active"] = "claude"
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"claude": 5, "opencode": 0}
        with patch.object(run_module, "_run_stuck_check", return_value=(False, None)):
            _maybe_rotate_after_stuck_check("claude", "run-1", None, None)
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["active"], "claude")
        self.assertEqual(run_module._AGENT_RUNTIME_STATE["nonzero_streak"]["claude"], 0)


def _make_mock_process(returncode: int = 0) -> MagicMock:
    """Build a mock subprocess that completes immediately with given returncode; stdout/stderr readline returns ''."""
    process = MagicMock()
    process.returncode = returncode
    process.wait = MagicMock(return_value=None)
    process.kill = MagicMock()
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.close = MagicMock()
    for pipe_name in ("stdout", "stderr"):
        pipe = MagicMock()
        pipe.readline = MagicMock(return_value="")
        pipe.close = MagicMock()
        setattr(process, pipe_name, pipe)
    return process


class InvokeAgentTests(unittest.TestCase):
    """Tests for _invoke_agent: agent_override, timeout_override, and that rotation is not triggered when override is set."""

    def setUp(self) -> None:
        self._log_dir = Path(tempfile.mkdtemp(prefix="pdca_invoke_test_"))
        self._orig_log_dir = run_module.LOG_DIR
        run_module.LOG_DIR = self._log_dir

    def tearDown(self) -> None:
        run_module.LOG_DIR = self._orig_log_dir
        shutil.rmtree(self._log_dir, ignore_errors=True)

    def test_invoke_agent_with_agent_override_uses_that_agent(self) -> None:
        """With agent_override=kimi, Popen is called with kimi's cmd (from AGENT_CONFIGS)."""
        mock_process = _make_mock_process(returncode=0)
        with patch.object(run_module.subprocess, "Popen", return_value=mock_process) as mock_popen:
            agent, code, out, err, out_path, err_path = _invoke_agent(
                "hello",
                "pd",
                "run-invoke-1",
                worktree_path=None,
                agent_override="kimi",
            )
        self.assertEqual(agent, "kimi")
        self.assertEqual(code, 0)
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], AGENT_CONFIGS["kimi"]["cmd"][0], "cmd should start with kimi binary")
        self.assertIn("hello", cmd, "prompt should be in cmd for arg-based agent")

    def test_invoke_agent_with_agent_override_uses_timeout_override(self) -> None:
        """With timeout_override=99, process.wait(timeout=99) is called."""
        mock_process = _make_mock_process(returncode=0)
        with patch.object(run_module.subprocess, "Popen", return_value=mock_process):
            _invoke_agent(
                "task",
                "ca",
                "run-invoke-2",
                agent_override="kimi",
                timeout_override=99,
            )
        mock_process.wait.assert_called_once()
        self.assertEqual(mock_process.wait.call_args[1]["timeout"], 99)

    def test_invoke_agent_with_agent_override_does_not_call_record_agent_exit(self) -> None:
        """With agent_override, _record_agent_exit and _maybe_rotate_after_stuck_check must not be called."""
        mock_process = _make_mock_process(returncode=1)
        with patch.object(run_module.subprocess, "Popen", return_value=mock_process), patch.object(
            run_module, "_record_agent_exit"
        ) as mock_record, patch.object(
            run_module, "_maybe_rotate_after_stuck_check"
        ) as mock_rotate:
            agent, code, _out, _err, _o, _e = _invoke_agent(
                "fail",
                "pd",
                "run-invoke-3",
                agent_override="kimi",
            )
        self.assertEqual(agent, "kimi")
        self.assertEqual(code, 1)
        mock_record.assert_not_called()
        mock_rotate.assert_not_called()

    def test_invoke_agent_with_agent_override_unknown_agent_returns_error(self) -> None:
        """With agent_override to unknown agent, return error tuple without raising."""
        agent, code, _out, err, out_path, err_path = _invoke_agent(
            "x",
            "pd",
            "run-invoke-4",
            agent_override="nonexistent_agent",
        )
        self.assertEqual(agent, "nonexistent_agent")
        self.assertEqual(code, -1)
        self.assertIn("Unknown agent", err)
        self.assertIsNone(out_path)
        self.assertIsNone(err_path)

    def test_invoke_agent_with_agent_override_file_not_found_returns_error_no_mark_unavailable(self) -> None:
        """With agent_override, FileNotFoundError from Popen returns error and does not call _mark_agent_unavailable."""
        with patch.object(run_module.subprocess, "Popen", side_effect=FileNotFoundError("binary not found")), patch.object(
            run_module, "_mark_agent_unavailable"
        ) as mock_mark:
            agent, code, _out, err, out_path, err_path = _invoke_agent(
                "x",
                "pd",
                "run-invoke-5",
                agent_override="kimi",
            )
        self.assertEqual(agent, "kimi")
        self.assertEqual(code, -1)
        self.assertIn("binary not found", err)
        mock_mark.assert_not_called()

    def test_invoke_agent_without_override_uses_active_agent(self) -> None:
        """Without agent_override, _active_agent_name() is used and Popen receives that agent's cmd."""
        run_module._AGENT_RUNTIME_STATE["available"] = ["kimi", "opencode"]
        run_module._AGENT_RUNTIME_STATE["active"] = "kimi"
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"kimi": 0, "opencode": 0}
        mock_process = _make_mock_process(returncode=0)
        with patch.object(run_module.subprocess, "Popen", return_value=mock_process) as mock_popen:
            agent, code, _out, _err, _o, _e = _invoke_agent("hello", "pd", "run-invoke-6", worktree_path=None)
        self.assertEqual(agent, "kimi")
        self.assertEqual(code, 0)
        cmd = mock_popen.call_args[0][0]
        self.assertEqual(cmd[0], AGENT_CONFIGS["kimi"]["cmd"][0])

    def test_invoke_agent_without_override_calls_record_agent_exit_on_nonzero(self) -> None:
        """Without agent_override, non-zero exit triggers _record_agent_exit (and maybe _maybe_rotate)."""
        run_module._AGENT_RUNTIME_STATE["available"] = ["kimi"]
        run_module._AGENT_RUNTIME_STATE["active"] = "kimi"
        run_module._AGENT_RUNTIME_STATE["nonzero_streak"] = {"kimi": 0}
        mock_process = _make_mock_process(returncode=2)
        with patch.object(run_module.subprocess, "Popen", return_value=mock_process), patch.object(
            run_module, "_record_agent_exit", return_value=False
        ) as mock_record:
            _invoke_agent("fail", "pd", "run-invoke-7", worktree_path=None)
        mock_record.assert_called_once()
        self.assertEqual(mock_record.call_args[0], ("kimi", 2))


def _run_stuck_check_with_diagnostics(
    daemon_module: object, root: Path, log_dir: Path, run_id: str, prompt: str
) -> tuple[bool, dict[str, object] | None, str]:
    """Run stuck-check with PROJECT_ROOT/LOG_DIR patched; return (ok, summary, diagnostic)."""
    orig_root = daemon_module.PROJECT_ROOT
    orig_log = daemon_module.LOG_DIR
    orig_timeout = daemon_module.STUCK_CHECK_TIMEOUT_SECONDS
    try:
        daemon_module.PROJECT_ROOT = root
        daemon_module.LOG_DIR = log_dir
        daemon_module.STUCK_CHECK_TIMEOUT_SECONDS = 120
        ok, summary = _run_stuck_check("kimi", prompt, run_id)
    finally:
        daemon_module.PROJECT_ROOT = orig_root
        daemon_module.LOG_DIR = orig_log
        daemon_module.STUCK_CHECK_TIMEOUT_SECONDS = orig_timeout

    diag: list[str] = []
    summary_path = root / daemon_module.SUMMARY_FILENAME
    diag.append(f"summary_path exists={summary_path.exists()}")
    if (log_dir / f"{run_id}.stderr.log").exists():
        tail = (log_dir / f"{run_id}.stderr.log").read_text(encoding="utf-8", errors="replace").strip()
        diag.append(f"stuck_check stderr (tail): {tail[-500:]!r}")
    if summary_path.exists():
        diag.append(f"summary content: {summary_path.read_text(encoding='utf-8', errors='replace')!r}")
    return ok, summary, "; ".join(diag)


@unittest.skipUnless(shutil.which("kimi"), "kimi CLI not on PATH")
class StuckCheckKimiIntegrationTests(unittest.TestCase):
    """Integration tests: run real Kimi agent to verify stuck-check on mocked previous-agent logs."""

    def test_stuck_check_kimi_reports_stuck_when_logs_show_quota_error(self) -> None:
        """Mock previous agent logs with quota error; run real kimi; expect previous_agent_stuck true."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "logs"
            log_dir.mkdir()
            prev_stdout = root / "prev.stdout.log"
            prev_stderr = root / "prev.stderr.log"
            prev_stdout.write_text("> build · k2p5\n", encoding="utf-8")
            prev_stderr.write_text(
                "Error: You've reached your usage limit for this billing cycle. "
                "Your quota will be refreshed in the next cycle. Upgrade to get more: https://example.com\n",
                encoding="utf-8",
            )
            prompt = _build_stuck_check_prompt("opencode", prev_stdout, prev_stderr)
            self.assertIn("usage limit", prompt)
            self.assertIn("Error:", prompt)

            ok, summary, diag = _run_stuck_check_with_diagnostics(
                run_module, root, log_dir, "test-stuck-check-stuck", prompt
            )
            self.assertTrue(ok, f"stuck-check run should succeed (exit 0 and valid summary). {diag}")
            self.assertIsNotNone(summary)
            self.assertIn("previous_agent_stuck", summary)
            self.assertIs(
                summary["previous_agent_stuck"],
                True,
                f"Logs show quota error; expected previous_agent_stuck true, got {summary!s}. {diag}",
            )
            self.assertIn("reason", summary)

    def test_stuck_check_kimi_reports_not_stuck_when_logs_show_normal_output(self) -> None:
        """Mock previous agent logs with normal output; run real kimi; expect previous_agent_stuck false."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "logs"
            log_dir.mkdir()
            prev_stdout = root / "prev.stdout.log"
            prev_stderr = root / "prev.stderr.log"
            prev_stdout.write_text(
                "Listing directory contents.\n"
                "total 4\ndrwxr-x 2 user group 4096 Jan 1 12:00 .\n"
                "- file1.py\n- file2.txt\n",
                encoding="utf-8",
            )
            prev_stderr.write_text("(no errors)\n", encoding="utf-8")
            prompt = _build_stuck_check_prompt("opencode", prev_stdout, prev_stderr)

            ok, summary, diag = _run_stuck_check_with_diagnostics(
                run_module, root, log_dir, "test-stuck-check-not-stuck", prompt
            )
            self.assertTrue(ok, f"stuck-check run should succeed. {diag}")
            self.assertIsNotNone(summary)
            self.assertIn("previous_agent_stuck", summary)
            self.assertIs(
                summary["previous_agent_stuck"],
                False,
                f"Logs show normal output; expected previous_agent_stuck false, got {summary!s}. {diag}",
            )


if __name__ == "__main__":
    unittest.main()
