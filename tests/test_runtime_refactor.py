try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from autoresearch_runtime.case_checkpoint import case_scope, current_case_id, summarize_usage
from autoresearch_runtime.dash_http import DashboardHandler, DashboardHTTPServer
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.fuzzy_args import normalize_cli_args
from autoresearch_runtime.gitutil import _restore_candidate
from autoresearch_runtime.locking import _pid_alive, lab_lock
from autoresearch_runtime.runtimeops import capture_worker_identity
from autoresearch_runtime.signals import build_research_signals
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.state import _IGNORED_CACHE_DIRS, _fingerprints
from autoresearch_runtime.store import _read_json, _write_json
from autoresearch_runtime.trial import _aggregate_metric, _first_repeat_rejection


class FuzzyArgsNormalizationTest(unittest.TestCase):
    def test_normalize_cli_args_hoists_confirmed_from_subcommand(self):
        self.assertEqual(normalize_cli_args(["step", "--confirmed"]), ["--confirmed", "step"])
        self.assertEqual(
            normalize_cli_args(["trial", "--runs=3", "--confirmed"]),
            ["--confirmed", "trial", "--repeats=3"],
        )

    def test_normalize_cli_args_hoists_agent_and_maps_aliases(self):
        self.assertEqual(normalize_cli_args(["doctor", "--agent"]), ["--agent", "doctor"])
        self.assertEqual(
            normalize_cli_args(["step", "--from-report", "rep.json"]),
            ["step", "--source", "rep.json"],
        )
        self.assertEqual(
            normalize_cli_args(["step", "--input=data.json"]),
            ["step", "--source=data.json"],
        )

    def test_normalize_cli_args_hoists_config_and_value_flags(self):
        self.assertEqual(
            normalize_cli_args(["step", "--config", "lab.json", "--confirmed"]),
            ["--config", "lab.json", "--confirmed", "step"],
        )
        self.assertEqual(
            normalize_cli_args(["step", "--project=myproj", "--agent"]),
            ["--project=myproj", "--agent", "step"],
        )


class LockingAndProcessRefactorTest(unittest.TestCase):
    def test_pid_alive_handles_posix_permission_error_as_true(self):
        with patch("os.name", "posix"):
            with patch("os.kill", side_effect=PermissionError):
                self.assertTrue(_pid_alive(12345))

    def test_pid_alive_handles_process_lookup_error_and_invalid_pids(self):
        with patch("os.name", "posix"):
            with patch("os.kill", side_effect=ProcessLookupError):
                self.assertFalse(_pid_alive(12345))
        self.assertFalse(_pid_alive(0))
        self.assertFalse(_pid_alive(-10))

    def test_lab_lock_heartbeat_and_mutual_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            with lab_lock(state_dir):
                lock_file = state_dir / ".lock"
                self.assertTrue(lock_file.is_file())
                data = json.loads(lock_file.read_text(encoding="utf-8"))
                self.assertEqual(data.get("pid"), os.getpid())
                self.assertIsInstance(data.get("heartbeat"), float)

                # Concurrent acquire in same lab raises E_LOCKED
                with self.assertRaises(LabError) as ctx:
                    with lab_lock(state_dir):
                        pass
                self.assertEqual(ctx.exception.code, "E_LOCKED")

            # Clean removal on exit
            self.assertFalse((state_dir / ".lock").exists())

    def test_capture_worker_identity_posix_proc_fallback(self):
        proc = Mock(pid=8765)
        with patch("os.name", "posix"), \
             patch("sys.platform", "darwin"), \
             patch("os.getpgid", return_value=8700, create=True), \
             patch("os.getsid", return_value=8600, create=True):
            identity, handle = capture_worker_identity(proc, "test-op-1")
            self.assertIsNone(handle)
            self.assertIsNotNone(identity)
            self.assertEqual(identity["kind"], "posix_proc")
            self.assertEqual(identity["start_token"], "8765")
            self.assertEqual(identity["process_group"], 8700)
            self.assertEqual(identity["session_id"], 8600)

        # Invalid process returns None, None
        proc_zero = Mock(pid=0)
        self.assertEqual(capture_worker_identity(proc_zero, "test-op-0"), (None, None))


class SQLiteStoreRefactorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / ".autoresearch"
        self.prev_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_env)

        self.store = SQLiteStore(self.home)
        self.project = self.root / "project"
        self.project.mkdir()
        self.candidate = self.root / "candidate"
        (self.candidate / ".git").mkdir(parents=True)
        self.state_dir = self.root / "labs" / "lab-1"
        self.state_dir.parent.mkdir(parents=True, exist_ok=True)
        self.config_dict = {
            "id": "refactor-lab",
            "root_ref": "root-1",
            "project": str(self.project),
            "state_dir": str(self.state_dir),
            "config_path": str(self.state_dir / "lab.json"),
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "evaluator": ["python", "eval.py"],
            "limits": {"max_trials": 3},
            "pins": {"evaluator": {"sha256": "e" * 64}},
            "_passport": {
                "confirmed": True,
                "goal": "Verify Wave 1 Refactor",
                "project": str(self.project),
                "metrics": {"primary": "score", "direction": "max", "guards": []},
                "evaluator": ["python", "eval.py"],
                "limits": {"max_trials": 3},
                "pins": {"evaluator": {"sha256": "e" * 64}},
            },
        }
        self.initial_state = {
            "best_commit": "a" * 40,
            "current_epoch": 1,
            "setup_pending": False,
        }
        self.lab_id = self.store.create_lab(
            self.config_dict,
            self.initial_state,
            self.candidate,
            source_config_bytes=json.dumps(self.config_dict).encode("utf-8"),
        )

    def _restore_env(self):
        if self.prev_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.prev_home

    def test_document_key_routes_run_repeat_usage_json(self):
        doc_key = self.store._document_key(str(self.lab_id), "runs/R1/repeat_01/usage.json")
        self.assertEqual(doc_key, "run:R1:usage:repeat_01")
        doc_key_res = self.store._document_key(str(self.lab_id), "runs/R1/repeat_02/result.json")
        self.assertEqual(doc_key_res, "run:R1:result:repeat_02")

    def test_store_read_json_merges_custom_usage_fields_with_totals(self):
        run_id = self.store.create_run(self.lab_id, "trial", "R1", 1)
        usage_path = self.state_dir / "runs" / "R1" / "repeat_01" / "usage.json"
        # Save custom usage metadata into the SQLite-backed usage document
        _write_json(usage_path, {"custom_metric": 42, "provider_extra": "tracked"})

        # Record a finished attempt with cost
        self.store.append_attempt(
            self.lab_id, run_id, "att-1", "started", {"repeat_index": 1},
        )
        self.store.append_attempt(
            self.lab_id, run_id, "att-1", "finished", {"repeat_index": 1, "cost_usd": 0.03},
        )

        read_back = _read_json(usage_path)
        self.assertIsInstance(read_back, dict)
        self.assertEqual(read_back.get("custom_metric"), 42)
        self.assertEqual(read_back.get("provider_extra"), "tracked")
        self.assertEqual(read_back.get("requests"), 1)
        self.assertEqual(read_back.get("cost_usd"), 0.03)
        self.assertEqual(read_back.get("confirmed_cost_usd"), 0.03)

class HooksAndCaseCheckpointTest(unittest.TestCase):
    def test_current_case_id_falls_back_in_thread_pool_worker(self):
        with case_scope("case-thread-pool-42"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: current_case_id(), range(4)))
            self.assertEqual(results, ["case-thread-pool-42"] * 4)
        self.assertIsNone(current_case_id())

    def test_summarize_usage_includes_confirmed_cost_usd(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"event": "started", "attempt_id": "att-1"}) + "\n")
            f.write(json.dumps({"event": "finished", "attempt_id": "att-1", "cost_usd": 0.05}) + "\n")
            f.write(json.dumps({"event": "started", "attempt_id": "att-2"}) + "\n")
            path = Path(f.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        summary = summarize_usage(path)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["unknown_requests"], 1)
        self.assertIsNone(summary["cost_usd"])
        self.assertEqual(summary["confirmed_cost_usd"], 0.05)

    def test_httpx_hooks_and_streaming_response_safety(self):
        import importlib.util

        if importlib.util.find_spec("httpx") is None:
            self.skipTest("httpx is not installed")

        hook_path = (
            Path(__file__).resolve().parents[1]
            / ".agents"
            / "skills"
            / "autoresearch"
            / "scripts"
            / "hooks"
            / "sitecustomize.py"
        )
        self.assertTrue(hook_path.is_file(), f"Missing hook file: {hook_path}")
        spec = importlib.util.spec_from_file_location("autoresearch_sitecustomize", hook_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        sc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sc)

        import httpx

        sc._install_httpx()
        self.assertTrue(getattr(httpx.Client.send, "_autoresearch_hook", False))
        self.assertTrue(getattr(httpx.AsyncClient.send, "_autoresearch_hook", False))

        # Test that unconsumed streaming response does not raise ResponseNotRead
        class StreamingTransport(httpx.BaseTransport):
            def handle_request(self, request):
                return httpx.Response(200, stream=httpx.ByteStream(b"streaming-data-chunk"))

        client = httpx.Client(transport=StreamingTransport())
        resp = client.get("http://streaming.test.local")
        self.assertEqual(resp.status_code, 200)
        # Content should still be cleanly unread and readable by caller
        self.assertEqual(resp.read(), b"streaming-data-chunk")

        # Test AsyncClient send works
        class AsyncTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, json={"async_ok": True})

        async def run_async():
            async with httpx.AsyncClient(transport=AsyncTransport()) as aclient:
                return await aclient.get("http://async.test.local")

        async_resp = asyncio.run(run_async())
        self.assertEqual(async_resp.json(), {"async_ok": True})


class TrialAndStateRefactorTest(unittest.TestCase):
    def test_aggregate_metric_worst_mean_median(self):
        results = [
            {"metrics": {"acc": 0.80}},
            {"metrics": {"acc": 0.90}},
            {"metrics": {"acc": 0.70}},
        ]
        # Max direction
        self.assertAlmostEqual(_aggregate_metric(results, "acc", "max", statistic="worst"), 0.70)
        self.assertAlmostEqual(_aggregate_metric(results, "acc", "max", statistic="mean"), 0.80)
        self.assertAlmostEqual(_aggregate_metric(results, "acc", "max", statistic="median"), 0.80)

        # Min direction
        self.assertAlmostEqual(_aggregate_metric(results, "acc", "min", statistic="worst"), 0.90)
        self.assertAlmostEqual(_aggregate_metric(results, "acc", "min", statistic="mean"), 0.80)
        self.assertAlmostEqual(_aggregate_metric(results, "acc", "min", statistic="median"), 0.80)

    def test_first_repeat_rejection_respects_mean_and_stop_on_first_failure(self):
        best_runs = [{"metrics": {"score": 10.0}}]
        failed_result = {"metrics": {"score": 5.0}}

        # When statistic="mean", repeat 0 failure does not reject early
        cfg_mean = Mock(
            repeats=3, primary="score", direction="max", guards=[],
            acceptance={"statistic": "mean", "stop_on_first_failure": True},
        )
        self.assertIsNone(_first_repeat_rejection(cfg_mean, best_runs, failed_result))

        # When stop_on_first_failure=False, repeat 0 failure does not reject early
        cfg_no_stop = Mock(
            repeats=3, primary="score", direction="max", guards=[],
            acceptance={"statistic": "worst", "stop_on_first_failure": False},
        )
        self.assertIsNone(_first_repeat_rejection(cfg_no_stop, best_runs, failed_result))

        # When statistic="worst" and stop_on_first_failure=True, it rejects
        cfg_worst = Mock(
            repeats=3, primary="score", direction="max", guards=[],
            acceptance={"statistic": "worst", "stop_on_first_failure": True},
        )
        self.assertEqual(_first_repeat_rejection(cfg_worst, best_runs, failed_result), "primary did not improve")

    def test_ignored_cache_dirs_skips_pycache_in_fingerprints(self):
        self.assertIn("__pycache__", _IGNORED_CACHE_DIRS)
        with tempfile.TemporaryDirectory() as td:
            candidate_dir = Path(td) / "candidate"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "app.py").write_text("print(1)", encoding="utf-8")
            cache_dir = candidate_dir / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "app.cpython-310.pyc").write_bytes(b"pyc-data")

            mock_config = Mock(state_dir=Path(td))
            state = {"config_sha256": "c" * 64, "inputs": []}
            with patch("autoresearch_runtime.state._passport_pins_sha256", return_value=None):
                fps = _fingerprints(mock_config, state)
            self.assertIn("app.py", fps["candidate_hashes"])
            self.assertNotIn("__pycache__/app.cpython-310.pyc", fps["candidate_hashes"])

    def test_restore_candidate_preserves_gitignored_files(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "candidate"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / ".gitignore").write_text("ignored_folder/\n", encoding="utf-8")
            (repo / "tracked.txt").write_text("v1", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=T", "-c", "user.email=t@l", "commit", "-q", "-m", "init"],
                cwd=repo, check=True,
            )
            head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()

            # Create an untracked ignored file and a tracked modification
            ignored_dir = repo / "ignored_folder"
            ignored_dir.mkdir()
            ignored_file = ignored_dir / "cache.bin"
            ignored_file.write_text("preserve me", encoding="utf-8")
            (repo / "tracked.txt").write_text("v2", encoding="utf-8")

            # Restore candidate using _restore_candidate
            _restore_candidate(repo, head_commit)

            # Tracked file is restored to v1
            self.assertEqual((repo / "tracked.txt").read_text(encoding="utf-8"), "v1")
            # Ignored file is preserved (git clean -fd without -x)
            self.assertTrue(ignored_file.is_file())
            self.assertEqual(ignored_file.read_text(encoding="utf-8"), "preserve me")


class SignalsRefactorTest(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "config_sha256": "c" * 64,
            "input_hashes": {"input": "1"},
            "evaluator": ["python", "eval.py"],
            "epoch": 1,
        }

    def test_similar_hypothesis_rejected_emission(self):
        candidate = {
            "id": "cand-1",
            "status": "rejected",
            "reason": "reason-cand",
            "title": "optimize quicksort memory allocation cache buffer",
            "candidate_hashes": {"a": "1"},
            **self.contract,
        }
        history = [
            {
                "id": "hist-1",
                "status": "rejected",
                "reason": "reason-hist",
                "title": "optimize quicksort memory allocation cache pool",
                "candidate_hashes": {"a": "2"},
                **self.contract,
            }
        ]
        signals = build_research_signals(Mock(repeats=1), candidate, parent=None, history=history, limit=5)
        kinds = [s["kind"] for s in signals]
        self.assertIn("similar_hypothesis_rejected", kinds)
        sim_sig = next(s for s in signals if s["kind"] == "similar_hypothesis_rejected")
        self.assertGreaterEqual(sim_sig["similarity"], 0.65)
        self.assertIn("distinct mechanism", sim_sig["note"])

    def test_case_regression_emission(self):
        parent = {
            "id": "parent-run",
            "cases": {"case_alpha": "ok", "case_beta": "ok"},
            **self.contract,
        }
        candidate = {
            "id": "cand-regress",
            "cases": {"case_alpha": "ok", "case_beta": "fail"},
            **self.contract,
        }
        signals = build_research_signals(Mock(repeats=1), candidate, parent=parent, history=[])
        kinds = [s["kind"] for s in signals]
        self.assertIn("case_regression", kinds)
        reg_sig = next(s for s in signals if s["kind"] == "case_regression")
        self.assertIn("case_beta", reg_sig["regressed_case_ids"])

    def test_repeated_rejection_literature_hint_threshold(self):
        # 3 rejections with the same reason trigger the external literature hint
        candidate = {
            "id": "cand-rep-3",
            "status": "rejected",
            "reason": "timeout exceeded in evaluator",
            **self.contract,
        }
        history_3 = [
            {"id": "h-1", "status": "rejected", "reason": "timeout exceeded in evaluator", **self.contract},
            {"id": "h-2", "status": "rejected", "reason": "timeout exceeded in evaluator", **self.contract},
        ]
        signals = build_research_signals(Mock(repeats=1), candidate, parent=None, history=history_3)
        rep_sig = next(s for s in signals if s["kind"] == "repeated_rejection")
        self.assertEqual(rep_sig["count"], 3)
        self.assertIn("explore external literature approaches", rep_sig["note"])

        # 2 rejections with the same reason do not include the literature hint
        history_2 = [
            {"id": "h-1", "status": "rejected", "reason": "timeout exceeded in evaluator", **self.contract},
        ]
        signals_2 = build_research_signals(Mock(repeats=1), candidate, parent=None, history=history_2)
        rep_sig_2 = next(s for s in signals_2 if s["kind"] == "repeated_rejection")
        self.assertEqual(rep_sig_2["count"], 2)
        self.assertNotIn("explore external literature approaches", rep_sig_2["note"])


class DashboardRefactorTest(unittest.TestCase):
    def test_dashboard_http_server_caches_revision_payload(self):
        server = Mock(spec=DashboardHTTPServer)
        server._revision_cache_lock = threading.Lock()
        server._cached_revision_payload = None
        server._cached_revision_at = 0.0
        server.revision_payload = Mock(return_value={"revision": "rev-123", "timestamp": 1000.0})

        # Bind cached_revision_payload method
        bound_cached = DashboardHTTPServer.cached_revision_payload.__get__(server, DashboardHTTPServer)

        first = bound_cached(max_age=1.0)
        self.assertEqual(first["revision"], "rev-123")
        self.assertEqual(server.revision_payload.call_count, 1)

        # Immediate second call within max_age returns cached payload without invoking revision_payload
        second = bound_cached(max_age=1.0)
        self.assertEqual(second["revision"], "rev-123")
        self.assertEqual(server.revision_payload.call_count, 1)

    def test_dashboard_handler_protocol_version(self):
        self.assertEqual(DashboardHandler.protocol_version, "HTTP/1.1")
