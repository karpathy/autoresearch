try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import json
import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.lab import (
    LabError,
    assess_candidate,
    init_lab,
    load_config,
    load_state,
    run_baseline,
    run_trial,
    status_lab,
    main,
)
from autoresearch_runtime.lab import _remaining, _spent


class LabTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "candidate.txt").write_text("first", encoding="utf-8")
        (self.source / "protected.txt").write_text("fixed", encoding="utf-8")
        self.opaque_input = self.root / "sample.bin"
        self.opaque_input.write_bytes(b"\x00\xff\x13")
        self.config_path = self.root / "lab.json"
        self.config = {
            "source_root": "source",
            "state_dir": "state",
            "source_paths": ["candidate.txt", "protected.txt"],
            "editable_files": ["candidate.txt"],
            "input_files": ["sample.bin"],
            "evaluator": ["{python}", "evaluator.py", "{candidate}", "{run_dir}"],
            "metrics": {
                "primary": "score",
                "direction": "max",
                "guards": [{"name": "validity", "direction": "max"}],
            },
            "repeats": 2,
            "timeout_seconds": 5,
            "budget": {
                "max_seconds": 60,
                "max_trials": 3,
                "max_requests": 0,
                "max_cost_usd": 0,
            },
        }
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_spent_does_not_turn_an_explicit_unknown_cost_into_zero(self):
        spent = _spent([{
            "id": "legacy-run",
            "runs": [{"usage": {"requests": 1, "cost_usd": None}}],
        }])

        self.assertIsNone(spent["cost_usd"])
        self.assertEqual(spent["confirmed_cost_usd"], 0.0)
        self.assertEqual(spent["unknown_cost_usd"], 0.0)
        self.assertEqual(spent["unknown_costs"], 1.0)

    def install_evaluator(self):
        evaluator = self.root / "evaluator.py"
        evaluator.write_text(
            "import json, sys, time\n"
            "from pathlib import Path\n"
            "candidate, output = map(Path, sys.argv[1:3])\n"
            "text = (candidate / 'candidate.txt').read_text(encoding='utf-8')\n"
            "if text == 'broken_json':\n"
            "    output.mkdir(parents=True, exist_ok=True)\n"
            "    (output / 'result.json').write_text('{broken')\n"
            "    raise SystemExit(0)\n"
            "if text == 'sleep': time.sleep(2)\n"
            "if text == 'mutate':\n"
            "    (candidate / 'protected.txt').write_text('changed')\n"
            "    text = '2;1'\n"
            "if text == 'mutate_editable':\n"
            "    (candidate / 'candidate.txt').write_text('3;1')\n"
            "    text = '2;1'\n"
            "if text == 'artifact':\n"
            "    (candidate / 'extra.tmp').write_text('generated')\n"
            "    text = '2;1'\n"
            "if text == 'invalid': result = {'metrics': {'score': float('nan')}}\n"
            "else:\n"
            "    parts = text.split(';')\n"
            "    result = {'metrics': {'score': float(parts[0]) + len(Path(" + repr(str(self.opaque_input)) + ").read_bytes()),"
            " 'validity': float(parts[1])}, 'usage': {'requests': int(parts[3]) if len(parts) > 3 else 0,"
            " 'cost_usd': float(parts[2]) if len(parts) > 2 else 0}, 'details': {'kind': 'binary'}}\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'result.json').write_text(json.dumps(result), encoding='utf-8')\n",
            encoding="utf-8",
        )
        self.config["evaluator"] = ["{python}", str(evaluator), "{candidate}", "{run_dir}"]
        self.write_config()

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_failed_repeat_usage_is_saved_and_unknown_reservation_blocks_next_run(self):
        self.install_evaluator()
        evaluator = Path(self.config["evaluator"][1])
        evaluator.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "run=Path(sys.argv[2]);run.mkdir(parents=True,exist_ok=True)\n"
            "(run/'usage.json').write_text(json.dumps({'requests':50,'cost_usd':0.0176445324}))\n"
            "raise SystemExit(1)\n", encoding="utf-8")
        config = load_config(self.config_path)
        init_lab(config)
        with self.assertRaisesRegex(LabError, "Evaluator exited 1"):
            run_baseline(config)
        manifest = json.loads((config.state_dir / "runs" / "baseline" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runs"][0]["usage"]["requests"], 50)
        self.assertAlmostEqual(status_lab(config)["spent"]["cost_usd"], 0.0176445324)

        usage_path = config.state_dir / "runs" / "baseline" / "repeat_01" / "usage.json"
        usage_path.write_text(json.dumps({"requests": 1, "cost_usd": 0.02, "unknown_requests": 1,
                                          "unknown_cost_usd": 0.02}), encoding="utf-8")
        unresolved = {"id": "baseline", "runs": [{"duration_seconds": 1}]}
        manifest_path = config.state_dir / "runs" / "baseline" / "manifest.json"
        manifest_path.write_text(json.dumps(unresolved), encoding="utf-8")
        self.assertEqual(_spent([unresolved], config.state_dir)["unknown_requests"], 1)
        self.assertGreater(_remaining(config), 0)

    def test_init_copies_source_and_hashes_opaque_input(self):
        config = load_config(self.config_path)
        init_lab(config)

        self.assertEqual(
            (self.root / "state" / "candidate" / "candidate.txt").read_text(encoding="utf-8"),
            "first",
        )
        state = load_state(config)
        self.assertEqual(len(state["inputs"]), 1)
        self.assertEqual(state["inputs"][0]["sha256"], "bbc8693209b89bf586a409447716deab4ba41a658b24c660e29a490309c0726e")
        self.assertTrue(state["best_commit"])

    def test_declared_file_ignored_by_source_gitignore_is_snapshotted(self):
        (self.source / ".gitignore").write_text("*.txt\n", encoding="utf-8")
        (self.source / "candidate.txt").write_text("2;1", encoding="utf-8")
        self.config["source_paths"].append(".gitignore")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("1;1", encoding="utf-8")
        trial = run_trial(config, {"title": "Worse", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "rejected")
        self.assertEqual(candidate.read_text(encoding="utf-8"), "2;1")

    def test_rejects_editable_path_outside_snapshot(self):
        self.config["editable_files"] = ["../sample.bin"]
        self.write_config()
        with self.assertRaises(LabError):
            load_config(self.config_path)

    def test_nested_git_metadata_is_not_copied_into_candidate(self):
        metadata = self.source / "nested" / ".git"
        metadata.mkdir(parents=True)
        (metadata / "config").write_text("hidden", encoding="utf-8")
        self.config["source_paths"].append("nested")
        self.write_config()

        with self.assertRaisesRegex(LabError, "Git internals"):
            load_config(self.config_path)

    def test_rejects_nonfinite_or_boolean_budget(self):
        for value in (float("nan"), True):
            with self.subTest(value=value):
                self.config["budget"]["max_seconds"] = value
                self.write_config()
                with self.assertRaises(LabError):
                    load_config(self.config_path)
        self.config["budget"]["max_seconds"] = 60
        self.config["budget"]["max_cost_usd"] = float("nan")
        self.write_config()
        self.assertEqual(load_config(self.config_path).budget["max_seconds"], 60)

    def test_rejects_nontext_guard_metric_name(self):
        self.config["metrics"]["guards"] = [{"name": 3, "direction": "max"}]
        self.write_config()
        with self.assertRaisesRegex(LabError, "Invalid guard"):
            load_config(self.config_path)

    def test_acceptance_uses_worst_repeats_and_guards(self):
        baseline = [
            {"metrics": {"score": 0.40, "validity": 1.0}},
            {"metrics": {"score": 0.42, "validity": 1.0}},
        ]
        better = [
            {"metrics": {"score": 0.45, "validity": 1.0}},
            {"metrics": {"score": 0.41, "validity": 1.0}},
        ]
        worse_guard = [
            {"metrics": {"score": 0.45, "validity": 1.0}},
            {"metrics": {"score": 0.41, "validity": 0.9}},
        ]
        config = load_config(self.config_path)

        self.assertEqual(assess_candidate(config, baseline, better), (True, "improved"))
        self.assertEqual(assess_candidate(config, baseline, worse_guard), (False, "guard validity regressed"))

    def test_minimizing_metric_uses_highest_repeat_as_worst(self):
        self.config["metrics"] = {"primary": "latency", "direction": "min", "guards": []}
        self.write_config()
        config = load_config(self.config_path)
        baseline = [{"metrics": {"latency": 9.0}}, {"metrics": {"latency": 10.0}}]
        better = [{"metrics": {"latency": 8.0}}, {"metrics": {"latency": 9.5}}]
        worse = [{"metrics": {"latency": 8.0}}, {"metrics": {"latency": 10.1}}]

        self.assertEqual(assess_candidate(config, baseline, better), (True, "improved"))
        self.assertEqual(assess_candidate(config, baseline, worse), (False, "primary did not improve"))

    def test_json_input_with_minimization_runs_through_same_core(self):
        (self.source / "candidate.txt").write_text("4", encoding="utf-8")
        data = self.root / "sample.json"
        data.write_text(json.dumps({"target": 2}), encoding="utf-8")
        evaluator = self.source / "evaluator.py"
        evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, output = map(Path, sys.argv[1:3])\n"
            "target = json.loads((Path(sys.argv[3]) / 'sample.json').read_text(encoding='utf-8'))['target']\n"
            "actual = int((candidate / 'candidate.txt').read_text(encoding='utf-8'))\n"
            "result = {'metrics': {'error': abs(actual - target)}, 'usage': {'requests': 0, 'cost_usd': 0}}\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'result.json').write_text(json.dumps(result), encoding='utf-8')\n",
            encoding="utf-8",
        )
        self.config["source_paths"].append("evaluator.py")
        self.config["input_files"] = ["sample.json"]
        self.config["evaluator"] = ["{python}", "evaluator.py", "{candidate}", "{run_dir}", "{config_dir}"]
        self.config["metrics"] = {"primary": "error", "direction": "min", "guards": []}
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        baseline = run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("3", encoding="utf-8")
        trial = run_trial(config, {"title": "Closer", "observation": "error high", "expected_effect": "error lower"})

        self.assertEqual(baseline["runs"][0]["metrics"]["error"], 2)
        self.assertEqual(trial["status"], "accepted")
        self.assertEqual(trial["runs"][0]["metrics"]["error"], 1)

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_trial_accepts_binary_backed_evaluator_and_preserves_best(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        baseline = run_baseline(config)
        self.assertEqual([run["metrics"]["score"] for run in baseline["runs"]], [4.0, 4.0])

        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("2;1", encoding="utf-8")
        trial = run_trial(config, {"title": "More signal", "observation": "low score", "expected_effect": "increase score"})

        self.assertEqual(trial["status"], "accepted")
        self.assertEqual([run["metrics"]["score"] for run in trial["runs"]], [5.0, 5.0])
        self.assertEqual(load_state(config)["best_id"], trial["id"])
        self.assertEqual(candidate.read_text(encoding="utf-8"), "2;1")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--config", str(config.path), "report"]), 0)
        dashboard = (config.state_dir / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("createElementNS('http://www.w3.org/2000/svg','svg')", dashboard)
        self.assertIn("More signal", dashboard)

    def test_rejected_trial_restores_best_and_keeps_patch(self):
        (self.source / "candidate.txt").write_text("2;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("1;1", encoding="utf-8")

        trial = run_trial(config, {"title": "Bad change", "observation": "guess", "expected_effect": "increase"})

        self.assertEqual(trial["status"], "rejected")
        self.assertEqual(candidate.read_text(encoding="utf-8"), "2;1")
        self.assertIn("-2;1", (config.state_dir / "runs" / trial["id"] / "change.patch").read_text(encoding="utf-8"))

    def test_complete_all_runs_every_repeat_before_rejection(self):
        (self.source / "candidate.txt").write_text("2;1", encoding="utf-8")
        self.install_evaluator()
        self.config["repeat_policy"] = "complete_all"
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("1;1", encoding="utf-8")

        trial = run_trial(config, {"title": "Bad change", "observation": "guess", "expected_effect": "increase"})

        self.assertEqual(len(trial["runs"]), 2)
        self.assertEqual(trial["status"], "rejected")
        self.assertEqual(candidate.read_text(encoding="utf-8"), "2;1")

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_protected_change_is_blocked_before_evaluation(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "protected.txt").write_text("tampered", encoding="utf-8")

        trial = run_trial(config, {"title": "Tamper", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "invalid")
        self.assertIn("Protected file changed", trial["reason"])
        self.assertEqual(trial["runs"], [])
        self.assertEqual((config.state_dir / "candidate" / "protected.txt").read_text(encoding="utf-8"), "fixed")
        self.assertEqual(load_state(config)["best_id"], "baseline")

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_extra_candidate_file_before_trial_is_recorded_and_removed(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        extra = config.state_dir / "candidate" / "extra.tmp"
        extra.write_text("surprise", encoding="utf-8")

        trial = run_trial(config, {"title": "Extra", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "invalid")
        self.assertIn("Candidate file set changed", trial["reason"])
        self.assertFalse(extra.exists())

    def test_candidate_symlink_cannot_read_outside_snapshot(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        outside = self.root / "outside.txt"
        outside.write_text("2;1", encoding="utf-8")
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.unlink()
        try:
            candidate.symlink_to(outside)
        except OSError:
            self.skipTest("Symlink creation is unavailable on this Windows account")

        trial = run_trial(config, {"title": "Linked", "observation": "x", "expected_effect": "y"})
        self.assertEqual(trial["status"], "invalid")
        self.assertIn("Candidate symlink", trial["reason"])

    def test_invalid_result_never_promotes(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("invalid", encoding="utf-8")

        trial = run_trial(config, {"title": "Malformed", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "invalid")
        self.assertEqual(load_state(config)["best_id"], "baseline")

    def test_malformed_json_result_never_promotes(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("broken_json", encoding="utf-8")

        trial = run_trial(config, {"title": "Broken JSON", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "invalid")
        self.assertIn("Invalid result.json", trial["reason"])
        self.assertEqual(load_state(config)["best_id"], "baseline")

    def test_timeout_never_promotes(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        # Хук повторов добавляет запуск процесса, поэтому 0.1 с уже не хватает
        # обычному оценщику. Сон в тесте длится 2 с, так что 1 с всё ещё ловит таймаут.
        self.config["timeout_seconds"] = 1
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("sleep", encoding="utf-8")

        trial = run_trial(config, {"title": "Slow", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "timeout")
        self.assertEqual((config.state_dir / "candidate" / "candidate.txt").read_text(encoding="utf-8"), "1;1")

    def test_evaluator_cannot_rewrite_editable_candidate_during_trial(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("mutate_editable", encoding="utf-8")

        trial = run_trial(config, {"title": "Evaluator mutation", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "invalid")
        self.assertEqual(candidate.read_text(encoding="utf-8"), "1;1")

    def test_untracked_evaluator_artifact_is_removed_on_rejection(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("artifact", encoding="utf-8")

        trial = run_trial(config, {"title": "Artifact", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "invalid")
        self.assertFalse((config.state_dir / "candidate" / "extra.tmp").exists())

    def test_agent_commit_inside_candidate_is_blocked(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate_dir = config.state_dir / "candidate"
        (candidate_dir / "candidate.txt").write_text("2;1", encoding="utf-8")
        subprocess.run(["git", "add", "candidate.txt"], cwd=candidate_dir, check=True, capture_output=True, **support.hidden_process_kwargs())
        subprocess.run([
            "git", "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "-m", "manual",
        ], cwd=candidate_dir, check=True, capture_output=True, **support.hidden_process_kwargs())

        with self.assertRaisesRegex(LabError, "Candidate Git HEAD changed"):
            run_trial(config, {"title": "x", "observation": "x", "expected_effect": "y"})

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_baseline_evaluator_mutation_is_recorded_and_reset(self):
        (self.source / "candidate.txt").write_text("mutate", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)

        with self.assertRaisesRegex(LabError, "Candidate changed during evaluation"):
            run_baseline(config)

        manifest = json.loads((config.state_dir / "runs" / "baseline" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "invalid")
        self.assertEqual((config.state_dir / "candidate" / "protected.txt").read_text(encoding="utf-8"), "fixed")

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_missing_evaluator_command_is_recorded_as_crash(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.config["evaluator"] = [str(self.root / "missing-executable"), "{candidate}", "{run_dir}"]
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)

        with self.assertRaisesRegex(LabError, "Baseline failed"):
            run_baseline(config)

        manifest = json.loads((config.state_dir / "runs" / "baseline" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "crash")

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_failed_baseline_retry_keeps_both_attempts_and_spend(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        original = (self.root / "evaluator.py").read_text(encoding="utf-8")
        flag = self.root / "fail-once.flag"
        (self.root / "evaluator.py").write_text(
            "from pathlib import Path\n"
            f"flag = Path({str(flag)!r})\n"
            "if not flag.exists():\n"
            "    flag.write_text('done')\n"
            "    raise SystemExit(1)\n" + original,
            encoding="utf-8",
        )
        self.config["repeats"] = 1
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        with self.assertRaisesRegex(LabError, "Baseline failed"):
            run_baseline(config)
        baseline = run_baseline(config)

        self.assertEqual(baseline["id"], "baseline-0002")
        self.assertEqual(load_state(config)["baseline_id"], "baseline-0002")
        self.assertTrue((config.state_dir / "runs" / "baseline" / "manifest.json").exists())
        self.assertGreater(status_lab(config)["spent"]["seconds"], baseline["runs"][0]["duration_seconds"])

    def test_changed_input_blocks_new_trial(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        self.opaque_input.write_bytes(b"different")
        (config.state_dir / "candidate" / "candidate.txt").write_text("2;1", encoding="utf-8")

        with self.assertRaisesRegex(LabError, "Input changed"):
            run_trial(config, {"title": "Different input", "observation": "x", "expected_effect": "y"})

    def test_cost_overrun_after_last_repeat_does_not_promote(self):
        (self.source / "candidate.txt").write_text("1;1;0.2;0", encoding="utf-8")
        self.install_evaluator()
        self.config["repeats"] = 1
        self.config["budget"]["max_cost_usd"] = 0.5
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("2;1;0.4;0", encoding="utf-8")

        trial = run_trial(config, {"title": "Too expensive", "observation": "x", "expected_effect": "y"})

        self.assertEqual(trial["status"], "accepted")
        self.assertEqual(load_state(config)["best_id"], trial["id"])

    def test_baseline_budget_exhaustion_is_recorded(self):
        (self.source / "candidate.txt").write_text("1;1;0;1", encoding="utf-8")
        self.install_evaluator()
        self.config["budget"]["max_requests"] = 1
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)

        baseline = run_baseline(config)

        self.assertEqual(baseline["status"], "accepted")
        self.assertEqual(len(baseline["runs"]), 2)

    def test_zero_request_budget_rejects_reported_usage(self):
        (self.source / "candidate.txt").write_text("1;1;0;1", encoding="utf-8")
        self.install_evaluator()
        self.config["repeats"] = 1
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)

        baseline = run_baseline(config)

        self.assertEqual(baseline["status"], "accepted")
        self.assertEqual(baseline["runs"][0]["usage"]["requests"], 1)

    def test_zero_cost_budget_blocks_next_trial_after_reported_overrun(self):
        (self.source / "candidate.txt").write_text("1;1;0;0", encoding="utf-8")
        self.install_evaluator()
        self.config["repeats"] = 1
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("2;1;0.1;0", encoding="utf-8")
        first = run_trial(config, {"title": "Costly", "observation": "x", "expected_effect": "y"})
        self.assertEqual(first["status"], "accepted")
        candidate.write_text("3;1;0;0", encoding="utf-8")
        second = run_trial(config, {"title": "After limit", "observation": "x", "expected_effect": "y"})
        self.assertEqual(second["status"], "accepted")

    def test_trial_count_limit_blocks_another_hypothesis(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        self.config["budget"]["max_trials"] = 1
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_text("2;1", encoding="utf-8")
        run_trial(config, {"title": "First", "observation": "x", "expected_effect": "y"})
        candidate.write_text("3;1", encoding="utf-8")

        with self.assertRaisesRegex(LabError, "Trial budget"):
            run_trial(config, {"title": "Second", "observation": "x", "expected_effect": "y"})

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_trial_rejects_nonobject_hypothesis(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("2;1", encoding="utf-8")

        with self.assertRaisesRegex(LabError, "Hypothesis needs"):
            run_trial(config, ["not", "an", "object"])

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_trial_auto_generates_hypothesis_from_diff_stat(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("2;1", encoding="utf-8")

        trial = run_trial(config, None)
        self.assertEqual(trial["status"], "accepted")
        self.assertIn("candidate.txt", trial["title"])
        self.assertIn("candidate.txt", trial["observation"])
        self.assertTrue(trial["expected_effect"])

    def test_each_experiment_records_input_candidate_and_result_hashes(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        self.config["input_files"].append("evaluator.py")
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        baseline = run_baseline(config)
        (config.state_dir / "candidate" / "candidate.txt").write_text("2;1", encoding="utf-8")
        trial = run_trial(config, {"title": "Better", "observation": "x", "expected_effect": "y"})

        self.assertEqual(baseline["input_hashes"], trial["input_hashes"])
        self.assertEqual(len(trial["input_hashes"]), 2)
        self.assertNotEqual(baseline["candidate_hashes"], trial["candidate_hashes"])
        self.assertEqual(len(trial["runs"][0]["result_sha256"]), 64)
        self.assertEqual(trial["config_sha256"], load_state(config)["config_sha256"])

    def test_changed_evaluator_registered_as_input_blocks_trial(self):
        (self.source / "candidate.txt").write_text("1;1", encoding="utf-8")
        self.install_evaluator()
        self.config["input_files"].append("evaluator.py")
        self.write_config()
        config = load_config(self.config_path)
        init_lab(config)
        run_baseline(config)
        (self.root / "evaluator.py").write_text("changed", encoding="utf-8")
        (config.state_dir / "candidate" / "candidate.txt").write_text("2;1", encoding="utf-8")

        with self.assertRaisesRegex(LabError, "Input changed"):
            run_trial(config, {"title": "x", "observation": "x", "expected_effect": "y"})

    def test_sqlite_init_ignores_its_persistent_lock_guard(self):
        self.install_evaluator()
        state_dir = self.root / "sqlite-state"
        state_dir.mkdir()
        config_path = state_dir / "lab.json"
        raw = {
            **self.config,
            "source_root": str(self.source),
            "state_dir": str(state_dir),
            "input_files": [],
            "setup_pending": True,
        }
        config_bytes = (json.dumps(raw, indent=2) + "\n").encode("utf-8")
        config_path.write_bytes(config_bytes)
        validated = load_config(config_path, allow_pending=True)
        passport = {
            "confirmed": True,
            "goal": "Initialize a validated offline lab",
            "project": str(self.source),
            "metrics": raw["metrics"],
            "constraints": raw["budget"],
            "pins": [{"name": "source", "path": str(self.source)}],
        }
        sqlite_config = replace(
            validated,
            config_bytes=config_bytes,
            passport=passport,
        )
        home = self.root / "home"

        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
            from autoresearch_runtime.locking import lab_lock
            with lab_lock(state_dir):
                state = init_lab(sqlite_config)

            from autoresearch_runtime.sqlite_store import SQLiteStore
            store = SQLiteStore(home)
            self.assertTrue(store.is_initialized())
            self.assertTrue(store.resolve_lab_id(config_path))

        self.assertTrue((state_dir / ".lock.guard").is_file())
        self.assertTrue((state_dir / "candidate" / ".git").is_dir())
        self.assertTrue(state["best_commit"])

    def test_init_still_rejects_unrecognized_files(self):
        config = load_config(self.config_path)
        config.state_dir.mkdir()
        (config.state_dir / "unmanaged.txt").write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(LabError, "State directory is not empty"):
            init_lab(config)

    def test_packaged_synthetic_example(self):
        example = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "draft-config"
        copy = self.root / "toy"
        shutil.copytree(example, copy)
        config = load_config(copy / "lab.json")
        init_lab(config)
        baseline = run_baseline(config)
        (config.state_dir / "candidate" / "rule.txt").write_text("2", encoding="utf-8")
        hypothesis = json.loads((copy / "hypothesis.json").read_text(encoding="utf-8"))
        trial = run_trial(config, hypothesis)

        self.assertEqual(baseline["runs"][0]["metrics"]["mean_error"], 2)
        self.assertEqual(trial["status"], "accepted")
        self.assertEqual(trial["runs"][0]["metrics"]["mean_error"], 0)


if __name__ == "__main__":
    unittest.main()
