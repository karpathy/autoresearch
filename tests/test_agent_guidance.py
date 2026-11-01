"""Tests for AutoResearch 2.0 Agent Cockpit and Guidance CLI."""

from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.cockpit import (
    _recent_manifests,
    build_cockpit,
    get_impact_analysis,
    get_recommended_actions,
    get_state_card,
    render_ascii_card,
)
from autoresearch_runtime.config import load_config
from autoresearch_runtime.snapshot import init_lab
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.state import load_state
from autoresearch_runtime.probe import adopt_probe
from tests.durable_workflow_helpers import (
    approved_offline_profile,
    run_baseline as run_durable_baseline,
    run_probe as run_durable_probe,
    run_trial as run_durable_trial,
)


class AgentGuidanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        example = Path(__file__).resolve().parents[1] / "examples" / "toy"
        self.toy = self.root / "toy"
        shutil.copytree(example, self.toy)
        self.config = load_config(self.toy / "lab.json")

    def _use_sqlite_toy(self):
        """Migrate the copied toy fixture to its confirmed SQLite contract."""
        home = self.root / "autoresearch-home"
        home.mkdir()
        home_patch = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)})
        home_patch.start()
        self.addCleanup(home_patch.stop)

        init_lab(self.config)
        initial_state = load_state(self.config)
        candidate = self.config.state_dir / "candidate"
        config = json.loads(self.config.path.read_text(encoding="utf-8"))
        for name in config.get("input_files", []):
            shutil.copy2(self.config.path.parent / name, self.config.state_dir / name)
        config.update({
            "source_root": str(self.config.source_root),
            "state_dir": str(self.config.state_dir),
            "config_path": str(self.config.state_dir / "lab.json"),
            "input_files": [str(self.config.state_dir / name) for name in config.get("input_files", [])],
            "confirmed": True,
            "goal": "Improve the toy evaluator score",
            "pins": {"program": {"path": str(self.toy / "program.md"), "sha256": "a" * 64}},
        })
        config["_passport"] = {
            "confirmed": True,
            "goal": config["goal"],
            "constraints": [],
            "limits": config["budget"],
            "metrics": config["metrics"],
            "evaluator": config["evaluator"],
            "pins": config["pins"],
        }
        store = SQLiteStore(home)
        lab_id = store.create_lab(config, initial_state, candidate)
        config_bytes = store.load_config_bytes(lab_id)
        (self.config.state_dir / "lab.json").write_bytes(config_bytes)
        state = store.load_state(lab_id)
        state["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        state["config_copy_sha256"] = state["config_sha256"]
        store.save_state(lab_id, state)
        self.config = load_config(str(lab_id))
        return self.config

    def _run_sqlite_step(self, *step_args):
        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        cmd = [
            sys.executable, str(scripts / "lab.py"),
            "--headless", "--agent", "--confirmed",
            "--config", str(self.config.path),
            "step", "--run", *step_args,
        ]
        with approved_offline_profile(self.config):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(scripts)
            result = subprocess.run(
                cmd, cwd=scripts, env=env, capture_output=True, text=True,
                encoding="utf-8", **support.hidden_process_kwargs(),
            )
        self.assertEqual(result.returncode, 0, f"Step failed: {result.stderr}\n{result.stdout}")
        envelope = json.loads(result.stdout)
        self.assertTrue(envelope["ok"], envelope)
        return envelope

    def test_state_card_before_baseline(self):
        init_lab(self.config)
        state = load_state(self.config)
        card = get_state_card(self.config, state)

        self.assertEqual(card["stage"], "BASELINE_NEEDED")
        self.assertEqual(card["primary_metric"], self.config.primary)
        self.assertEqual(card["direction"], self.config.direction)
        self.assertIsNone(card["baseline"]["score"])
        self.assertIsNone(card["best"]["score"])

        actions = get_recommended_actions(self.config, state, last_run=None)
        self.assertTrue(len(actions) >= 1)
        self.assertEqual(actions[0]["action"], "RUN_BASELINE")

    def test_cockpit_after_baseline(self):
        self._use_sqlite_toy()
        baseline_res = run_durable_baseline(self.config)
        state = load_state(self.config)

        cockpit = build_cockpit(self.config, state, last_run=baseline_res)
        card = cockpit["state"]
        self.assertEqual(card["stage"], "ITERATING")
        self.assertIsNotNone(card["baseline"]["score"])
        self.assertEqual(card["best"]["score"], card["baseline"]["score"])

        # Impact analysis
        impact = cockpit["impact"]
        self.assertIsNotNone(impact)
        self.assertEqual(impact["status"], "accepted")
        self.assertTrue(impact["is_new_best"])

        # ASCII card
        ascii_text = render_ascii_card(cockpit)
        self.assertIn("[ AGENT COCKPIT ]", ascii_text)
        self.assertIn("ITERATING", ascii_text)

        # Recommended actions
        actions = cockpit["next_steps"]
        self.assertTrue(len(actions) >= 1)
        self.assertEqual(actions[0]["action"], "RUN_TRIAL")

    def test_cockpit_compares_last_run_to_its_parent_after_best_updates(self):
        init_lab(self.config)
        runs = self.config.state_dir / "runs"
        parent = {
            "id": "baseline", "status": "accepted", "epoch": 1,
            "config_sha256": "cfg", "input_hashes": {"cases": "hash"}, "evaluator": ["eval"],
            "runs": [
                {"status": "ok", "metrics": {"mean_error": 4.0}},
                {"status": "ok", "metrics": {"mean_error": 8.0}},
            ],
        }
        accepted = {
            "id": "trial-0001", "status": "accepted", "parent": "baseline", "epoch": 1,
            "config_sha256": "cfg", "input_hashes": {"cases": "hash"}, "evaluator": ["eval"],
            "runs": [
                {"status": "ok", "metrics": {"mean_error": 2.0}},
                {"status": "ok", "metrics": {"mean_error": 4.0}},
            ],
        }
        rejected = {
            "id": "trial-0002", "status": "rejected", "parent": "trial-0001", "epoch": 1,
            "config_sha256": "cfg", "input_hashes": {"cases": "hash"}, "evaluator": ["eval"],
            "runs": [{"status": "ok", "metrics": {"mean_error": 1.0}}],
        }
        for manifest in (parent, accepted, rejected):
            run_dir = runs / manifest["id"]
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        state = load_state(self.config)
        state.update(baseline_id="baseline", best_id="trial-0001")
        summary = get_state_card(self.config, state)["baseline"]
        self.assertEqual(summary["score"], 8.0)  # compatibility score follows worst-of-repeats decision
        self.assertEqual(summary["mean"], 6.0)
        self.assertEqual(summary["decision_worst"], 8.0)  # decision score follows min direction
        accepted_impact = get_impact_analysis(self.config, state, accepted)
        self.assertEqual(accepted_impact["score"], 4.0)  # decision score follows worst-of-repeats
        self.assertEqual(accepted_impact["mean"], 3.0)
        self.assertEqual(accepted_impact["delta_vs_best"], -4.0)  # compare worst to pre-run parent
        self.assertEqual(accepted_impact["delta_vs_best_mean"], -3.0)
        self.assertTrue(accepted_impact["is_new_best"])

        rejected_impact = get_impact_analysis(self.config, state, rejected)
        self.assertFalse(rejected_impact["is_new_best"])

    def test_guide_run_selector_is_bounded_and_compact(self):
        self._use_sqlite_toy()
        run_durable_baseline(self.config)
        store = SQLiteStore()
        state = load_state(self.config)
        baseline_id = str(store.resolve_run_id(self.config.lab_id, "baseline"))
        for number in range(1, 13):
            run_id = f"trial-{number:04d}"
            database_run_id = store.create_run(
                self.config.lab_id, "trial", run_id, state["current_epoch"],
                {"parent_run_id": baseline_id, "candidate_commit": state["best_commit"]},
            )
            manifest = {
                "id": run_id, "status": "rejected", "parent": "baseline", "epoch": 1,
                "created_at_utc": "2026-10-07T00:00:00+00:00",
                "runs": [{"status": "ok", "metrics": {"mean_error": number}}],
                "observation": "large unneeded raw narrative " * 100,
            }
            store.save_manifest(self.config.lab_id, database_run_id, manifest)

        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(scripts)
        result = subprocess.run(
            [sys.executable, str(scripts / "lab.py"), "--config", str(self.config.path),
             "guide", "--run", "trial-0012", "--json"],
            cwd=scripts, env=env, capture_output=True, text=True, encoding="utf-8", check=True,
            **support.hidden_process_kwargs(),
        )
        card = json.loads(result.stdout)["data"]
        self.assertEqual(card["selected_run_id"], "trial-0012")
        self.assertEqual(card["current_epoch"], 1)
        self.assertLessEqual(len(card["evidence"]), 5)
        self.assertLessEqual(len(card["signals"]), 3)
        self.assertNotIn("runs", card)
        observation = card["run_detail"]["hypothesis"]["observation"]
        self.assertLessEqual(len(observation), 800)
        self.assertLess(len(observation), len("large unneeded raw narrative " * 100))
        agent_result = subprocess.run(
            [sys.executable, str(scripts / "lab.py"), "--config", str(self.config.path),
             "--headless", "--agent", "guide", "--run", "trial-0012", "--json"],
            cwd=scripts, env=env, capture_output=True, text=True, encoding="utf-8", check=True,
            **support.hidden_process_kwargs(),
        )
        agent_payload = json.loads(agent_result.stdout)
        agent_guide = agent_payload.get("cockpit", agent_payload["data"])
        encoded_guide = json.dumps(agent_guide, ensure_ascii=True, separators=(",", ":"))
        self.assertLessEqual(len(encoded_guide), 4500, {
            key: len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))
            for key, value in agent_guide.items()
        })

    def test_guide_filters_current_epoch_before_bounded_selection(self):
        init_lab(self.config)
        state = load_state(self.config)
        state["current_epoch"] = 2
        run_root = self.config.state_dir / "runs"
        for number in range(1, 13):
            run_id = f"trial-{number:04d}"
            run_dir = run_root / run_id
            run_dir.mkdir(parents=True)
            epoch = 2 if number == 1 else 1
            manifest = {
                "id": run_id, "status": "rejected", "epoch": epoch,
                "runs": [{"status": "ok", "metrics": {"mean_error": number}}],
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        card = build_cockpit(self.config, state)
        self.assertEqual(card["current_epoch"], 2)
        self.assertEqual(card["selected_run_id"], "trial-0001")
        self.assertLessEqual(len(card["evidence"]), 5)

    def test_cockpit_reports_decision_unit_and_spend_from_run_evidence(self):
        from dataclasses import replace

        config = replace(
            self.config, unit="ms",
            budget={"max_seconds": 60, "max_requests": 10, "max_cost_usd": 2.0},
        )
        init_lab(config)
        run_dir = config.state_dir / "runs" / "baseline"
        run_dir.mkdir(parents=True)
        manifest = {
            "id": "baseline", "status": "accepted", "epoch": 1,
            "runs": [
                {"status": "ok", "metrics": {"mean_error": 4.0}, "duration_seconds": 3,
                 "usage": {"requests": 1, "cost_usd": 0.2}},
                {"status": "ok", "metrics": {"mean_error": 6.0}, "duration_seconds": 3,
                 "usage": {"requests": 1, "cost_usd": 0.2}},
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        state = load_state(config)
        state.update(baseline_id="baseline", best_id="baseline", requests=0, cost_usd=0.0)

        card = get_state_card(config, state)

        self.assertEqual(card["unit"], "ms")
        self.assertEqual(card["baseline"]["score"], 6.0)
        self.assertNotIn("requests_remaining", card["budget"])
        self.assertNotIn("cost_usd_remaining", card["budget"])
        self.assertEqual(card["budget"]["time_remaining_seconds"], 54)
        self.assertEqual(card["spent"]["requests"], 2)
        self.assertAlmostEqual(card["spent"]["cost_usd"], 0.4)

    def test_guide_keeps_control_f1_tradeoff_diagnostic_and_unknown_cost_unpriced(self):
        from dataclasses import replace

        config = replace(
            self.config,
            guards=(("control_f1", "max"),),
            repeats=1,
            budget={"max_cost_usd": 10.0},
        )
        init_lab(config)
        state = load_state(config)
        state.update(baseline_id="H3", best_id="H3")
        common = {
            "epoch": 1,
            "config_sha256": state["config_sha256"],
            "input_hashes": {"cases": "same-input"},
            "evaluator": list(config.evaluator),
        }
        for run_id, manifest in (
            ("H3", {
                **common, "id": "H3", "status": "accepted",
                "runs": [{"status": "ok", "metrics": {"mean_error": 2.0, "control_f1": 0.715}}],
            }),
            ("H6", {
                **common, "id": "H6", "status": "accepted", "parent": "H3",
                "runs": [{"status": "ok", "metrics": {"mean_error": 1.0, "control_f1": 0.711}}],
            }),
        ):
            run_dir = config.state_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        repeat_dir = config.state_dir / "runs" / "H6" / "repeat_01"
        repeat_dir.mkdir()
        (repeat_dir / "attempts.jsonl").write_text(
            json.dumps({"event": "started", "attempt_id": "unknown-h6-call"}) + "\n",
            encoding="utf-8",
        )

        guide = build_cockpit(config, state, run_id="H6")

        control_signal = next(
            item for item in guide["signals"]
            if item.get("kind") == "metric_tradeoff" and item.get("metric") == "control_f1"
        )
        self.assertEqual(control_signal["effect"], "guard_regressed_while_primary_improved")
        self.assertFalse(control_signal["gate"])
        self.assertNotIn("cost_usd_remaining", guide["state"]["budget"])
        self.assertEqual(guide["provider_usage"]["requests"], 1)
        self.assertEqual(guide["provider_usage"]["unknown_requests"], 1)
        self.assertEqual(guide["provider_usage"]["confirmed_cost_usd"], 0.0)
        self.assertIsNone(guide["provider_usage"]["cost_usd"])
        self.assertEqual(guide["provider_retry"]["external_evaluator"], "unverified")
        from autoresearch_runtime.step import run_step
        step = run_step(config)
        self.assertEqual(step["provider_usage"]["unknown_requests"], 1)
        self.assertIsNone(step["provider_usage"]["cost_usd"])

    def test_guide_goal_uses_only_explicit_fresh_repeat_evidence(self):
        from dataclasses import replace
        from autoresearch_runtime.store import _write_json

        config = replace(self.config, goal_criteria={
            "thresholds": {"mean_error": {"max": 3.0}}, "confirmation_repeats": 2,
        })
        init_lab(config)
        state = load_state(config)
        state.update(baseline_id="baseline", best_id="baseline")
        _write_json(config.state_dir / "state.json", state)
        manifest = {
            "id": "baseline", "status": "accepted", "epoch": 1,
            "config_sha256": state["config_sha256"],
            "input_hashes": {item["path"]: item["sha256"] for item in state["inputs"]},
            "evaluator": list(config.evaluator),
            "runs": [
                {"status": "ok", "details": {"evidence_origin": "fresh"},
                 "metrics": {"mean_error": 2.0}},
                {"status": "ok", "details": {"evidence_origin": "fresh"},
                 "metrics": {"mean_error": 2.5}},
            ],
        }
        manifest_path = config.state_dir / "runs" / "baseline" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(build_cockpit(config, state)["goal_status"], "met")

        manifest["runs"][1]["details"]["evidence_origin"] = "offline_recompute"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(build_cockpit(config, state)["goal_status"], "unconfirmed")

    def test_step_waits_or_inspects_instead_of_starting_another_trial(self):
        from autoresearch_runtime.step import run_step
        from autoresearch_runtime.store import _write_json

        init_lab(self.config)
        state = load_state(self.config)
        state.update(baseline_id="baseline", best_id="baseline")
        _write_json(self.config.state_dir / "state.json", state)
        baseline = {
            "id": "baseline", "status": "accepted", "epoch": 1,
            "config_sha256": state["config_sha256"],
            "input_hashes": {item["path"]: item["sha256"] for item in state["inputs"]},
            "evaluator": list(self.config.evaluator),
            "runs": [{"status": "ok", "metrics": {"mean_error": 4.0, "coverage": 1.0}}],
        }
        baseline_dir = self.config.state_dir / "runs" / "baseline"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "manifest.json").write_text(json.dumps(baseline), encoding="utf-8")
        active = self.config.state_dir / "runs" / "trial-0001"
        active.mkdir()
        (active / "manifest.json").write_text(json.dumps({
            "id": "trial-0001", "status": "running", "parent": "baseline", "epoch": 1,
            "config_sha256": baseline["config_sha256"], "input_hashes": baseline["input_hashes"],
            "evaluator": baseline["evaluator"],
            "runs": [],
        }), encoding="utf-8")
        waiting = run_step(self.config, run=False)
        self.assertEqual(waiting["recommended_action"]["action"], "WAIT")
        self.assertIn("wait", waiting["argv"])
        self.assertLessEqual(len(waiting["evidence"]), 5)
        self.assertEqual(run_step(self.config, run=True)["status"], "waiting_for_trial")

        failed = active / "manifest.json"
        failed.write_text(json.dumps({
            "id": "trial-0001", "status": "timeout", "parent": "baseline", "epoch": 1,
            "runs": [],
        }), encoding="utf-8")
        inspect = run_step(self.config, run=False)
        self.assertEqual(inspect["recommended_action"]["action"], "INSPECT_RUN")
        self.assertIn("guide", inspect["argv"])
        self.assertEqual(run_step(self.config, run=True)["status"], "action_not_runnable")

    def test_setup_pending_with_declared_evaluator_runs_baseline(self):
        from dataclasses import replace

        init_lab(self.config)
        state = load_state(self.config)
        state["setup_pending"] = True
        self.assertEqual(get_recommended_actions(self.config, state)[0]["action"], "RUN_BASELINE")
        no_evaluator = replace(self.config, evaluator=(), primary="")
        self.assertEqual(get_recommended_actions(no_evaluator, state)[0]["action"], "CONFIG_SETUP")

    def test_probe_adoption_updates_cockpit(self):
        from autoresearch_runtime.errors import LabError

        self._use_sqlite_toy()
        run_durable_baseline(self.config)

        # Candidate modifies rule
        (self.config.state_dir / "candidate" / "rule.txt").write_text("5", encoding="utf-8")
        import subprocess
        subprocess.run(
            [
                "git", "-c", "user.name=AutoResearch Tests",
                "-c", "user.email=tests@example.invalid",
                "commit", "-am", "test probe rule",
            ],
            cwd=str(self.config.state_dir / "candidate"),
            capture_output=True,
            check=True,
            **support.hidden_process_kwargs(),
        )
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.config.state_dir / "candidate"),
            capture_output=True,
            text=True,
            check=True,
            **support.hidden_process_kwargs(),
        ).stdout.strip()

        probe_res = run_durable_probe(self.config, commit_hash, {}, "ideas", "test-probe", repeats=1)
        probe_id = probe_res["id"]

        # Adopt probe
        state_before = load_state(self.config)
        with self.assertRaises(LabError) as error:
            adopt_probe(self.config, probe_id)
        self.assertEqual(error.exception.code, "E_CONTRACT_PROBE_ADOPTION")

        state = load_state(self.config)
        self.assertEqual(state["best_id"], state_before["best_id"])
        self.assertEqual(state["best_commit"], state_before["best_commit"])

    def test_fuzzy_args_normalization(self):
        from autoresearch_runtime.fuzzy_args import normalize_cli_args

        args = ["trial", "start", "--runs", "3", "--config", "lab.json", "--agent", "--from-report", "rep.json"]
        normalized = normalize_cli_args(args)
        self.assertEqual(normalized[:3], ["--config", "lab.json", "--agent"])
        self.assertIn("--repeats", normalized)
        self.assertIn("--source", normalized)
        self.assertNotIn("--runs", normalized)
        self.assertNotIn("--from-report", normalized)

        # Flag with equal sign
        args_eq = ["step", "--runs=5", "--config=my_lab.json", "--input=data.json"]
        norm_eq = normalize_cli_args(args_eq)
        self.assertEqual(norm_eq[0], "--config=my_lab.json")
        self.assertIn("--repeats=5", norm_eq)
        self.assertIn("--source=data.json", norm_eq)

    def test_step_command_lifecycle(self):
        from autoresearch_runtime.step import run_step

        self._use_sqlite_toy()
        # 1. Before baseline: recommended action RUN_BASELINE
        step1 = run_step(self.config, run=False)
        self.assertEqual(step1["stage"], "BASELINE_NEEDED")
        self.assertEqual(step1["recommended_action"]["action"], "RUN_BASELINE")
        self.assertEqual(step1["status"], "recommended")

        # 2. Execute baseline with --run
        step2 = self._run_sqlite_step()
        self.assertEqual(step2["data"]["status"], "accepted")
        self.assertEqual(step2["cockpit"]["state"]["stage"], "ITERATING")

        # 3. Iterating stage without changes
        step3 = run_step(self.config, run=False)
        self.assertEqual(step3["stage"], "ITERATING")
        self.assertIn(step3["recommended_action"]["action"], ("ITERATE", "RUN_TRIAL"))

        step3_run = self._run_sqlite_step()
        self.assertEqual(step3_run["data"]["status"], "hypothesis_required")

        # 4. Iterating stage with candidate changes
        (self.config.state_dir / "candidate" / "rule.txt").write_text("7\n", encoding="utf-8")
        step4 = self._run_sqlite_step()
        self.assertEqual(step4["data"]["status"], "hypothesis_required")
        self.assertNotIn("result", step4["data"])

        hypothesis_path = self.toy / "hypothesis.json"
        hypothesis_path.write_text(json.dumps({
            "title": "Adjust toy rule", "observation": "Rule 7 improves the example cases",
            "expected_effect": "Lower mean_error without reducing coverage",
        }), encoding="utf-8")
        step4 = self._run_sqlite_step("--hypothesis", str(hypothesis_path))
        self.assertIn(step4["data"]["status"], ("accepted", "rejected"))
        trial_id = step4["cockpit"]["selected_run_id"]
        trial = SQLiteStore().load_manifest(self.config.lab_id, trial_id)
        self.assertIsNotNone(trial)
        self.assertEqual(trial["title"], "Adjust toy rule")

    def test_step_command_parses_explicit_hypothesis(self):
        init_lab(self.config)
        hypothesis_path = self.toy / "hypothesis.json"
        hypothesis_path.write_text("{}", encoding="utf-8")
        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(scripts)
        result = subprocess.run(
            [sys.executable, str(scripts / "lab.py"), "--config", str(self.config.path),
             "step", "--hypothesis", str(hypothesis_path)],
            cwd=scripts, env=env, capture_output=True, text=True, encoding="utf-8", check=True,
            **support.hidden_process_kwargs(),
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["data"]["stage"], "BASELINE_NEEDED")

    def test_prepare_auto_returns_migration_workflow_without_evaluator_or_lab_writes(self):
        from autoresearch_runtime.bootstrap import main as bootstrap_main
        import contextlib
        import io

        # Create a mini project with an evaluation script
        mini_proj = self.root / "mini_proj"
        mini_proj.mkdir(parents=True, exist_ok=True)
        (mini_proj / "model.py").write_text("SCORE = 0.9\n", encoding="utf-8")

        import subprocess
        subprocess.run(["git", "init"], cwd=str(mini_proj), capture_output=True, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(mini_proj), check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=str(mini_proj), check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "add", "."], cwd=str(mini_proj), check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(mini_proj), check=True, **support.hidden_process_kwargs())

        eval_script = self.root / "eval_mini.py"
        evaluator_marker = self.root / "evaluator-ran.marker"
        eval_script.write_text(
            "from pathlib import Path\n"
            f"Path({str(evaluator_marker)!r}).write_text('ran', encoding='utf-8')\n",
            encoding="utf-8",
        )

        home_dir = self.root / ".labs"
        home_dir.mkdir(parents=True, exist_ok=True)
        project_before = support.tree_digest(mini_proj)
        home_before = support.tree_digest(home_dir)
        old_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(home_dir)
        try:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = bootstrap_main([
                    "--auto",
                    "--project", str(mini_proj),
                    "--eval", f"python {eval_script}",
                    "--metric", "accuracy:max",
                    "--no-browser",
                    "--no-dashboard",
                ])
            self.assertNotEqual(rc, 0)
            response = json.loads(stderr.getvalue())
            self.assertEqual(response["error"]["code"], "E_AUTO_MIGRATION_REQUIRED")
            self.assertIn("prepare", response["error"]["hint"].lower())
            self.assertFalse(evaluator_marker.exists())
            self.assertEqual(support.tree_digest(mini_proj), project_before)
            self.assertEqual(support.tree_digest(home_dir), home_before)
        finally:
            if old_home is None:
                os.environ.pop("AUTORESEARCH_HOME", None)
            else:
                os.environ["AUTORESEARCH_HOME"] = old_home

    def test_resolve_hypothesis_args_and_validation(self):
        from types import SimpleNamespace
        from autoresearch_runtime.step import resolve_hypothesis_args, load_and_validate_hypothesis
        from autoresearch_runtime.errors import LabError

        # 1. Test resolve_hypothesis_args with hypothesis_json
        args1 = SimpleNamespace(
            hypothesis_json='{"title": "Loop unroll", "observation": "Loop overhead", "expected_effect": "Speedup"}',
            hypothesis=None, title=None, observation=None, expected_effect=None,
        )
        resolved = resolve_hypothesis_args(args1)
        self.assertEqual(resolved, {
            "title": "Loop unroll",
            "observation": "Loop overhead",
            "expected_effect": "Speedup",
        })
        validated = load_and_validate_hypothesis(resolved)
        self.assertEqual(validated["title"], "Loop unroll")
        self.assertEqual(validated["rationale"], "Loop overhead -> Speedup")

        # 2. Test resolve_hypothesis_args with title / observation / expected_effect
        args2 = SimpleNamespace(
            hypothesis_json=None, hypothesis=None,
            title="Inline Title", observation="Inline Obs", expected_effect="Inline Effect",
            rationale="Custom rationale", target_files=["code.py"],
        )
        resolved2 = resolve_hypothesis_args(args2)
        self.assertEqual(resolved2["title"], "Inline Title")
        self.assertEqual(resolved2["rationale"], "Custom rationale")
        self.assertEqual(resolved2["target_files"], ["code.py"])
        validated2 = load_and_validate_hypothesis(resolved2)
        self.assertEqual(validated2["rationale"], "Custom rationale")

        # 3. Test resolve_hypothesis_args with invalid JSON
        args_bad_json = SimpleNamespace(
            hypothesis_json="not-a-json", hypothesis=None, title=None, observation=None, expected_effect=None,
        )
        with self.assertRaises(LabError) as ctx:
            resolve_hypothesis_args(args_bad_json)
        self.assertEqual(ctx.exception.code, "E_BAD_HYPOTHESIS")

        # 4. Test load_and_validate_hypothesis error on missing fields
        with self.assertRaises(LabError) as ctx:
            load_and_validate_hypothesis({"title": "Incomplete"})
        self.assertEqual(ctx.exception.code, "E_CONTRACT_HYPOTHESIS")

        # 5. Test load_and_validate_hypothesis directly from json string
        valid_json_str = '{"title": "Str Title", "observation": "Str Obs", "expected_effect": "Str Effect"}'
        validated_str = load_and_validate_hypothesis(valid_json_str)
        self.assertEqual(validated_str["title"], "Str Title")

        # 6. Incomplete inline flags in --agent trial are rejected before resolve_hypothesis_args
        import contextlib
        import io
        from autoresearch_runtime.cli import main as cli_main
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc_incomplete = cli_main([
                "--agent", "--config", str(self.config.path), "trial", "--title", "Only title provided",
            ])
        out_incomplete = json.loads(buf.getvalue())
        self.assertNotEqual(rc_incomplete, 0)
        self.assertEqual(out_incomplete["error"]["code"], "E_CONTRACT_HYPOTHESIS")

    def test_evaluator_error_tail_unit_and_cockpit_surfacing(self):
        from autoresearch_runtime.evaluator import _extract_error_tail

        # 1. Unit test _extract_error_tail
        lines = [f"line {i}" for i in range(1, 20)]
        stderr_trace = "\n".join(lines) + "\nZeroDivisionError: division by zero"
        tail = _extract_error_tail(stdout="", stderr=stderr_trace, max_lines=10, max_chars=600)
        tail_lines = tail.splitlines()
        self.assertLessEqual(len(tail_lines), 10)
        self.assertEqual(tail_lines[-1], "ZeroDivisionError: division by zero")

        long_line = "x" * 700
        truncated_tail = _extract_error_tail(stdout="", stderr=long_line, max_lines=10, max_chars=600)
        self.assertLessEqual(len(truncated_tail), 600)
        self.assertTrue(truncated_tail.startswith("..."))

        stdout_tail = _extract_error_tail(stdout="Standard output crash\nError line", stderr="")
        self.assertIn("Standard output crash", stdout_tail)
        self.assertEqual(_extract_error_tail(stdout="", stderr=""), "")

        # 2. Integration test: evaluator exits with non-zero code and prints traceback to stderr
        self._use_sqlite_toy()
        baseline = run_durable_baseline(self.config)
        self.assertEqual(baseline["status"], "accepted")

        # In toy evaluator.py, factor = int((candidate / 'rule.txt').read_text()).
        # Writing a non-integer causes Python to exit with code 1 and dump a traceback to stderr.
        (self.config.state_dir / "candidate" / "rule.txt").write_text("invalid_integer\n", encoding="utf-8")

        crash_trial = run_durable_trial(self.config, {
            "title": "Crashing candidate",
            "observation": "Invalid integer in candidate rule.txt causes evaluator crash",
            "expected_effect": "Crash and capture stderr tail",
        })
        self.assertEqual(crash_trial["status"], "crash")
        self.assertIn("Evaluator exited 1", crash_trial["reason"])
        self.assertIn("ValueError: invalid literal for int() with base 10", crash_trial["reason"])

        state = load_state(self.config)
        cockpit = build_cockpit(self.config, state, last_run=crash_trial, run_id=crash_trial["id"])
        self.assertIn("error_tail", cockpit["impact"])
        self.assertIn("ValueError: invalid literal for int() with base 10", cockpit["impact"]["error_tail"])
        self.assertIn("Traceback (most recent call last):", cockpit["impact"]["error_tail"])
        self.assertIn("error_tail", cockpit["run_detail"])
        self.assertEqual(cockpit["run_detail"]["error_tail"], cockpit["impact"]["error_tail"])
        self.assertIn("Evaluator exited 1", cockpit["run_detail"]["reason"])
        self.assertIn("ValueError: invalid literal for int() with base 10", cockpit["run_detail"]["reason"])
        self.assertIn("Evaluator exited 1", cockpit["impact"]["verdict"])

    def test_end_to_end_step_run_with_inline_hypothesis_json(self):
        self._use_sqlite_toy()
        baseline = run_durable_baseline(self.config)
        self.assertEqual(baseline["status"], "accepted")

        # In toy evaluator: cases are (1->2, 2->4, 3->6). Factor 2 yields error 0.0 (baseline factor 1 had error 2.0).
        (self.config.state_dir / "candidate" / "rule.txt").write_text("2\n", encoding="utf-8")

        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        hyp_json = json.dumps({
            "title": "Lower toy error inline",
            "observation": "Factor 2 perfectly matches expected = input * 2",
            "expected_effect": "Lower mean_error to 0.0",
        })
        cmd = [
            sys.executable, str(scripts / "lab.py"),
            "--headless",
            "--agent",
            "--confirmed",
            "step",
            "--config", str(self.config.path),
            "--run",
            "--hypothesis-json", hyp_json,
        ]
        with approved_offline_profile(self.config):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(scripts)
            res = subprocess.run(
                cmd,
                cwd=str(scripts),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                **support.hidden_process_kwargs(),
            )
        self.assertEqual(res.returncode, 0, f"Command failed: {res.stderr}\n{res.stdout}")

        envelope = json.loads(res.stdout)
        self.assertTrue(envelope["ok"])
        data = envelope["data"]
        self.assertEqual(data.get("status"), "accepted")
        cockpit = envelope.get("cockpit")
        self.assertIsNotNone(cockpit)
        self.assertEqual(cockpit["impact"]["status"], "accepted")
        self.assertTrue(cockpit["impact"]["is_new_best"])

    def test_sqlite_step_run_after_crash_trial_resumes_cleanly(self):
        """Check 1: SQLite mode trial crash followed by candidate edit and step --run resumes with accepted status."""
        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            home = tmp_path / "home"
            home.mkdir()
            with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
                state_dir = home / "state"
                state_dir.mkdir()
                candidate = support.git_project(state_dir / "candidate", {"solver.py": "def solve():\n    return 10\n"})
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True, **support.hidden_process_kwargs()).strip()
                eval_path = tmp_path / "evaluator.py"
                eval_path.write_text(
                    "import json, sys\n"
                    "from pathlib import Path\n"
                    "candidate = Path(sys.argv[1])\n"
                    "run_dir = Path(sys.argv[2])\n"
                    "run_dir.mkdir(parents=True, exist_ok=True)\n"
                    "sys.path.insert(0, str(candidate))\n"
                    "import solver\n"
                    "score = float(solver.solve())\n"
                    "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': score}, 'status': 'ok'}))\n",
                    encoding="utf-8",
                )
                cfg = {
                    "confirmed": True, "source_root": str(candidate), "state_dir": str(state_dir),
                    "config_path": str(state_dir / "lab.json"), "goal": "test", "repeats": 1, "timeout_seconds": 15,
                    "metrics": {"primary": "score", "direction": "max", "guards": []},
                    "evaluator": ["{python}", str(eval_path), "{candidate}", "{run_dir}"],
                    "budget": {"max_trials": 10, "max_cost_usd": 0},
                    "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
                    "_passport": {"confirmed": True, "goal": "test", "constraints": [], "limits": {"max_trials": 10}, "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}}},
                }
                store = SQLiteStore(home)
                lab_id = store.create_lab(cfg, {"best_commit": commit, "current_epoch": 1, "inputs": [], "source": []}, candidate)
                config_bytes = store.load_config_bytes(lab_id)
                (state_dir / "lab.json").write_bytes(config_bytes)
                state = store.load_state(lab_id)
                state.update({
                    "best_commit": commit, "best_id": "baseline", "baseline_id": "baseline",
                    "baseline_verified": True, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "inputs": [], "source": [],
                })
                store.save_state(lab_id, state)
                base_run_id = store.create_run(lab_id, "baseline", "baseline", 1, {})
                store.save_manifest(lab_id, base_run_id, {
                    "id": "baseline", "status": "accepted", "runs": [{"status": "ok", "metrics": {"score": 0.5}}],
                })

                (candidate / "solver.py").write_text("def solve():\n    return 1 / 0\n", encoding="utf-8")
                c_run_id = store.create_run(lab_id, "trial-0001", "trial-0001", 1, {})
                store.save_manifest(lab_id, c_run_id, {
                    "id": "trial-0001", "status": "crash", "reason": "Evaluator exited 1: ZeroDivisionError",
                    "error_tail": "ZeroDivisionError: division by zero",
                    "runs": [{"status": "crash", "error_tail": "ZeroDivisionError: division by zero", "metrics": {}}],
                })
                state["best_id"] = "baseline"
                store.save_state(lab_id, state)

                # Fix candidate
                (candidate / "solver.py").write_text("def solve():\n    return 42\n", encoding="utf-8")

                cmd = [
                    sys.executable, str(scripts / "lab.py"),
                    "--lab", str(lab_id),
                    "--headless", "--agent", "--confirmed", "step",
                    "--run",
                    "--title", "Fix after crash",
                    "--observation", "Fixed zero div",
                    "--expected-effect", "Score improves",
                ]
                config = load_config(str(lab_id))
                with approved_offline_profile(config):
                    res = subprocess.run(cmd, env={**os.environ, "AUTORESEARCH_HOME": str(home)}, capture_output=True, text=True, encoding="utf-8", **support.hidden_process_kwargs())
                self.assertEqual(res.returncode, 0, f"SQLite step failed: {res.stderr}\n{res.stdout}")
                out = json.loads(res.stdout)
                self.assertTrue(out["ok"])
                self.assertEqual(out["data"]["status"], "accepted")
                self.assertNotEqual(out["data"].get("action"), "action_not_runnable")

    def test_sqlite_multi_repeat_crash_preserves_error_tail_in_impact_and_run_detail(self):
        """Check 3: Multi-repeat SQLite lab where Repeat 1 succeeds and Repeat 2 crashes preserves error_tail."""
        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            home = tmp_path / "home"
            home.mkdir()
            with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
                state_dir = home / "state"
                state_dir.mkdir()
                candidate = support.git_project(state_dir / "candidate", {"counter.txt": "0"})
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True, **support.hidden_process_kwargs()).strip()

                evaluator_code = (
                    "import json, sys, os\n"
                    "from pathlib import Path\n"
                    "candidate = Path(sys.argv[1])\n"
                    "run_dir = Path(sys.argv[2])\n"
                    "run_dir.mkdir(parents=True, exist_ok=True)\n"
                    "if '02' in run_dir.name or 'repeat_02' in str(run_dir):\n"
                    "    sys.stderr.write('Traceback (most recent call last):\\n  File eval.py, line 12\\nZeroDivisionError: repeat 2 division by zero\\n')\n"
                    "    sys.exit(1)\n"
                    "else:\n"
                    "    (run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': 1.0}, 'status': 'ok'}))\n"
                    "    sys.exit(0)\n"
                )
                eval_path = tmp_path / "evaluator.py"
                eval_path.write_text(evaluator_code, encoding="utf-8")

                cfg = {
                    "confirmed": True, "source_root": str(candidate), "state_dir": str(state_dir),
                    "config_path": str(state_dir / "lab.json"), "goal": "test multi repeat crash",
                    "repeats": 2, "repeat_policy": "stop_on_failure", "timeout_seconds": 15,
                    "metrics": {"primary": "score", "direction": "max", "guards": []},
                    "evaluator": ["{python}", str(eval_path), "{candidate}", "{run_dir}"],
                    "budget": {"max_trials": 10, "max_cost_usd": 0},
                    "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
                    "_passport": {"confirmed": True, "goal": "test", "constraints": [], "limits": {"max_trials": 10}, "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}}},
                }
                store = SQLiteStore(home)
                lab_id = store.create_lab(cfg, {"best_commit": commit, "current_epoch": 1, "inputs": [], "source": []}, candidate)
                config_bytes = store.load_config_bytes(lab_id)
                (state_dir / "lab.json").write_bytes(config_bytes)
                state = store.load_state(lab_id)
                state.update({
                    "best_commit": commit, "best_id": "baseline", "baseline_id": "baseline",
                    "baseline_verified": True, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "inputs": [], "source": [],
                })
                store.save_state(lab_id, state)
                base_id = store.create_run(lab_id, "baseline", "baseline", 1, {})
                store.save_manifest(lab_id, base_id, {
                    "id": "baseline", "status": "accepted",
                    "runs": [
                        {"status": "ok", "metrics": {"score": 0.5}},
                        {"status": "ok", "metrics": {"score": 0.5}},
                    ],
                })

                (candidate / "counter.txt").write_text("1", encoding="utf-8")
                cmd = [
                    sys.executable, str(scripts / "lab.py"),
                    "--lab", str(lab_id),
                    "--headless", "--agent", "--confirmed", "trial",
                    "--title", "Repeat 2 crash",
                    "--observation", "O",
                    "--expected-effect", "E",
                ]
                config = load_config(str(lab_id))
                with approved_offline_profile(config):
                    res = subprocess.run(cmd, env={**os.environ, "AUTORESEARCH_HOME": str(home)}, capture_output=True, text=True, encoding="utf-8", **support.hidden_process_kwargs())
                self.assertEqual(res.returncode, 0, f"Trial failed: {res.stderr}\n{res.stdout}")
                out = json.loads(res.stdout)
                data = out["data"]
                self.assertEqual(data["status"], "crash")

                loaded_m = store.load_manifest(lab_id, data["id"])
                self.assertIsNotNone(loaded_m)
                self.assertEqual(loaded_m["status"], "crash")
                self.assertEqual(len(loaded_m["runs"]), 2)
                self.assertEqual(loaded_m["runs"][0]["status"], "ok")
                self.assertIn("ZeroDivisionError", loaded_m["runs"][1].get("error_tail", ""))
                self.assertIn("ZeroDivisionError", loaded_m.get("error_tail", ""))

                cockpit = out.get("cockpit", {})
                self.assertIn("error_tail", cockpit["impact"])
                self.assertIn("ZeroDivisionError", cockpit["impact"]["error_tail"])
                self.assertIn("Traceback (most recent call last):", cockpit["impact"]["error_tail"])

                config = load_config(str(state_dir / "lab.json"))
                cockpit_detail = build_cockpit(config, store.load_state(lab_id), last_run=loaded_m, run_id=data["id"])
                self.assertIn("error_tail", cockpit_detail["impact"])
                self.assertIn("error_tail", cockpit_detail["run_detail"])
                self.assertEqual(cockpit_detail["run_detail"]["error_tail"], cockpit_detail["impact"]["error_tail"])

    def test_sqlite_step_hypothesis_fields_retention_and_guide_surfacing(self):
        """Check 4: --rationale, --target-files, --falsification in SQLite are retained and surfaced in guide."""
        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            home = tmp_path / "home"
            home.mkdir()
            with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
                state_dir = home / "state"
                state_dir.mkdir()
                candidate = support.git_project(state_dir / "candidate", {"solver.py": "x = 1\n"})
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True, **support.hidden_process_kwargs()).strip()

                evaluator_code = (
                    "import json, sys\n"
                    "from pathlib import Path\n"
                    "run_dir = Path(sys.argv[2])\n"
                    "run_dir.mkdir(parents=True, exist_ok=True)\n"
                    "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': 1.0}, 'status': 'ok'}))\n"
                )
                eval_path = tmp_path / "evaluator.py"
                eval_path.write_text(evaluator_code, encoding="utf-8")

                cfg = {
                    "confirmed": True, "source_root": str(candidate), "state_dir": str(state_dir),
                    "config_path": str(state_dir / "lab.json"), "goal": "test hypothesis retention",
                    "repeats": 1, "timeout_seconds": 15,
                    "metrics": {"primary": "score", "direction": "max", "guards": []},
                    "evaluator": ["{python}", str(eval_path), "{candidate}", "{run_dir}"],
                    "budget": {"max_trials": 10, "max_cost_usd": 0},
                    "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
                    "_passport": {"confirmed": True, "goal": "test", "constraints": [], "limits": {"max_trials": 10}, "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}}},
                }
                store = SQLiteStore(home)
                lab_id = store.create_lab(cfg, {"best_commit": commit, "current_epoch": 1, "inputs": [], "source": []}, candidate)
                config_bytes = store.load_config_bytes(lab_id)
                (state_dir / "lab.json").write_bytes(config_bytes)
                state = store.load_state(lab_id)
                state.update({
                    "best_commit": commit, "best_id": "baseline", "baseline_id": "baseline",
                    "baseline_verified": True, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "inputs": [], "source": [],
                })
                store.save_state(lab_id, state)
                base_id = store.create_run(lab_id, "baseline", "baseline", 1, {})
                store.save_manifest(lab_id, base_id, {
                    "id": "baseline", "status": "accepted", "runs": [{"status": "ok", "metrics": {"score": 0.5}}],
                })

                (candidate / "solver.py").write_text("x = 2\n", encoding="utf-8")
                cmd = [
                    sys.executable, str(scripts / "lab.py"),
                    "--lab", str(lab_id),
                    "--headless", "--agent", "--confirmed", "step",
                    "--run",
                    "--title", "T1",
                    "--observation", "O1",
                    "--expected-effect", "E1",
                    "--rationale", "R1",
                    "--target-files", "solver.py",
                ]
                config = load_config(str(lab_id))
                with approved_offline_profile(config):
                    res = subprocess.run(cmd, env={**os.environ, "AUTORESEARCH_HOME": str(home)}, capture_output=True, text=True, encoding="utf-8", **support.hidden_process_kwargs())
                self.assertEqual(res.returncode, 0, f"Step run failed: {res.stderr}\n{res.stdout}")
                out = json.loads(res.stdout)
                data = out["data"]
                run_id = data.get("id") or data.get("run_id") or (out.get("cockpit") or {}).get("selected_run_id")
                self.assertTrue(run_id)

                m = store.load_manifest(lab_id, run_id)
                rationale = m.get("rationale") or (m.get("hypothesis") or {}).get("rationale")
                target_files = m.get("target_files") or (m.get("hypothesis") or {}).get("target_files")
                self.assertEqual(rationale, "R1")
                self.assertEqual(target_files, ["solver.py"])

                guide_cmd = [
                    sys.executable, str(scripts / "lab.py"),
                    "--lab", str(lab_id),
                    "--headless", "guide",
                    "--json",
                    "--run", run_id,
                ]
                g_res = subprocess.run(guide_cmd, env={**os.environ, "AUTORESEARCH_HOME": str(home)}, capture_output=True, text=True, encoding="utf-8", **support.hidden_process_kwargs())
                self.assertEqual(g_res.returncode, 0, f"Guide failed: {g_res.stderr}\n{g_res.stdout}")
                g_out = json.loads(g_res.stdout)
                g_hyp = g_out["data"]["run_detail"]["hypothesis"]
                self.assertEqual(g_hyp["title"], "T1")
                self.assertEqual(g_hyp["observation"], "O1")
                self.assertEqual(g_hyp["expected_effect"], "E1")
                self.assertEqual(g_hyp["rationale"], "R1")
                self.assertEqual(g_hyp["target_files"], ["solver.py"])

    def test_sqlite_recent_manifests_single_query_and_compact_budget_under_heavy_load(self):
        """Check 5: _recent_manifests single SQL query and <= 4500 byte cockpit budget under heavy load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            home = tmp_path / "home"
            home.mkdir()
            with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
                state_dir = home / "state"
                state_dir.mkdir()
                candidate = support.git_project(state_dir / "candidate", {"v.txt": "1"})
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True, **support.hidden_process_kwargs()).strip()

                cfg = {
                    "confirmed": True, "source_root": str(candidate), "state_dir": str(state_dir),
                    "config_path": str(state_dir / "lab.json"), "goal": "test", "repeats": 1, "timeout_seconds": 15,
                    "metrics": {"primary": "score", "direction": "max", "guards": []},
                    "evaluator": ["python", "eval.py"], "budget": {"max_trials": 20, "max_cost_usd": 0},
                    "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
                    "_passport": {"confirmed": True, "goal": "test", "constraints": [], "limits": {"max_trials": 20}, "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}}},
                }
                store = SQLiteStore(home)
                lab_id = store.create_lab(cfg, {"best_commit": commit, "current_epoch": 1, "inputs": [], "source": []}, candidate)
                config_bytes = store.load_config_bytes(lab_id)
                (state_dir / "lab.json").write_bytes(config_bytes)
                config = load_config(str(lab_id))

                queries = []
                orig_connect = store._connect
                def traced_connect(write=False):
                    conn = orig_connect(write=write)
                    conn.set_trace_callback(queries.append)
                    return conn

                with patch.object(SQLiteStore, "_connect", side_effect=traced_connect):
                    zero_runs = _recent_manifests(config, 1)
                self.assertEqual(len(queries), 1)
                self.assertEqual(len(zero_runs), 0)

                for i in range(1, 9):
                    r_id = f"trial-{i:04d}"
                    db_r_id = store.create_run(lab_id, r_id, r_id, 1, {})
                    m = {
                        "id": r_id, "epoch": 1, "status": "rejected" if i % 2 == 1 else "crash",
                        "candidate_hash": f"hash-{i}",
                        "hypothesis": {"title": f"Idea {i} to optimize solver", "observation": f"Obs {i}", "expected_effect": f"Eff {i}"},
                        "reason": f"Evaluator failed reason {i}",
                        "runs": [{"status": "crash" if i % 2 == 0 else "rejected", "metrics": {"score": 0.5}}],
                    }
                    store.save_manifest(lab_id, db_r_id, m)

                err_tail_600 = "Traceback (most recent call last):\n" + ("File solver.py line 1 in <module>\n  div = 1 / 0\n" * 10) + "ZeroDivisionError: division by zero"
                if len(err_tail_600) < 600:
                    err_tail_600 += " " * (600 - len(err_tail_600))
                err_tail_600 = err_tail_600[:600]

                trial_9_id = store.create_run(lab_id, "trial-0009", "trial-0009", 1, {})
                m9 = {
                    "id": "trial-0009", "epoch": 1, "status": "crash",
                    "candidate_hash": "hash-9",
                    "hypothesis": {"title": "Idea 9", "observation": "Obs 9", "expected_effect": "Eff 9"},
                    "reason": "Evaluator exited 1: ZeroDivisionError",
                    "error_tail": err_tail_600,
                    "runs": [{"status": "crash", "error_tail": err_tail_600, "metrics": {"score": 0.0}}],
                }
                store.save_manifest(lab_id, trial_9_id, m9)

                state = store.load_state(lab_id)
                fake_signals = [
                    {
                        "kind": "case_regression", "count": 4,
                        "case_samples": [
                            {"id": "c1", "status": "failed", "error": "AssertionError: 1!=2", "metrics": {"loss": 0.5}},
                            {"id": "c2", "status": "failed", "error": "TimeoutError after 30s", "metrics": {"loss": 1.0}},
                            {"id": "c3", "status": "failed", "error": "ValueError: token", "metrics": {"loss": 0.8}},
                        ],
                        "evidence": ["trial-0009"],
                        "note": "Regression in test cases observed across multiple runs",
                    },
                    {"kind": "high_spread", "evidence": ["trial-0009"], "note": "High repeat variance detected across repeats"},
                    {"kind": "guard_regression", "metric": "coverage", "evidence": ["trial-0009"], "note": "Coverage dropped significantly"},
                ]
                with patch("autoresearch_runtime.cockpit.build_research_signals", return_value=fake_signals):
                    cockpit = build_cockpit(config, state, last_run=m9, run_id="trial-0009")

                self.assertEqual(len(cockpit.get("dead_ends", [])), 8)
                self.assertEqual(len(cockpit.get("signals", [])), 3)
                self.assertIn("error_tail", cockpit["impact"])
                self.assertIn("error_tail", cockpit["run_detail"])
                self.assertIn("Traceback (most recent call last):", cockpit["impact"]["error_tail"])
                self.assertIn("ZeroDivisionError", cockpit["impact"]["error_tail"])
                self.assertEqual(cockpit["impact"]["error_tail"], cockpit["run_detail"]["error_tail"])

                raw_bytes = json.dumps(cockpit, ensure_ascii=False).encode("utf-8")
                self.assertLessEqual(len(raw_bytes), 4500)


if __name__ == "__main__":
    unittest.main()
