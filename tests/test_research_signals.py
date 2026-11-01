"""Small, explicit research signals for the agent guide."""

from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import shutil
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace

from autoresearch_runtime.cockpit import build_cockpit
from autoresearch_runtime.config import load_config
from autoresearch_runtime.snapshot import init_lab
from autoresearch_runtime.state import load_state
from autoresearch_runtime.baseline import run_baseline
from autoresearch_runtime.trial import run_trial
from autoresearch_runtime.signals import build_research_signals, evaluate_goal_status


def _manifest(run_id: str, *, parent: str | None = None, score: float = 0.8,
              coverage: float = 0.95, origin: str = "fresh", epoch: int = 3) -> dict:
    return {
        "id": run_id,
        "parent": parent,
        "status": "accepted",
        "epoch": epoch,
        "config_sha256": "config-hash",
        "input_hashes": {"cases.json": "input-hash"},
        "evaluator": ["python", "evaluator.py"],
        "runs": [
            {"status": "ok", "details": {"evidence_origin": origin},
             "metrics": {"accuracy": score, "coverage": coverage}},
            {"status": "ok", "details": {"evidence_origin": origin},
             "metrics": {"accuracy": score, "coverage": coverage}},
        ],
    }


class ResearchSignalsTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            primary="accuracy",
            direction="max",
            guards=(("coverage", "max"),),
            goal_criteria=None,
            repeats=2,
        )

    def test_tradeoff_signal_is_evidence_linked_and_never_a_gate(self):
        parent = _manifest("trial-0001", score=0.80, coverage=0.99)
        candidate = _manifest("trial-0002", parent="trial-0001", score=0.84, coverage=0.96)

        signals = build_research_signals(self.config, candidate, parent)

        self.assertLessEqual(len(signals), 3)
        coverage = next(signal for signal in signals if signal["metric"] == "coverage")
        self.assertEqual(coverage["kind"], "metric_tradeoff")
        self.assertAlmostEqual(coverage["delta"], -0.03)
        self.assertFalse(coverage["gate"])
        self.assertEqual(coverage["evidence"], ["trial-0002", "trial-0001"])

    def test_primary_gain_and_guard_regression_survive_repeat_spread_signals(self):
        self.config.repeats = 3
        parent = _manifest("trial-0001", score=0.80, coverage=0.95)
        candidate = _manifest("trial-0002", parent="trial-0001", score=0.90, coverage=0.80)
        parent["runs"].append({"metrics": {"accuracy": 0.81, "coverage": 0.94,
                                             "latency": 11.0, "cost": 2.0}})
        candidate["runs"].append({"metrics": {"accuracy": 0.92, "coverage": 0.78,
                                                "latency": 13.0, "cost": 5.0}})
        parent["runs"][0]["metrics"].update(latency=10.0, cost=3.0)
        parent["runs"][1]["metrics"].update(latency=13.0, cost=4.0)
        candidate["runs"][0]["metrics"].update(latency=12.0, cost=4.0)
        candidate["runs"][1]["metrics"].update(latency=17.0, cost=7.0)

        signals = build_research_signals(self.config, candidate, parent)

        tradeoff = next(signal for signal in signals if signal["kind"] == "metric_tradeoff")
        self.assertEqual(tradeoff["effect"], "guard_regressed_while_primary_improved")
        self.assertEqual(tradeoff["primary_metric"], "accuracy")
        self.assertEqual(tradeoff["primary_direction"], "max")
        self.assertAlmostEqual(tradeoff["primary_from_mean"], 0.8033333333)
        self.assertAlmostEqual(tradeoff["primary_to_mean"], 0.9066666667)
        self.assertAlmostEqual(tradeoff["primary_from_decision_worst"], 0.80)
        self.assertAlmostEqual(tradeoff["primary_to_decision_worst"], 0.90)
        self.assertEqual(tradeoff["guard_metric"], "coverage")
        self.assertAlmostEqual(tradeoff["guard_from_mean"], 0.9466666667)
        self.assertAlmostEqual(tradeoff["guard_to_mean"], 0.7933333333)
        self.assertAlmostEqual(tradeoff["guard_from_decision_worst"], 0.94)
        self.assertAlmostEqual(tradeoff["guard_to_decision_worst"], 0.78)
        self.assertEqual(tradeoff["evidence"], ["trial-0002", "trial-0001"])
        self.assertLessEqual(len(signals), 3)

    def test_incomparable_epoch_or_fingerprint_produces_no_tradeoff(self):
        parent = _manifest("trial-0001")
        candidate = _manifest("trial-0002", parent="trial-0001", score=0.9, epoch=4)
        self.assertEqual(build_research_signals(self.config, candidate, parent), [])

        candidate = _manifest("trial-0002", parent="trial-0001", score=0.9)
        candidate["input_hashes"] = {"cases.json": "changed-input"}
        self.assertEqual(build_research_signals(self.config, candidate, parent), [])

    def test_repeat_spread_and_missing_repeats_are_visible(self):
        self.config.repeats = 3
        parent = _manifest("trial-0001")
        candidate = _manifest("trial-0002", parent="trial-0001", score=0.8)
        candidate["runs"][1]["metrics"]["accuracy"] = 0.92

        signals = build_research_signals(self.config, candidate, parent)

        kinds = {signal["kind"] for signal in signals}
        self.assertIn("repeat_spread", kinds)
        missing = next(signal for signal in signals if signal["kind"] == "missing_repeats")
        self.assertEqual(missing["missing_count"], 1)
        self.assertEqual(missing["expected_repeats"], 3)
        self.assertFalse(missing["gate"])

    def test_structured_diagnostics_are_compacted_and_evidence_linked(self):
        parent = _manifest("trial-0001")
        candidate = _manifest("trial-0002", parent="trial-0001")
        candidate["runs"][0]["details"]["diagnostics"] = [
            {"case_id": "screen-4", "slice": "OCR", "failure_kind": "truncated_label"},
            {"case_id": "screen-7", "slice": "OCR", "failure_kind": "truncated_label"},
        ]

        signals = build_research_signals(self.config, candidate, parent)

        diagnostic = next(signal for signal in signals if signal["kind"] == "diagnostic_cluster")
        self.assertEqual(diagnostic["failure_kind"], "truncated_label")
        self.assertEqual(diagnostic["slice"], "OCR")
        self.assertEqual(diagnostic["count"], 2)
        self.assertEqual(diagnostic["case_ids"], ["screen-4", "screen-7"])
        self.assertFalse(diagnostic["gate"])

    def test_recent_rejections_expose_same_candidate_and_shared_reason(self):
        parent = _manifest("baseline")
        first = _manifest("trial-0001")
        second = _manifest("trial-0002")
        first.update(status="rejected", reason="primary did not improve", candidate_hashes={"rule.py": "same"})
        second.update(status="rejected", reason="primary did not improve", candidate_hashes={"rule.py": "same"})
        current = _manifest("trial-0003", parent="baseline")

        signals = build_research_signals(self.config, current, parent, history=[second, first])

        kinds = {signal["kind"] for signal in signals}
        self.assertIn("repeated_rejection", kinds)
        self.assertIn("same_candidate_rejected", kinds)
        repeated = next(signal for signal in signals if signal["kind"] == "repeated_rejection")
        self.assertEqual(repeated["count"], 2)
        self.assertEqual(repeated["evidence"], ["trial-0002", "trial-0001"])

    def test_recurring_rejections_require_exact_current_research_contract(self):
        parent = _manifest("baseline")
        first = _manifest("trial-0001")
        second = _manifest("trial-0002")
        current = _manifest("trial-0003", parent="baseline")
        for item in (first, second):
            item.update(status="rejected", reason="primary did not improve",
                        candidate_hashes={"rule.py": "same"})

        for key, different_value in (
            ("config_sha256", "other-config"),
            ("input_hashes", {"cases.json": "other-input"}),
            ("evaluator", ["other-evaluator"]),
        ):
            mismatched = dict(first)
            mismatched[key] = different_value
            signals = build_research_signals(self.config, current, parent,
                                             history=[mismatched, second])
            kinds = {signal["kind"] for signal in signals}
            self.assertNotIn("repeated_rejection", kinds, key)
            self.assertNotIn("same_candidate_rejected", kinds, key)

    def test_goal_uses_every_repeat_and_requires_explicit_fresh_evidence(self):
        self.config.goal_criteria = {
            "thresholds": {"accuracy": {"min": 0.82}},
            "confirmation_repeats": 2,
        }
        candidate = _manifest("trial-0002", score=0.84)
        self.assertEqual(evaluate_goal_status(self.config, candidate)["status"], "met")

        candidate["runs"][1]["metrics"]["accuracy"] = 0.81
        self.assertEqual(evaluate_goal_status(self.config, candidate)["status"], "not_met")

        candidate["runs"][1]["metrics"]["accuracy"] = 0.84
        candidate["runs"][1]["details"]["evidence_origin"] = "offline_recompute"
        self.assertEqual(evaluate_goal_status(self.config, candidate)["status"], "unconfirmed")

    def test_goal_is_unconfirmed_for_incomplete_or_legacy_evidence(self):
        self.config.goal_criteria = {
            "thresholds": {"accuracy": {"min": 0.82}},
            "confirmation_repeats": 2,
        }
        candidate = _manifest("trial-0002", score=0.84)
        candidate["runs"].pop()
        self.assertEqual(evaluate_goal_status(self.config, candidate)["status"], "unconfirmed")

        candidate = _manifest("trial-0002", score=0.84)
        candidate["runs"][0]["details"].pop("evidence_origin")
        self.assertEqual(evaluate_goal_status(self.config, candidate)["status"], "unknown")

    def test_goal_does_not_infer_thresholds_from_absent_criteria(self):
        candidate = _manifest("trial-0002", score=1.0)
        self.assertEqual(evaluate_goal_status(self.config, candidate)["status"], "unknown")

    def test_case_regression_includes_case_samples_with_diagnostics(self):
        parent = _manifest("trial-0001", score=0.85, coverage=0.95)
        parent["runs"][0]["cases"] = [
            {"id": "case-1", "status": "ok"},
            {"id": "case-2", "status": "ok"},
            {"id": "case-3", "status": "ok"},
            {"id": "case-4", "status": "ok"},
        ]
        parent["runs"][1]["cases"] = [
            {"id": "case-1", "status": "ok"},
            {"id": "case-2", "status": "ok"},
            {"id": "case-3", "status": "ok"},
            {"id": "case-4", "status": "ok"},
        ]
        candidate = _manifest("trial-0002", parent="trial-0001", score=0.80, coverage=0.90)
        candidate["runs"][0]["cases"] = [
            {"id": "case-1", "status": "failed", "error": "AssertionError: 1 != 2", "metrics": {"loss": 0.45, "latency": 15.0}},
            {"id": "case-2", "status": "failed", "error": "TimeoutError after 30s", "metrics": {"loss": 1.20}},
            {"id": "case-3", "status": "failed", "error": "ValueError: invalid token", "metrics": {"loss": 0.88}},
            {"id": "case-4", "status": "failed", "error": "ZeroDivisionError", "metrics": {"loss": 9.99}},
        ]
        candidate["runs"][1]["cases"] = list(candidate["runs"][0]["cases"])

        signals = build_research_signals(self.config, candidate, parent)
        regression = next(s for s in signals if s["kind"] == "case_regression")
        self.assertEqual(regression["count"], 4)
        self.assertIn("case_samples", regression)
        samples = regression["case_samples"]
        self.assertEqual(len(samples), 3)  # capped at 3
        self.assertEqual(samples[0]["id"], "case-1")
        self.assertEqual(samples[0]["status"], "failed")
        self.assertEqual(samples[0]["error"], "AssertionError: 1 != 2")
        self.assertEqual(samples[0]["metrics"], {"loss": 0.45, "latency": 15.0})
        self.assertEqual(samples[1]["id"], "case-2")
        self.assertEqual(samples[1]["status"], "failed")
        self.assertEqual(samples[1]["error"], "TimeoutError after 30s")
        self.assertEqual(samples[1]["metrics"], {"loss": 1.20})

    def test_long_horizon_duplicate_and_similarity_detection_past_15_runs(self):
        runs = []
        for i in range(1, 36):
            r_id = f"trial-{i:04d}"
            r = _manifest(r_id, epoch=1)
            r["status"] = "accepted" if i not in (3, 35) else "rejected"
            r["reason"] = "guard regression: coverage dropped" if i in (3, 35) else "accepted"
            r["hypothesis"] = {
                "title": "Vectorize matrix dot product calculation",
                "observation": "matrix dot product is slow",
                "expected_effect": "lower latency",
            } if i in (3, 35) else {
                "title": f"Arbitrary variation number {i}",
                "observation": f"Observation {i}",
                "expected_effect": f"Effect {i}",
            }
            r["patch_sha256"] = "patch-duplicate-hash" if i in (3, 35) else f"patch-hash-{i}"
            runs.append(r)

        candidate_35 = runs[-1]
        history = list(reversed(runs[:-1]))  # 34 items; trial-0003 is at index 31
        self.assertEqual(len(history), 34)
        self.assertEqual(history[31]["id"], "trial-0003")

        parent = runs[33]
        signals = build_research_signals(self.config, candidate_35, parent, history=history)

        same_patch = next((s for s in signals if s["kind"] == "same_candidate_rejected"), None)
        self.assertIsNotNone(same_patch, "same_candidate_rejected should be detected past 15 runs")
        self.assertIn("trial-0003", same_patch["evidence"])
        self.assertIn("trial-0035", same_patch["evidence"])

        candidate_35_diff_patch = dict(candidate_35)
        candidate_35_diff_patch["patch_sha256"] = "different-patch-hash"
        signals_sim = build_research_signals(self.config, candidate_35_diff_patch, parent, history=history)
        similar_hyp = next((s for s in signals_sim if s["kind"] == "similar_hypothesis_rejected"), None)
        self.assertIsNotNone(similar_hyp, "similar_hypothesis_rejected should be detected past 15 runs")
        self.assertIn("trial-0003", similar_hyp["evidence"])

    def test_cockpit_dead_ends_digest_capped_at_8(self):
        from autoresearch_runtime.cockpit import _build_dead_ends_digest
        manifests = []
        for i in range(1, 15):
            manifests.append({
                "id": f"trial-{i:04d}",
                "epoch": 1,
                "status": "rejected" if i % 2 == 1 else "crash",
                "title": f"Hypothesis Idea {i}",
                "reason": f"Evaluator failed reason {i}",
                "candidate_hash": f"cand-hash-{i}",
            })
        digest = _build_dead_ends_digest(manifests, current_epoch=1, limit=8)
        self.assertEqual(len(digest), 8)
        self.assertTrue(all(item["id"].startswith("trial-") for item in digest))
        self.assertTrue(all("status" in item and "title" in item and "reason" in item for item in digest))

        duplicates = [
            {"id": "trial-0001", "epoch": 1, "status": "rejected", "title": "Same Idea", "reason": "R1"},
            {"id": "trial-0002", "epoch": 1, "status": "rejected", "title": "Same Idea", "reason": "R2"},
            {"id": "trial-0003", "epoch": 1, "status": "rejected", "title": "Different Idea", "reason": "R3"},
        ]
        dedup_digest = _build_dead_ends_digest(duplicates, current_epoch=1, limit=8)
        self.assertEqual(len(dedup_digest), 2)
        self.assertEqual([d["title"] for d in dedup_digest], ["Same Idea", "Different Idea"])


class RuntimeGoalSignalIntegrationTest(unittest.TestCase):
    def test_real_baseline_and_trial_report_repeat_spread_and_confirm_goal(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / "toy"
            shutil.copytree(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy-toy", project)
            config_path = project / "lab.json"
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            raw_config["goal_criteria"] = {
                "thresholds": {"mean_error": {"max": 0.15}},
                "confirmation_repeats": 2,
            }
            config_path.write_text(json.dumps(raw_config), encoding="utf-8")

            evaluator = project / "source" / "evaluator.py"
            evaluator.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "candidate, run_dir, config_dir = map(Path, sys.argv[1:4])\n"
                "factor = int((candidate / 'rule.txt').read_text(encoding='utf-8'))\n"
                "repeat = int(run_dir.name[-2:])\n"
                "result = {'metrics': {'mean_error': factor + (repeat - 1) * 0.1, 'coverage': 3}, "
                "'usage': {'requests': 0, 'cost_usd': 0}, 'details': {}}\n"
                "run_dir.mkdir(parents=True, exist_ok=True)\n"
                "(run_dir / 'result.json').write_text(json.dumps(result), encoding='utf-8')\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            init_lab(config)
            baseline = run_baseline(config)
            self.assertEqual(baseline["status"], "accepted")

            candidate = config.state_dir / "candidate" / "rule.txt"
            candidate.write_text("0\n", encoding="utf-8")
            trial = run_trial(config, {
                "title": "Lower toy error",
                "observation": "A zero multiplier reduces error on the toy cases.",
                "expected_effect": "Lower mean_error while preserving coverage.",
            })
            self.assertEqual(trial["status"], "accepted")
            self.assertEqual(len(trial["runs"]), 2)
            self.assertTrue(all(row["details"]["evidence_origin"] == "fresh" for row in trial["runs"]))

            cockpit = build_cockpit(config, load_state(config), last_run=trial)
            self.assertEqual(cockpit["goal"]["status"], "met")
            spread = next(signal for signal in cockpit["signals"] if signal["kind"] == "repeat_spread")
            self.assertEqual(spread["metric"], "mean_error")
            self.assertAlmostEqual(spread["spread"], 0.1)
            self.assertEqual(spread["evidence"], [trial["id"]])

    def test_evaluator_subprocess_cases_jsonl_ingestion_and_case_samples(self):
        """Verify evaluator writing only to cases.jsonl (AUTORESEARCH_CASE_LOG) is ingested and surfaces in cockpit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            proj_dir = tmp_path / "proj"
            proj_dir.mkdir()
            (proj_dir / "solver.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
            evaluator_code = (
                "import json, sys, os\n"
                "from pathlib import Path\n"
                "candidate = Path(sys.argv[1])\n"
                "run_dir = Path(sys.argv[2])\n"
                "run_dir.mkdir(parents=True, exist_ok=True)\n"
                "sys.path.insert(0, str(candidate))\n"
                "import solver\n"
                "val = solver.solve()\n"
                "case_log = os.environ.get('AUTORESEARCH_CASE_LOG')\n"
                "if case_log:\n"
                "    p = Path(case_log)\n"
                "    p.parent.mkdir(parents=True, exist_ok=True)\n"
                "    with p.open('w', encoding='utf-8') as f:\n"
                "        f.write(json.dumps({'id': 'case-1', 'status': 'ok' if val == 1 else 'failed', 'error': '' if val == 1 else 'AssertionError', 'metrics': {'loss': 0.1 if val == 1 else 1.0}}) + '\\n')\n"
                "        f.write(json.dumps({'id': 'case-2', 'status': 'ok' if val == 1 else 'failed', 'error': '' if val == 1 else 'ValueError', 'metrics': {'loss': 0.2 if val == 1 else 2.0}}) + '\\n')\n"
                "        f.write(json.dumps({'id': 'case-3', 'status': 'ok', 'error': '', 'metrics': {'loss': 0.1}}) + '\\n')\n"
                "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': 1.0 if val == 1 else 0.5}, 'status': 'ok'}))\n"
            )
            eval_path = tmp_path / "evaluator.py"
            eval_path.write_text(evaluator_code, encoding="utf-8")

            cfg_path = tmp_path / "lab.json"
            config_data = {
                "protocol_version": 1,
                "source_root": str(proj_dir),
                "state_dir": str(tmp_path / "state"),
                "evaluator": ["{python}", str(eval_path), "{candidate}", "{run_dir}"],
                "metrics": {"primary": "score", "direction": "max", "guards": []},
                "repeats": 1,
                "timeout_seconds": 15,
                "budget": {"max_trials": 10},
            }
            cfg_path.write_text(json.dumps(config_data), encoding="utf-8")
            config = load_config(str(cfg_path))
            init_lab(config)
            b = run_baseline(config)
            self.assertEqual(b["status"], "accepted")
            self.assertIn("cases", b["runs"][0]["details"])

            (config.state_dir / "candidate" / "solver.py").write_text("def solve():\n    return 2\n", encoding="utf-8")
            t = run_trial(config, {"title": "Failing cases", "observation": "O", "expected_effect": "E"})
            self.assertEqual(t["status"], "rejected")
            self.assertIn("cases", t["runs"][0]["details"])

            state = load_state(config)
            cockpit = build_cockpit(config, state, last_run=t, run_id=t["id"])
            signals = cockpit.get("signals", [])
            reg_signal = next((s for s in signals if s["kind"] == "case_regression"), None)
            self.assertIsNotNone(reg_signal)
            self.assertIn("case_samples", reg_signal)
            self.assertEqual(len(reg_signal["case_samples"]), 2)
            self.assertEqual(reg_signal["case_samples"][0]["id"], "case-1")
            self.assertEqual(reg_signal["case_samples"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
