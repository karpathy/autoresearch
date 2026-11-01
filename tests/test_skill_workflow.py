try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import hashlib
import json
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.lab import load_config, load_state, setup_project, status_lab
from tests.durable_workflow_helpers import (
    approved_offline_profile, run_baseline, run_trial,
)


class SkillWorkflowTest(unittest.TestCase):
    def _restore_home(self):
        if self._previous_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self._previous_home

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self._previous_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.root / ".labs")
        self.addCleanup(self._restore_home)
        self.program = self.root / "program.md"
        self.program.write_text("Project: project\nGoal: improve score\nMetric: score, max\n", encoding="utf-8")
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "value.txt").write_text("1", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "value = int((candidate / 'value.txt').read_text())\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': value}, "
            "'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )

    def spec(self, *, generated=False):
        path = self.root / "derived.json"
        known_case_path = self.root / "known-case.json"
        evaluator_sha = hashlib.sha256(self.evaluator.read_bytes()).hexdigest() if self.evaluator.is_file() else ""
        known_case_path.write_text(json.dumps({
            "schema": 1, "id": "fixture-score-v1", "reproducible": True,
            "evaluator_sha256": evaluator_sha,
            "input_hashes": {}, "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1}, "tolerance": 1e-9,
        }), encoding="utf-8")
        path.write_text(json.dumps({
            "project": "project",
            "evaluator": ["{python}", "evaluator.py", "{candidate}", "{run_dir}"],
            "evaluator_generated": generated,
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {"path": str(known_case_path),
                               "sha256": hashlib.sha256(known_case_path.read_bytes()).hexdigest()},
            },
        }), encoding="utf-8")
        return path

    def store(self):
        from autoresearch_runtime.sqlite_store import SQLiteStore

        return SQLiteStore(Path(os.environ["AUTORESEARCH_HOME"]))

    def assert_no_sqlite_json_shadows(self, config):
        """New labs keep these records in SQLite, not virtual JSON files."""
        home = Path(os.environ["AUTORESEARCH_HOME"])
        self.assertFalse((config.state_dir / "lab.json").exists())
        self.assertFalse((config.state_dir / "state.json").exists())
        self.assertFalse((home / "catalog.json").exists())
        self.assertFalse(any((config.state_dir / "runs").rglob("manifest.json")))
        self.assertFalse(any((config.state_dir / "runs").rglob("result.json")))
        self.assertFalse(any((config.state_dir / "runs").rglob("*.jsonl")))

    def update_known_case(self, spec_path):
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        validation = raw.get("baseline_validation", {})
        if validation.get("mode") != "fresh":
            return
        known_case_path = self.root / "known-case.json"
        evaluator = Path(raw["evaluator"][1])
        if not evaluator.is_absolute():
            evaluator = self.root / evaluator
        input_hashes = {}
        for name in raw.get("input_files", []):
            input_path = Path(name)
            input_path = input_path.resolve() if input_path.is_absolute() else (self.program.parent / input_path).resolve()
            input_hashes[str(input_path)] = hashlib.sha256(input_path.read_bytes()).hexdigest()
        primary = raw["metrics"]["primary"]
        values = {primary: 1}
        for guard in raw["metrics"].get("guards", []):
            values[guard["name"]] = 1
        record = {
            "schema": 1, "id": "fixture-score-v1", "reproducible": True,
            "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            "input_hashes": input_hashes, "expected_metrics": values,
            "observed_metrics": values, "tolerance": 1e-9,
        }
        known_case_path.write_text(json.dumps(record), encoding="utf-8")
        validation["known_case"] = {"path": str(known_case_path),
                                    "sha256": hashlib.sha256(known_case_path.read_bytes()).hexdigest()}
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        return spec_path

    def delegated_hypothesis(self, **overrides):
        agent_id = f"child-{hashlib.sha256(os.urandom(16)).hexdigest()[:10]}"
        receipt_dir = self.root / ".labs" / "internal" / "delegations"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{agent_id}.json"
        receipt_path.write_text(json.dumps({
            "agent_id": agent_id, "harness": "Codex", "effective_model": "gpt-6-luna",
            "effective_reasoning_effort": "max", "launch_response": {
                "agent_id": agent_id, "harness": "Codex", "effective_model": "gpt-6-luna",
                "effective_reasoning_effort": "max",
            },
        }), encoding="utf-8")
        hypothesis = {
            "title": "Improve", "observation": "low", "expected_effect": "higher",
            "delegation": {"agents": [{
                "id": agent_id, "harness": "Codex", "model": "gpt-6-luna",
                "reasoning_effort": "max", "dispatch_ref": str(receipt_path),
                "dispatch_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "parameters_confirmed": True,
            }]},
        }
        hypothesis.update(overrides)
        return hypothesis

    def historical_spec(self, expected_scores):
        reference = self.project / "reference.json"
        reference.write_text("frozen reference", encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        relative = "project/reference.json"
        raw["input_files"] = [relative]
        raw["baseline_validation"] = {
            "mode": "historical", "tolerance": 1e-9,
            "input_hashes": {relative: hashlib.sha256(reference.read_bytes()).hexdigest()},
            "runs": [
                {"name": f"reference-{index + 1:02d}", "metrics": {"score": score}}
                for index, score in enumerate(expected_scores)
            ],
        }
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        return spec_path

    def pinned_project_fixture(self, *, mode="historical", wrong_hash=False):
        """Register a pin for sample-project in AUTORESEARCH_HOME/pins and return a spec for that project."""
        project = self.root / "sample-project"
        project.mkdir(exist_ok=True)
        (project / "value.txt").write_text("1", encoding="utf-8")
        reference_file = project / "reference.bin"
        reference_file.write_bytes(b"frozen known input")
        digest = hashlib.sha256(reference_file.read_bytes()).hexdigest()
        if wrong_hash:
            digest = "0" * 64
        runs = [
            {"name": "stability/run_01", "metrics": {"f1_score": 0.4, "accuracy": 0.6, "recall": 1.0}},
            {"name": "stability/run_02", "metrics": {"f1_score": 0.5, "accuracy": 0.7, "recall": 1.0}},
        ]
        pin_dir = Path(os.environ["AUTORESEARCH_HOME"]) / "pins"
        pin_dir.mkdir(parents=True, exist_ok=True)
        (pin_dir / "sample-project.json").write_text(json.dumps({
            "schema": 1,
            "project_name": "sample-project",
            "tolerance": 1e-9, "runs": runs,
            "inputs_relative_to_program_md": [{"path": "sample-project/reference.bin", "sha256": digest}],
        }), encoding="utf-8")
        self.evaluator.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "run=Path(sys.argv[2]);run.mkdir(parents=True,exist_ok=True)\n"
            "i=int(run.name[-2:])\n"
            "metrics=({'f1_score':0.4,'accuracy':0.6,'recall':1.0} if i==1 else {'f1_score':0.5,'accuracy':0.7,'recall':1.0})\n"
            "(run/'result.json').write_text(json.dumps({'metrics':metrics,'usage':{'requests':0,'cost_usd':0}}))\n",
            encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["project"] = "sample-project"
        raw["evaluator"] = ["{python}", "evaluator.py", "{candidate}", "{run_dir}"]
        raw["input_files"] = ["sample-project/reference.bin"]
        raw["metrics"] = {"primary": "f1_score", "direction": "max", "guards": [
            {"name": "accuracy", "direction": "max"}, {"name": "recall", "direction": "max"},
        ]}
        raw["baseline_validation"] = ({
            "mode": "historical", "tolerance": 1e-9,
            "runs": runs, "input_hashes": {"sample-project/reference.bin": digest},
        } if mode == "historical" else {"mode": "fresh", "evidence": "Agent assertion"})
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        return spec_path

    def test_preparation_from_agent_derived_spec_stays_in_sqlite_and_resumes(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        first = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(first["config"]))
        dashboard = first.get("dashboard")
        self.assertTrue(dashboard is None or str(dashboard).startswith(("http://", "https://")))
        self.assertTrue(status_lab(config)["setup_pending"])
        self.assertEqual(first["config"], prepare_lab(self.program, self.spec(), confirmed=True)["config"])
        self.assertTrue(self.store().is_initialized())
        self.assertTrue(self.store().load_config_bytes(config.lab_id))
        self.assert_no_sqlite_json_shadows(config)
        self.assertEqual(run_baseline(config)["status"], "accepted")
        (config.state_dir / "candidate" / "value.txt").write_text("2", encoding="utf-8")
        trial = run_trial(config, {"title": "Improve", "observation": "low", "expected_effect": "higher"})
        self.assertEqual(trial["status"], "accepted")
        self.assertNotIn("delegation", trial)
        self.assert_no_sqlite_json_shadows(config)

    def test_skill_setup_requires_a_baseline_validation_mode(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw.pop("baseline_validation")
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "baseline_validation"):
            prepare_lab(self.program, spec_path)
        self.assertFalse((self.root / ".labs").exists())

    def test_fresh_validation_requires_a_hash_pinned_known_case_record(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["baseline_validation"].pop("known_case")
        raw["baseline_validation"]["evidence"] = "trust me"
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "known_case record path and SHA-256"):
            prepare_lab(self.program, spec_path)
        self.assertFalse((self.root / ".labs").exists())

    def test_historical_baseline_mismatch_is_incomparable_and_blocks_trials(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.historical_spec([2, 2])
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))

        with self.assertRaisesRegex(Exception, "historical baseline mismatch"):
            run_baseline(config)

        state = load_state(config)
        manifest = self.store().get_run(config.lab_id, "baseline")["manifest"]
        self.assertIsNone(state["baseline_id"])
        self.assertFalse(state["baseline_verified"])
        self.assertEqual(manifest["status"], "incomparable")
        with self.assertRaisesRegex(Exception, "verified baseline"):
            run_trial(config, self.delegated_hypothesis())

    def test_catalog_incomparability_does_not_block_a_later_trial(self):
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.paths import home_root_id, project_id

        prepared = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(prepared["config"]))
        run_baseline(config)
        store = self.store()
        root_ref = home_root_id(Path(os.environ["AUTORESEARCH_HOME"]))
        catalog = store.load_catalog(root_ref)
        self.assertEqual(len(catalog["projects"]), 1)
        self.assertEqual(catalog["projects"][0]["project_ref"], project_id(self.project))
        # The v0.4 catalog stores project-to-active-lab identity, not comparison flags.
        catalog["projects"][0].update(
            comparison_status="incomparable", comparison_note="test historical mismatch"
        )
        store.save_catalog(catalog, root_ref=root_ref)
        stored_catalog = store.load_catalog(root_ref)
        self.assertEqual(len(stored_catalog["projects"]), 1)
        self.assertNotIn("comparison_status", stored_catalog["projects"][0])
        (config.state_dir / "candidate" / "value.txt").write_text("2", encoding="utf-8")
        manifest = run_trial(config, self.delegated_hypothesis())
        self.assertTrue(str(manifest.get("id", "")).startswith("trial-"))
        self.assertEqual(len(store.list_runs(config.lab_id)["items"]), 2)

    def test_pins_load_only_from_autoresearch_home(self):
        from autoresearch_runtime import pins

        self.assertEqual([name for name in vars(pins) if "h3" in name.lower()], [])
        self.assertEqual(pins._pin_paths(), [])
        pin_dir = Path(os.environ["AUTORESEARCH_HOME"]) / "pins"
        pin_dir.mkdir(parents=True)
        (pin_dir / "sample-project.json").write_text("{}", encoding="utf-8")
        self.assertEqual([path.name for path in pins._pin_paths()], ["sample-project.json"])

    def test_damaged_or_unknown_pin_raises_naming_the_file_instead_of_being_skipped(self):
        from autoresearch_runtime import pins

        pin_dir = Path(os.environ["AUTORESEARCH_HOME"]) / "pins"
        pin_dir.mkdir(parents=True)
        damaged = pin_dir / "damaged.json"
        for text, message in (
            ("{not json", "cannot be read"),
            ("[]", "not a JSON object"),
            (json.dumps({"schema": 2}), "unknown schema"),
        ):
            with self.subTest(message=message):
                damaged.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"{message}: damaged.json"):
                    pins.validate(self.program, self.project, {}, [])

    def test_pinned_contract_cannot_be_bypassed_with_fresh_mode(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.pinned_project_fixture(mode="fresh")
        with self.assertRaisesRegex(ValueError, "fresh baseline cannot bypass"):
            prepare_lab(self.program, spec_path, confirmed=True)

    def test_pinned_contract_with_matching_repeats_is_accepted_and_input_hash_mismatch_blocks(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.pinned_project_fixture()
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))
        baseline = run_baseline(config)
        self.assertEqual(baseline["status"], "accepted")
        self.assertTrue(load_state(config)["baseline_verified"])

        spec_path = self.pinned_project_fixture(wrong_hash=True)
        with self.assertRaisesRegex(ValueError, "reference input is missing or changed"):
            prepare_lab(self.program, spec_path, confirmed=True)

    def test_historical_model_repro_nonblocking_accepts_drift_with_identity_gates(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        fixture_path = self.root / "evaluator-fixture.bin"
        fixture_path.write_bytes(b"evaluator-fixture-v1")
        fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        request_sha = hashlib.sha256(b"hello-request" * 2).hexdigest()
        self.evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "value = int((candidate / 'value.txt').read_text())\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "with (run_dir / 'sent-requests.jsonl').open('w', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps({'body': 'hello-request'}) + '\\n')\n"
            "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': value}, "
            "'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        spec_path = self.historical_spec([2, 2])
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["baseline_validation"]["model_repro"] = {"blocking": False}
        raw["baseline_validation"]["request_identity"] = {"sha256": request_sha}
        raw["baseline_validation"]["evaluator_fixture"] = {
            "path": str(fixture_path),
            "sha256": fixture_sha,
        }
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))
        baseline = run_baseline(config)
        self.assertEqual(baseline["status"], "accepted")
        self.assertEqual(baseline["baseline_validation"]["model_repro_status"], "drift")
        self.assertEqual(
            baseline["baseline_validation"]["historical_reference"],
            raw["baseline_validation"]["runs"],
        )
        state = load_state(config)
        self.assertIsNotNone(state["baseline_id"])
        self.assertFalse((config.state_dir / "runs" / "baseline" / "repeat_01" / "sent-requests.jsonl").exists())

    def test_matching_historical_baseline_records_verification(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.historical_spec([1, 1])
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))

        baseline = run_baseline(config)
        state = load_state(config)
        self.assertEqual(baseline["status"], "accepted")
        self.assertTrue(state["baseline_verified"])
        self.assertEqual(baseline["baseline_validation"]["status"], "verified")

    def test_fresh_baseline_is_blocked_when_comparable_history_exists(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        first = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(first["config"]))
        run_baseline(config)
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["timeout_seconds"] = 9
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        with self.assertRaisesRegex(ValueError, "Comparable baseline history"):
            prepare_lab(self.program, spec_path)

    def test_large_unclassified_candidate_file_blocks_preparation(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        large = self.project / "database.sql"
        with large.open("wb") as stream:
            stream.seek(25 * 1024 * 1024)
            stream.write(b"x")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["source_paths"] = ["value.txt", "database.sql"]
        spec_path.write_text(json.dumps(raw), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unclassified large candidate file: database.sql"):
            prepare_lab(self.program, spec_path)
        self.assertFalse((self.root / ".labs").exists())

        raw["input_files"] = ["project/database.sql"]
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        self.assertFalse((Path(prepared["candidate"]) / "database.sql").exists())

    def test_openrouter_brief_does_not_require_a_key_budget(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        self.program.write_text(
            self.program.read_text(encoding="utf-8") + "\nProvider: OpenRouter paid requests\n",
            encoding="utf-8",
        )
        prepared = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(prepared["config"]))
        self.assertIsNotNone(self.store().get_lab(config.lab_id))
        self.assertNotIn("max_requests", config.budget)
        self.assertNotIn("max_cost_usd", config.budget)

    def test_fabricated_local_limit_without_brief_source_is_rejected(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["budget"] = {"max_trials": 3}
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "budget source for max_trials"):
            prepare_lab(self.program, spec_path)

    def test_spend_limit_in_spec_is_rejected_before_any_lab_is_written(self):
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.errors import LabError

        for field, value in (
            ("budget", {"max_requests": 5, "openrouter": True}),
            ("budget", {"max_cost_usd": 2.0}),
            ("budget_sources", {"max_cost_usd": "at most 2 dollars"}),
        ):
            with self.subTest(field=field, value=value):
                spec_path = self.spec()
                raw = json.loads(spec_path.read_text(encoding="utf-8"))
                raw[field] = value
                spec_path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(LabError) as rejected:
                    prepare_lab(self.program, spec_path, confirmed=True)
                self.assertEqual(rejected.exception.code, "E_CONTRACT_SPEND_LIMIT")
                self.assertIn("does not limit spending", str(rejected.exception))
                self.assertIn("limit on the API key", str(rejected.exception))
        store = self.store()
        self.assertTrue(not store.is_initialized() or store.list_labs(limit=1)["total"] == 0)

    def test_trial_does_not_require_a_subagent_receipt(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        prepared = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(prepared["config"]))
        run_baseline(config)
        candidate_value = config.state_dir / "candidate" / "value.txt"
        candidate_value.write_text("2", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Missing agent", "observation": "x", "expected_effect": "y",
            "delegation": {"agents": [{"model": "anything"}]},
        })
        self.assertEqual(trial["status"], "accepted")
        self.assertNotIn("delegation", trial)
        self.assertEqual(candidate_value.read_text(encoding="utf-8"), "2")

    def test_evaluator_process_gets_utf8_environment(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        self.evaluator.write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "run=Path(sys.argv[2]); run.mkdir(parents=True, exist_ok=True)\n"
            "print('Проверка UTF-8')\n"
            "(run/'result.json').write_text(json.dumps({'metrics': {'score': 1}, "
            "'usage': {'requests': 0, 'cost_usd': 0}, "
            "'details': {'utf8': os.environ.get('PYTHONUTF8'), 'io': os.environ.get('PYTHONIOENCODING')}}))\n",
            encoding="utf-8",
        )
        config = load_config(Path(prepare_lab(self.program, self.spec(), confirmed=True)["config"]))
        baseline = run_baseline(config)
        self.assertEqual(baseline["runs"][0]["details"], {"utf8": "1", "io": "utf-8", "evidence_origin": "fresh"})
        self.assertIn("Проверка UTF-8", (config.state_dir / "runs" / "baseline" / "repeat_01" / "run.log").read_text(encoding="utf-8"))

    def test_trial_cli_json_is_safe_on_cp1252_console(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        prepared = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(prepared["config"]))
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("2", encoding="utf-8")
        hypothesis = self.root / "hypothesis.json"
        hypothesis.write_text(json.dumps(self.delegated_hypothesis(title="Улучшение")), encoding="utf-8")
        with approved_offline_profile(config):
            environment = dict(os.environ, PYTHONIOENCODING="cp1252")
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts" / "lab.py"),
                 "--config", str(config.path), "--confirmed", "trial", "--hypothesis", str(hypothesis)],
                cwd=self.root, env=environment, capture_output=True, text=True, encoding="cp1252",
                **support.hidden_process_kwargs(),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["data"]["title"], "Улучшение")

    def test_prepare_cli_starts_dashboard_and_returns_url_when_browser_is_unavailable(self):
        from unittest.mock import patch
        from autoresearch_runtime.bootstrap import main as prepare_main

        output = io.StringIO()
        with patch.dict(os.environ, {"AUTORESEARCH_NO_DASHBOARD": ""}), \
             patch("autoresearch_runtime.dashboard_server.start_dashboard", return_value={
            "running": True, "url": "http://127.0.0.1:8765/", "browser_opened": False, "pid": 123,
        }) as start, contextlib.redirect_stdout(output):
            self.assertEqual(prepare_main(["--confirmed", "--program", str(self.program), "--spec", str(self.spec())]), 0)
        result = json.loads(output.getvalue())["data"]
        self.assertTrue(start.called)
        self.assertEqual(result["dashboard_url"], "http://127.0.0.1:8765/")

    def test_prepare_cli_respects_no_dashboard_env(self):
        from unittest.mock import patch
        from autoresearch_runtime.bootstrap import main as prepare_main

        output = io.StringIO()
        with patch.dict(os.environ, {"AUTORESEARCH_NO_DASHBOARD": "1"}), \
             patch("autoresearch_runtime.dashboard_server.start_dashboard") as start, \
             contextlib.redirect_stdout(output):
            self.assertEqual(prepare_main(["--confirmed", "--program", str(self.program), "--spec", str(self.spec())]), 0)
        result = json.loads(output.getvalue())["data"]
        self.assertFalse(start.called)
        self.assertIsNone(result.get("dashboard_url"))

    def test_paid_evaluator_runs_without_a_key_limit(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        self.program.write_text(self.program.read_text(encoding="utf-8") + "\nProvider: OpenRouter paid requests\n", encoding="utf-8")
        evaluator = self.root / "paid_evaluator.py"
        evaluator.write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "run=Path(sys.argv[2]); run.mkdir(parents=True, exist_ok=True)\n"
            "(run/'result.json').write_text(json.dumps({'metrics': {'score': 1}, "
            "'usage': {'requests': 1, 'cost_usd': 0.1}, "
            "'details': {'limit': os.environ.get('AUTORESEARCH_MAX_COST_USD')}}))\n",
            encoding="utf-8",
        )
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["evaluator"] = ["{python}", str(evaluator), "{candidate}", "{run_dir}"]
        raw["budget"] = {"openrouter": True}
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))
        baseline = run_baseline(config)
        self.assertEqual(baseline["status"], "accepted")
        self.assertIsNone(baseline["runs"][0]["details"]["limit"])
        self.assertFalse((config.state_dir / "openrouter_budget.json").exists())
        self.assertAlmostEqual(baseline["runs"][0]["usage"]["cost_usd"], 0.1)

    def test_openrouter_flag_prepares_a_lab_without_a_balance_check(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        self.program.write_text(self.program.read_text(encoding="utf-8") + "\nProvider: OpenRouter paid requests\n", encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["budget"] = {"openrouter": True}
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))
        self.assertTrue(config.budget["openrouter"])
        self.assertFalse(Path(prepared["config"]).exists())

    def test_openrouter_flag_stores_no_spend_fields_and_caps_nothing(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        evaluator = self.root / "key_capped_evaluator.py"
        evaluator.write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "run=Path(sys.argv[2]); run.mkdir(parents=True, exist_ok=True)\n"
            "(run/'result.json').write_text(json.dumps({'metrics': {'score': 1}, "
            "'usage': {'requests': 1, 'cost_usd': 0.1}, "
            "'details': {'limit': os.environ.get('AUTORESEARCH_MAX_COST_USD')}}))\n",
            encoding="utf-8",
        )
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["evaluator"] = ["{python}", str(evaluator), "{candidate}", "{run_dir}"]
        raw["budget"] = {"openrouter": True}
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))
        self.assertNotIn("max_requests", config.budget)
        self.assertNotIn("max_cost_usd", config.budget)
        baseline = run_baseline(config)
        self.assertEqual(baseline["status"], "accepted")
        self.assertIsNone(baseline["runs"][0]["details"]["limit"])
        self.assertFalse((config.state_dir / "openrouter_budget.json").exists())

    def test_lab_stored_before_0_8_resumes_under_the_same_contract(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        first = prepare_lab(self.program, self.spec(), confirmed=True)
        store = self.store()
        lab_id = store.resolve_lab_id(Path(first["config"]))
        # Simulate a stored contract from 0.7.x: the retired spend fields are present.
        legacy = json.loads(store.load_config_bytes(lab_id).decode("utf-8"))
        legacy["budget"].update(max_requests=0, max_cost_usd=0)
        legacy_bytes = (json.dumps(legacy, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        store.save_config_bytes(lab_id, legacy, legacy_bytes)
        state = store.load_projection(lab_id, "state")
        state["config_sha256"] = hashlib.sha256(legacy_bytes).hexdigest()
        store.save_projection(lab_id, "state", state)

        again = prepare_lab(self.program, self.spec(), confirmed=True)

        self.assertEqual(again["config"], first["config"])
        self.assertFalse(again["created"])
        self.assertEqual(store.load_config_bytes(lab_id), legacy_bytes)

    def test_old_shape_lab_opens_with_its_contract_fingerprint_unchanged(self):
        from autoresearch_runtime.harness_policy import contract_fingerprint

        # The lab.json bytes are exactly the 0.7.x shape. Its fingerprint was recorded from
        # the 0.7.x code. Hashing the stored bytes as stored keeps it unchanged.
        lab = self.root / "legacy-lab"
        (lab / "source").mkdir(parents=True)
        (lab / "source" / "rule.txt").write_text("rule\n", encoding="utf-8")
        (lab / "source" / "evaluator.py").write_text("print('x')\n", encoding="utf-8")
        (lab / ".lab").mkdir()
        stored = (json.dumps({
            "source_root": "source",
            "state_dir": ".lab",
            "source_paths": ["rule.txt", "evaluator.py"],
            "editable_files": ["rule.txt"],
            "evaluator": ["{python}", "evaluator.py", "{candidate}", "{run_dir}", "{config_dir}"],
            "metrics": {"primary": "mean_error", "direction": "min",
                        "guards": [{"name": "coverage", "direction": "max"}]},
            "repeats": 2,
            "timeout_seconds": 5,
            "budget": {"max_seconds": 60, "max_trials": 5, "max_requests": 0, "max_cost_usd": 0},
        }, indent=2) + "\n").encode("utf-8")
        (lab / "lab.json").write_bytes(stored)
        digest = hashlib.sha256(stored).hexdigest()
        (lab / ".lab" / "state.json").write_text(json.dumps({
            "schema": 2, "setup_pending": False, "config_sha256": digest, "config_copy_sha256": digest,
            "inputs": [], "current_epoch": 1, "epochs": [], "source": [],
        }), encoding="utf-8")

        config = load_config(lab / "lab.json")

        self.assertEqual(config.budget["max_seconds"], 60)
        self.assertEqual(contract_fingerprint(config),
                         "90ae45483902cc49cd2e6df405daf43e58978de83b857eaa4fdc34125a608390")
        self.assertEqual((lab / "lab.json").read_bytes(), stored)

    def test_generated_evaluator_is_copied_into_lab_and_changes_open_new_lab(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        first = prepare_lab(self.program, self.spec(generated=True), confirmed=True)
        config = load_config(Path(first["config"]))
        self.assertTrue(Path(config.evaluator[1]).is_relative_to(config.state_dir))
        self.assertEqual(run_baseline(config)["status"], "accepted")
        old_evaluator = Path(config.evaluator[1])
        self.evaluator.write_text(self.evaluator.read_text(encoding="utf-8") + "\n# revised\n", encoding="utf-8")
        second = prepare_lab(self.program, self.spec(generated=True), confirmed=True)
        self.assertNotEqual(first["config"], second["config"])
        self.assertEqual(self.store().get_run(config.lab_id, "baseline")["manifest"]["status"], "accepted")
        self.assertEqual(old_evaluator.read_text(encoding="utf-8").count("# revised"), 0)

    def test_reference_change_opens_new_lab_and_default_run_is_unbounded(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        reference = self.root / "reference.txt"
        reference.write_text("first", encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["input_files"] = ["reference.txt"]
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        first = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(first["config"]))
        self.assertIsNone(config.budget["max_trials"])
        self.assertIsNone(config.budget["max_seconds"])
        self.assertNotIn("max_requests", config.budget)
        self.assertNotIn("max_cost_usd", config.budget)
        reference.write_text("second", encoding="utf-8")
        self.update_known_case(spec_path)
        second = prepare_lab(self.program, spec_path, confirmed=True)
        self.assertNotEqual(first["config"], second["config"])
        raw["metrics"]["direction"] = "min"
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        third = prepare_lab(self.program, spec_path, confirmed=True)
        self.assertNotEqual(second["config"], third["config"])

    def test_labs_are_grouped_under_readable_project_and_contract_versions(self):
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.paths import home_root_id, project_id

        first = prepare_lab(self.program, self.spec(), confirmed=True)
        first_path = Path(first["config"])
        self.assertEqual(first_path.parent.parent.name.split("-")[0], "project")
        self.assertEqual(first_path.parent.parent.parent.name, "projects")
        self.assertEqual(prepare_lab(self.program, self.spec(), confirmed=True)["config"], first["config"])

        self.evaluator.write_text(self.evaluator.read_text(encoding="utf-8") + "\n# new contract\n", encoding="utf-8")
        second = prepare_lab(self.program, self.spec(), confirmed=True)
        second_path = Path(second["config"])
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.parent.parent, second_path.parent.parent)
        catalog = self.store().load_catalog(home_root_id(Path(os.environ["AUTORESEARCH_HOME"])))
        self.assertEqual(len(catalog["projects"]), 1)
        self.assertEqual(catalog["projects"][0]["project_ref"], project_id(self.project))
        self.assertEqual(str(catalog["projects"][0]["lab_id"]), load_config(Path(second["config"])).lab_id)
        self.assertEqual(len(self.store().list_labs(filters={"source_root": str(self.project)})["items"]), 2)

    def test_protected_input_is_hashed_but_never_copied_into_candidate(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        sql = self.project / "large.sql"
        sql.write_text("SELECT 1;\n" * 10, encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["input_files"] = ["project/large.sql"]
        raw["source_paths"] = ["value.txt", "large.sql"]
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(prepared["config"]))
        self.assertFalse((config.state_dir / "candidate" / "large.sql").exists())
        state = load_state(config)
        self.assertEqual(state["inputs"][0]["path"], str(sql.resolve()))
        self.assertEqual(run_baseline(config)["status"], "accepted")

        copied = config.state_dir / "candidate" / "renamed-reference.sql"
        copied.write_bytes(sql.read_bytes())
        store = self.store()
        last_operation = store.get_latest_operation(config.lab_id)
        hypothesis = self.delegated_hypothesis()
        hypothesis.update(title="Copy protected input", observation="Reference may be reused",
                          expected_effect="Test enforcement")
        from autoresearch_runtime.errors import LabError
        with self.assertRaisesRegex(LabError, "Candidate contains a copy of a protected input"):
            run_trial(config, hypothesis)
        self.assertEqual(store.get_latest_operation(config.lab_id)["operation_id"], last_operation["operation_id"])
        self.assertFalse(copied.exists())
        self.assertEqual(sql.read_text(encoding="utf-8"), "SELECT 1;\n" * 10)

        sql.write_text("SELECT 2;\n", encoding="utf-8")
        with self.assertRaisesRegex(
            LabError, "Known-case input hashes must exactly match the protected evaluator inputs",
        ):
            run_trial(config, self.delegated_hypothesis())
        self.assertEqual(store.get_latest_operation(config.lab_id)["operation_id"], last_operation["operation_id"])
        self.assertEqual(sql.read_text(encoding="utf-8"), "SELECT 2;\n")

    def test_trial_completion_leaves_archival_explicit(self):
        from unittest.mock import patch
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.archive import archive_idle_labs
        from autoresearch_runtime.dash_api import archive_lab
        from autoresearch_runtime.errors import LabError

        prepared = prepare_lab(self.program, self.spec(), confirmed=True)
        config = load_config(Path(prepared["config"]))
        self.assertEqual(run_baseline(config)["status"], "accepted")
        (config.state_dir / "candidate" / "value.txt").write_text("2", encoding="utf-8")

        trial = run_trial(config, self.delegated_hypothesis())
        status, payload = archive_lab(Path(os.environ["AUTORESEARCH_HOME"]), "project", str(config.lab_id))
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "unsupported_db_lab_action")

        # Keep the legacy file-store behavior independently testable beside DB labs.
        legacy_home = self.root / "legacy-home"
        failed = legacy_home / "projects" / "old-project-id" / "failed-contract"
        failed.mkdir(parents=True)
        (failed / "lab.json").write_text(json.dumps({"source_root": str(self.project)}), encoding="utf-8")
        (failed / "state.json").write_text(json.dumps({"setup_pending": True, "inputs": []}), encoding="utf-8")
        archived = legacy_home / "archive" / "old-project-id" / "failed-contract"
        self.assertTrue(failed.exists())
        self.assertNotIn("archived_labs", trial)
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(legacy_home)}):
            preview = archive_idle_labs(legacy_home)
        self.assertEqual(preview["archived"], [])
        self.assertTrue(failed.exists())
        before = support.tree_digest(legacy_home)
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(legacy_home)}):
            with self.assertRaises(LabError) as error:
                archive_idle_labs(legacy_home, apply=True)
        self.assertEqual(error.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(legacy_home), before)
        self.assertTrue(failed.exists())
        self.assertFalse(archived.exists())

    def test_preparation_refuses_changed_internal_configuration_on_resume(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.spec()
        first = prepare_lab(self.program, spec_path, confirmed=True)
        config = load_config(Path(first["config"]))
        store = self.store()
        raw = store.load_config(config.lab_id)
        raw["metrics"]["primary"] = "other"
        store.save_config(config.lab_id, raw)
        with self.assertRaisesRegex(ValueError, "configuration"):
            prepare_lab(self.program, spec_path)

    def test_missing_evaluator_is_reported_before_lab_creation(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        self.evaluator.unlink()
        with self.assertRaisesRegex(ValueError, "evaluator"):
            prepare_lab(self.program, self.spec())
        self.assertFalse((self.root / ".labs").exists())

    def test_missing_reference_input_blocks_preparation_before_baseline(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["input_files"] = ["missing-golden-data.csv"]
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing files"):
            prepare_lab(self.program, spec_path)
        self.assertFalse((self.root / ".labs").exists())

    def test_internal_spec_must_not_be_written_into_source_project(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        spec_path = self.project / "derived.json"
        spec_path.write_text(self.spec().read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside the project"):
            prepare_lab(self.program, spec_path)
        self.assertFalse((self.root / ".labs").exists())

    def test_program_inside_project_places_lab_outside_project(self):
        from autoresearch_runtime.bootstrap import prepare_lab

        inside_program = self.project / "program.md"
        inside_program.write_text(self.program.read_text(encoding="utf-8"), encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["project"] = "."
        raw["evaluator"][1] = str(self.evaluator)
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        info = prepare_lab(inside_program, spec_path, confirmed=True)
        self.assertFalse(Path(info["config"]).is_relative_to(self.project))
        dashboard = info.get("dashboard")
        self.assertTrue(dashboard is None or str(dashboard).startswith(("http://", "https://")))

    def test_resume_uses_saved_lab_after_generated_staging_evaluator_is_gone(self):
        from autoresearch_runtime.bootstrap import prepare_lab, resume_lab

        first = prepare_lab(self.program, self.spec(generated=True), confirmed=True)
        self.evaluator.unlink()
        resumed = resume_lab(self.program, self.project)
        self.assertEqual(resumed["config"], first["config"])
        self.assertFalse(resumed["created"])
        self.program.write_text(self.program.read_text(encoding="utf-8") + "\nNew goal\n", encoding="utf-8")
        self.assertIsNone(resume_lab(self.program, self.project))

    def test_resume_rejects_changed_protected_input(self):
        from autoresearch_runtime.bootstrap import prepare_lab, resume_lab

        reference = self.root / "reference.txt"
        reference.write_text("first", encoding="utf-8")
        spec_path = self.spec()
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        raw["input_files"] = ["reference.txt"]
        spec_path.write_text(json.dumps(raw), encoding="utf-8")
        self.update_known_case(spec_path)
        prepare_lab(self.program, spec_path, confirmed=True)
        reference.write_text("changed", encoding="utf-8")
        self.assertIsNone(resume_lab(self.program, self.project))

    def test_legacy_setup_has_dashboard_before_baseline(self):
        from autoresearch_runtime.errors import LabError

        labs = self.root / "legacy-labs"
        with self.assertRaises(LabError) as error:
            setup_project(self.project, labs)
        self.assertEqual(error.exception.code, "E_OPERATION_REMOVED")
        self.assertFalse(labs.exists())

    def test_legacy_setup_does_not_invent_time_or_trial_limits(self):
        from autoresearch_runtime.lab import main

        labs = self.root / "legacy-labs"
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = main(["setup", "--project", str(self.project)], labs_root=labs)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(errors.getvalue())["error"]["code"], "E_OPERATION_REMOVED")
        self.assertFalse(labs.exists())

    def test_skill_scripts_work_from_unrelated_working_directory(self):
        original = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch"
        installed = self.root / "installed-skill"
        shutil.copytree(original, installed)
        scripts = installed / "scripts"
        prepared = subprocess.run(
            [sys.executable, str(scripts / "prepare.py"), "--program", str(self.program),
             "--spec", str(self.spec()), "--confirmed", "--no-dashboard"], cwd=self.root, capture_output=True, text=True,
             **support.hidden_process_kwargs(),
        )
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        config = json.loads(prepared.stdout)["data"]["config"]
        config_obj = load_config(Path(config))
        with approved_offline_profile(config_obj):
            measured = subprocess.run(
                [sys.executable, str(scripts / "lab.py"), "--config", config, "--confirmed", "baseline"],
                cwd=self.root, capture_output=True, text=True,
                **support.hidden_process_kwargs(),
            )
        self.assertEqual(measured.returncode, 0, measured.stderr)
        self.assertEqual(json.loads(measured.stdout)["data"]["status"], "accepted")
        resumed = subprocess.run(
            [sys.executable, str(scripts / "prepare.py"), "--program", str(self.program),
             "--project", str(self.project), "--resume", "--no-dashboard"],
            cwd=self.root, capture_output=True, text=True,
            **support.hidden_process_kwargs(),
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(json.loads(resumed.stdout)["data"]["config"], config)


class RequestIdentityDigestTest(unittest.TestCase):
    def test_retry_lines_do_not_change_the_request_digest(self):
        from types import SimpleNamespace

        from autoresearch_runtime.baseline import _request_identity_digest

        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            log = state / "runs" / "baseline" / "repeat_01" / "sent-requests.jsonl"
            log.parent.mkdir(parents=True)
            body = {"messages": [{"role": "user", "content": "same"}]}
            line = json.dumps({"body": body, "status": 200}, ensure_ascii=False)
            retry = json.dumps({"body": body, "status": 429}, ensure_ascii=False)
            log.write_text(retry + "\n" + line + "\n" + retry + "\n", encoding="utf-8")
            digest_with_retries = _request_identity_digest(SimpleNamespace(state_dir=state), "baseline", 1)
            log.write_text(line + "\n", encoding="utf-8")
            digest_once = _request_identity_digest(SimpleNamespace(state_dir=state), "baseline", 1)
            self.assertEqual(digest_with_retries, digest_once)
            self.assertIsNotNone(digest_once)


if __name__ == "__main__":
    unittest.main()
