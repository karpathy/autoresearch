try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import json
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.dashboard import chart_series
from autoresearch_runtime.lab import (
    LabError, load_config, load_state, run_baseline, run_trial,
    setup_project, status_lab, main, recover_lab,
)
from autoresearch_runtime.history import read_history
from autoresearch_runtime.snapshot import init_lab


class SetupHistoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.labs = self.root / "labs"
        (self.project / "value.txt").write_text("1", encoding="utf-8")

    def configure(self, info):
        path = Path(info["config"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        evaluator = self.root / "evaluator.py"
        evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, output = map(Path, sys.argv[1:3])\n"
            "value = int((candidate / 'value.txt').read_text())\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'result.json').write_text(json.dumps({'metrics': {'score': value},"
            " 'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        raw["evaluator"] = ["{python}", str(evaluator), "{candidate}", "{run_dir}"]
        raw["input_files"] = [str(evaluator)]
        raw["metrics"] = {"primary": "score", "direction": "max", "guards": []}
        path.write_text(json.dumps(raw), encoding="utf-8")
        return load_config(path), evaluator

    def legacy_setup_fixture(self):
        """Build a legacy test fixture without calling the retired setup API."""
        state_dir = self.labs / "legacy-project"
        config_path = state_dir / "lab.json"
        if not config_path.exists():
            state_dir.mkdir(parents=True)
            config_path.write_text(json.dumps({
                "source_root": str(self.project), "state_dir": str(state_dir),
                "input_files": [], "evaluator": [],
                "metrics": {"primary": "", "direction": "max", "guards": []},
                "repeats": 2, "timeout_seconds": 120,
                "budget": {"max_seconds": None, "max_trials": None,
                           "max_requests": 0, "max_cost_usd": 0},
                "setup_pending": True,
            }), encoding="utf-8")
            init_lab(load_config(config_path, allow_pending=True))
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        catalog = read_history(state_dir)
        missing = []
        if not raw.get("evaluator"):
            missing.append("evaluator")
        if not raw.get("metrics", {}).get("primary"):
            missing.append("metrics.primary")
        return {
            "config": str(config_path), "candidate": str(state_dir / "candidate"),
            "history": str(state_dir / "history.json"),
            "dashboard": str(state_dir / "dashboard.html"),
            "historical_hypotheses": len(catalog["hypotheses"]), "missing": missing,
        }

    def test_setup_does_not_import_old_project_history(self):
        history_dir = self.project / "history"
        history_dir.mkdir()
        (history_dir / "manifest.json").write_text(json.dumps({
            "title": "Earlier result", "metrics": {"score": 99}, "status": "accepted",
        }), encoding="utf-8")
        (history_dir / "NOTES.md").write_text("Prior experiment notes\n", encoding="utf-8")

        info = self.legacy_setup_fixture()

        self.assertEqual(info["historical_hypotheses"], 0)
        self.assertFalse((Path(info["config"]).parent / "history.json").exists())
        self.assertEqual(read_history(Path(info["config"]).parent), {
            "schema": 1, "documents": [], "hypotheses": [],
        })

    def test_setup_snapshots_dirty_git_project_without_service_files(self):
        subprocess.run(["git", "init", "-q"], cwd=self.project, check=True, **support.hidden_process_kwargs())
        (self.project / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
        subprocess.run(["git", "add", "value.txt", ".gitignore"], cwd=self.project, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@local",
                        "commit", "-q", "-m", "initial"], cwd=self.project, check=True, **support.hidden_process_kwargs())
        (self.project / "value.txt").write_text("2", encoding="utf-8")
        (self.project / "new.txt").write_text("new", encoding="utf-8")
        (self.project / "ignored.txt").write_text("skip", encoding="utf-8")
        (self.project / ".env").write_text("SECRET=x", encoding="utf-8")
        (self.project / ".venv").mkdir()
        (self.project / ".venv" / "secret.txt").write_text("skip", encoding="utf-8")
        (self.project / "results").mkdir()
        (self.project / "results" / "old.json").write_text("{}", encoding="utf-8")

        info = self.legacy_setup_fixture()
        candidate = Path(info["candidate"])
        self.assertEqual((candidate / "value.txt").read_text(), "2")
        self.assertEqual((candidate / "new.txt").read_text(), "new")
        for name in ("ignored.txt", ".env", ".venv", "results"):
            self.assertFalse((candidate / name).exists(), name)
        self.assertEqual((self.project / "value.txt").read_text(), "2")
        self.assertFalse((self.project / "lab.json").exists())
        self.assertIn("evaluator", info["missing"])
        with self.assertRaises(LabError):
            load_config(Path(info["config"]))
        self.assertEqual(self.legacy_setup_fixture()["config"], info["config"])

    def test_setup_cli_is_retired_without_writing_a_legacy_lab(self):
        errors = io.StringIO()
        labs = self.root / ".labs"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = main(["setup", "--project", str(self.project)], labs_root=labs)
        self.assertEqual(code, 2)
        payload = json.loads(errors.getvalue())
        self.assertEqual(payload["error"]["code"], "E_OPERATION_REMOVED")
        self.assertIn("prepare", payload["error"]["hint"].casefold())
        self.assertIn("autoresearch_home", payload["error"]["hint"].casefold())
        self.assertFalse(labs.exists())

    def test_direct_setup_helper_is_retired_without_writing_a_legacy_lab(self):
        with self.assertRaises(LabError) as error:
            setup_project(self.project, self.labs)
        self.assertEqual(error.exception.code, "E_OPERATION_REMOVED")
        self.assertFalse(self.labs.exists())

    def test_project_command_without_setup_gives_next_step(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main(["--project", str(self.project), "status"], labs_root=self.root / ".labs"), 2)
        self.assertIn("prepare", errors.getvalue().casefold())
        self.assertNotIn("setup --project", errors.getvalue().casefold())

    def test_default_allows_new_deleted_and_renamed_files_and_restores_rejection(self):
        (self.project / "old.txt").write_text("old", encoding="utf-8")
        info = self.legacy_setup_fixture()
        config, _ = self.configure(info)
        run_baseline(config)
        candidate = config.state_dir / "candidate"
        (candidate / "value.txt").write_text("2", encoding="utf-8")
        (candidate / "old.txt").rename(candidate / "renamed.txt")
        (candidate / "new.txt").write_text("new", encoding="utf-8")
        accepted = run_trial(config, {"title": "All changes", "observation": "low", "expected_effect": "higher"})
        self.assertEqual(accepted["status"], "accepted")
        self.assertTrue((candidate / "new.txt").exists())
        self.assertFalse((candidate / "old.txt").exists())
        patch = (config.state_dir / "runs" / accepted["id"] / "change.patch").read_text()
        self.assertIn("new.txt", patch)
        self.assertIn("old.txt", patch)

        (candidate / "value.txt").write_text("0", encoding="utf-8")
        (candidate / "new.txt").unlink()
        rejected = run_trial(config, {"title": "Worse", "observation": "x", "expected_effect": "y"})
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual((candidate / "value.txt").read_text(), "2")
        self.assertTrue((candidate / "new.txt").exists())

    def test_external_evaluator_is_frozen_and_candidate_copy_cannot_change_scoring(self):
        (self.project / "evaluator.py").write_text("candidate copy", encoding="utf-8")
        info = self.legacy_setup_fixture()
        config, evaluator = self.configure(info)
        run_baseline(config)
        candidate = config.state_dir / "candidate"
        (candidate / "evaluator.py").write_text("modified candidate copy", encoding="utf-8")
        (candidate / "value.txt").write_text("2", encoding="utf-8")
        self.assertEqual(run_trial(config, {"title": "score", "observation": "x", "expected_effect": "y"})["status"], "accepted")
        evaluator.write_text("changed external evaluator", encoding="utf-8")
        with self.assertRaisesRegex(LabError, "Input changed"):
            run_trial(config, {"title": "x", "observation": "x", "expected_effect": "y"})

    def test_default_mode_rejects_evaluator_inside_editable_candidate(self):
        info = self.legacy_setup_fixture()
        path = Path(info["config"])
        script = Path(info["candidate"]) / "evaluator.py"
        script.write_text("print('untrusted')", encoding="utf-8")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["evaluator"] = ["{python}", str(script), "{candidate}", "{run_dir}"]
        raw["metrics"] = {"primary": "score", "direction": "max", "guards": []}
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(LabError, "outside the editable candidate"):
            load_config(path)

    def test_default_mode_rejects_reference_file_inside_candidate(self):
        info = self.legacy_setup_fixture()
        config, _ = self.configure(info)
        raw = json.loads(config.path.read_text(encoding="utf-8"))
        raw["input_files"] = [str(config.state_dir / "candidate" / "value.txt")]
        config.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(LabError, "outside the editable candidate"):
            load_config(config.path)

    def test_existing_sqlite_history_is_readable_but_not_a_baseline(self):
        import hashlib

        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.dashboard import render_unified_dashboard
        from autoresearch_runtime.sqlite_store import SQLiteStore

        history_dir = self.project / "history"
        history_dir.mkdir()
        (history_dir / "manifest.json").write_text(json.dumps({
            "title": "Old idea", "metrics": {"score": 99}, "status": "accepted",
        }), encoding="utf-8")
        (history_dir / "NOTES.md").write_text("# Another old idea\nManual score 88", encoding="utf-8")
        program = self.root / "program.md"
        program.write_text("Project: project\nGoal: improve score\nMetric: score, max\n", encoding="utf-8")
        evaluator = self.root / "evaluator.py"
        evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "value = int((candidate / 'value.txt').read_text())\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': value}, "
            "'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        known_case_path = self.root / "known-case.json"
        known_case_path.write_text(json.dumps({
            "schema": 1, "id": "fixture-score-v1", "reproducible": True,
            "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            "input_hashes": {}, "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1}, "tolerance": 1e-9,
        }), encoding="utf-8")
        spec_path = self.root / "derived.json"
        spec_path.write_text(json.dumps({
            "project": "project",
            "evaluator": ["{python}", "evaluator.py", "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(known_case_path),
                    "sha256": hashlib.sha256(known_case_path.read_bytes()).hexdigest(),
                },
            },
        }), encoding="utf-8")
        home = self.root / "autoresearch-home"
        catalog = {
            "schema": 1,
            "documents": ["history/NOTES.md"],
            "hypotheses": [{
                "id": "legacy-existing-record",
                "title": "<script>Bad</script>",
                "source": "history/NOTES.md",
                "summary": "manual result",
                "conclusion": "<img src=x>",
                "metrics": {"score": 88},
                "benchmark": {},
                "runs": [{"source": "history/NOTES.md", "metrics": {"score": 88}}],
                "artifact": "history/manifest.json",
                "completeness": "recorded",
            }],
        }
        with patch.dict(os.environ, {
            "AUTORESEARCH_HOME": str(home),
            "AUTORESEARCH_NO_DASHBOARD": "1",
        }):
            for name in (
                "AUTORESEARCH_HARNESS_PROFILE", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
            ):
                os.environ.pop(name, None)
            prepared = prepare_lab(program, spec_path, confirmed=True)
            config = load_config(Path(prepared["config"]))
            self.assertEqual(config.storage_mode, "sqlite")
            self.assertEqual(str(config.lab_id), prepared["lab_id"])
            store = SQLiteStore(home)
            self.assertEqual(read_history(config.state_dir), {
                "schema": 1, "documents": [], "hypotheses": [],
            })
            store.set_document(config.lab_id, "history_catalog", catalog)
            first = read_history(config.state_dir)
            self.assertEqual(first, catalog)
            self.assertEqual(first["hypotheses"][0]["metrics"], {"score": 88})
            self.assertIn("history/NOTES.md", first["documents"])
            self.assertEqual(store.get_document(config.lab_id, "history_catalog"), catalog)
            legacy_id = first["hypotheses"][0]["id"]

            before = status_lab(config)
            self.assertIsNone(before["baseline_id"])
            self.assertIsNone(before["baseline_score"])
            self.assertIsNone(before["best_score"])
            self.assertEqual(before["historical_hypotheses"], 1)
            self.assertEqual(before["history_documents"], 1)

            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--headless", "--config", str(config.path), "status"], labs_root=home)
            self.assertEqual(code, 0, output.getvalue())
            status = json.loads(output.getvalue())["data"]
            self.assertIsNone(status["baseline_score"])
            self.assertIsNone(status["best_score"])
            self.assertEqual(status["historical_hypotheses"], 1)
            self.assertEqual(status["history_documents"], 1)

            rendered = render_unified_dashboard(
                {"lab": str(config.lab_id)},
                {str(config.lab_id): {"historical_hypotheses": [{
                    "id": legacy_id, "title": first["hypotheses"][0]["title"],
                    "source": first["hypotheses"][0]["source"],
                    "artifact": first["hypotheses"][0]["artifact"],
                    "score": "88", "conclusion": first["hypotheses"][0]["conclusion"],
                    "recheck": "Не перепроверено",
                }]}},
                live=False, root_id=str(config.lab_id), selected_lab=str(config.lab_id),
            )
            self.assertIn("Прежние гипотезы", rendered)
            self.assertIn("\\u003cscript\\u003eBad", rendered)
            self.assertNotIn("<script>Bad</script>", rendered)
            self.assertIn("\\u003cimg src=x\\u003e", rendered)
            self.assertIn("артефакт", rendered)
            self.assertIn("history/manifest.json", rendered)
            self.assertIn("Не перепроверено", rendered)
            self.assertEqual(read_history(config.state_dir), first)

    def test_pending_legacy_history_import_is_retired_but_status_is_readable(self):
        note = self.project / "NOTES.md"
        note.write_text("A previous hypothesis", encoding="utf-8")
        info = self.legacy_setup_fixture()
        normalized = self.root / "normalized.json"
        normalized.write_text(json.dumps({"hypotheses": [{"title": "Old", "source": "NOTES.md"}]}), encoding="utf-8")
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertEqual(main(["--config", info["config"], "history-import",
                                   "--file", str(normalized)]), 2)
        import_error = json.loads(errors.getvalue())["error"]
        self.assertEqual(import_error["code"], "E_OPERATION_REMOVED")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--config", info["config"], "status"]), 0)
        config = load_config(Path(info["config"]), allow_pending=True)
        catalog = read_history(config.state_dir)
        self.assertFalse(any(item.get("title") == "Old" for item in catalog["hypotheses"]))
        self.assertEqual(status_lab(config)["missing"], ["evaluator", "metrics.primary"])

    def test_recovery_of_legacy_interrupted_trial_is_refused_without_writes(self):
        info = self.legacy_setup_fixture()
        config = load_config(Path(info["config"]), allow_pending=True)
        extra = config.state_dir / "candidate" / "new.txt"
        extra.write_text("uncommitted", encoding="utf-8")
        state = load_state(config)
        run_dir = config.state_dir / "runs" / "trial-0001"
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "id": "trial-0001", "status": "running", "runs": [],
            "parent": "baseline", "base_commit": state["best_commit"],
        }), encoding="utf-8")
        before = support.tree_digest(config.state_dir)

        with self.assertRaises(LabError) as raised:
            recover_lab(config)

        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(config.state_dir), before)
        self.assertTrue(extra.is_file())
        self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["status"], "running")

    def test_failed_first_baseline_allows_fixing_evaluator_before_success(self):
        info = self.legacy_setup_fixture()
        config, evaluator = self.configure(info)
        raw = json.loads(config.path.read_text(encoding="utf-8"))
        raw["evaluator"][1] = str(self.root / "missing_evaluator.py")
        raw["input_files"] = []
        config.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(LabError):
            run_baseline(load_config(config.path))
        self.assertTrue(load_state(config)["setup_pending"])
        raw["evaluator"][1] = str(evaluator)
        config.path.write_text(json.dumps(raw), encoding="utf-8")
        fixed = load_config(config.path)
        self.assertEqual(run_baseline(fixed)["status"], "accepted")
        self.assertFalse(load_state(fixed)["setup_pending"])

    def test_new_project_notes_after_setup_do_not_change_prepared_snapshot(self):
        info = self.legacy_setup_fixture()
        (self.project / "review_notes.json").write_text("{}", encoding="utf-8")
        config, _ = self.configure(info)
        self.assertEqual(run_baseline(config)["status"], "accepted")
        self.assertFalse((config.state_dir / "candidate" / "review_notes.json").exists())

    def test_setup_baseline_rejects_file_added_before_first_measurement(self):
        (self.project / ".gitignore").write_text("surprise.txt\n", encoding="utf-8")
        info = self.legacy_setup_fixture()
        config, _ = self.configure(info)
        (config.state_dir / "candidate" / "surprise.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(LabError, "changed"):
            run_baseline(config)


if __name__ == "__main__":
    unittest.main()
