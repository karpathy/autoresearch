try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.bootstrap import main as prepare_main, prepare_lab, resume_lab
from autoresearch_runtime.catalog import aggregate_dashboard_status, list_catalogs
from autoresearch_runtime.commands.roots import add_root_command, list_roots
from autoresearch_runtime.dash_api import archive_lab
from autoresearch_runtime.lab import load_config, load_state
from autoresearch_runtime.paths import brief_path, project_id, resolve_home
from autoresearch_runtime.sqlite_store import SQLiteStore
from tests.test_legacy_fixture import materialize


class HomeRootsTest(unittest.TestCase):
    def _restore_home(self):
        if self._previous_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self._previous_home

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.home = self.workspace / "home"
        self.home.mkdir()
        self._previous_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self.project = self.workspace / "project"
        self.project.mkdir()
        (self.project / "value.txt").write_text("1", encoding="utf-8")
        self.evaluator = self.workspace / "evaluator.py"
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

    def _spec(self) -> Path:
        known_case_path = self.workspace / "known-case.json"
        known_case_path.write_text(json.dumps({
            "schema": 1, "id": "fixture-score-v1", "reproducible": True,
            "evaluator_sha256": hashlib.sha256(self.evaluator.read_bytes()).hexdigest(),
            "input_hashes": {}, "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1}, "tolerance": 1e-9,
        }), encoding="utf-8")
        spec_path = self.workspace / "derived.json"
        spec_path.write_text(json.dumps({
            "project": str(self.project),
            "evaluator": ["{python}", str(self.evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {"path": str(known_case_path),
                               "sha256": hashlib.sha256(known_case_path.read_bytes()).hexdigest()},
            },
        }), encoding="utf-8")
        return spec_path

    def test_prepare_requires_program_or_brief(self):
        spec = self._spec()
        self.assertEqual(prepare_main(["--spec", str(spec)]), 2)

    def test_brief_text_without_program_path_prepares_lab(self):
        spec = self._spec()
        brief = "Project: measured\nGoal: improve score\nMetric: score, max\n"
        self.assertEqual(prepare_main([
            "--brief-text", brief, "--spec", str(spec), "--project", str(self.project),
            "--no-browser", "--no-dashboard", "--confirmed",
        ]), 0)
        program = brief_path(self.project)
        self.assertTrue(program.is_file())
        self.assertEqual(program.read_text(encoding="utf-8"), brief)
        prepared = prepare_lab(program, spec, confirmed=True)
        config = load_config(Path(prepared["config"]))
        snapshot = Path(config.state_dir) / "program.snapshot.md"
        self.assertTrue(snapshot.is_file())
        self.assertEqual(snapshot.read_bytes(), program.read_bytes())

    def test_brief_text_budget_requires_an_exact_quote(self):
        spec = self._spec()
        raw = json.loads(spec.read_text(encoding="utf-8"))
        raw["budget"] = {"max_trials": 3}
        raw["budget_sources"] = {"max_trials": "at most 3 trials"}
        spec.write_text(json.dumps(raw), encoding="utf-8")
        base = "Project: measured\nGoal: improve score\nMetric: score, max\n"
        self.assertEqual(prepare_main([
            "--brief-text", base, "--spec", str(spec), "--project", str(self.project),
            "--no-browser", "--no-dashboard",
        ]), 2)
        quoted = base + "Limits: at most 3 trials\n"
        self.assertEqual(prepare_main([
            "--brief-text", quoted, "--spec", str(spec), "--project", str(self.project),
            "--no-browser", "--no-dashboard", "--confirmed",
        ]), 0)
        saved = brief_path(self.project).read_text(encoding="utf-8")
        self.assertIn("at most 3 trials", saved)

    def test_resume_lab_in_registered_extra_root(self):
        # A registered legacy root remains readable, but resume is read-only.
        from autoresearch_runtime.errors import LabError
        from tests.test_legacy_fixture import materialize

        legacy_case = self.workspace / "legacy-extra-root"
        _config_path, program, project = materialize(legacy_case)
        extra = legacy_case / ".labs"
        add_root_command(extra)
        central_store = SQLiteStore(self.home)
        self.assertFalse(central_store.is_initialized())
        self.assertTrue((self.home / "roots.json").is_file())
        self.assertFalse((extra / "autoresearch.sqlite3").exists())
        before_resume = support.tree_digest(legacy_case)

        with self.assertRaises(LabError) as raised:
            resume_lab(program, project)

        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(legacy_case), before_resume)
        self.assertFalse((extra / "autoresearch.sqlite3").exists())
        self.assertFalse(central_store.is_initialized())

    def test_resume_uses_central_sqlite_lab_with_registered_legacy_extra_root(self):
        from tests.test_legacy_fixture import materialize

        legacy_case = self.workspace / "legacy-extra-root"
        _legacy_config, program, project = materialize(legacy_case)
        extra = legacy_case / ".labs"
        add_root_command(extra)

        spec_path = self._spec()
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["project"] = str(project)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        prepared = prepare_lab(program, spec_path, confirmed=True)
        store = SQLiteStore(self.home)
        self.assertTrue(store.is_initialized())
        self.assertEqual(str(store.resolve_lab_id(Path(prepared["config"]))), prepared["lab_id"])
        self.assertFalse((extra / "autoresearch.sqlite3").exists())

        resumed = resume_lab(program, project)

        self.assertIsNotNone(resumed)
        self.assertEqual(Path(resumed["config"]).resolve(), Path(prepared["config"]).resolve())
        self.assertEqual(str(resumed["lab_id"]), prepared["lab_id"])
        self.assertFalse(resumed["created"])

    def test_legacy_archive_is_read_only_and_preserves_lab(self):
        legacy_case = self.workspace / "legacy-case"
        config_path, _program, _project = materialize(legacy_case)
        extra = config_path.parent.parent.parent.parent
        state_dir = config_path.parent
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        state["setup_pending"] = False
        state["baseline_id"] = None
        (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        # This test exercises the old catalog-backed archive path explicitly.
        for manifest_path in (state_dir / "runs").glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "crash"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        for active_path in (extra / ".active").glob("*.json"):
            active_path.unlink()
        catalog = json.loads((extra / "catalog.json").read_text(encoding="utf-8"))
        project_row = catalog["projects"][0]
        lab_row = project_row["labs"][0]
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        add_root_command(extra)
        status, payload = archive_lab(self.home, project_row["id"], lab_row["id"])
        self.assertEqual(status, 409)
        self.assertEqual(payload["code"], "E_LEGACY_READ_ONLY")
        archived_dir = extra / "archive" / project_row["id"] / state_dir.name
        self.assertFalse(archived_dir.exists())
        self.assertTrue(state_dir.is_dir())

    def test_archive_explicitly_rejects_sqlite_backed_lab(self):
        program = self.workspace / "sqlite-program.md"
        program.write_text("Goal: score\n", encoding="utf-8")
        prepared = prepare_lab(program, self._spec(), confirmed=True)
        config_path = Path(prepared["config"])
        store = SQLiteStore(self.home)
        lab_id = store.resolve_lab_id(config_path)
        status, payload = archive_lab(self.home, project_id(self.project), str(lab_id))
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "unsupported_db_lab_action")
        self.assertEqual(str(store.get_lab(lab_id)["lab_id"]), str(lab_id))
        self.assertFalse(config_path.exists())

    def test_list_roots_and_merged_catalog(self):
        extra = self.workspace / "extra-labs"
        extra.mkdir()
        add_root_command(extra)
        roots = list_roots()
        self.assertEqual(len(roots["roots"]), 2)
        catalogs = list_catalogs(self.home)
        self.assertEqual(len(catalogs["roots"]), 2)
        summary = aggregate_dashboard_status(self.home)
        self.assertIn("home_root_id", summary)

    def test_adoption_uses_program_directory_and_respects_explicit_home(self):
        from autoresearch_runtime.paths import adopt_legacy_stores

        legacy = self.workspace / "Old Project" / ".labs"
        (legacy / "projects").mkdir(parents=True)
        # AUTORESEARCH_HOME is set by setUp, so nothing is adopted implicitly ...
        self.assertEqual(adopt_legacy_stores(self.home, extra_seeds=[legacy.parent]), [])
        # ... and the default-home path (env unset) sees labs next to program.md.
        del os.environ["AUTORESEARCH_HOME"]
        try:
            adopted = adopt_legacy_stores(self.home, extra_seeds=[legacy.parent])
        finally:
            os.environ["AUTORESEARCH_HOME"] = str(self.home)
        # The current directory may also hold an old `.labs`; only the one next to program.md is asserted.
        self.assertIn(legacy.resolve(), [Path(item["path"]).resolve() for item in adopted])

    def test_legacy_labs_directory_is_registered_without_moving_files(self):
        from autoresearch_runtime.paths import adopt_legacy_stores

        legacy = self.workspace / "Old Project" / ".labs"
        (legacy / "projects").mkdir(parents=True)
        (legacy / "catalog.json").write_text('{"schema": 1, "projects": [], "archived": []}\n', encoding="utf-8")
        marker = legacy / "projects" / "keep.txt"
        marker.write_text("stay", encoding="utf-8")
        adopted = adopt_legacy_stores(self.home, seeds=[legacy.parent])
        self.assertEqual(len(adopted), 1)
        self.assertEqual(Path(adopted[0]["path"]).resolve(), legacy.resolve())
        self.assertEqual(marker.read_text(encoding="utf-8"), "stay")
        summary = aggregate_dashboard_status(self.home)
        self.assertTrue(any(Path(item["path"]).resolve() == legacy.resolve() for item in list_catalogs(self.home)["roots"]))
        self.assertIn("home_root_id", summary)


if __name__ == "__main__":
    unittest.main()
