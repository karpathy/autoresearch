from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.bootstrap import prepare_lab
from autoresearch_runtime.lab import load_config
from autoresearch_runtime.sqlite_store import SQLiteStore, StoreIntegrityError
from autoresearch_runtime.store import _write_json


class SQLiteVirtualPathIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="sqlite-path-isolation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / ".labs"
        self._previous_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)

        self.program = self.root / "program.md"
        self.program.write_text(
            "Project: SQLite path isolation fixture\n"
            "Goal: preserve lab state boundaries\n"
            "Metric: score, max\n",
            encoding="utf-8",
        )
        self.evaluator = self.root / "evaluator.py"
        # Preparation pins this file but never executes the evaluator.
        self.evaluator.write_text("raise AssertionError('evaluator must not run')\n", encoding="utf-8")
        self.store = SQLiteStore(self.home)

    def _restore_home(self) -> None:
        if self._previous_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self._previous_home

    def _prepare_lab(self, name: str):
        project = self.root / f"project-{name}"
        project.mkdir()
        (project / "value.txt").write_text(name, encoding="utf-8")
        reference = project / "reference.txt"
        reference.write_text(f"frozen input for {name}\n", encoding="utf-8")
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        spec_path = self.root / f"spec-{name}.json"
        spec_path.write_text(json.dumps({
            "project": project.name,
            "evaluator": ["{python}", str(self.evaluator)],
            "input_files": [f"{project.name}/reference.txt"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 1,
            "baseline_validation": {
                "mode": "historical",
                "tolerance": 1e-9,
                "input_hashes": {f"{project.name}/reference.txt": reference_hash},
                "runs": [{"name": "recorded", "metrics": {"score": 1}}],
            },
        }), encoding="utf-8")
        prepared = prepare_lab(self.program, spec_path, confirmed=True)
        return load_config(Path(prepared["config"]), allow_pending=True)

    def _prepare_labs(self):
        return {name: self._prepare_lab(name) for name in ("A", "B")}

    def _create_state_link(self, link: Path, target: Path) -> None:
        target.mkdir(parents=True)
        if os.name == "nt":
            result = subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.addCleanup(os.rmdir, link)
        else:
            link.symlink_to(target, target_is_directory=True)
            self.addCleanup(link.unlink)

    def test_reparse_path_cannot_redirect_virtual_state_write_to_another_lab(self):
        configs = self._prepare_labs()
        config_a, config_b = configs["A"], configs["B"]
        path_a = config_a.state_dir / "state.json"
        target_b = config_b.state_dir / "state.json"
        self._create_state_link(path_a, target_b)

        before_a = self.store.load_state(config_a.lab_id)
        before_b = self.store.load_state(config_b.lab_id)
        target_entries_before = sorted(item.name for item in target_b.iterdir())
        resolved_target_before = path_a.resolve(strict=True)
        payload = {**before_a, "path_isolation_marker": "must-not-reach-lab-B"}
        caught = None
        try:
            _write_json(path_a, payload)
        except StoreIntegrityError as error:
            caught = error

        after_a = self.store.load_state(config_a.lab_id)
        after_b = self.store.load_state(config_b.lab_id)
        with self.subTest("typed fail-closed error"):
            self.assertIsInstance(caught, StoreIntegrityError)
        with self.subTest("Lab A state remains unchanged"):
            self.assertEqual(after_a, before_a)
        with self.subTest("Lab B state remains unchanged"):
            self.assertEqual(after_b, before_b)
        with self.subTest("filesystem alias remains unchanged"):
            self.assertTrue(path_a.is_dir())
            self.assertEqual(path_a.resolve(strict=True), resolved_target_before)
            self.assertEqual(resolved_target_before, target_b.resolve(strict=True))
            self.assertEqual(sorted(item.name for item in target_b.iterdir()), target_entries_before)

    def test_normal_virtual_state_write_updates_only_owning_lab(self):
        configs = self._prepare_labs()
        config_a, config_b = configs["A"], configs["B"]
        before_a = self.store.load_state(config_a.lab_id)
        before_b = self.store.load_state(config_b.lab_id)
        payload = {**before_a, "path_isolation_marker": "lab-A"}

        _write_json(config_a.state_dir / "state.json", payload)

        self.assertEqual(self.store.load_state(config_a.lab_id), payload)
        self.assertEqual(self.store.load_state(config_b.lab_id), before_b)
        self.assertFalse((config_a.state_dir / "state.json").exists())
        self.assertFalse((config_b.state_dir / "state.json").exists())

    def test_reparse_run_alias_cannot_redirect_virtual_manifest_within_lab(self):
        config = self._prepare_lab("A")
        state = self.store.load_state(config.lab_id)
        run_one = self.store.create_run(
            config.lab_id, "trial", "run-one", state["current_epoch"], {},
        )
        run_two = self.store.create_run(
            config.lab_id, "trial", "run-two", state["current_epoch"], {},
        )
        manifest_one = {"id": "run-one", "status": "created", "runs": [], "marker": "one"}
        manifest_two = {"id": "run-two", "status": "created", "runs": [], "marker": "two"}
        self.store.save_manifest(config.lab_id, run_one, manifest_one)
        self.store.save_manifest(config.lab_id, run_two, manifest_two)

        alias_dir = config.state_dir / "runs" / "run-one"
        target_dir = config.state_dir / "runs" / "run-two"
        self._create_state_link(alias_dir, target_dir)
        manifest_path = alias_dir / "manifest.json"
        resolved_target_before = alias_dir.resolve(strict=True)
        target_entries_before = sorted(item.name for item in target_dir.iterdir())
        run_ids_before = (
            self.store.resolve_run_id(config.lab_id, "run-one"),
            self.store.resolve_run_id(config.lab_id, "run-two"),
        )
        before_one = self.store.load_manifest(config.lab_id, run_one)
        before_two = self.store.load_manifest(config.lab_id, run_two)
        payload = {**manifest_one, "marker": "must-not-reach-run-two"}
        caught = None
        try:
            _write_json(manifest_path, payload)
        except StoreIntegrityError as error:
            caught = error

        with self.subTest("typed fail-closed error"):
            self.assertIsInstance(caught, StoreIntegrityError)
        with self.subTest("run-one manifest remains unchanged"):
            self.assertEqual(self.store.load_manifest(config.lab_id, run_one), before_one)
        with self.subTest("run-two manifest remains unchanged"):
            self.assertEqual(self.store.load_manifest(config.lab_id, run_two), before_two)
        with self.subTest("run aliases remain unchanged"):
            self.assertEqual((
                self.store.resolve_run_id(config.lab_id, "run-one"),
                self.store.resolve_run_id(config.lab_id, "run-two"),
            ), run_ids_before)
        with self.subTest("filesystem alias and target remain unchanged"):
            self.assertTrue(alias_dir.is_dir())
            self.assertEqual(alias_dir.resolve(strict=True), resolved_target_before)
            self.assertEqual(resolved_target_before, target_dir.resolve(strict=True))
            self.assertEqual(sorted(item.name for item in target_dir.iterdir()), target_entries_before)
            self.assertFalse(manifest_path.exists())
            self.assertFalse((target_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
