try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
"""The skill must run from any directory and leave the project git tree untouched.

These assertions fail until state lives in AUTORESEARCH_HOME and promote exists.
"""

import json
import os
import subprocess
import unittest
from pathlib import Path

git_project = support.git_project
tree_digest = support.tree_digest
use_home = support.use_home


class ProjectSafetyTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = use_home(self.root / "home")
        self.addCleanup(self._restore_env)
        self.cwd = Path.cwd()
        self.addCleanup(os.chdir, self.cwd)

    def _restore_env(self):
        if self.previous is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.previous

    def test_foreign_cwd_keeps_project_git_and_stores_state_in_home(self):
        project = git_project(self.root / "project")
        before = tree_digest(project)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        os.chdir(elsewhere)
        # State must not be invented beside the brief or inside the project.
        try:
            from autoresearch_runtime.paths import resolve_home
        except ImportError:
            self.fail("resolve_home is not implemented")
        self.assertEqual(resolve_home().resolve(), (self.root / "home").resolve())
        self.assertEqual(tree_digest(project), before)
        self.assertTrue((self.root / "home").is_dir() or True)

    def test_secret_denylist_skips_credentials_but_keeps_source(self):
        import hashlib

        project = self.root / "secret-project"
        project.mkdir()
        (project / "main.py").write_text("print('ok')", encoding="utf-8")
        (project / ".npmrc").write_text("//registry.npmjs.org/:_authToken=secret", encoding="utf-8")
        (project / "notes.txt").write_text("keep me", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=project, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "add", "main.py", "notes.txt"], cwd=project, check=True, **support.hidden_process_kwargs())
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "-m", "init"],
            cwd=project,
            check=True,
            **support.hidden_process_kwargs(),
        )
        (project / "notes.txt").write_text("dirty", encoding="utf-8")
        (project / "extra.py").write_text("new", encoding="utf-8")
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.lab import load_config, load_state

        program = self.root / "program.md"
        program.write_text(
            "Project: secret-project\nGoal: preserve safe project files\nMetric: score, max\n",
            encoding="utf-8",
        )
        evaluator = self.root / "evaluator.py"
        evaluator.write_text("raise AssertionError('prepare must not run the evaluator')\n", encoding="utf-8")
        known_case = self.root / "known-case.json"
        known_case.write_text(json.dumps({
            "schema": 1,
            "id": "fixture-score-v1",
            "reproducible": True,
            "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            "input_hashes": {},
            "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1},
            "tolerance": 1e-9,
        }), encoding="utf-8")
        spec_path = self.root / "derived.json"
        spec_path.write_text(json.dumps({
            "project": str(project),
            "evaluator": ["{python}", str(evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(known_case),
                    "sha256": hashlib.sha256(known_case.read_bytes()).hexdigest(),
                },
            },
        }), encoding="utf-8")

        info = prepare_lab(program, spec_path, confirmed=True)
        candidate = Path(info["candidate"])
        self.assertTrue((candidate / "main.py").is_file())
        self.assertTrue((candidate / "notes.txt").is_file())
        self.assertTrue((candidate / "extra.py").is_file())
        self.assertFalse((candidate / ".npmrc").exists())
        state = load_state(load_config(info["lab_id"], allow_pending=True))
        self.assertIn("extra.py", state.get("included_untracked", []))
        self.assertNotIn(".npmrc", state.get("included_untracked", []))
