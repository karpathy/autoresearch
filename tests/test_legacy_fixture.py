try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
"""A lab produced by the pre-refactor runtime must stay readable."""

import hashlib
import contextlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from autoresearch_runtime.lab import LabError, load_config, replay_plan, status_lab
from autoresearch_runtime.bootstrap import resume_lab


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "legacy-lab.zip"


def materialize(destination: Path) -> tuple[Path, Path, Path]:
    """Extract the tokenized fixture archive and point its absolute paths at destination."""
    zipfile.ZipFile(FIXTURE).extractall(destination)
    for gitdir in destination.rglob("_gitrepo"):
        if gitdir.is_dir() and not (gitdir.parent / ".git").exists():
            gitdir.rename(gitdir.parent / ".git")
    root = str(destination.resolve())
    # JSON stores Windows separators as escaped backslashes, so a raw path would be invalid.
    escaped = json.dumps(root)[1:-1]
    for path in destination.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        if "{ROOT}" not in text:
            continue
        if os.name != "nt" and path.suffix == ".json":
            # These fixtures were captured on Windows; render their encoded
            # path separators for the host before substituting the temp root.
            text = "".join(
                line.replace(r"\\", "/") if "{ROOT}" in line else line
                for line in text.splitlines(keepends=True)
            )
        replacement = escaped if path.suffix == ".json" else root
        path.write_text(text.replace("{ROOT}", replacement), encoding="utf-8")
    lab_json = next(destination.glob(".labs/projects/*/*/lab.json"))
    digest = hashlib.sha256(lab_json.read_bytes()).hexdigest()
    state_path = lab_json.parent / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config_sha256"] = digest
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    program = (destination / "program.md").resolve()
    project = (destination / "project").resolve()
    active_dir = destination / ".labs" / ".active"
    from autoresearch_runtime.paths import project_id
    pointer = active_dir / f"{project_id(project)}.json"
    current = next(active_dir.glob("*.json"))
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["program"] = str(program)
    payload["config"] = str(lab_json.resolve())
    if current != pointer:
        current.unlink()
    pointer.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lab_json, program, project


class LegacyFixtureTest(unittest.TestCase):
    def _restore_home(self):
        if self._previous_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self._previous_home

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = self.root / "case"
        self._previous_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.case / ".labs")
        self.addCleanup(self._restore_home)
        self.lab_json, self.program, self.project = materialize(self.case)

    def test_materialized_source_root_is_a_native_existing_path(self):
        config = json.loads(self.lab_json.read_text(encoding="utf-8"))
        source_root = Path(config["source_root"])
        self.assertEqual(source_root.resolve(), self.project.resolve())
        self.assertTrue(source_root.is_dir())

    def test_status_replay_plan_resume_and_dashboard_summary(self):
        from autoresearch_runtime.dashboard_server import dashboard_status

        config = load_config(self.lab_json)
        status = status_lab(config)
        self.assertEqual(status["baseline_id"], "baseline")
        self.assertTrue(status["baseline_verified"])
        self.assertEqual(status["best_id"], "trial-0001")
        self.assertFalse(status["setup_pending"])
        self.assertEqual(replay_plan(config)["gaps"], [])
        before_resume = support.tree_digest(self.case)
        with self.assertRaises(LabError) as raised:
            resume_lab(self.program, self.project)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.case), before_resume)
        from autoresearch_runtime.bootstrap import main as prepare_main
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertEqual(prepare_main([
                "--program", str(self.program), "--project", str(self.project),
                "--resume", "--no-dashboard",
            ]), 2)
        self.assertEqual(json.loads(errors.getvalue())["error"]["code"], "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.case), before_resume)
        summary = dashboard_status(self.case / ".labs")
        lab = summary["projects"][0]["labs"][0]
        self.assertEqual(lab["status"], "completed")
        self.assertEqual(lab["metric"], "score")

    def test_prepare_resume_refuses_pending_legacy_without_rewriting_state(self):
        raw = json.loads(self.lab_json.read_text(encoding="utf-8"))
        raw["setup_pending"] = True
        self.lab_json.write_text(json.dumps(raw), encoding="utf-8")
        state_path = self.lab_json.parent / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["setup_pending"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")
        before = support.tree_digest(self.case)

        from autoresearch_runtime.bootstrap import main as prepare_main
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            code = prepare_main([
                "--program", str(self.program), "--project", str(self.project),
                "--resume", "--no-dashboard",
            ])

        self.assertEqual(code, 2)
        self.assertEqual(json.loads(errors.getvalue())["error"]["code"], "E_LEGACY_READ_ONLY")
        self.assertEqual(support.tree_digest(self.case), before)
