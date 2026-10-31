from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime.commands import register_all, run_command
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.storage_ops import export_lab_json


class StorageExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.home = self.base / "home"
        self.project = self.base / "project"
        self.project.mkdir()
        program = self.project / "program.md"
        program.write_text("goal\n", encoding="utf-8")
        self.state_dir = self.base / "lab"
        self.candidate = support.git_project(self.state_dir / "candidate", {"value.txt": "1"})
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.candidate, text=True,
            **support.hidden_process_kwargs(),
        ).strip()
        config = {
            "confirmed": True,
            "source_root": str(self.project),
            "state_dir": str(self.state_dir),
            "config_path": str(self.state_dir / "lab.json"),
            "goal": "test SQLite export",
            "repeats": 1,
            "timeout_seconds": 30,
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "evaluator": ["python", "local-evaluator.py"],
            "budget": {"max_trials": 1, "max_cost_usd": 0},
            "pins": {"program": {"path": str(program), "sha256": "a" * 64}},
        }
        state = {"best_commit": commit, "current_epoch": 1, "source": [], "inputs": []}
        self.store = SQLiteStore(self.home)
        self.lab_id = self.store.create_lab(config, state, self.candidate)

    def test_explicit_export_contains_runs_and_artifacts_and_never_overwrites(self):
        run_id = self.store.create_run(self.lab_id, "baseline", "baseline", 1, {"source": "unit test"})
        result = {
            "metrics": {"score": 0.75},
            "usage": {"requests": 1, "cost_usd": 0.0},
            "status": "ok",
        }
        self.store.ensure_repeat_started(self.lab_id, run_id, 1)
        self.store.append_result(self.lab_id, run_id, 1, result)
        self.store.save_manifest(self.lab_id, run_id, {"id": "baseline", "status": "accepted", "runs": [result]})
        artifact_content = b"portable evidence"
        artifact = self.store.register_artifact(
            self.lab_id,
            "evidence.txt",
            artifact_content,
            run_id=run_id,
            evidence_key="results/evidence.txt",
            media_type="text/plain",
        )
        output = self.base / "exports" / "sqlite-lab.json"
        revision = self.store.get_revision()

        exported = export_lab_json(home=self.home, lab=str(self.lab_id), output=output)

        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exported["lab_id"], str(self.lab_id))
        self.assertEqual(base64.b64decode(payload["config_bytes_base64"]), self.store.load_config_bytes(self.lab_id))
        baseline = next(item for item in payload["runs"] if item["alias"] == "baseline")
        self.assertEqual(baseline["repeats"][0]["result"], result)
        exported_artifact = next(item for item in payload["artifacts"] if item["artifact_id"] == str(artifact["artifact_id"]))
        contents = base64.b64decode(exported_artifact["content_base64"])
        self.assertEqual(contents, artifact_content)
        self.assertEqual(hashlib.sha256(contents).hexdigest(), exported_artifact["sha256"])
        self.assertEqual(self.store.get_revision(), revision, "export must not modify SQLite")

        parser = argparse.ArgumentParser()
        register_all(parser.add_subparsers(dest="command", required=True))
        cli_output = self.base / "exports" / "from-cli.json"
        args = parser.parse_args(["storage", "export", "--lab", str(self.lab_id), "--output", str(cli_output)])
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(self.home)}):
            cli_result = run_command(args, None)
        self.assertTrue(cli_output.is_file())
        self.assertEqual(cli_result["lab_id"], str(self.lab_id))
        with self.assertRaises(FileExistsError):
            export_lab_json(home=self.home, lab=str(self.lab_id), output=output)


if __name__ == "__main__":
    unittest.main()
