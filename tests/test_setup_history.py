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
    status_lab, main, recover_lab,
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

    def test_project_command_without_setup_gives_next_step(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main(["--project", str(self.project), "status"], labs_root=self.root / ".labs"), 2)
        self.assertIn("prepare", errors.getvalue().casefold())
        self.assertNotIn("setup --project", errors.getvalue().casefold())

if __name__ == "__main__":
    unittest.main()
