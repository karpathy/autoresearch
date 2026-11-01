"""Public CLI flow for a host-attested offline harness profile."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import


_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / ".agents" / "skills" / "autoresearch" / "scripts"
_PREPARE = _SCRIPTS / "prepare.py"
_LAB = _SCRIPTS / "lab.py"


class HarnessPublicFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "measured-project"
        self.project.mkdir()
        self.home = self.root / "lab-home"
        self.host = self.root / "host"
        self.host.mkdir()
        self.marker = self.root / "evaluator-runs.txt"
        self.profile_env_marker = self.root / "profile-env.txt"
        self.program = self.root / "program.md"
        self.program.write_text(
            "Project: measured-project\nGoal: improve the offline score\nMetric: score, max\n",
            encoding="utf-8",
        )
        (self.project / "score.txt").write_text("1\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            f"Path({str(self.profile_env_marker)!r}).write_text('present' if 'AUTORESEARCH_HARNESS_PROFILE' in os.environ else 'absent')\n"
            "score = float((candidate / 'score.txt').read_text(encoding='utf-8'))\n"
            "(run_dir / 'result.json').write_text(json.dumps({"
            "'metrics': {'score': score}, 'usage': {'requests': 0, 'cost_usd': 0}}))\n"
            f"with Path({str(self.marker)!r}).open('a', encoding='utf-8') as marker:\n"
            "    marker.write('evaluated\\n')\n",
            encoding="utf-8",
        )
        self.known_case = self.root / "known-case.json"
        self.known_case.write_text(json.dumps({
            "schema": 1,
            "id": "offline-score-v1",
            "reproducible": True,
            "evaluator_sha256": hashlib.sha256(self.evaluator.read_bytes()).hexdigest(),
            "input_hashes": {},
            "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1},
            "tolerance": 1e-9,
        }), encoding="utf-8")
        self.spec = self.root / "derived-spec.json"
        self.spec.write_text(json.dumps({
            "project": str(self.project),
            "evaluator": ["{python}", str(self.evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 1,
            "timeout_seconds": 30,
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(self.known_case),
                    "sha256": hashlib.sha256(self.known_case.read_bytes()).hexdigest(),
                },
            },
        }), encoding="utf-8")

        self._old_profile = os.environ.get("AUTORESEARCH_HARNESS_PROFILE")
        os.environ.pop("AUTORESEARCH_HARNESS_PROFILE", None)
        self.addCleanup(self._restore_profile_env)
        self.env_patch = patch.dict(os.environ, {
            "AUTORESEARCH_HOME": str(self.home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(_SCRIPTS),
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _restore_profile_env(self):
        if self._old_profile is None:
            os.environ.pop("AUTORESEARCH_HARNESS_PROFILE", None)
        else:
            os.environ["AUTORESEARCH_HARNESS_PROFILE"] = self._old_profile

    def _run(self, script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=_SCRIPTS,
            capture_output=True,
            text=True,
            encoding="utf-8",
            **support.hidden_process_kwargs(),
        )

    def _json(self, result: subprocess.CompletedProcess[str]) -> dict:
        output = result.stdout.strip() or result.stderr.strip()
        try:
            return json.loads(output)
        except ValueError as error:
            self.fail(f"CLI did not return JSON (exit {result.returncode}): {output!r}; {error}")

    def _run_lab(self, config: str, *args: str) -> subprocess.CompletedProcess[str]:
        return self._run(_LAB, ["--config", config, *args])

if __name__ == "__main__":
    unittest.main()
