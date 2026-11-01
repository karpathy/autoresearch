try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autoresearch_runtime.control_import import import_control_report


class ControlImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / ".labs"
        self.lab = self.root / "projects" / "ControlProject" / "lab-control"
        self.lab.mkdir(parents=True)
        (self.lab / "runs" / "baseline").mkdir(parents=True)
        (self.lab / "lab.json").write_text(
            json.dumps(
                {
                    "source_root": "C:/private/project",
                    "state_dir": str(self.lab),
                    "evaluator": ["noop"],
                    "metrics": {"primary": "score", "direction": "max"},
                    "budget": {"max_requests": None, "max_cost_usd": 2},
                }
            ),
            encoding="utf-8",
        )
        (self.lab / "state.json").write_text(
            json.dumps({"setup_pending": False, "baseline_id": "baseline", "best_id": "baseline"}),
            encoding="utf-8",
        )
        (self.lab / "runs" / "baseline" / "manifest.json").write_text(
            json.dumps(
                {
                    "id": "baseline",
                    "title": "Baseline",
                    "status": "accepted",
                    "runs": [{"metrics": {"score": 0.5}}],
                }
            ),
            encoding="utf-8",
        )

if __name__ == "__main__":
    unittest.main()
