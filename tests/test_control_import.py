try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autoresearch_runtime.control_import import import_control_report
from autoresearch_runtime.dashboard_server import dashboard_details


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

    def test_control_import_appears_on_dashboard_without_runs_dir(self):
        report_path = Path(self.temp.name) / "control-report.json"
        report_path.write_text(
            json.dumps({"runs": [{"metrics": {"score": 1}}]}),
            encoding="utf-8",
        )

        manifest = import_control_report(
            self.lab,
            report_path,
            result_type="reproducibility_control",
            label="Replay check",
        )
        self.assertEqual(manifest["id"], "control-0001")
        self.assertEqual(manifest["result_type"], "reproducibility_control")
        self.assertEqual(manifest["runs"][0]["metrics"]["score"], 1)

        control_manifest = self.lab / "controls" / "control-0001" / "manifest.json"
        self.assertTrue(control_manifest.is_file())
        self.assertFalse((self.lab / "runs" / "control-0001").exists())
        self.assertEqual(list((self.lab / "runs").glob("control-*")), [])

        details = dashboard_details(self.root, "ControlProject", "lab-control")
        self.assertIsNotNone(details)
        control_rows = [row for row in details["series"] if row["id"] == "control-0001"]
        self.assertEqual(len(control_rows), 1)
        self.assertEqual(control_rows[0]["result_type"], "reproducibility_control")
        self.assertEqual(control_rows[0]["worst"], 1.0)
        self.assertTrue(control_rows[0]["has_value"])


if __name__ == "__main__":
    unittest.main()
