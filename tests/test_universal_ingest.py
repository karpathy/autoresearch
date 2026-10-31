import json
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.errors import LabError
from autoresearch_runtime.ingest import ingest_reports


class UniversalIngestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir(parents=True)
        (self.src_dir / "app.py").write_text("print('hello')", encoding="utf-8")
        self.state_dir = self.root / "lab_state"
        self.state_dir.mkdir(parents=True)
        (self.state_dir / "lab.json").write_text(json.dumps({
            "source_root": str(self.src_dir),
            "state_dir": str(self.state_dir),
            "metrics": {"primary": "latency_ms", "direction": "min", "guards": []},
            "evaluator": ["python", "eval.py"],
            "input_files": [],
            "repeats": 2,
            "timeout_seconds": 120,
            "budget": {"max_seconds": None, "max_trials": None, "max_requests": 0, "max_cost_usd": 0},
        }), encoding="utf-8")
        (self.state_dir / "state.json").write_text(json.dumps({
            "schema": 2, "best_id": None, "baseline_id": None, "inputs": []
        }), encoding="utf-8")

    def test_ingest_flat_manifest_json(self):
        source = self.root / "manifest_run.json"
        source.write_text(json.dumps({
            "id": "external-01",
            "metrics": {"latency_ms": 120.5, "accuracy": 0.95},
            "commit": "abc1234",
        }), encoding="utf-8")

        written = ingest_reports(self.state_dir, source, result_type="experiment")
        self.assertEqual(len(written), 1)
        summary = json.loads(written[0].read_text(encoding="utf-8"))
        self.assertEqual(summary["id"], "probe-0001")
        self.assertEqual(summary["primary"], "latency_ms")
        self.assertEqual(summary["direction"], "min")
        self.assertEqual(summary["worst"], 120.5)
        self.assertEqual(summary["commit"], "abc1234")

    def test_ingest_csv_table(self):
        source = self.root / "benchmark.csv"
        source.write_text(
            "run_id,latency_ms,accuracy\n"
            "run_01,150.0,92%\n"
            "run_02,130.0,94%\n",
            encoding="utf-8",
        )

        written = ingest_reports(self.state_dir, source, result_type="provider_check")
        self.assertEqual(len(written), 1)
        summary = json.loads(written[0].read_text(encoding="utf-8"))
        self.assertEqual(summary["repeats"], 2)
        # For min direction, worst should be max(150, 130) = 150.0
        self.assertEqual(summary["worst"], 150.0)
        self.assertEqual(summary["mean"], 140.0)

    def test_ingest_markdown_table(self):
        source = self.root / "results.md"
        source.write_text(
            "# A/B Test Results\n\n"
            "| Step | latency_ms | accuracy |\n"
            "| :--- | :--- | :--- |\n"
            "| Rep 1 | 142.5 | 0.96 |\n"
            "| Rep 2 | 138.0 | 0.97 |\n",
            encoding="utf-8",
        )

        written = ingest_reports(self.state_dir, source, result_type="experiment")
        self.assertEqual(len(written), 1)
        summary = json.loads(written[0].read_text(encoding="utf-8"))
        self.assertEqual(summary["repeats"], 2)
        self.assertEqual(summary["worst"], 142.5)  # min direction -> max is worst

    def test_no_clobber_existing_probe(self):
        # Existing probe-0001
        p1 = self.state_dir / "probes" / "probe-0001"
        p1.mkdir(parents=True)
        (p1 / "summary.json").write_text('{"id": "probe-0001", "keep": true}', encoding="utf-8")

        source = self.root / "new_run.json"
        source.write_text(json.dumps({
            "metrics": {"latency_ms": 110.0}
        }), encoding="utf-8")

        written = ingest_reports(self.state_dir, source, result_type="experiment")
        self.assertEqual(len(written), 1)
        # probe-0001 must remain untouched
        old_data = json.loads((p1 / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(old_data.get("keep"))
        # New probe must be probe-0002
        new_data = json.loads(written[0].read_text(encoding="utf-8"))
        self.assertEqual(new_data["id"], "probe-0002")

    def test_invalid_state_dir_raises(self):
        invalid_dir = self.root / "non_existent"
        source = self.root / "dummy.json"
        source.write_text("{}", encoding="utf-8")
        with self.assertRaises(LabError):
            ingest_reports(invalid_dir, source)


if __name__ == "__main__":
    unittest.main()
