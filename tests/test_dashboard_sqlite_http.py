"""The dashboard serves a lab created through the real prepare/baseline/trial flow, over HTTP."""

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

import importlib.util
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

from autoresearch_runtime.dash_http import create_server

_DEMO = Path(__file__).resolve().parents[1] / "examples" / "toy" / "run_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("toy_run_demo", _DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DashboardSqliteHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(tempfile.mkdtemp(prefix="autoresearch-dash-"))
        cls.host = Path(tempfile.mkdtemp(prefix="autoresearch-dash-host-"))
        cls.home = cls.base / "lab-home"
        _load_demo().run_flow(cls.base, cls.host)  # prepare, guide, baseline and one trial via the CLI
        cls.env = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(cls.home)})
        cls.env.start()
        cls.server = create_server(cls.home, port=0, stop_token="dash-test")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.env.stop()
        for path in (cls.base, cls.host):
            shutil.rmtree(path, ignore_errors=True)

    def _get(self, path):
        try:
            with urlopen(self.origin + path, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_lab_summary_and_details_come_from_sqlite_over_http(self):
        status, body = self._get("/api/v1/labs")
        self.assertEqual(status, 200, body)
        items = json.loads(body)["data"]["items"]
        self.assertEqual(len(items), 1)
        lab = items[0]
        lab_id = lab["lab_id"]
        summary = lab["dashboard_summary"]
        self.assertEqual(summary["metric_definitions"]["primary"]["name"], "mean_error")
        self.assertIsNotNone(summary["baseline"])
        self.assertIsNotNone(summary["best"])
        self.assertTrue(summary["trend"])

        status, body = self._get(f"/api/v1/labs/{lab_id}/runs")
        self.assertEqual(status, 200, body)
        runs = json.loads(body)["data"]["items"]
        self.assertGreaterEqual(len(runs), 2)
        status, body = self._get(f"/api/v1/labs/{lab_id}/runs/{runs[0]['run_id']}")
        self.assertEqual(status, 200, body)
        self.assertTrue(json.loads(body)["data"])

        status, body = self._get(f"/api/v1/labs/{lab_id}/guide")
        self.assertEqual(status, 200, body)
        self.assertTrue(json.loads(body)["data"]["next_steps"])

    def test_page_loads_and_file_based_routes_are_gone(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<html", body.lower())
        self.assertEqual(self._get("/api/status")[0], 404)
        self.assertEqual(self._get("/api/labs/project/lab")[0], 404)


if __name__ == "__main__":
    unittest.main()
