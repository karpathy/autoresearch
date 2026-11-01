try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

import json
import os
import threading
import unittest
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from unittest.mock import patch

from autoresearch_runtime.dash_http import create_server
from autoresearch_runtime.auth import AuthError, authenticate


class DashboardLocalAccessTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(Path(self.temp.name) / "home")})
        env.start()
        self.addCleanup(env.stop)
        self.root = Path(self.temp.name) / "dashboard"
        self.root.mkdir()
        self._create_lab()
        self.server = create_server(self.root, port=0, stop_token="private-stop-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._close_server)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def _create_lab(self):
        project = "Project-1234567890"
        lab = self.root / "projects" / project / "ready-lab"
        (lab / "runs" / "trial-0001").mkdir(parents=True)
        (lab / "lab.json").write_text(json.dumps({
            "source_root": "C:/private/project", "state_dir": str(lab),
            "metrics": {"primary": "score", "direction": "max"},
        }), encoding="utf-8")
        (lab / "state.json").write_text(json.dumps({
            "setup_pending": False, "baseline_id": "baseline", "best_id": "baseline",
        }), encoding="utf-8")
        (lab / "runs" / "trial-0001" / "manifest.json").write_text(json.dumps({
            "id": "trial-0001", "title": "Ready", "status": "accepted", "runs": [],
        }), encoding="utf-8")
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": project, "name": "Project", "labs": [
                {"id": "ready-lab", "path": f"projects/{project}/ready-lab", "status": "completed"},
            ],
        }], "archived": []}), encoding="utf-8")
        self.project = project
        self.lab = lab

    def _close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, path, *, method=None, body=None, headers=None):
        request = Request(self.base + path, method=method, data=body, headers=headers or {})
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            try:
                return error.code, error.headers, error.read()
            finally:
                error.close()

    def test_cross_origin_browser_mutations_are_rejected(self):
        cases = (
            {"Origin": "http://example.test", "Sec-Fetch-Site": "same-origin",
             "Sec-Fetch-Mode": "cors"},
            {"Origin": self.base, "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors"},
            {"Sec-Fetch-Mode": "cors"},
        )
        for headers in cases:
            with self.subTest(headers=headers):
                status, _, _ = self._request("/api/notify", method="POST", body=b"{}", headers=headers)
                self.assertEqual(status, 403)

    def test_originless_local_api_call_and_exact_host_are_supported(self):
        status, _, _ = self._request("/api/notify", method="POST", body=b"{}")
        self.assertEqual(status, 200)

        status, _, _ = self._request("/", headers={"Host": "example.test"})
        self.assertEqual(status, 403)

        headers = Message()
        headers["Host"] = f"127.0.0.1:{self.server.server_port}"
        remote_peer = SimpleNamespace(
            client_address=("198.51.100.4", 51234), server=self.server,
            headers=headers, command="GET",
        )
        with self.assertRaises(AuthError) as failure:
            authenticate(remote_peer)
        self.assertEqual(failure.exception.code, "loopback_required")

    def test_browser_token_issuance_is_removed_but_stop_token_is_retained(self):
        status, _, _ = self._request("/__browser-token", method="POST", body=b"{}", headers={
            "X-AutoResearch-Stop": "private-stop-token",
        })
        self.assertEqual(status, 404)

        status, _, _ = self._request("/__stop", method="POST", body=b"{}")
        self.assertEqual(status, 401)
        status, _, _ = self._request("/__stop", method="POST", body=b"{}", headers={
            "X-AutoResearch-Stop": "wrong-stop-token",
        })
        self.assertEqual(status, 401)

        status, _, _ = self._request("/__stop", method="POST", body=b"{}", headers={
            "X-AutoResearch-Stop": "private-stop-token",
        })
        self.assertEqual(status, 200)
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def test_browser_opens_plain_url_once_and_force_allows_reopening(self):
        from autoresearch_runtime.dash_process import _maybe_open_browser

        marker_root = Path(self.temp.name) / "browser-open"
        marker_root.mkdir()
        info = {"url": self.base}
        result = {}
        with patch("autoresearch_runtime.dash_process.urlopen") as request, \
             patch("autoresearch_runtime.dashboard_server.webbrowser.open", return_value=True) as opener:
            _maybe_open_browser(marker_root, info, result, force_open=False)
            _maybe_open_browser(marker_root, info, result, force_open=False)
            _maybe_open_browser(marker_root, info, result, force_open=True)

        self.assertTrue(result["browser_opened"])
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(opener.call_args_list[0].args[0], self.base)
        self.assertEqual(opener.call_args_list[1].args[0], self.base)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
