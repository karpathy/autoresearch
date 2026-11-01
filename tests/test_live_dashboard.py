try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import json
import contextlib
import io
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from unittest.mock import patch

from autoresearch_runtime.dashboard_server import (
    create_server, dashboard_server_status, start_dashboard, stop_dashboard,
)
from autoresearch_runtime.lab import main as lab_main


_TEST_DASHBOARD_TOKEN = "dashboard-test-token"


def _test_server(root, *, port=0):
    return create_server(root, port=port, stop_token=_TEST_DASHBOARD_TOKEN)


def _local_request(url, *, method=None, data=None, headers=None):
    return Request(url, method=method, data=data, headers=dict(headers or {}))



class LiveDashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / ".labs"
        home_env = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(Path(self.temp.name) / "sqlite-home")})
        home_env.start()
        self.addCleanup(home_env.stop)
        lab = self.root / "projects" / "sample-project-1234567890" / "contract-a"
        (lab / "runs" / "trial-0001" / "repeat_01").mkdir(parents=True)
        (lab / "lab.json").write_text(json.dumps({
            "source_root": "C:/private/project", "state_dir": str(lab),
            "evaluator": ["contains-no-secret"], "metrics": {"primary": "f1", "direction": "max"},
            "budget": {"max_requests": None, "max_cost_usd": 2},
        }), encoding="utf-8")
        (lab / "state.json").write_text(json.dumps({
            "setup_pending": False, "baseline_id": "baseline", "best_id": "baseline",
        }), encoding="utf-8")
        (lab / "runs" / "trial-0001" / "manifest.json").write_text(json.dumps({
            "id": "trial-0001", "title": "Test", "status": "running", "runs": [],
        }), encoding="utf-8")
        (lab / "runs" / "trial-0001" / "repeat_01" / "usage.json").write_text(json.dumps({
            "requests": 3, "cost_usd": 0.42,
        }), encoding="utf-8")

    def test_not_modified_revision_clears_stale_connection_warning(self):
        import shutil

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is needed to execute the dashboard polling function")
        asset = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts" / "autoresearch_runtime" / "dashboard_assets" / "live_app.js"
        source = asset.read_text(encoding="utf-8")
        start = source.index("async function refreshRevision()")
        end = source.index("\nfunction scheduleRefresh()", start)
        refresh_revision = source[start:end]
        mark_live_start = source.find("function markLive()")
        if mark_live_start >= 0:
            mark_live_end = source.index("\n}", mark_live_start) + 2
            mark_live = source[mark_live_start:mark_live_end]
        else:
            mark_live = (
                "function markLive() { updated.classList.add('is-live'); "
                "updated.classList.remove('is-down'); modeText.textContent = 'В эфире'; "
                "setMessage('', 'info'); }"
            )
        harness = f"""
let revision = '1196';
let refreshing = false;
let sourceWarning = '';
let sourcesUnavailable = false;
let bannerText = 'Нет связи с локальным API';
const classes = new Set(['is-down']);
const updated = {{classList: {{add: name => classes.add(name), remove: name => classes.delete(name)}}}};
const modeText = {{textContent: 'Нет связи'}};
const fetch = async () => ({{status: 304}});
let reloadCount = 0;
const reloadLabs = async () => {{ reloadCount++; }};
const setMessage = text => {{ bannerText = text; }};
const AUTH_REQUIRED_MESSAGE = 'session required';
{mark_live}
{refresh_revision}
refreshRevision().then(() => process.stdout.write(JSON.stringify({{
  mode: modeText.textContent, banner: bannerText,
  live: classes.has('is-live'), down: classes.has('is-down'), reloadCount
}}))).catch(error => {{ console.error(error); process.exitCode = 1; }});
"""
        result = subprocess.run(
            [node, "-e", harness], capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        view = json.loads(result.stdout)
        self.assertEqual(view, {"mode": "В эфире", "banner": "", "live": True, "down": False, "reloadCount": 1})

    def test_run_chart_renders_with_dom_append_semantics(self):
        import shutil

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is needed to execute the dashboard chart renderer")
        asset = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts" / "autoresearch_runtime" / "dashboard_assets" / "live_app.js"
        source = asset.read_text(encoding="utf-8")
        start = source.index("function svgElement(")
        end = source.index("function renderRuns()", start)
        renderer = source[start:end]
        harness = f"""
function element(tag) {{
  return {{tag, children: [], attributes: {{}}, classList: {{add() {{}}}},
    append(...children) {{ this.children.push(...children); }},
    setAttribute(name, value) {{ this.attributes[name] = value; }}, textContent: ''}};
}}
const document = {{createElementNS: (_namespace, tag) => element(tag)}};
const detail = element('main');
const guide = {{current_epoch: 1, state: {{primary_metric: 'Qmax', epoch: 1,
  baseline: {{id: 'base', expected_repeats: 1}}, best: {{score: 0.8}}}}}};
const runMetrics = [
  {{run_id: 'trial', alias: 'trial-1', kind: 'trial', status: 'accepted', epoch_id: 1,
    parent_run_id: 'base', data: {{metrics: {{Qmax: {{decision_worst: 0.8, completed_repeats: 1}}}}}}}},
  {{run_id: 'base', alias: 'base', kind: 'baseline', status: 'accepted', epoch_id: 1,
    parent_run_id: null, data: {{metrics: {{Qmax: {{decision_worst: 0.68, completed_repeats: 1}}}}}}}}
];
const selectedLab = () => ({{primary_metric: 'Qmax'}});
const summaryFor = () => guide.state;
const metricDefinitions = (_summary, lab) => [{{name: lab.primary_metric, role: 'primary'}}];
const section = (title) => Object.assign(element('section'), {{title}});
const node = (tag, cls, value) => Object.assign(element(tag), {{className: cls, textContent: value ?? ''}});
{renderer}
renderRunCharts();
const host = detail.children[0];
const visit = (current, out) => {{ out.push(current); for (const child of current.children || []) visit(child, out); }};
const all = []; visit(host, all);
process.stdout.write(JSON.stringify({{
  host: host.title,
  svgCount: all.filter(item => item.tag === 'svg').length,
  pointCount: all.filter(item => item.tag === 'circle').length,
  labels: all.filter(item => item.tag === 'text').map(item => item.textContent)
}}));
"""
        result = subprocess.run(
            [node, "-e", harness], capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        view = json.loads(result.stdout)
        self.assertEqual(view["host"], "Метрики запусков")
        self.assertEqual(view["svgCount"], 1)
        self.assertEqual(view["pointCount"], 2)
        self.assertEqual(view["labels"], ["раньше", "позже"])

    def test_browser_token_and_bootstrap_endpoints_are_removed(self):
        home = Path(self.temp.name) / "bootstrap-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
            server = _test_server(home, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
            base = f"http://127.0.0.1:{server.server_port}"
            issue = Request(base + "/__browser-token", method="POST", data=b"{}",
                            headers={"X-AutoResearch-Stop": _TEST_DASHBOARD_TOKEN,
                                     "Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as cross_site:
                urlopen(issue)
            self.assertEqual(cross_site.exception.code, 404)
            bootstrap = Request(base + "/api/v1/auth/bootstrap", method="POST", data=b"{}",
                                headers={"Origin": base, "Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as missing:
                urlopen(bootstrap)
            self.assertEqual(missing.exception.code, 405)
            self.assertIn("method_not_allowed", missing.exception.read().decode("utf-8"))

    def test_server_reuse_rejects_same_root_health_from_a_different_process(self):
        from autoresearch_runtime import dash_process

        home = Path(self.temp.name) / "identity-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
            home.mkdir(parents=True)
            server = _test_server(home, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
            info = {
                "host": "127.0.0.1", "port": server.server_port,
                "url": f"http://127.0.0.1:{server.server_port}/", "token": "foreign-token",
                "pid": os.getpid() + 100000, "root_id": server.root_id,
                "home_id": server.home_id, "service_id": server.service_id,
            }
            (home / ".dashboard-server.json").write_text(json.dumps(info), encoding="utf-8")

            self.assertIsNone(dash_process._server_info(home))
            info.update(pid=os.getpid(), service_id="foreign-service")
            (home / ".dashboard-server.json").write_text(json.dumps(info), encoding="utf-8")
            self.assertIsNone(dash_process._server_info(home))
            info.update(service_id=server.service_id, home_id="foreign-home")
            (home / ".dashboard-server.json").write_text(json.dumps(info), encoding="utf-8")
            self.assertIsNone(dash_process._server_info(home))

    def test_explicit_open_launches_the_plain_url_on_every_request_and_never_otherwise(self):
        home = Path(self.temp.name) / "reopen-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
            home.mkdir(parents=True)
            server = _test_server(home, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
            info = {
                "host": "127.0.0.1", "port": server.server_port,
                "url": f"http://127.0.0.1:{server.server_port}/", "token": _TEST_DASHBOARD_TOKEN,
                "pid": os.getpid(), "root_id": server.root_id,
                "home_id": server.home_id, "service_id": server.service_id,
            }
            (home / ".dashboard-server.json").write_text(json.dumps(info), encoding="utf-8")
            with patch("autoresearch_runtime.dash_process.webbrowser.open", return_value=True) as opener:
                first = start_dashboard(home, open_browser=True)
                second = start_dashboard(home, open_browser=True)
                quiet = start_dashboard(home)

            self.assertEqual(first["pid"], os.getpid())
            self.assertTrue(first["browser_opened"])
            self.assertTrue(second["browser_opened"])
            self.assertFalse(quiet["browser_opened"])
            self.assertEqual(opener.call_count, 2)
            self.assertEqual(opener.call_args_list[0].args[0], info["url"])
            self.assertEqual(opener.call_args_list[1].args[0], info["url"])

    def test_live_api_revision_reads_absent_store_without_creating_database(self):
        home = Path(self.temp.name) / "revision-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
            server = _test_server(home, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
            request = Request(f"http://127.0.0.1:{server.server_port}/api/v1/revision")
            with urlopen(request) as response:
                data = json.loads(response.read())
                self.assertEqual(response.headers.get("ETag"), '"0"')
            self.assertEqual(data["data"], {"revision": "0", "initialized": False})
            self.assertFalse((home / "autoresearch.sqlite3").exists())

    def test_cli_commands_never_start_the_dashboard(self):
        from autoresearch_runtime import cli

        home = Path(self.temp.name) / "no-autostart-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}), \
             patch.object(cli, "run_command", return_value={"ran": True}), \
             patch("autoresearch_runtime.dash_process.subprocess.Popen") as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lab_main(["status"]), 0)
            launch.assert_not_called()
        self.assertFalse((home / ".dashboard-server.json").exists())

    def test_dashboard_start_token_stays_out_of_argv_and_browser_failure_can_retry(self):
        from autoresearch_runtime import dash_process
        from unittest.mock import MagicMock

        root = Path(self.temp.name) / "token-home"
        child = MagicMock()
        child.pid = 4444
        child.poll.return_value = None
        info = {"url": "http://127.0.0.1:8765/", "token": "private-start-token", "pid": 4444}
        launched = []

        def record(command, **options):
            launched.append((command, options))
            return child

        with patch.object(dash_process.subprocess, "Popen", side_effect=record), \
             patch.object(dash_process.secrets, "token_urlsafe", return_value="private-start-token"), \
             patch.object(dash_process, "_server_info", side_effect=[None, info]):
            dash_process.start_dashboard(root, open_browser=False, timeout=0.5)
        self.assertNotIn("private-start-token", " ".join(map(str, launched[0][0])))
        self.assertEqual(launched[0][1]["env"]["AUTORESEARCH_DASHBOARD_STOP_TOKEN"], "private-start-token")
        self.assertEqual(launched[0][1]["env"].get("AUTORESEARCH_HOME"), os.environ.get("AUTORESEARCH_HOME"))

        with patch("autoresearch_runtime.dash_process.webbrowser.open", side_effect=[False, True]) as opener:
            first = start_dashboard(root, open_browser=True, port_start=19200)
            second = start_dashboard(root, open_browser=True, port_start=19200)
        self.addCleanup(lambda: stop_dashboard(root))
        self.assertFalse(first["browser_opened"])
        self.assertIn("warning", first)
        self.assertTrue(second["browser_opened"])
        self.assertEqual(opener.call_count, 2)
        browser_url = opener.call_args_list[-1].args[0]
        self.assertTrue(browser_url.startswith("http://127.0.0.1:"))
        self.assertNotIn("#", browser_url)
        self.assertNotIn("private-start-token", json.dumps(second))

    def test_background_server_reuses_instance_stops_and_skips_busy_port(self):
        import socket

        occupied = socket.socket()
        occupied.bind(("127.0.0.1", 18950))
        occupied.listen()
        self.addCleanup(occupied.close)
        with patch("autoresearch_runtime.dash_process.webbrowser.open", return_value=False) as open_browser:
            first = start_dashboard(self.root, open_browser=True, timeout=8, port_start=18950)
        self.addCleanup(lambda: stop_dashboard(self.root))
        self.assertTrue(open_browser.called)
        self.assertFalse(first["browser_opened"])
        self.assertTrue(first["running"])
        server_info = json.loads((self.root / ".dashboard-server.json").read_text(encoding="utf-8"))
        self.assertIn(server_info["token_protection"], {"restricted", "best_effort"})
        if server_info["token_protection"] == "best_effort":
            self.assertIn("permissions could not be restricted", first.get("warning", ""))
        self.assertNotIn(server_info["token"], json.dumps(first))
        self.assertNotIn(":18950/", first["url"])
        # A second `dashboard start` from the CLI reuses the running service and opens nothing.
        second_output = io.StringIO()
        with contextlib.redirect_stdout(second_output):
            self.assertEqual(lab_main(["dashboard", "start", "--root", str(self.root)]), 0)
        second = json.loads(second_output.getvalue())["data"]
        self.assertEqual(second["url"], first["url"])
        self.assertFalse(second["browser_opened"])
        self.assertEqual(first["pid"], second["pid"])
        self.assertTrue(dashboard_server_status(self.root)["running"])
        status_output = io.StringIO()
        with contextlib.redirect_stdout(status_output):
            self.assertEqual(lab_main(["dashboard", "status", "--root", str(self.root)]), 0)
        self.assertTrue(json.loads(status_output.getvalue())["data"]["running"])
        stop_output = io.StringIO()
        with contextlib.redirect_stdout(stop_output):
            self.assertEqual(lab_main(["dashboard", "stop", "--root", str(self.root)]), 0)
        stopped = json.loads(stop_output.getvalue())["data"]
        self.assertTrue(stopped["stopped"])
        deadline = __import__("time").monotonic() + 5
        while dashboard_server_status(self.root)["running"] and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.05)
        self.assertFalse(dashboard_server_status(self.root)["running"])

    def test_server_uses_ephemeral_port_when_preferred_range_is_unavailable(self):
        from autoresearch_runtime import dash_process

        root = Path(self.temp.name) / "ephemeral-port-labs"
        port_start = 19800
        real_create_server = dash_process.create_server
        exit_codes = []

        def reject_preferred_range(server_root, *, port, stop_token=""):
            if port_start <= port < port_start + 30:
                raise OSError("simulated unavailable port range")
            return real_create_server(server_root, port=port, stop_token=stop_token)

        thread = threading.Thread(
            target=lambda: exit_codes.append(
                dash_process.serve(root, port_start=port_start, token="test-token")
            ),
            daemon=True,
        )
        try:
            with patch.object(dash_process, "create_server", side_effect=reject_preferred_range):
                thread.start()
                info = None
                deadline = __import__("time").monotonic() + 3
                while __import__("time").monotonic() < deadline:
                    info = dash_process._server_info(root)
                    if info is not None or not thread.is_alive():
                        break
                    __import__("time").sleep(0.02)

            self.assertIsNotNone(info, "server should retry once with an OS-selected port")
            self.assertFalse(port_start <= info["port"] < port_start + 30)
            stop_request = Request(
                f"{info['url']}__stop", method="POST", data=b"",
                headers={"X-AutoResearch-Stop": info["token"]},
            )
            with urlopen(stop_request, timeout=2):
                pass
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(exit_codes, [0])
        finally:
            if thread.is_alive():
                info = dash_process._server_info(root)
                if info is not None:
                    stop_request = Request(
                        f"{info['url']}__stop", method="POST", data=b"",
                        headers={"X-AutoResearch-Stop": info["token"]},
                    )
                    with contextlib.suppress(OSError):
                        urlopen(stop_request, timeout=2).close()
                thread.join(timeout=3)

    def test_concurrent_dashboard_start_shares_one_server(self):
        root = Path(self.temp.name) / "concurrent-labs"
        with patch("autoresearch_runtime.dash_process.webbrowser.open", return_value=True) as open_browser:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: start_dashboard(root, open_browser=True, port_start=19050), range(4)))
        self.addCleanup(lambda: stop_dashboard(root))
        self.assertEqual(len({item["pid"] for item in results}), 1)
        # Opening is an explicit request each time, so every request that asked for a tab got one.
        self.assertEqual(open_browser.call_count, 4)

    def test_dashboard_start_lock_serializes_separate_processes(self):
        root = Path(self.temp.name) / "process-labs"
        scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
        code = ("import json,sys;sys.path.insert(0,sys.argv[2]);"
                "from pathlib import Path;from autoresearch_runtime.dashboard_server import start_dashboard;"
                "print(json.dumps(start_dashboard(Path(sys.argv[1]),open_browser=False,port_start=19100)))")
        env = dict(os.environ, PYTHONPATH=str(scripts))

        def start():
            return subprocess.run([sys.executable, "-c", code, str(root), str(scripts)], env=env,
                                  capture_output=True, text=True, encoding="utf-8", timeout=15, check=True, **support.hidden_process_kwargs())

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [json.loads(item.stdout) for item in pool.map(lambda _: start(), range(2))]
        self.addCleanup(lambda: stop_dashboard(root))
        self.assertEqual(results[0]["pid"], results[1]["pid"])

    def test_server_reuse_rejects_wrong_host_or_dashboard_identity(self):
        from autoresearch_runtime.dashboard_server import _server_info

        info_path = self.root / ".dashboard-server.json"
        info_path.write_text(json.dumps({"host": "0.0.0.0", "port": 8765,
                                        "url": "http://0.0.0.0:8765/", "token": "not-a-real-token"}),
                            encoding="utf-8")
        self.assertIsNone(_server_info(self.root))

    def test_events_sse_headers_and_heartbeat(self):
        home_patch = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(Path(self.temp.name) / "sse-home")})
        home_patch.start()
        self.addCleanup(home_patch.stop)
        server = _test_server(self.root, port=0)
        server.heartbeat_interval = 0.2
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))

        req = _local_request(f"http://127.0.0.1:{server.server_port}/api/events")
        response = urlopen(req, timeout=5)
        self.addCleanup(response.close)

        self.assertEqual(response.headers.get("Content-Type"), "text/event-stream")
        self.assertEqual(response.headers.get("Cache-Control"), "no-cache")
        self.assertEqual(response.headers.get("Connection"), "keep-alive")
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

        line1 = response.readline().decode("utf-8")
        while line1 != ": ping\n":
            self.assertEqual(line1, "event: update\n")
            self.assertTrue(response.readline().decode("utf-8").startswith("data: "))
            self.assertEqual(response.readline().decode("utf-8"), "\n")
            line1 = response.readline().decode("utf-8")
        self.assertEqual(response.readline().decode("utf-8"), "\n")

    def test_sqlite_sse_emits_only_authoritative_revision_changes(self):
        from autoresearch_runtime.sqlite_store import SQLiteStore

        revision_reads = 0

        def current_revision(_store):
            nonlocal revision_reads
            revision_reads += 1
            return 0 if revision_reads <= 2 else 1

        home = Path(self.temp.name) / "sqlite-sse-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}), \
             patch.object(SQLiteStore, "is_initialized", return_value=True), \
             patch.object(SQLiteStore, "get_revision", current_revision):
            server = _test_server(self.root, port=0)
            server.heartbeat_interval = 0.2
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            response = None
            try:
                request = _local_request(f"http://127.0.0.1:{server.server_port}/api/events")
                response = urlopen(request, timeout=5)
                event = response.readline().decode("utf-8")
                while event == ": ping\n":
                    self.assertEqual(response.readline().decode("utf-8"), "\n")
                    event = response.readline().decode("utf-8")
                data = response.readline().decode("utf-8")
                blank = response.readline().decode("utf-8")
                self.assertEqual(event, "event: update\n")
                self.assertEqual(blank, "\n")
                payload = json.loads(data.removeprefix("data: "))
                self.assertEqual(payload["revision"], "1")
                self.assertEqual(set(payload), {"revision", "timestamp"})
            finally:
                if response is not None:
                    response.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_start_reuses_a_live_server_and_launches_only_when_missing(self):
        with patch("autoresearch_runtime.dash_process._server_info",
                   return_value={"url": "http://127.0.0.1:8765/", "pid": 4242}) as mock_info, \
             patch("autoresearch_runtime.dash_process._launch_server") as mock_launch:
            res = start_dashboard(self.root, open_browser=False)
            self.assertTrue(res["running"])
            self.assertEqual(res["url"], "http://127.0.0.1:8765/")
            self.assertEqual(res["pid"], 4242)
            self.assertFalse(res["browser_opened"])
            mock_info.assert_called_once_with(self.root.resolve(), timeout=1.5)
            mock_launch.assert_not_called()

        with patch("autoresearch_runtime.dash_process._server_info", return_value=None), \
             patch("autoresearch_runtime.dash_process._launch_server",
                   return_value={"url": "http://127.0.0.1:8765/", "pid": 4243}) as mock_launch:
            res = start_dashboard(self.root, open_browser=False)
            self.assertTrue(res["running"])
            self.assertEqual(res["pid"], 4243)
            mock_launch.assert_called_once_with(self.root.resolve(), timeout=5, port_start=8765)

if __name__ == "__main__":
    unittest.main()

