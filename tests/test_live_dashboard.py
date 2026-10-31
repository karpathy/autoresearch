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
    create_server, dashboard_server_status, ensure_dashboard_server,
    notify_dashboard, start_dashboard, stop_dashboard,
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

    def test_external_config_copy_exposes_guards_and_repeats_in_dashboard(self):
        from autoresearch_runtime.dash_model import _lab_summary
        from autoresearch_runtime.dashboard_server import dashboard_details
        from autoresearch_runtime.lab import init_lab, load_config
        from autoresearch_runtime.store import _sha256

        work = Path(self.temp.name) / "work"
        source = work / "source"
        support.git_project(source, {"candidate.txt": "1"})
        evaluator = work / "evaluator.py"
        evaluator.write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)\n"
            "(Path(sys.argv[2]) / 'result.json').write_text(json.dumps({'metrics': {'f1': 1}}))\n",
            encoding="utf-8",
        )
        state_dir = self.root / "projects" / "CopyProject" / "lab-copy"
        state_dir.mkdir(parents=True)
        external = work / "outside-lab.json"
        external.write_text(json.dumps({
            "source_root": str(source),
            "state_dir": str(state_dir),
            "source_paths": ["candidate.txt"],
            "editable_files": ["candidate.txt"],
            "input_files": [],
            "evaluator": ["{python}", str(evaluator), "{candidate}", "{run_dir}"],
            "metrics": {
                "primary": "f1",
                "direction": "max",
                "guards": [
                    {"name": "accuracy", "direction": "max"},
                    {"name": "recall", "direction": "max"},
                ],
            },
            "repeats": 2,
            "timeout_seconds": 30,
            "budget": {"max_seconds": 60, "max_trials": 3, "max_requests": 0, "max_cost_usd": 0},
        }), encoding="utf-8")

        config = load_config(external)
        init_lab(config)

        copy_path = state_dir / "lab.json"
        self.assertTrue(copy_path.is_file())
        self.assertEqual(copy_path.read_bytes(), external.read_bytes())
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["config_copy_sha256"], _sha256(copy_path))

        (self.root / "catalog.json").write_text(json.dumps({
            "projects": [{
                "id": "CopyProject", "name": "Copy", "source_root": str(source),
                "labs": [{"id": "lab-copy", "path": "projects/CopyProject/lab-copy"}],
            }],
        }), encoding="utf-8")

        project = {"id": "CopyProject", "name": "Copy", "source_root": str(source)}
        item = {"id": "lab-copy", "path": "projects/CopyProject/lab-copy"}
        summary = _lab_summary(self.root, project, item)
        self.assertEqual(summary["guards"], ["accuracy", "recall"])
        self.assertEqual(summary["repeats"], 2)
        self.assertFalse(summary["config_copy_mismatch"])

        details = dashboard_details(self.root, "CopyProject", "lab-copy")
        self.assertIsNotNone(details)
        self.assertEqual(details["conditions"]["repeats"], 2)

    def test_read_only_api_includes_live_usage_and_contains_no_configuration_or_secrets(self):
        server = _test_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret-demo-token"}):
            with urlopen(_local_request(f"http://127.0.0.1:{server.server_port}/api/status")) as response:
                payload = json.loads(response.read())
        lab = payload["projects"][0]["labs"][0]
        self.assertEqual(lab["status"], "running")
        self.assertEqual(lab["stage"], "Сейчас идёт")
        self.assertEqual(lab["current_repeat"], "repeat_01")
        self.assertIn("trial-0001", lab["activity"])
        self.assertEqual(lab["requests"], 3)
        self.assertAlmostEqual(lab["cost_usd"], 0.42)
        self.assertNotIn("evaluator", json.dumps(payload))
        self.assertNotIn("OPENROUTER_API_KEY", json.dumps(payload))
        self.assertNotIn("secret-demo-token", json.dumps(payload))
        self.assertFalse((self.root / "catalog.json").exists())
        usage_path = self.root / "projects" / "sample-project-1234567890" / "contract-a" / "runs" / "trial-0001" / "repeat_01" / "usage.json"
        usage_path.write_text(json.dumps({"requests": 4, "cost_usd": 0.5}), encoding="utf-8")
        with urlopen(_local_request(f"http://127.0.0.1:{server.server_port}/api/status")) as response:
            updated = json.loads(response.read())
        self.assertEqual(updated["projects"][0]["labs"][0]["requests"], 4)
        self.assertAlmostEqual(updated["projects"][0]["labs"][0]["cost_usd"], 0.5)

    def test_dashboard_page_polls_without_meta_refresh_or_static_file_rewrite(self):
        server = _test_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))

        with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("/assets/live_app.js", page)
        self.assertNotIn("/assets/app.js", page)
        self.assertNotIn('http-equiv="refresh"', page)
        with urlopen(f"http://127.0.0.1:{server.server_port}/assets/live_app.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("scheduleRefresh", script)
        self.assertIn("/api/v1/revision", script)
        self.assertIn("/api/v1/labs/", script)
        self.assertNotIn("/api/v1/auth/bootstrap", script)
        self.assertNotIn("/__browser-token", script)
        self.assertNotIn("#token=", script)
        self.assertNotIn("function renderAuthError()", script)
        self.assertIn("function renderRunCharts()", script)
        self.assertIn("run.parent_run_id", script)
        self.assertIn("decision_worst", script)
        self.assertIn("completed_repeats", script)
        self.assertIn('/api/status', script)
        self.assertIn('/api/labs/', script)
        self.assertIn('X-AutoResearch-Confirm', script)
        self.assertNotIn("localStorage.setItem(\"autoresearch:token", script)
        with self.assertRaises(HTTPError) as failure:
            urlopen(f"http://127.0.0.1:{server.server_port}/assets/../dashboard_server.py")
        self.assertEqual(failure.exception.code, 404)

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
            self.assertEqual(missing.exception.code, 404)

    def test_server_reuse_rejects_same_root_health_from_a_different_process(self):
        from autoresearch_runtime import dash_process

        home = Path(self.temp.name) / "identity-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home), "AUTORESEARCH_NO_DASHBOARD": ""}):
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

    def test_notify_does_not_send_the_stop_token_to_a_foreign_service_port(self):
        from autoresearch_runtime import dash_http
        import hashlib

        home = Path(self.temp.name) / "notify-home"
        expected_root = home / "expected"
        foreign_root = home / "foreign"
        expected_root.mkdir(parents=True)
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
            server = _test_server(foreign_root, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
            expected_id = hashlib.sha256(str(expected_root.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
            (expected_root / ".dashboard-server.json").write_text(json.dumps({
                "host": "127.0.0.1", "port": server.server_port,
                "url": f"http://127.0.0.1:{server.server_port}/", "token": "do-not-forward",
                "pid": os.getpid(), "root_id": expected_id, "home_id": server.home_id,
                "service_id": "foreign-service",
            }), encoding="utf-8")
            with patch("autoresearch_runtime.dash_http.urlopen") as opener:
                self.assertFalse(notify_dashboard(expected_root))

            opener.assert_not_called()

    def test_explicit_ensure_opens_plain_url_and_force_allows_reopening(self):
        from autoresearch_runtime import dash_process

        home = Path(self.temp.name) / "reopen-home"
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home), "AUTORESEARCH_NO_DASHBOARD": ""}):
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
            with patch("autoresearch_runtime.dash_process._restrict_server_info_permissions", return_value=True), \
                 patch("autoresearch_runtime.dashboard_server.webbrowser.open", return_value=True) as opener:
                first = ensure_dashboard_server(home, open_browser=True)
                second = ensure_dashboard_server(home, open_browser=True, force_open=True)

            self.assertTrue(first["browser_opened"])
            self.assertTrue(second["browser_opened"])
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

    def test_cli_autostarts_once_but_headless_help_and_dashboard_management_skip_it(self):
        from autoresearch_runtime import cli

        home = Path(self.temp.name) / "autostart-home"
        args = ["status"]
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home), "AUTORESEARCH_NO_DASHBOARD": ""}), \
             patch("autoresearch_runtime.dashboard_server.ensure_dashboard_server",
                  return_value={"running": True, "url": "http://127.0.0.1:8765/", "browser_opened": False}) as auto, \
             patch.object(cli, "run_command", return_value={"ran": True}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(lab_main(args), 0)
            auto.assert_called_once_with(home.resolve(), open_browser=False)
            self.assertFalse((home / "autoresearch.sqlite3").exists())

        with patch("autoresearch_runtime.dashboard_server.ensure_dashboard_server") as auto, \
             patch.object(cli, "run_command", return_value={"ran": True}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(lab_main(["--headless", "status"]), 0)
                self.assertEqual(lab_main(["dashboard", "status"]), 0)
                self.assertEqual(lab_main(["--help"]), 0)
            auto.assert_not_called()

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

        with patch("autoresearch_runtime.dashboard_server.webbrowser.open", side_effect=[False, True]) as opener:
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

    def test_catalog_can_mark_a_preserved_history_as_incomparable(self):
        lab_dir = self.root / "projects" / "sample-project-1234567890" / "contract-a"
        manifest_path = lab_dir / "runs" / "trial-0001" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "rejected"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        catalog = {
            "projects": [{"id": "sample-project-1234567890", "name": "sample-project", "source_root": "C:/private/project", "labs": [{
                "id": "contract-a", "path": "projects/sample-project-1234567890/contract-a",
                "name": "Historical replay", "status": "rejected",
                "comparison_status": "incomparable",
                "comparison_note": "The baseline does not match the historical H3 contract.",
            }]}],
        }
        (self.root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")

        from autoresearch_runtime.dashboard_server import dashboard_details, dashboard_status
        lab = dashboard_status(self.root)["projects"][0]["labs"][0]
        self.assertEqual(lab["status"], "completed")
        self.assertEqual(lab["stage"], "Завершена")
        self.assertEqual(lab["comparison_status"], "incomparable")
        self.assertEqual(lab["comparison_note"], "The baseline does not match the historical H3 contract.")
        details = dashboard_details(self.root, "sample-project-1234567890", "contract-a")
        self.assertIsNotNone(details)
        self.assertEqual(details["lab"]["status"], "completed")
        self.assertEqual(details["problems"][0]["level"], "info")
        self.assertNotIn("нельзя сравнивать", details["problems"][0]["title"])

    def test_background_server_reuses_instance_stops_and_skips_busy_port(self):
        import socket

        occupied = socket.socket()
        occupied.bind(("127.0.0.1", 18950))
        occupied.listen()
        self.addCleanup(occupied.close)
        with patch("autoresearch_runtime.dashboard_server.webbrowser.open", return_value=False) as open_browser:
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
        second = start_dashboard(self.root, open_browser=False)
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

    def test_browser_is_opened_once_per_labs_root_and_can_be_reopened_explicitly(self):
        with patch("autoresearch_runtime.dashboard_server.webbrowser.open", return_value=True) as open_browser:
            first = start_dashboard(self.root, open_browser=True, port_start=18900)
            second = start_dashboard(self.root, open_browser=True, port_start=18900)
            third = start_dashboard(self.root, open_browser=True, force_open=True, port_start=18900)
        self.addCleanup(lambda: stop_dashboard(self.root))
        self.assertTrue(first["browser_opened"])
        self.assertFalse(second["browser_opened"])
        self.assertTrue(third["browser_opened"])
        self.assertEqual(open_browser.call_count, 2)

    def test_detail_endpoint_exposes_hypotheses_and_diff_but_not_config_or_logs(self):
        from autoresearch_runtime.dashboard_server import DashboardHTTPServer

        lab = self.root / "projects" / "sample-project-1234567890" / "contract-a"
        trial = lab / "runs" / "trial-0001"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "manifest.json").write_text(json.dumps({
            "id": "trial-0001", "title": "Hypothesis A", "observation": "evidence",
            "expected_effect": "higher F1", "status": "rejected",
            "reason": "primary did not improve", "runs": [{
                "metrics": {"f1": 0.5, "accuracy": 0.8},
            }],
        }), encoding="utf-8")
        lab_config = json.loads((lab / "lab.json").read_text(encoding="utf-8"))
        lab_config["metrics"]["guards"] = [{"name": "accuracy", "direction": "max"}]
        (lab / "lab.json").write_text(json.dumps(lab_config), encoding="utf-8")
        (trial / "change.patch").write_text("diff --git a/a.py b/a.py\n+safe change\n", encoding="utf-8")
        server = _test_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        with urlopen(_local_request(f"http://127.0.0.1:{server.server_port}/api/labs/sample-project-1234567890/contract-a")) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["hypotheses"][0]["reason"], "primary did not improve")
        self.assertEqual(payload["hypotheses"][0]["primary_score"], 0.5)
        self.assertEqual(payload["hypotheses"][0]["guard_summary"], [{"name": "accuracy", "value": "0.8–0.8"}])
        self.assertEqual([row["id"] for row in payload["series"]], ["trial-0001"])
        self.assertTrue(payload["series"][0]["has_value"])
        self.assertIn("verdict", payload)
        self.assertIn("budget", payload)
        self.assertIn("problems", payload)
        self.assertEqual(payload["hypotheses"][0]["reason_label"], "Главная метрика не улучшилась")
        self.assertNotIn("evaluator", json.dumps(payload))
        with urlopen(_local_request(f"http://127.0.0.1:{server.server_port}/api/labs/sample-project-1234567890/contract-a/runs/trial-0001/diff")) as response:
            self.assertIn("safe change", response.read().decode())

    def test_old_project_relative_catalog_path_reads_completed_status_and_human_name(self):
        lab_dir = self.root / "projects" / "sample-project-1234567890" / "contract-a"
        state = json.loads((lab_dir / "state.json").read_text(encoding="utf-8"))
        state["setup_pending"] = False
        (lab_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        manifest_path = lab_dir / "runs" / "trial-0001" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(status="rejected", reason="primary did not improve")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (lab_dir / "lab-meta.json").write_text(json.dumps({"goal": "contract-a", "primary_metric": "f1"}), encoding="utf-8")
        (lab_dir / "program.snapshot.md").write_text(
            "## Цель\n\nЧто улучшить: `Human-readable objective`\n", encoding="utf-8",
        )
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": "sample-project-1234567890", "name": "sample-project",
            "labs": [{"id": "contract-a", "path": "contract-a", "name": "hash"}],
        }]}), encoding="utf-8")
        from autoresearch_runtime.dashboard_server import dashboard_status
        lab = dashboard_status(self.root)["projects"][0]["labs"][0]
        self.assertEqual(lab["status"], "completed")
        self.assertEqual(lab["stage"], "Завершена")
        self.assertEqual(lab["latest_trial_status"], "rejected")
        self.assertEqual(lab["latest_trial_label"], "Отклонено")
        self.assertEqual(lab["name"], "Human-readable objective")
        self.assertEqual(lab["latest_trial_reason"], "primary did not improve")
        self.assertIsNone(lab["latest_error"])

    def test_archived_failed_usage_is_counted_without_rewriting_manifests(self):
        archived_entries = []
        original_manifests = []
        for name, requests, cost in (("failed", 41, 0.016633782), ("failed-2", 9, 0.0010107504)):
            archived = self.root / "archive" / "sample" / name
            repeat = archived / "runs" / "baseline" / "repeat_01"
            repeat.mkdir(parents=True)
            manifest = archived / "runs" / "baseline" / "manifest.json"
            manifest.write_text(json.dumps({
                "id": "baseline", "status": "crash", "runs": [{"duration_seconds": 4}],
            }), encoding="utf-8")
            original_manifests.append((manifest, manifest.read_bytes()))
            (repeat / "usage.json").write_text(json.dumps({"requests": requests, "cost_usd": cost}), encoding="utf-8")
            archived_entries.append({"project": "sample", "archived_path": str(archived), "status": "crash"})
        (self.root / "catalog.json").write_text(json.dumps({"projects": [], "archived": archived_entries}), encoding="utf-8")
        from autoresearch_runtime.dashboard_server import dashboard_status
        dashboard = dashboard_status(self.root)
        summary = dashboard["archived_totals"]
        self.assertEqual(summary["requests"], 50)
        self.assertEqual(summary["cost_usd"], 0.0176445324)
        self.assertEqual([entry["requests"] for entry in dashboard["archived"]], [41, 9])
        self.assertEqual([path.read_bytes() for path, _ in original_manifests],
                         [contents for _, contents in original_manifests])

    def test_live_usage_file_supersedes_stale_manifest_without_double_counting(self):
        lab_dir = self.root / "projects" / "sample-project-1234567890" / "contract-a"
        repeat = lab_dir / "runs" / "trial-0001" / "repeat_01"
        repeat.mkdir(parents=True, exist_ok=True)
        manifest_path = lab_dir / "runs" / "trial-0001" / "manifest.json"
        manifest_path.write_text(json.dumps({"id": "trial-0001", "status": "rejected", "runs": [
            {"duration_seconds": 2, "usage": {"requests": 2, "cost_usd": 0.2}},
        ]}), encoding="utf-8")
        (repeat / "usage.json").write_text(json.dumps({"requests": 4, "cost_usd": 0.5}), encoding="utf-8")
        catalog = {"projects": [{"id": "sample-project-1234567890", "name": "sample-project", "labs": [
            {"id": "contract-a", "path": "projects/sample-project-1234567890/contract-a"},
        ]}]}
        (self.root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        from autoresearch_runtime.dashboard_server import dashboard_status
        summary = dashboard_status(self.root)["projects"][0]["labs"][0]
        self.assertEqual(summary["requests"], 4)
        self.assertAlmostEqual(summary["cost_usd"], 0.5)

    def test_delete_refuses_escape_running_and_legacy_mutation(self):
        from autoresearch_runtime.dashboard_server import delete_lab

        project = "sample-project-1234567890"
        sibling = self.root / "projects" / project / "contract-b"
        (sibling / "runs" / "baseline").mkdir(parents=True)
        (sibling / "lab.json").write_text("{}", encoding="utf-8")
        (sibling / "state.json").write_text(json.dumps({
            "baseline_id": "baseline", "setup_pending": False,
        }), encoding="utf-8")
        (sibling / "runs" / "baseline" / "manifest.json").write_text(
            json.dumps({"id": "baseline", "status": "accepted"}), encoding="utf-8")
        (sibling / "marker.txt").write_text("keep", encoding="utf-8")
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": project, "name": "sample-project", "labs": [
                {"id": "contract-a", "path": f"projects/{project}/contract-a"},
                {"id": "contract-b", "path": f"projects/{project}/contract-b"},
            ],
        }], "archived": []}), encoding="utf-8")

        self.assertEqual(delete_lab(self.root, "..", "contract-b")[0], 404)
        self.assertEqual(delete_lab(self.root, project, "contract-a"), (409, {"error": "running"}))
        self.assertTrue((self.root / "projects" / project / "contract-a").is_dir())
        blocked_status, blocked = delete_lab(self.root, project, "contract-b")
        self.assertEqual(blocked_status, 409)
        self.assertEqual(blocked["code"], "E_LEGACY_READ_ONLY")
        self.assertTrue(sibling.is_dir())
        self.assertEqual((sibling / "marker.txt").read_text(encoding="utf-8"), "keep")
        saved = json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual([lab["id"] for lab in saved["projects"][0]["labs"]], ["contract-a", "contract-b"])

        server = _test_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        delete_url = f"http://127.0.0.1:{server.server_port}/api/labs/{quote(project)}/contract-a/delete"
        refused = _local_request(delete_url, method="POST", data=b"")
        with self.assertRaises(HTTPError) as missing:
            urlopen(refused)
        self.assertEqual(missing.exception.code, 403)
        confirmed = _local_request(delete_url, method="POST", data=b"",
                                        headers={"X-AutoResearch-Confirm": "contract-a"})
        with self.assertRaises(HTTPError) as running:
            urlopen(confirmed)
        self.assertEqual(running.exception.code, 409)
        self.assertTrue((self.root / "projects" / project / "contract-a").is_dir())

        legacy_url = f"http://127.0.0.1:{server.server_port}/api/labs/{quote(project)}/contract-b/delete"
        legacy_request = _local_request(legacy_url, method="POST", data=b"",
                                         headers={"X-AutoResearch-Confirm": "contract-b"})
        with self.assertRaises(HTTPError) as legacy_block:
            urlopen(legacy_request)
        self.assertEqual(legacy_block.exception.code, 409)
        self.assertEqual(json.loads(legacy_block.exception.read())["code"], "E_LEGACY_READ_ONLY")
        self.assertTrue(sibling.is_dir())

    def test_delete_preserves_a_legacy_lab_folder_while_read_only(self):
        from autoresearch_runtime.dashboard_server import delete_lab

        legacy_id = "sample-project-legacy-abc"
        legacy = self.root / legacy_id
        (legacy / "runs").mkdir(parents=True)
        (legacy / "lab.json").write_text("{}", encoding="utf-8")
        (legacy / "marker.txt").write_text("gone", encoding="utf-8")
        kept = self.root / "projects" / "kept-project" / "kept-lab"
        (kept / "runs").mkdir(parents=True)
        (kept / "lab.json").write_text("{}", encoding="utf-8")
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": "kept-project", "labs": [
                {"id": legacy_id, "path": legacy_id},
                {"id": "kept-lab", "path": "projects/kept-project/kept-lab"},
            ],
        }], "archived": []}), encoding="utf-8")

        status, error = delete_lab(self.root, "kept-project", legacy_id)
        self.assertEqual(status, 409)
        self.assertEqual(error["code"], "E_LEGACY_READ_ONLY")
        self.assertTrue(legacy.is_dir())
        self.assertEqual((legacy / "marker.txt").read_text(encoding="utf-8"), "gone")
        self.assertTrue((kept / "lab.json").is_file())
        self.assertTrue((self.root / "projects").is_dir())
        saved = json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual([lab["id"] for lab in saved["projects"][0]["labs"]], [legacy_id, "kept-lab"])

    def test_delete_refuses_a_legacy_lab_with_a_read_only_git_object(self):
        import os
        import stat

        from autoresearch_runtime.dashboard_server import delete_lab

        lab = self.root / "projects" / "kept-project" / "readonly-lab"
        git_object = lab / "candidate" / ".git" / "objects" / "0e"
        git_object.mkdir(parents=True)
        (lab / "lab.json").write_text("{}", encoding="utf-8")
        locked = git_object / "ff6fb97ad71cc045d187caa1cfd08d91339368"
        locked.write_bytes(b"readonly")
        os.chmod(locked, stat.S_IREAD)
        self.addCleanup(lambda: os.chmod(locked, stat.S_IREAD | stat.S_IWRITE) if locked.exists() else None)
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": "kept-project", "labs": [
                {"id": "readonly-lab", "path": "projects/kept-project/readonly-lab"},
            ],
        }], "archived": []}), encoding="utf-8")

        status, error = delete_lab(self.root, "kept-project", "readonly-lab")
        self.assertEqual(status, 409)
        self.assertEqual(error["code"], "E_LEGACY_READ_ONLY")
        self.assertTrue(lab.is_dir())
        self.assertEqual(locked.read_bytes(), b"readonly")

    def test_legacy_archive_and_restore_are_read_only_before_moving_bytes(self):
        from autoresearch_runtime.dashboard_server import archive_lab, dashboard_details, dashboard_status, restore_lab

        project = "sample-project-1234567890"
        kept = self.root / "projects" / project / "contract-a"
        lab = self.root / "projects" / project / "contract-b"
        (lab / "runs" / "trial-0001").mkdir(parents=True)
        config = {
            "source_root": "C:/private/project", "state_dir": str(lab),
            "metrics": {"primary": "f1", "direction": "max"},
        }
        (lab / "lab.json").write_text(json.dumps(config), encoding="utf-8")
        original = (lab / "lab.json").read_bytes()
        (lab / "state.json").write_text(json.dumps({
            "setup_pending": False, "baseline_id": "baseline", "best_id": "baseline",
        }), encoding="utf-8")
        (lab / "runs" / "trial-0001" / "manifest.json").write_text(json.dumps({
            "id": "trial-0001", "title": "Kept", "status": "accepted", "runs": [
                {"metrics": {"f1": 0.5}, "usage": {"requests": 1, "cost_usd": 0.1}},
            ],
        }), encoding="utf-8")
        (lab / "runs" / "trial-0001" / "change.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (self.root / "catalog.json").write_text(json.dumps({"projects": [{
            "id": project, "name": "sample-project", "labs": [
                {"id": "contract-a", "path": f"projects/{project}/contract-a", "name": "Running"},
                {"id": "contract-b", "path": f"projects/{project}/contract-b", "name": "Ready", "status": "completed"},
            ],
        }], "archived": []}), encoding="utf-8")

        self.assertEqual(archive_lab(self.root, "..", "contract-b")[0], 404)
        self.assertEqual(archive_lab(self.root, project, "contract-a"), (409, {"error": "running"}))
        occupied = self.root / "archive" / project / "contract-b"
        occupied.mkdir(parents=True)
        occupied_status, occupied_error = archive_lab(self.root, project, "contract-b")
        self.assertEqual(occupied_status, 409)
        self.assertEqual(occupied_error["code"], "E_LEGACY_READ_ONLY")
        occupied.rmdir()
        status, error = archive_lab(self.root, project, "contract-b")
        self.assertEqual(status, 409)
        self.assertEqual(error["code"], "E_LEGACY_READ_ONLY")
        self.assertTrue(lab.is_dir())
        self.assertEqual((lab / "lab.json").read_bytes(), original)
        self.assertTrue((lab / "runs" / "trial-0001" / "change.patch").is_file())
        archived = self.root / "archive" / project / "contract-b"
        self.assertFalse(archived.exists())
        saved = json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in saved["projects"][0]["labs"]], ["contract-a", "contract-b"])
        self.assertEqual(saved["archived"], [])
        status = dashboard_status(self.root)
        self.assertEqual(status["archived"], [])
        self.assertTrue(kept.is_dir())
        details = dashboard_details(self.root, project, "contract-b")
        self.assertFalse(details["lab"]["archived"])
        self.assertEqual(details["hypotheses"][0]["id"], "trial-0001")

        server = _test_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        diff_url = f"http://127.0.0.1:{server.server_port}/api/labs/{quote(project)}/contract-b/runs/trial-0001/diff"
        with urlopen(_local_request(diff_url)) as response:
            self.assertIn("diff --git", response.read().decode("utf-8"))
        archive_url = f"http://127.0.0.1:{server.server_port}/api/labs/{quote(project)}/contract-a/archive"
        with self.assertRaises(HTTPError) as missing:
            urlopen(_local_request(archive_url, method="POST", data=b""))
        self.assertEqual(missing.exception.code, 403)

        archived.parent.mkdir(parents=True, exist_ok=True)
        lab.rename(archived)
        restore_status, restore_error = restore_lab(self.root, project, "contract-b")
        self.assertEqual(restore_status, 409)
        self.assertEqual(restore_error["code"], "E_LEGACY_READ_ONLY")
        self.assertTrue(archived.is_dir())
        self.assertFalse(lab.exists())
        self.assertEqual((archived / "lab.json").read_bytes(), original)

        program = Path(self.temp.name) / "program.md"
        program.write_text("Goal: keep\n", encoding="utf-8")
        source = Path(self.temp.name) / "source-project"
        source.mkdir()
        from hashlib import sha256
        from autoresearch_runtime.bootstrap import resume_lab
        digest = sha256(program.read_bytes()).hexdigest()
        # resume_lab hashes the file the same way; a missing config must not raise.
        identity = sha256(str(program.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
        pointer = self.root / ".active" / f"{identity}.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        parked = self.root / "archive" / project / "parked"
        parked.mkdir(parents=True)
        (parked / "lab.json").write_text("{}", encoding="utf-8")
        pointer.write_text(json.dumps({
            "program": str(program.resolve()),
            "program_sha256": digest,
            "config": str(parked / "lab.json"),
        }), encoding="utf-8")
        self.assertIsNone(resume_lab(program, source))

    def test_concurrent_dashboard_start_shares_server_and_opens_one_tab(self):
        root = Path(self.temp.name) / "concurrent-labs"
        with patch("autoresearch_runtime.dashboard_server.webbrowser.open", return_value=True) as open_browser:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: start_dashboard(root, open_browser=True, port_start=19050), range(4)))
        self.addCleanup(lambda: stop_dashboard(root))
        self.assertEqual(len({item["pid"] for item in results}), 1)
        self.assertEqual(open_browser.call_count, 1)

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

    def test_events_sse_headers_heartbeat_and_broadcast(self):
        # Keep this legacy compatibility test independent of the user's real SQLite home.
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

        server.broadcast("update", {"revision": "test-rev-1", "timestamp": 123.45})
        evt_line = response.readline().decode("utf-8")
        data_line = response.readline().decode("utf-8")
        blank_line = response.readline().decode("utf-8")
        self.assertEqual(evt_line, "event: update\n")
        self.assertIn('"revision": "test-rev-1"', data_line)
        self.assertEqual(blank_line, "\n")

        notify_dashboard(self.root, {"revision": "test-rev-2", "timestamp": 678.9})
        evt_line2 = response.readline().decode("utf-8")
        data_line2 = response.readline().decode("utf-8")
        blank_line2 = response.readline().decode("utf-8")
        self.assertEqual(evt_line2, "event: update\n")
        self.assertIn('"revision": "test-rev-2"', data_line2)
        self.assertEqual(blank_line2, "\n")

        response.close()
        server.broadcast("update", {"revision": "noop"})
        deadline = __import__("time").monotonic() + 2
        while len(server.clients) > 0 and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.05)
        self.assertEqual(len(server.clients), 0)

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
                deadline = __import__("time").monotonic() + 2
                while not server.clients and __import__("time").monotonic() < deadline:
                    __import__("time").sleep(0.01)
                server.broadcast("update", {"secret": "must-not-be-forwarded"})
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
                self.assertNotIn("must-not-be-forwarded", data)
            finally:
                if response is not None:
                    response.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_status_returns_revision_and_timestamp(self):
        server = _test_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))

        with urlopen(_local_request(f"http://127.0.0.1:{server.server_port}/api/status")) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertIn("revision", payload)
        self.assertTrue(payload["revision"])
        self.assertIn("timestamp", payload)
        self.assertIsInstance(payload["timestamp"], (int, float))

    def test_ensure_dashboard_server_fast_check_and_fallback(self):
        with patch.dict(os.environ, {"AUTORESEARCH_NO_DASHBOARD": ""}), \
             patch("autoresearch_runtime.dash_process._server_info", return_value={"url": "http://127.0.0.1:8765/", "pid": 4242}) as mock_info, \
             patch("autoresearch_runtime.dash_process._start_dashboard_locked") as mock_start:
            res = ensure_dashboard_server(self.root, open_browser=False)
            self.assertTrue(res["running"])
            self.assertEqual(res["url"], "http://127.0.0.1:8765/")
            self.assertEqual(res["pid"], 4242)
            self.assertFalse(res["browser_opened"])
            mock_info.assert_called_once_with(self.root.resolve(), timeout=1.5)
            mock_start.assert_not_called()

        info = {"url": "http://127.0.0.1:8765/", "pid": 4243}
        with patch.dict(os.environ, {"AUTORESEARCH_NO_DASHBOARD": ""}), \
             patch("autoresearch_runtime.dash_process._server_info", side_effect=[None, info]), \
             patch("autoresearch_runtime.dash_process._start_dashboard_locked", return_value={"running": True, "url": "http://127.0.0.1:8765/", "pid": 4243, "browser_opened": False}) as mock_start:
            res = ensure_dashboard_server(self.root, open_browser=False)
            self.assertTrue(res["running"])
            self.assertEqual(res["pid"], 4243)
            mock_start.assert_called_once_with(self.root.resolve(), open_browser=False, timeout=5.0, port_start=8765, force_open=False)

    def test_legacy_file_actions_reject_sqlite_labs_before_touching_paths(self):
        from autoresearch_runtime import dash_api
        from autoresearch_runtime.sqlite_store import SQLiteStore

        with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(Path(self.temp.name) / "action-home")}), \
             patch.object(SQLiteStore, "is_initialized", return_value=True), \
             patch.object(SQLiteStore, "get_lab", return_value={"lab_id": "lab-uuid"}):
            for action in (dash_api.archive_lab, dash_api.restore_lab, dash_api.delete_lab):
                status, payload = action(self.root, "project", "lab-uuid")
                self.assertEqual(status, 409)
                self.assertEqual(payload["error"], "unsupported_db_lab_action")
        self.assertFalse((self.root / "archive").exists())

    def test_save_manifest_notifies_dashboard(self):
        from types import SimpleNamespace
        from autoresearch_runtime.state import _save_manifest

        lab_dir = self.root / "projects" / "sample-project-1234567890" / "contract-a"
        config = SimpleNamespace(
            state_dir=lab_dir,
            primary="f1",
            direction="max",
            guards=[],
            budget={},
            source_files=[],
            editable_files=None,
        )

        manifest = {
            "id": "trial-0001",
            "title": "Test run",
            "status": "accepted",
            "runs": [{"metrics": {"f1": 0.8}, "usage": {"requests": 1, "cost_usd": 0.05}, "duration_seconds": 1}],
        }

        with patch("autoresearch_runtime.dash_http.notify_dashboard") as mock_notify:
            _save_manifest(config, manifest)
            mock_notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()

