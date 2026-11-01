try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
_HOOKS = _SCRIPTS / "hooks"


def run_hook(code, env):
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        **support.hidden_process_kwargs(),
    )


class ProviderHookTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def hook_env(self, attempts, *extra_paths, **variables):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_HOOKS), *map(str, extra_paths), str(_SCRIPTS), env.get("PYTHONPATH", "")]
        )
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env.pop("AUTORESEARCH_CASE_SCOPE_REQUIRED", None)
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)
        env.update(variables)
        return env

    def counting_server(self):
        """Start a local server; the returned list grows by one for each request it receives."""
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(True)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        return hits, f"http://127.0.0.1:{server.server_port}/"

    def test_scoped_hook_persists_attempt_before_network_and_finishes_it(self):
        hits = []
        log_path = self.root / "attempts.jsonl"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if log_path.is_file():
                    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                    hits.append(any(row.get("event") == "started" for row in rows))
                else:
                    hits.append(False)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        url = f"http://127.0.0.1:{server.server_port}/"
        code = (
            "from urllib.request import urlopen\n"
            "from autoresearch_runtime.case_checkpoint import case_scope\n"
            f"with case_scope('case-alpha'): urlopen({url!r}, timeout=2).read()\n"
        )
        completed = run_hook(code, self.hook_env(log_path, AUTORESEARCH_CASE_SCOPE_REQUIRED="1"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(hits, [True])
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["started", "finished"])
        self.assertEqual([row["case_id"] for row in rows], ["case-alpha", "case-alpha"])
        self.assertEqual(rows[0]["attempt_id"], rows[1]["attempt_id"])

    def test_unscoped_provider_attempt_starts_before_transport_and_stays_open_when_outcome_is_unknown(self):
        transport_dir = self.root / "fake-transport"
        transport_dir.mkdir()
        observation = self.root / "transport-observation.json"
        attempts = self.root / "attempts.jsonl"
        (transport_dir / "sitecustomize.py").write_text(
            "import json, os, urllib.request\n"
            "from pathlib import Path\n"
            "def fake_urlopen(request, *args, **kwargs):\n"
            "    path = Path(os.environ['AUTORESEARCH_ATTEMPTS_LOG'])\n"
            "    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []\n"
            "    Path(os.environ['AUTORESEARCH_TEST_OBSERVATION']).write_text(json.dumps({\n"
            "        'started_before_transport': any(row.get('event') == 'started' for row in rows),\n"
            "    }))\n"
            "    raise OSError('ambiguous connection outcome')\n"
            "urllib.request.urlopen = fake_urlopen\n",
            encoding="utf-8",
        )
        code = (
            "from urllib.request import Request, urlopen\n"
            "try: urlopen(Request('https://provider.invalid/v1/chat', data=b'{}'))\n"
            "except OSError: print('unknown-outcome')\n"
        )
        completed = run_hook(code, self.hook_env(
            attempts, transport_dir, AUTORESEARCH_TEST_OBSERVATION=str(observation), AUTORESEARCH_HTTP_RETRY="1",
        ))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "unknown-outcome")
        self.assertTrue(json.loads(observation.read_text(encoding="utf-8"))["started_before_transport"])
        rows = [json.loads(line) for line in attempts.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["started"])
        self.assertIsNone(rows[0]["case_id"])
        from autoresearch_runtime.case_checkpoint import summarize_usage
        self.assertEqual(summarize_usage(attempts), {
            "requests": 1, "unknown_requests": 1, "cost_usd": None,
            "confirmed_cost_usd": 0.0,
        })

    def test_requests_and_httpx_hooks_pair_unscoped_attempts(self):
        transport_dir = self.root / "fake-clients"
        transport_dir.mkdir()
        observation = self.root / "client-observation.json"
        attempts = self.root / "attempts.jsonl"
        (transport_dir / "sitecustomize.py").write_text(
            "import json, os, sys, types\n"
            "from pathlib import Path\n"
            "def observe(name):\n"
            "    path = Path(os.environ['AUTORESEARCH_ATTEMPTS_LOG'])\n"
            "    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []\n"
            "    values = json.loads(Path(os.environ['AUTORESEARCH_TEST_OBSERVATION']).read_text()) if Path(os.environ['AUTORESEARCH_TEST_OBSERVATION']).exists() else []\n"
            "    values.append({'client': name, 'started_before_transport': rows[-1].get('event') == 'started' if rows else False})\n"
            "    Path(os.environ['AUTORESEARCH_TEST_OBSERVATION']).write_text(json.dumps(values))\n"
            "class Response:\n"
            "    status_code = 200\n"
            "    text = 'ok'\n"
            "class Request:\n"
            "    url = 'https://provider.invalid/v1/chat'\n"
            "    method = 'POST'\n"
            "    body = b'{}'\n"
            "    content = b'{}'\n"
            "    headers = {'Content-Type': 'application/json'}\n"
            "class RequestsSession:\n"
            "    def send(self, request, **kwargs): observe('requests'); return Response()\n"
            "class HttpxClient:\n"
            "    def send(self, request, **kwargs): observe('httpx'); return Response()\n"
            "requests = types.ModuleType('requests'); requests.sessions = types.SimpleNamespace(Session=RequestsSession)\n"
            "httpx = types.ModuleType('httpx'); httpx.Client = HttpxClient\n"
            "sys.modules['requests'] = requests; sys.modules['httpx'] = httpx\n",
            encoding="utf-8",
        )
        code = (
            "import requests, httpx\n"
            "request = type('Request', (), {'url':'https://provider.invalid/v1/chat','method':'POST','body':b'{}','content':b'{}','headers':{'Content-Type':'application/json'}})()\n"
            "requests.sessions.Session().send(request)\n"
            "httpx.Client().send(request)\n"
            "print('sent')\n"
        )
        completed = run_hook(code, self.hook_env(
            attempts, transport_dir, AUTORESEARCH_TEST_OBSERVATION=str(observation),
        ))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "sent")
        self.assertEqual(json.loads(observation.read_text(encoding="utf-8")), [
            {"client": "requests", "started_before_transport": True},
            {"client": "httpx", "started_before_transport": True},
        ])
        rows = [json.loads(line) for line in attempts.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["started", "finished"] * 2)
        self.assertIsNone(rows[0]["case_id"])
        self.assertIsNone(rows[2]["case_id"])
        self.assertEqual(rows[0]["attempt_id"], rows[1]["attempt_id"])
        self.assertEqual(rows[2]["attempt_id"], rows[3]["attempt_id"])
        self.assertNotEqual(rows[0]["attempt_id"], rows[2]["attempt_id"])

    def test_rate_limit_response_is_not_retried_by_runtime_hook(self):
        transport_dir = self.root / "fake-transport"
        transport_dir.mkdir()
        observation = self.root / "transport-observation.json"
        attempts = self.root / "attempts.jsonl"
        (transport_dir / "sitecustomize.py").write_text(
            "import json, os, sys, urllib.request\n"
            "from email.message import Message\n"
            "from pathlib import Path\n"
            "calls = 0\n"
            "def fake_urlopen(request, *args, **kwargs):\n"
            "    global calls\n"
            "    calls += 1\n"
            "    path = Path(os.environ['AUTORESEARCH_ATTEMPTS_LOG'])\n"
            "    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []\n"
            "    Path(os.environ['AUTORESEARCH_TEST_OBSERVATION']).write_text(json.dumps({\n"
            "        'calls': calls, 'started_before_transport': any(row.get('event') == 'started' for row in rows),\n"
            "    }))\n"
            "    if calls > 1: raise AssertionError('runtime retried a provider response')\n"
            "    headers = Message(); headers['Retry-After'] = '0'\n"
            "    raise urllib.error.HTTPError(request.full_url, 429, 'rate limit', headers, None)\n"
            "urllib.request.urlopen = fake_urlopen\n",
            encoding="utf-8",
        )
        code = (
            "from urllib.request import Request, urlopen\n"
            "try: urlopen(Request('https://provider.invalid/v1/chat', data=b'{}'))\n"
            "except BaseException as error: print(type(error).__name__)\n"
        )
        completed = run_hook(code, self.hook_env(
            attempts, transport_dir, AUTORESEARCH_TEST_OBSERVATION=str(observation), AUTORESEARCH_HTTP_RETRY="1",
        ))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "HTTPError")
        self.assertEqual(json.loads(observation.read_text(encoding="utf-8"))["calls"], 1)
        rows = [json.loads(line) for line in attempts.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["started", "finished"])
        self.assertEqual(rows[0]["attempt_id"], rows[1]["attempt_id"])
        self.assertEqual(rows[1]["http_status"], 429)

    def test_resume_offline_hook_refuses_network_before_attempt_start(self):
        hits, url = self.counting_server()
        attempts = self.root / "attempts.jsonl"
        code = (
            "from urllib.request import urlopen\n"
            "from autoresearch_runtime.case_checkpoint import case_scope\n"
            "try:\n"
            f"    with case_scope('case-alpha'): urlopen({url!r}, timeout=2).read()\n"
            "except RuntimeError as error:\n"
            "    print('offline' if 'offline' in str(error).lower() else 'runtime-error')\n"
            "else:\n"
            "    print('sent')\n"
        )
        completed = run_hook(code, self.hook_env(
            attempts, AUTORESEARCH_CASE_SCOPE_REQUIRED="1", AUTORESEARCH_RESUME_OFFLINE="1",
        ))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "offline")
        self.assertEqual(hits, [])
        self.assertFalse(attempts.exists())

    def test_required_case_scope_fails_before_network(self):
        hits, url = self.counting_server()
        attempts = self.root / "attempts.jsonl"
        code = (
            "from urllib.request import urlopen\n"
            "try:\n"
            f"    urlopen({url!r}, timeout=2).read()\n"
            "except RuntimeError as error:\n"
            "    print('scope-required' if 'case scope' in str(error).lower() else 'runtime-error')\n"
            "else:\n"
            "    print('sent')\n"
        )
        completed = run_hook(code, self.hook_env(attempts, AUTORESEARCH_CASE_SCOPE_REQUIRED="1"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "scope-required")
        self.assertEqual(hits, [])
        self.assertFalse(attempts.exists())

    def test_resume_hook_refuses_a_case_already_checkpointed(self):
        hits, url = self.counting_server()
        case_log = self.root / "cases.jsonl"
        case_log.write_text(json.dumps({"id": "case-done", "status": "ok"}) + "\n", encoding="utf-8")
        attempts = self.root / "attempts.jsonl"
        code = (
            "from urllib.request import urlopen\n"
            "from autoresearch_runtime.case_checkpoint import case_scope\n"
            "try:\n"
            f"    with case_scope('case-done'): urlopen({url!r}, timeout=2).read()\n"
            "except RuntimeError as error:\n"
            "    print('completed-case' if 'completed case' in str(error).lower() else 'runtime-error')\n"
            "else:\n"
            "    print('sent')\n"
        )
        completed = run_hook(code, self.hook_env(
            attempts, AUTORESEARCH_CASE_LOG=str(case_log), AUTORESEARCH_CASE_SCOPE_REQUIRED="1",
            AUTORESEARCH_RESUME_ID="resume-1",
        ))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "completed-case")
        self.assertEqual(hits, [])
        self.assertFalse(attempts.exists())
