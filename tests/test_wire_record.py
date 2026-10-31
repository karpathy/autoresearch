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
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from autoresearch_runtime.wire_record import record_sent_request, redact_headers, redact_url

_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
_HOOKS = _SCRIPTS / "hooks"


class WireRecordTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_redact_headers_masks_authorization_and_drops_unknown(self):
        redacted = redact_headers({
            "Authorization": "Bearer sk-secret",
            "X-Api-Key": "leak-me",
            "Content-Type": "application/json",
        })
        self.assertEqual(redacted["Authorization"], "***")
        self.assertEqual(redacted["Content-Type"], "application/json")
        self.assertNotIn("X-Api-Key", redacted)

    def test_redact_headers_masks_bearer_tokens_inside_allowlisted_metadata(self):
        redacted = redact_headers({
            "X-Title": "trace Authorization: Bearer title-secret-token",
            "X-OpenRouter-Metadata": '{"authorization":"Bearer metadata-secret-token", "api_key":"metadata-api-key"}',
        })

        for secret in ("title-secret-token", "metadata-secret-token", "metadata-api-key"):
            self.assertNotIn(secret, json.dumps(redacted))

    def test_redact_url_masks_common_api_key_query_spellings(self):
        redacted = redact_url(
            "https://example.test/v1?x-api-key=leak-x&api-key=leak-api&"
            "access_token=leak-token&client_secret=leak-client&ok=visible"
        )

        for secret in ("leak-x", "leak-api", "leak-token", "leak-client"):
            self.assertNotIn(secret, redacted)
        self.assertIn("ok=visible", redacted)

    def test_record_sent_request_writes_mutated_body_only(self):
        log_path = self.root / "sent-requests.jsonl"
        draft = '{"messages":[]}'
        mutated = '{"seed":42,"provider":"openrouter","messages":[]}'
        with patch.dict(os.environ, {"AUTORESEARCH_WIRE_LOG": str(log_path), "AUTORESEARCH_WIRE_LOG_BODY": "1"}):
            record_sent_request(
                url="https://example.test/v1/chat",
                method="POST",
                body=mutated,
                headers=redact_headers({"Content-Type": "application/json"}),
                status=200,
                response_body='{"ok":true}',
                attempt=1,
            )
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("seed", text)
        self.assertIn("provider", text)
        self.assertNotIn(draft, text)
        row = json.loads(text.strip())
        self.assertEqual(row["body"], mutated)

    def test_default_wire_log_omits_bodies_and_redacts_url_secrets(self):
        log_path = self.root / "metadata.jsonl"
        with patch.dict(os.environ, {"AUTORESEARCH_WIRE_LOG": str(log_path), "AUTORESEARCH_WIRE_LOG_BODY": ""}):
            record_sent_request(
                url="https://person:password@example.test/v1/chat?api_key=sk-secret&ok=1#fragment-secret",
                method="POST",
                body='{"prompt":"private"}',
                headers={"Authorization": "Bearer sk-header-secret", "Content-Type": "application/json"},
                status=200,
                response_body='{"answer":"private"}',
                attempt=3,
            )

        text = log_path.read_text(encoding="utf-8")
        row = json.loads(text.strip())
        self.assertNotIn("body", row)
        self.assertNotIn("response_body", row)
        self.assertEqual(row["body_bytes"], len('{"prompt":"private"}'.encode("utf-8")))
        self.assertEqual(row["response_body_bytes"], len('{"answer":"private"}'.encode("utf-8")))
        self.assertNotIn("password", text)
        self.assertNotIn("sk-secret", text)
        self.assertNotIn("sk-header-secret", text)
        self.assertNotIn("fragment-secret", text)

    def test_sqlite_wire_event_is_visible_to_file_backed_evaluator(self):
        log_path = self.root / "sent-requests.jsonl"
        with patch.dict(os.environ, {"AUTORESEARCH_WIRE_LOG": str(log_path), "AUTORESEARCH_WIRE_LOG_BODY": "1"}):
            with patch("autoresearch_runtime.store._sqlite_document") as sqlite_document:
                record_sent_request(
                    url="https://example.test/v1/chat",
                    method="POST",
                    body='{"model":"test"}',
                    headers={"Content-Type": "application/json"},
                    status=200,
                    response_body='{"choices":[]}',
                    attempt=1,
                )

        sqlite_document.assert_not_called()
        self.assertTrue(log_path.is_file(), "SQLite-backed runs need a temporary wire file for file-based evaluators")
        row = json.loads(log_path.read_text(encoding="utf-8").strip())
        self.assertEqual(row["body"], '{"model":"test"}')

    def test_record_sent_request_noop_when_env_unset(self):
        log_path = self.root / "missing.jsonl"
        previous = os.environ.pop("AUTORESEARCH_WIRE_LOG", None)
        self.addCleanup(lambda: os.environ.__setitem__("AUTORESEARCH_WIRE_LOG", previous) if previous else None)
        record_sent_request(
            url="https://example.test/",
            method="GET",
            body=None,
            headers={},
            status=200,
            response_body=None,
            attempt=1,
        )
        self.assertFalse(log_path.exists())

    def test_urllib_hook_records_post_body_with_seed_without_retry(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        port = server.server_address[1]
        log_path = self.root / "wire.jsonl"
        payload = b'{"seed":99,"provider":"test","messages":[]}'
        code = (
            "import json, os\n"
            "from urllib.request import Request, urlopen\n"
            f"data = {payload!r}\n"
            f"req = Request('http://127.0.0.1:{port}/', data=data, method='POST')\n"
            "req.add_header('Content-Type', 'application/json')\n"
            "urlopen(req).read()\n"
            "print(open(os.environ['AUTORESEARCH_WIRE_LOG'], encoding='utf-8').read())\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(_HOOKS), str(_SCRIPTS), env.get("PYTHONPATH", "")])
        env["AUTORESEARCH_WIRE_LOG"] = str(log_path)
        env["AUTORESEARCH_WIRE_LOG_BODY"] = "1"
        env.pop("AUTORESEARCH_HTTP_RETRY", None)
        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        row = json.loads(completed.stdout.strip())
        self.assertIn("seed", row["body"])
        self.assertIn("provider", row["body"])
        self.assertEqual(row["attempt"], 1)

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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(_HOOKS), str(_SCRIPTS), env.get("PYTHONPATH", "")])
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(log_path)
        env["AUTORESEARCH_CASE_SCOPE_REQUIRED"] = "1"
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)

        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())

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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_HOOKS), str(transport_dir), str(_SCRIPTS), env.get("PYTHONPATH", "")]
        )
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env["AUTORESEARCH_WIRE_LOG"] = str(self.root / "wire.jsonl")
        env["AUTORESEARCH_TEST_OBSERVATION"] = str(observation)
        env["AUTORESEARCH_HTTP_RETRY"] = "1"
        env.pop("AUTORESEARCH_CASE_SCOPE_REQUIRED", None)
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)

        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True,
            **support.hidden_process_kwargs(),
        )

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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_HOOKS), str(transport_dir), str(_SCRIPTS), env.get("PYTHONPATH", "")]
        )
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env["AUTORESEARCH_WIRE_LOG"] = str(self.root / "wire.jsonl")
        env["AUTORESEARCH_TEST_OBSERVATION"] = str(observation)
        env.pop("AUTORESEARCH_CASE_SCOPE_REQUIRED", None)
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)

        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True,
            **support.hidden_process_kwargs(),
        )

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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_HOOKS), str(transport_dir), str(_SCRIPTS), env.get("PYTHONPATH", "")]
        )
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env["AUTORESEARCH_WIRE_LOG"] = str(self.root / "wire.jsonl")
        env["AUTORESEARCH_TEST_OBSERVATION"] = str(observation)
        env["AUTORESEARCH_HTTP_RETRY"] = "1"
        env.pop("AUTORESEARCH_CASE_SCOPE_REQUIRED", None)
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)

        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True,
            **support.hidden_process_kwargs(),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "HTTPError")
        self.assertEqual(json.loads(observation.read_text(encoding="utf-8"))["calls"], 1)
        rows = [json.loads(line) for line in attempts.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["event"] for row in rows], ["started", "finished"])
        self.assertEqual(rows[0]["attempt_id"], rows[1]["attempt_id"])
        self.assertEqual(rows[1]["http_status"], 429)

    def test_resume_offline_hook_refuses_network_before_attempt_start(self):
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
        url = f"http://127.0.0.1:{server.server_port}/"
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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(_HOOKS), str(_SCRIPTS), env.get("PYTHONPATH", "")])
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env["AUTORESEARCH_CASE_SCOPE_REQUIRED"] = "1"
        env["AUTORESEARCH_RESUME_OFFLINE"] = "1"

        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "offline")
        self.assertEqual(hits, [])
        self.assertFalse(attempts.exists())

    def test_required_case_scope_fails_before_network(self):
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
        url = f"http://127.0.0.1:{server.server_port}/"
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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(_HOOKS), str(_SCRIPTS), env.get("PYTHONPATH", "")])
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env["AUTORESEARCH_CASE_SCOPE_REQUIRED"] = "1"
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)

        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "scope-required")
        self.assertEqual(hits, [])
        self.assertFalse(attempts.exists())

    def test_resume_hook_refuses_a_case_already_checkpointed(self):
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
        url = f"http://127.0.0.1:{server.server_port}/"
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
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(_HOOKS), str(_SCRIPTS), env.get("PYTHONPATH", "")])
        env["AUTORESEARCH_CASE_LOG"] = str(case_log)
        env["AUTORESEARCH_ATTEMPTS_LOG"] = str(attempts)
        env["AUTORESEARCH_CASE_SCOPE_REQUIRED"] = "1"
        env["AUTORESEARCH_RESUME_ID"] = "resume-1"
        env.pop("AUTORESEARCH_RESUME_OFFLINE", None)

        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "completed-case")
        self.assertEqual(hits, [])
        self.assertFalse(attempts.exists())
