try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from autoresearch_runtime.lab import init_lab, load_config, load_state, run_baseline, run_trial
from autoresearch_runtime.lab import replay_plan, restore_vault, run_replay


class ReplayRetryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "candidate.txt").write_bytes(b"1;1\r\n")
        (self.source / "protected.txt").write_text("fixed", encoding="utf-8")
        self.opaque = self.root / "sample.bin"
        self.opaque.write_bytes(b"\x00\xff\x13")
        self.config_path = self.root / "lab.json"
        self.config_path.write_text(json.dumps({
            "source_root": "source",
            "state_dir": "state",
            "source_paths": ["candidate.txt", "protected.txt"],
            "editable_files": ["candidate.txt"],
            "input_files": [str(self.opaque)],
            "evaluator": ["{python}", str(self.root / "evaluator.py"), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": [{"name": "validity", "direction": "max"}]},
            "repeats": 1,
            "timeout_seconds": 30,
            "budget": {"max_seconds": 60, "max_trials": 3, "max_requests": 0, "max_cost_usd": 0},
        }), encoding="utf-8")
        (self.root / "evaluator.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "candidate, output = map(Path, sys.argv[1:3])\n"
            "text = (candidate / 'candidate.txt').read_bytes().decode('utf-8')\n"
            "score, validity = text.strip().split(';')\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'result.json').write_text(json.dumps({'metrics': {'score': float(score), 'validity': float(validity)},"
            " 'usage': {'requests': 0, 'cost_usd': 0}}), encoding='utf-8')\n",
            encoding="utf-8",
        )

    def test_patch_keeps_crlf_bytes_and_replay_leaves_the_live_candidate(self):
        config = load_config(self.config_path)
        init_lab(config)
        committed = subprocess.run(
            ["git", "show", "HEAD:candidate.txt"], cwd=config.state_dir / "candidate",
            capture_output=True, check=True,
            **support.hidden_process_kwargs(),
        ).stdout
        self.assertIn(b"\r\n", committed)
        run_baseline(config)
        candidate = config.state_dir / "candidate" / "candidate.txt"
        candidate.write_bytes(b"0;1\r\n")
        trial = run_trial(config, {"title": "Worse", "observation": "x", "expected_effect": "y"})
        self.assertEqual(trial["status"], "rejected")
        patch = (config.state_dir / "runs" / trial["id"] / "change.patch").read_bytes()
        # Отклонённый опыт возвращает файл, поэтому патч нельзя пересчитать.
        # Сохранённый патч всё равно должен содержать CR изменённой строки.
        self.assertIn(b"\r\n", patch)
        self.assertTrue(patch.startswith(b"diff --git"))
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.state_dir / "candidate",
            capture_output=True, text=True, check=True,
            **support.hidden_process_kwargs(),
        ).stdout
        plan = replay_plan(config)
        self.assertEqual(plan["gaps"], [])
        # Force a gap by marking the trial interrupted in a copy of the check via a short run.
        manifest_path = config.state_dir / "runs" / trial["id"] / "manifest.json"
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved["status"] = "invalid"
        saved["runs"] = []
        manifest_path.write_text(json.dumps(saved), encoding="utf-8")
        plan = replay_plan(config)
        self.assertEqual(plan["gaps"][0]["id"], trial["id"])
        self.assertTrue(plan["gaps"][0]["ready"])
        proof = run_replay(config, trial["id"])
        self.assertEqual(proof["status"], "complete", proof.get("reason"))
        after = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.state_dir / "candidate",
            capture_output=True, text=True, check=True,
            **support.hidden_process_kwargs(),
        ).stdout
        self.assertEqual(after, head)
        self.assertEqual(candidate.read_bytes(), b"1;1\r\n")
        verification = config.state_dir / "runs" / trial["id"] / "rechecks" / proof["replay_id"] / "verification.json"
        self.assertTrue(verification.is_file())

    def test_vault_restore_puts_back_a_changed_input(self):
        config = load_config(self.config_path)
        init_lab(config)
        original = self.opaque.read_bytes()
        self.opaque.write_bytes(b"changed")
        restored = restore_vault(config, self.opaque)
        self.assertEqual(self.opaque.read_bytes(), original)
        self.assertEqual(restored["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(load_state(config)["inputs"][0]["sha256"], restored["sha256"])

    def test_http_429_is_recorded_without_replaying_the_request(self):
        hits = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits["n"] += 1
                if hits["n"] < 3:
                    self.send_response(429)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
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
        stats = self.root / "net-stats.json"
        usage = self.root / "usage.json"
        body_path = self.root / "body.bin"
        hooks = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts" / "hooks"
        code = (
            "import json, os\n"
            "from urllib.request import urlopen\n"
            "from urllib.error import HTTPError\n"
            f"try: response = urlopen('http://127.0.0.1:{port}/')\n"
            "except HTTPError as error: status = error.code\n"
            "else: status = response.status\n"
            "outcome = {'status': status}\n"
            "open(os.environ['OUT'], 'w', encoding='utf-8').write(json.dumps(outcome))\n"
        )
        env = dict(os_environ(), **{
            "PYTHONPATH": str(hooks),
            "AUTORESEARCH_HTTP_RETRY": "1",
            "AUTORESEARCH_NET_STATS": str(stats),
            "AUTORESEARCH_USAGE_PATH": str(usage),
            "OUT": str(body_path),
        })
        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(body_path.read_text(encoding="utf-8")), {"status": 429})
        recorded = json.loads(stats.read_text(encoding="utf-8"))
        self.assertEqual(recorded["rate_limited_responses"], 1)
        self.assertEqual(recorded["attempts"], 1)
        self.assertEqual(hits["n"], 1)

    def test_requests_and_httpx_return_rate_limits_without_replaying(self):
        stats = self.root / "net-stats.json"
        hooks = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts" / "hooks"
        code = (
            "import json, os, sys, types\n"
            "import sitecustomize as hook\n"
            "\n"
            "class Response:\n"
            "    status_code = 429\n"
            "    headers = {}\n"
            "\n"
            "def install(module_name, session_attr):\n"
            "    module = types.ModuleType(module_name)\n"
            "    holder = types.ModuleType(module_name + '.sessions')\n"
            "    class Session:\n"
            "        def send(self, request, **kwargs):\n"
            "            return Response()\n"
            "    setattr(holder, session_attr, Session)\n"
            "    module.sessions = holder\n"
            "    if module_name == 'httpx':\n"
            "        module.Client = Session\n"
            "    sys.modules[module_name] = module\n"
            "    sys.modules[module_name + '.sessions'] = holder\n"
            "    return Session\n"
            "\n"
            "requests_session = install('requests', 'Session')\n"
            "httpx_session = install('httpx', 'Client')\n"
            "hook._install_requests()\n"
            "hook._install_httpx()\n"
            "statuses = []\n"
            "for session in (requests_session, httpx_session):\n"
            "    statuses.append(session().send(None).status_code)\n"
            "stats = json.loads(open(os.environ['AUTORESEARCH_NET_STATS'], encoding='utf-8').read())\n"
            "open(os.environ['OUT'], 'w', encoding='utf-8').write(json.dumps({'statuses': statuses, 'stats': stats}))\n"
        )
        out = self.root / "timeout-stats.json"
        # Хук импорта не включаем: иначе повторный import requests зацикливается.
        # Проверяем сами обёртки, которые рантайм ставит при включённых повторах.
        env = dict(os_environ(), **{
            "PYTHONPATH": str(hooks),
            "AUTORESEARCH_HTTP_RETRY": "1",
            "AUTORESEARCH_NET_STATS": str(stats),
            "OUT": str(out),
        })
        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        recorded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(recorded["statuses"], [429, 429])
        self.assertEqual(recorded["stats"]["rate_limited_responses"], 2)

    def test_net_stats_without_http_retry_does_not_loop(self):
        """A usage log must not turn on 429 retries by itself."""
        stats = self.root / "net-stats.json"
        hooks = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts" / "hooks"
        code = (
            "import os, sys, types\n"
            "import sitecustomize as hook\n"
            "module = types.ModuleType('requests')\n"
            "holder = types.ModuleType('requests.sessions')\n"
            "class Session:\n"
            "    def send(self, request, **kwargs):\n"
            "        class Response:\n"
            "            status_code = 429\n"
            "            headers = {}\n"
            "            text = ''\n"
            "        return Response()\n"
            "holder.Session = Session\n"
            "module.sessions = holder\n"
            "sys.modules['requests'] = module\n"
            "sys.modules['requests.sessions'] = holder\n"
            "hook._install_requests()\n"
            "response = Session().send(None)\n"
            "open(os.environ['OUT'], 'w', encoding='utf-8').write(str(response.status_code))\n"
        )
        out = self.root / "no-retry.txt"
        env = dict(os_environ(), **{
            "PYTHONPATH": str(hooks),
            "AUTORESEARCH_NET_STATS": str(stats),
            "OUT": str(out),
        })
        env.pop("AUTORESEARCH_HTTP_RETRY", None)
        completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(out.read_text(encoding="utf-8"), "429")


def os_environ():
    import os
    return dict(os.environ)
