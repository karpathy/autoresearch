"""Example of a provider evaluator integration test with a substituted transport."""
try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from autoresearch_runtime.case_checkpoint import summarize_usage


class ProviderIntegrationTest(unittest.TestCase):
    def test_stub_transport_checks_endpoint_model_result_logs_and_usage(self):
        scripts = Path(__file__).resolve().parents[1] / ".agents/skills/autoresearch/scripts"
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            code = r'''
import importlib.util, io, json, os, urllib.request
from pathlib import Path
from autoresearch_runtime.case_checkpoint import case_scope, append_case
run = Path(os.environ['TEST_PROVIDER_RUN'])
endpoint = 'https://sandbox.example.invalid/v1/chat/completions'
model = 'sandbox/deepseek-v4-flash'
class StubResponse:
    def __init__(self):
        self.fp = io.BytesIO(json.dumps({'answer': 'correct', 'usage': {'cost': 0.003}}).encode())
    def read(self): return self.fp.read()
    def getcode(self): return 200
def stub(request, *args, **kwargs):
    # Assertions execute before any possible transport; no socket is opened.
    assert request.full_url == endpoint
    assert json.loads(request.data)['model'] == model
    events = [json.loads(x) for x in (run/'attempts.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'started' and events[-1]['case_id'] == 'known-case'
    return StubResponse()
urllib.request.urlopen = stub
spec = importlib.util.spec_from_file_location('fixture_hook', os.environ['TEST_PROVIDER_HOOK'])
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
with case_scope('known-case'):
    request = urllib.request.Request(endpoint, data=json.dumps({'model': model}).encode(), method='POST')
    request.add_header('Authorization', 'Bearer test-secret')
    result = json.loads(urllib.request.urlopen(request).read())
append_case(run/'cases.jsonl', {'id':'known-case','status':'ok'})
(run/'result.json').write_text(json.dumps({'metrics':{'score':1},'usage':{'requests':1,'cost_usd':0.003},
    'attestations':{'endpoint':endpoint,'model':model}}))
'''
            env = dict(os.environ, PYTHONPATH=str(scripts), TEST_PROVIDER_RUN=str(run),
                       TEST_PROVIDER_HOOK=str(scripts / "hooks/sitecustomize.py"),
                       AUTORESEARCH_ATTEMPTS_LOG=str(run / "attempts.jsonl"),
                       AUTORESEARCH_NET_STATS=str(run / "net-stats.json"),
                       AUTORESEARCH_CASE_SCOPE_REQUIRED="1")
            for name in ("AUTORESEARCH_HTTP_RETRY", "AUTORESEARCH_RESUME_OFFLINE", "AUTORESEARCH_USAGE_FILE"):
                env.pop(name, None)
            completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, **support.hidden_process_kwargs())
            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(x) for x in (run / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual([x["event"] for x in events], ["started", "finished"])
            self.assertFalse((run / "sent-requests.jsonl").exists())
            for path in run.rglob("*"):
                if path.is_file():
                    self.assertNotIn("test-secret", path.read_text(encoding="utf-8"), path.name)
            stats = json.loads((run / "net-stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["attempts"], 1)
            self.assertEqual(stats["rate_limited_responses"], 0)
            # An HTTP status alone cannot establish its cost; retain that uncertainty.
            usage = summarize_usage(run / "attempts.jsonl")
            self.assertEqual(usage["requests"], 1)
            self.assertEqual(usage["unknown_requests"], 1)
            result = json.loads((run / "result.json").read_text())
            self.assertEqual(result["usage"], {"requests": 1, "cost_usd": 0.003})
            self.assertEqual(result["metrics"]["score"], 1)


if __name__ == "__main__":
    unittest.main()
