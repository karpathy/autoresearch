try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import contextlib
import io
import json
import unittest

from tests import test_agent_guidance as guidance
from autoresearch_runtime.cli import main
from autoresearch_runtime.snapshot import init_lab
from autoresearch_runtime.baseline import run_baseline


class CompactCliTest(unittest.TestCase):
    setUp = guidance.AgentGuidanceTest.setUp

    def invoke(self, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(["--config", str(self.config.path), *args])
        self.assertEqual(code, 0)
        return json.loads(stream.getvalue())

    def test_agent_guide_returns_one_canonical_summary(self):
        init_lab(self.config)
        run_baseline(self.config)
        full = self.invoke("guide", "--json")
        compact = self.invoke("--agent", "guide", "--json")
        self.assertEqual(compact["data"], full["data"])
        self.assertNotIn("cockpit", compact)

    def test_agent_guide_canonical_payload_stays_bounded(self):
        guidance.AgentGuidanceTest._use_sqlite_toy(self)
        guidance.run_durable_baseline(self.config)

        payload = self.invoke("--agent", "guide", "--json")
        guide = payload.get("cockpit", payload["data"])
        serialized = json.dumps(guide, ensure_ascii=True, separators=(",", ":"))

        # 4,500 escaped JSON characters leaves margin for a 1,500-token guide.
        self.assertLessEqual(len(serialized), 4500)
        self.assertLessEqual(len(guide["evidence"]), 5)
        self.assertLessEqual(len(guide["signals"]), 3)
        self.assertLessEqual(len(guide["next_steps"]), 1)

    def test_status_does_not_masquerade_as_a_run(self):
        init_lab(self.config)
        baseline = run_baseline(self.config)
        compact = self.invoke("--agent", "status")
        self.assertEqual(compact["cockpit"]["selected_run_id"], baseline["id"])
        self.assertNotIn("candidate_hashes", compact["data"])

    def test_agent_trial_requires_explicit_prediction(self):
        init_lab(self.config)
        run_baseline(self.config)
        hypothesis = self.root / "hypothesis.json"
        hypothesis.write_text(json.dumps({"title": "Unexplained change"}))
        for tail in (("trial",), ("trial", "--hypothesis", str(hypothesis))):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(["--config", str(self.config.path), "--agent", *tail])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(err.getvalue())["error"]["code"], "E_CONTRACT_HYPOTHESIS")


if __name__ == "__main__":
    unittest.main()
