from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime.commands import guide


class HarnessGuideTest(unittest.TestCase):
    def test_sqlite_guide_exposes_fingerprint_without_profile_or_evaluator(self):
        config = SimpleNamespace(lab_id="lab-1")
        state = {"current_epoch": 1}
        fingerprint = "a" * 64
        with patch("autoresearch_runtime.commands.guide.resolve_config", return_value=config), \
             patch("autoresearch_runtime.commands.guide.load_state", return_value=state), \
             patch("autoresearch_runtime.commands.guide.build_cockpit", return_value={"epoch": 1}), \
             patch("autoresearch_runtime.harness_policy.contract_fingerprint",
                   return_value=fingerprint, create=True) as fingerprint_for:
            result = guide.run(SimpleNamespace(run_id=None, json=True), None)

        self.assertEqual(result["contract_fingerprint"], fingerprint)
        fingerprint_for.assert_called_once_with(config)

if __name__ == "__main__":
    unittest.main()
