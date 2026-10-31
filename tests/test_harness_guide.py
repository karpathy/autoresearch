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
        config = SimpleNamespace(storage_mode="sqlite", lab_id="lab-1")
        state = {"current_epoch": 1}
        fingerprint = "a" * 64
        with patch("autoresearch_runtime.commands.guide.resolve_legacy_metadata", return_value=None), \
             patch("autoresearch_runtime.commands.guide.resolve_config", return_value=config), \
             patch("autoresearch_runtime.commands.guide.load_state", return_value=state), \
             patch("autoresearch_runtime.commands.guide.build_cockpit", return_value={"epoch": 1}), \
             patch("autoresearch_runtime.harness_policy.contract_fingerprint",
                   return_value=fingerprint, create=True) as fingerprint_for:
            result = guide.run(SimpleNamespace(run_id=None, json=True), None)

        self.assertEqual(result["contract_fingerprint"], fingerprint)
        fingerprint_for.assert_called_once_with(config)

    def test_legacy_guide_does_not_claim_a_harness_fingerprint(self):
        metadata = {"root": "legacy-lab"}
        card = {"goal_status": "ARCHIVE_ONLY", "read_only": True}
        with patch("autoresearch_runtime.commands.guide.resolve_legacy_metadata", return_value=metadata), \
             patch("autoresearch_runtime.commands.guide.legacy_read_only_status", return_value=card), \
             patch("autoresearch_runtime.commands.guide.resolve_config") as resolve_config, \
             patch("autoresearch_runtime.harness_policy.contract_fingerprint", create=True) as fingerprint_for:
            result = guide.run(SimpleNamespace(run_id=None, json=True), None)

        self.assertEqual(result, card)
        self.assertTrue(result["read_only"])
        self.assertNotIn("contract_fingerprint", result)
        resolve_config.assert_not_called()
        fingerprint_for.assert_not_called()


if __name__ == "__main__":
    unittest.main()
