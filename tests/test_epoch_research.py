try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import unittest
from tests import test_measurement as measurement
from tests.support import tree_digest
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.measurement import open_contract_epoch
from autoresearch_runtime.config import load_config
from autoresearch_runtime.snapshot import init_lab


class EpochResearchTest(unittest.TestCase):
    def setUp(self):
        measurement.MeasurementTest.setUp(self)
        self.config = load_config(self.config_path)
        init_lab(self.config)

    def test_goal_criteria_can_be_added_without_rewriting_history(self):
        criteria = {"thresholds": {"score": {"min": 2}}, "confirmation_repeats": 2}
        before = list((self.config.state_dir / "runs").glob("*/manifest.json"))
        opened = open_contract_epoch(self.config_path, {"goal_criteria": criteria}, reason="Declare success criteria")
        self.assertGreater(opened["epoch_id"], 1)
        self.assertEqual(json.loads(self.config_path.read_text())["goal_criteria"], criteria)
        self.assertEqual(list((self.config.state_dir / "runs").glob("*/manifest.json")), before)

    def test_invalid_research_contract_does_not_overwrite_lab(self):
        before = tree_digest(self.root)
        with self.assertRaises(LabError):
            open_contract_epoch(self.config_path, {"goal_criteria": {"thresholds": {"score": {"min": True}}}}, reason="Bad contract")
        self.assertEqual(tree_digest(self.root), before)


if __name__ == "__main__":
    unittest.main()
