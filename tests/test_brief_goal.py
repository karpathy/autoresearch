"""The confirmed goal can come from the spec, a `Goal:` line, or the passport's `## Goal` section."""

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.bootstrap import _confirmed_goal


class ConfirmedGoalTest(unittest.TestCase):
    def goal(self, brief: str, spec: dict | None = None) -> str:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "brief.md"
            path.write_text(brief, encoding="utf-8")
            return _confirmed_goal(path, spec or {})

    def test_goal_line(self):
        self.assertEqual(self.goal("Project: x\nGoal: lower error\n"), "lower error")

    def test_passport_goal_section(self):
        self.assertEqual(self.goal("## Project\nx\n\n## Goal\n\nlower error\n\n## Metric\ne\n"), "lower error")

    def test_empty_goal_section_does_not_borrow_the_next_section(self):
        self.assertEqual(self.goal("## Goal\n\n## Metric and direction\nerror, min\n"), "")

    def test_spec_goal_wins(self):
        self.assertEqual(self.goal("## Goal\nfrom brief\n", {"goal": "from spec"}), "from spec")


if __name__ == "__main__":
    unittest.main()
