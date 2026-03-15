"""Tests for autoanything.scoring — score.py execution and JSON parsing.

The scoring module runs scoring/score.py as a subprocess and extracts
the metric value from the JSON output.
"""

import json
import os
import textwrap

import pytest

from autoanything.scoring import run_score, parse_score_output, is_better


class TestParseScoreOutput:
    """Extract metric value from scoring subprocess stdout."""

    def test_single_line_json(self):
        stdout = '{"cost": 42.5}\n'
        score, metrics = parse_score_output(stdout, "cost")
        assert score == 42.5
        assert metrics == {"cost": 42.5}

    def test_json_on_last_line(self):
        stdout = "Loading model...\nTraining complete.\n{\"cost\": 10.0, \"iters\": 200}\n"
        score, metrics = parse_score_output(stdout, "cost")
        assert score == 10.0
        assert metrics["iters"] == 200

    def test_missing_metric_key(self):
        stdout = '{"accuracy": 0.95}\n'
        score, metrics = parse_score_output(stdout, "cost")
        assert score is None

    def test_invalid_json(self):
        stdout = "not json at all\n"
        score, metrics = parse_score_output(stdout, "cost")
        assert score is None
        assert metrics is None

    def test_empty_output(self):
        stdout = ""
        score, metrics = parse_score_output(stdout, "cost")
        assert score is None

    def test_multiple_json_lines_uses_last(self):
        stdout = '{"cost": 99}\n{"cost": 42}\n'
        score, metrics = parse_score_output(stdout, "cost")
        assert score == 42

    def test_score_coerced_to_float(self):
        stdout = '{"cost": "12.5"}\n'
        score, metrics = parse_score_output(stdout, "cost")
        # Should handle string-encoded numbers gracefully
        assert score == 12.5 or score is None  # implementation choice


class TestRunScorePy:
    """Run scoring/score.py as a subprocess and capture results."""

    def test_successful_scoring(self, tmp_path):
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()
        (scoring_dir / "score.py").write_text(textwrap.dedent("""\
            def score():
                return {"cost": 42.5}
        """))

        score, metrics, duration, error = run_score(
            str(tmp_path), score_name="cost", timeout=30,
        )
        assert score == 42.5
        assert error is None
        assert duration > 0

    def test_script_failure(self, tmp_path):
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()
        (scoring_dir / "score.py").write_text(textwrap.dedent("""\
            def score():
                raise RuntimeError("something went wrong")
        """))

        score, metrics, duration, error = run_score(
            str(tmp_path), score_name="cost", timeout=30,
        )
        assert score is None
        assert error is not None

    def test_script_timeout(self, tmp_path):
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()
        (scoring_dir / "score.py").write_text(textwrap.dedent("""\
            import time
            def score():
                time.sleep(60)
                return {"cost": 0}
        """))

        score, metrics, duration, error = run_score(
            str(tmp_path), score_name="cost", timeout=1,
        )
        assert score is None
        assert "timeout" in error.lower() or "timed out" in error.lower()

    def test_no_json_in_output(self, tmp_path):
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()
        (scoring_dir / "score.py").write_text(textwrap.dedent("""\
            def score():
                return {"wrong_key": 42}
        """))

        score, metrics, duration, error = run_score(
            str(tmp_path), score_name="cost", timeout=30,
        )
        assert score is None

    def test_imports_from_state(self, tmp_path):
        """score.py can import from state/ directory."""
        scoring_dir = tmp_path / "scoring"
        scoring_dir.mkdir()
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "solution.py").write_text("x = 42.5\n")
        (scoring_dir / "score.py").write_text(textwrap.dedent("""\
            def score():
                from state.solution import x
                return {"cost": x}
        """))

        score, metrics, duration, error = run_score(
            str(tmp_path), score_name="cost", timeout=30,
        )
        assert score == 42.5
        assert error is None


class TestIsBetter:
    """Score comparison respects direction."""

    def test_minimize_lower_is_better(self):
        assert is_better(5.0, 10.0, "minimize") is True
        assert is_better(10.0, 5.0, "minimize") is False
        assert is_better(5.0, 5.0, "minimize") is False

    def test_maximize_higher_is_better(self):
        assert is_better(10.0, 5.0, "maximize") is True
        assert is_better(5.0, 10.0, "maximize") is False
        assert is_better(5.0, 5.0, "maximize") is False
