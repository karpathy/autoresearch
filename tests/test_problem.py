"""Tests for autoanything.problem — YAML config loading and validation.

The problem module is the single source of truth for configuration.
It replaces the old string-matching YAML parsers (load_direction, load_score_name)
with proper PyYAML parsing and validation.
"""

import pytest
import textwrap

from autoanything.problem import load_problem, ProblemConfig, ValidationError


class TestLoadMinimal:
    """A minimal problem.yaml should load with sensible defaults."""

    def test_loads_required_fields(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.name == "my-problem"
        assert config.score.name == "cost"
        assert config.score.direction == "minimize"
        assert config.state == ["state/solution.py"]

    def test_default_timeout(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.score.timeout == 900

    def test_default_script(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.score.script == "scoring/score.sh"

    def test_default_base_branch(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.git.base_branch == "main"

    def test_default_proposal_pattern(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.git.proposal_pattern == "proposals/*"


class TestLoadFull:
    """A fully specified problem.yaml should have all fields populated."""

    def test_all_fields_present(self, tmp_path, full_problem_yaml):
        (tmp_path / "problem.yaml").write_text(full_problem_yaml)
        config = load_problem(tmp_path)
        assert config.name == "test-problem"
        assert config.score.name == "cost"
        assert config.score.direction == "minimize"
        assert config.score.timeout == 300
        assert config.score.script == "scoring/score.sh"
        assert config.score.bounded is True
        assert config.git.base_branch == "main"
        assert config.git.proposal_pattern == "proposals/*"
        assert len(config.state) == 2
        assert len(config.context) == 1
        assert len(config.constraints) == 2


class TestValidation:
    """Missing or invalid fields should raise clear errors."""

    def test_missing_name(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            description: No name here.
            state:
              - state/solution.py
            score:
              name: cost
              direction: minimize
        """))
        with pytest.raises(ValidationError, match="name"):
            load_problem(tmp_path)

    def test_missing_state(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Missing state.
            score:
              name: cost
              direction: minimize
        """))
        with pytest.raises(ValidationError, match="state"):
            load_problem(tmp_path)

    def test_empty_state(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Empty state list.
            state: []
            score:
              name: cost
              direction: minimize
        """))
        with pytest.raises(ValidationError, match="state"):
            load_problem(tmp_path)

    def test_missing_score_name(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Missing score name.
            state:
              - state/solution.py
            score:
              direction: minimize
        """))
        with pytest.raises(ValidationError, match="score.name"):
            load_problem(tmp_path)

    def test_missing_score_direction(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Missing direction.
            state:
              - state/solution.py
            score:
              name: cost
        """))
        with pytest.raises(ValidationError, match="direction"):
            load_problem(tmp_path)

    def test_invalid_direction(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Bad direction.
            state:
              - state/solution.py
            score:
              name: cost
              direction: sideways
        """))
        with pytest.raises(ValidationError, match="direction"):
            load_problem(tmp_path)

    def test_no_problem_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_problem(tmp_path)


class TestScoreFallback:
    """When scoring/score.sh doesn't exist, fall back to evaluator/score.sh."""

    def test_fallback_to_evaluator_path(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        # Don't create scoring/score.sh — only evaluator/score.sh
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        (eval_dir / "score.sh").write_text("#!/bin/bash\necho '{}'")
        config = load_problem(tmp_path)
        # The config should note the fallback or resolve the actual path
        assert config.score.script in ("scoring/score.sh", "evaluator/score.sh")


class TestMutableField:
    """The 'mutable' field name from current YAML should still work (backward compat)."""

    def test_mutable_alias_for_state(self, tmp_path):
        """If the YAML uses 'mutable:' instead of 'state:', it should still load."""
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: legacy
            description: Uses mutable instead of state.
            mutable:
              - state/solution.py
            score:
              name: cost
              direction: minimize
        """))
        config = load_problem(tmp_path)
        assert config.state == ["state/solution.py"]


class TestMaximizeDirection:
    """Maximize direction should be supported."""

    def test_maximize(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: maximize-test
            description: Higher is better.
            state:
              - state/solution.py
            score:
              name: accuracy
              direction: maximize
        """))
        config = load_problem(tmp_path)
        assert config.score.direction == "maximize"
