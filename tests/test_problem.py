"""Tests for autoanything.problem — YAML config loading and validation.

The problem module is the single source of truth for configuration.
State files are discovered from the state/ directory by convention.
"""

import warnings

import pytest
import textwrap

from autoanything.problem import load_problem, ProblemConfig, ValidationError, get_state_files


class TestLoadMinimal:
    """A minimal problem.yaml should load with sensible defaults."""

    def test_loads_required_fields(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.name == "my-problem"
        assert config.score.name == "score"  # default
        assert config.score.direction == "minimize"

    def test_default_timeout(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.score.timeout == 900

    def test_default_base_branch(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.git.base_branch == "main"

    def test_default_proposal_pattern(self, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        config = load_problem(tmp_path)
        assert config.git.proposal_pattern == "proposals/*"

    def test_default_score_name(self, tmp_path):
        """When score.name is omitted, it defaults to 'score'."""
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Test problem.
            score:
              direction: minimize
        """))
        config = load_problem(tmp_path)
        assert config.score.name == "score"


class TestLoadFull:
    """A fully specified problem.yaml should have all fields populated."""

    def test_all_fields_present(self, tmp_path, full_problem_yaml):
        (tmp_path / "problem.yaml").write_text(full_problem_yaml)
        config = load_problem(tmp_path)
        assert config.name == "test-problem"
        assert config.score.name == "cost"
        assert config.score.direction == "minimize"
        assert config.score.timeout == 300
        assert config.score.bounded is True
        assert config.git.base_branch == "main"
        assert config.git.proposal_pattern == "proposals/*"
        assert len(config.constraints) == 2


class TestValidation:
    """Missing or invalid fields should raise clear errors."""

    def test_missing_name(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            description: No name here.
            score:
              direction: minimize
        """))
        with pytest.raises(ValidationError, match="name"):
            load_problem(tmp_path)

    def test_missing_score_direction(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Missing direction.
            score:
              name: cost
        """))
        with pytest.raises(ValidationError, match="direction"):
            load_problem(tmp_path)

    def test_invalid_direction(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Bad direction.
            score:
              name: cost
              direction: sideways
        """))
        with pytest.raises(ValidationError, match="direction"):
            load_problem(tmp_path)

    def test_no_problem_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_problem(tmp_path)


class TestMaximizeDirection:
    """Maximize direction should be supported."""

    def test_maximize(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: maximize-test
            description: Higher is better.
            score:
              name: accuracy
              direction: maximize
        """))
        config = load_problem(tmp_path)
        assert config.score.direction == "maximize"


class TestGetStateFiles:
    """State file discovery from the state/ directory."""

    def test_discovers_files(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "solution.py").write_text("x = 1\n")
        (state_dir / "config.py").write_text("y = 2\n")

        files = get_state_files(tmp_path)
        assert sorted(files) == ["state/config.py", "state/solution.py"]

    def test_excludes_pycache(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "solution.py").write_text("x = 1\n")
        pycache = state_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "solution.cpython-312.pyc").write_bytes(b"")

        files = get_state_files(tmp_path)
        assert files == ["state/solution.py"]

    def test_empty_state_dir(self, tmp_path):
        (tmp_path / "state").mkdir()
        assert get_state_files(tmp_path) == []

    def test_no_state_dir(self, tmp_path):
        assert get_state_files(tmp_path) == []

    def test_nested_files(self, tmp_path):
        state_dir = tmp_path / "state"
        (state_dir / "sub").mkdir(parents=True)
        (state_dir / "sub" / "module.py").write_text("z = 3\n")
        (state_dir / "top.py").write_text("x = 1\n")

        files = get_state_files(tmp_path)
        assert "state/sub/module.py" in files
        assert "state/top.py" in files

    def test_state_populated_at_load_time(self, tmp_path):
        """When state: is omitted from YAML, load_problem populates it from state/ dir."""
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Auto-discover state.
            score:
              direction: minimize
        """))
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "solution.py").write_text("x = 1\n")

        config = load_problem(tmp_path)
        assert config.state == ["state/solution.py"]

    def test_explicit_state_in_yaml_used(self, tmp_path):
        """When state: is present in YAML, use it as-is."""
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Explicit state.
            state:
              - state/solution.py
            score:
              direction: minimize
        """))
        config = load_problem(tmp_path)
        assert config.state == ["state/solution.py"]


class TestUnknownKeys:
    """Unknown keys in problem.yaml should produce warnings."""

    def test_unknown_top_level_key(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            description: Has extra field.
            bogus_field: hello
            score:
              direction: minimize
        """))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_problem(tmp_path)
        assert any("bogus_field" in str(warning.message) for warning in w)

    def test_unknown_score_key(self, tmp_path):
        (tmp_path / "problem.yaml").write_text(textwrap.dedent("""\
            name: test
            score:
              direction: minimize
              extra: true
        """))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_problem(tmp_path)
        assert any("extra" in str(warning.message) for warning in w)

    def test_no_warning_for_valid_keys(self, tmp_path, full_problem_yaml):
        (tmp_path / "problem.yaml").write_text(full_problem_yaml)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_problem(tmp_path)
        assert len(w) == 0
