"""Integration tests — end-to-end workflows.

These test the full path from init to scoring, verifying that
all modules compose correctly.
"""

import os
import subprocess

import pytest
from click.testing import CliRunner

from autoanything.cli import main as cli


@pytest.fixture
def runner():
    return CliRunner()


class TestInitToScore:
    """Full workflow: init a problem, add scoring logic, run score."""

    def test_init_then_validate(self, runner, tmp_path):
        """init creates a problem that passes validate (minus score.py content)."""
        runner.invoke(cli, ["init", "test-prob", "--dir", str(tmp_path)])
        prob_dir = tmp_path / "test-prob"

        # Add real scoring logic to the scaffold score.py
        score_py = prob_dir / "scoring" / "score.py"
        score_py.write_text(
            'def score():\n    return {"score": 42.0}\n'
        )

        result = runner.invoke(cli, ["validate", "--dir", str(prob_dir)])
        assert result.exit_code == 0

    def test_init_then_score(self, runner, tmp_path):
        """init + add real scoring -> autoanything score works."""
        runner.invoke(cli, [
            "init", "test-prob",
            "--dir", str(tmp_path),
            "--direction", "minimize",
        ])
        prob_dir = tmp_path / "test-prob"

        score_py = prob_dir / "scoring" / "score.py"
        score_py.write_text(
            'def score():\n    return {"score": 42.0}\n'
        )

        result = runner.invoke(cli, ["score", "--dir", str(prob_dir)])
        assert result.exit_code == 0
        assert "42.0" in result.output


class TestExistingProblemStructure:
    """The existing test problems should be loadable with the new config parser."""

    @pytest.mark.parametrize("problem", ["rastrigin", "tsp", "packing"])
    def test_load_existing_problem_yaml(self, problem):
        """Existing problem.yaml files should parse with load_problem."""
        from autoanything.problem import load_problem

        problem_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", problem,
        )
        if not os.path.exists(os.path.join(problem_dir, "problem.yaml")):
            pytest.skip(f"No problem.yaml for {problem}")

        config = load_problem(problem_dir)
        assert config.name  # has a name
        assert config.score.direction in ("minimize", "maximize")


class TestPackageInstallable:
    """The package should be installable and expose the CLI entry point."""

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "autoanything" in result.output.lower() or "usage" in result.output.lower()

    def test_cli_commands_listed(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ("init", "score", "evaluate", "validate", "history", "leaderboard", "plot"):
            assert cmd in result.output
