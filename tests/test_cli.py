"""Tests for autoanything.cli — CLI commands.

Tests use click.testing.CliRunner to invoke commands without subprocesses.
"""

import os
import textwrap

import pytest
from click.testing import CliRunner

from autoanything.cli import main as cli


@pytest.fixture
def runner():
    return CliRunner()


class TestInit:
    """maxx init <name> scaffolds a new problem directory."""

    def test_creates_directory(self, runner, tmp_path):
        result = runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "my-problem").is_dir()

    def test_creates_problem_yaml(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert (tmp_path / "my-problem" / "problem.yaml").exists()

    def test_creates_state_dir(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert (tmp_path / "my-problem" / "state").is_dir()

    def test_creates_scoring_dir(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert (tmp_path / "my-problem" / "scoring").is_dir()

    def test_creates_score_py(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        score_py = tmp_path / "my-problem" / "scoring" / "score.py"
        assert score_py.exists()
        content = score_py.read_text()
        assert "def score()" in content

    def test_creates_gitignore(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        gitignore = tmp_path / "my-problem" / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()
        assert "scoring/" in content
        assert ".autoanything/" in content

    def test_creates_agent_instructions(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert (tmp_path / "my-problem" / "agent_instructions.md").exists()

    def test_direction_in_templates(self, runner, tmp_path):
        runner.invoke(cli, [
            "init", "my-problem",
            "--dir", str(tmp_path),
            "--direction", "maximize",
        ])
        content = (tmp_path / "my-problem" / "problem.yaml").read_text()
        assert "maximize" in content
        instructions = (tmp_path / "my-problem" / "agent_instructions.md").read_text()
        assert "maximize" in instructions

    def test_initializes_git_repo(self, runner, tmp_path):
        runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert (tmp_path / "my-problem" / ".git").is_dir()

    def test_prints_next_steps(self, runner, tmp_path):
        result = runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert "Next steps" in result.output
        assert "maxx validate" in result.output
        assert "score.py" in result.output

    def test_refuses_existing_directory(self, runner, tmp_path):
        (tmp_path / "my-problem").mkdir()
        result = runner.invoke(cli, ["init", "my-problem", "--dir", str(tmp_path)])
        assert result.exit_code != 0


class TestValidate:
    """maxx validate checks problem directory structure."""

    def test_valid_problem_passes(self, runner, problem_dir):
        result = runner.invoke(cli, ["validate", "--dir", str(problem_dir)])
        assert result.exit_code == 0

    def test_missing_problem_yaml_fails(self, runner, tmp_path):
        result = runner.invoke(cli, ["validate", "--dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "problem.yaml" in result.output

    def test_missing_state_dir_fails(self, runner, tmp_path, full_problem_yaml):
        (tmp_path / "problem.yaml").write_text(full_problem_yaml)
        (tmp_path / "scoring").mkdir()
        (tmp_path / "scoring" / "score.py").write_text("def score(): return {}\n")
        # state/ dir missing
        result = runner.invoke(cli, ["validate", "--dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_missing_score_py_fails(self, runner, tmp_path, full_problem_yaml):
        (tmp_path / "problem.yaml").write_text(full_problem_yaml)
        (tmp_path / "state").mkdir()
        (tmp_path / "state" / "solution.py").write_text("x = 1\n")
        # scoring/ missing
        result = runner.invoke(cli, ["validate", "--dir", str(tmp_path)])
        assert result.exit_code != 0 or "score" in result.output.lower()

    def test_scoring_tracked_by_git_warns(self, runner, problem_dir):
        """If scoring/ files are tracked by git, validate should warn."""
        import subprocess
        subprocess.run(["git", "init", "-b", "main"], cwd=str(problem_dir), check=True,
                       capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(problem_dir),
                       check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=str(problem_dir),
                       check=True, capture_output=True)
        # Remove scoring from gitignore so we can track it
        (problem_dir / ".gitignore").write_text(".autoanything/\n")
        subprocess.run(["git", "add", "-A"], cwd=str(problem_dir), check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(problem_dir), check=True,
                       capture_output=True)

        result = runner.invoke(cli, ["validate", "--dir", str(problem_dir)])
        # Should warn about scoring/ being tracked
        assert "scoring" in result.output.lower() or result.exit_code != 0

    def test_gitignore_missing_scoring_warns(self, runner, problem_dir):
        """If .gitignore doesn't exclude scoring/, validate should warn."""
        (problem_dir / ".gitignore").write_text("# nothing\n")
        result = runner.invoke(cli, ["validate", "--dir", str(problem_dir)])
        assert "scoring" in result.output.lower() or "gitignore" in result.output.lower()


class TestScore:
    """maxx score runs scoring once and prints the result."""

    def test_runs_and_prints_score(self, runner, problem_dir):
        result = runner.invoke(cli, ["score", "--dir", str(problem_dir)])
        assert result.exit_code == 0
        assert "42.5" in result.output

    def test_missing_score_py_fails(self, runner, tmp_path, minimal_problem_yaml):
        (tmp_path / "problem.yaml").write_text(minimal_problem_yaml)
        result = runner.invoke(cli, ["score", "--dir", str(tmp_path)])
        assert result.exit_code != 0


class TestHistory:
    """maxx history prints evaluation history."""

    def test_empty_history(self, runner, problem_dir):
        result = runner.invoke(cli, ["history", "--dir", str(problem_dir)])
        assert result.exit_code == 0

    def test_shows_evaluations(self, runner, problem_dir):
        # Pre-populate the history db
        from autoanything.history import init_db, record_evaluation
        db_path = str(problem_dir / ".autoanything" / "history.db")
        conn = init_db(db_path)
        record_evaluation(conn, "abc", "master", 100.0, "baseline", "baseline", 1.0)
        record_evaluation(conn, "def", "proposals/a", 80.0, "accepted", "improved", 2.0)
        conn.close()

        result = runner.invoke(cli, ["history", "--dir", str(problem_dir)])
        assert result.exit_code == 0
        assert "baseline" in result.output
        assert "accepted" in result.output


class TestLeaderboard:
    """maxx leaderboard regenerates leaderboard.md."""

    def test_generates_leaderboard(self, runner, problem_dir):
        from autoanything.history import init_db, record_evaluation, update_incumbent
        db_path = str(problem_dir / ".autoanything" / "history.db")
        conn = init_db(db_path)
        record_evaluation(conn, "abc", "master", 100.0, "baseline", "baseline", 1.0)
        update_incumbent(conn, "abc", 100.0)
        conn.close()

        result = runner.invoke(cli, ["leaderboard", "--dir", str(problem_dir)])
        assert result.exit_code == 0
        assert (problem_dir / "leaderboard.md").exists()


class TestPlot:
    """maxx plot generates a progress chart."""

    def test_no_history_fails(self, runner, problem_dir):
        result = runner.invoke(cli, ["plot", "--dir", str(problem_dir)])
        assert result.exit_code != 0

    def test_plot_help(self, runner):
        result = runner.invoke(cli, ["plot", "--help"])
        assert result.exit_code == 0
        assert "progress chart" in result.output.lower() or "chart" in result.output.lower()
