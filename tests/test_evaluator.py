"""Tests for autoanything.evaluator — the polling evaluation loop.

The evaluator module orchestrates scoring: it finds pending proposals,
scores them, merges improvements, and updates the leaderboard. These
tests verify the core logic with mocked git/scoring operations.
"""

import os

import pytest
from unittest.mock import patch, MagicMock

from autoanything.evaluator import (
    evaluate_proposal,
    establish_baseline,
)
from autoanything.history import init_db, get_incumbent, update_incumbent, record_evaluation


@pytest.fixture
def eval_db(tmp_path):
    """An initialized database for evaluator tests."""
    db_path = str(tmp_path / "history.db")
    conn = init_db(db_path)
    return conn, db_path


def _make_config(**overrides):
    """Build a mock ProblemConfig with sensible defaults."""
    defaults = dict(
        score=MagicMock(name="cost", timeout=900, direction="minimize"),
        git=MagicMock(base_branch="main"),
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


# Patch targets: the evaluator imports git helpers and scoring from their
# source modules, so we patch at the point of use in autoanything.evaluator.
GIT_PATCH = "autoanything.evaluator.git"
HEAD_PATCH = "autoanything.evaluator.get_head_commit"
MSG_PATCH = "autoanything.evaluator.get_commit_message"
SCORE_PATCH = "autoanything.evaluator.run_score"


class TestEstablishBaseline:
    """Baseline establishment scores the current state and records it."""

    @patch(SCORE_PATCH)
    @patch(HEAD_PATCH, return_value="abc1234567890abcdef1234567890abcdef123456")
    @patch(GIT_PATCH)
    def test_records_baseline(self, mock_git, mock_head, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        mock_run_score.return_value = (42.5, {"cost": 42.5}, 1.0, None)

        result = establish_baseline(conn, problem_dir=str(tmp_path), config=_make_config())

        assert result is True
        inc = get_incumbent(conn)
        assert inc is not None
        assert inc["score"] == 42.5

    @patch(SCORE_PATCH)
    @patch(HEAD_PATCH, return_value="abc1234567890abcdef1234567890abcdef123456")
    @patch(GIT_PATCH)
    def test_returns_false_on_score_failure(self, mock_git, mock_head, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        mock_run_score.return_value = (None, None, 1.0, "script failed")

        result = establish_baseline(conn, problem_dir=str(tmp_path), config=_make_config())

        assert result is False
        assert get_incumbent(conn) is None

    @patch(SCORE_PATCH)
    @patch(HEAD_PATCH, return_value="abc1234567890abcdef1234567890abcdef123456")
    @patch(GIT_PATCH)
    def test_writes_leaderboard_and_history(self, mock_git, mock_head, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        mock_run_score.return_value = (42.5, {"cost": 42.5}, 1.0, None)

        establish_baseline(conn, problem_dir=str(tmp_path), config=_make_config())

        leaderboard = (tmp_path / "leaderboard.md").read_text()
        assert "# Leaderboard" in leaderboard
        assert "42.5" in leaderboard

        history = (tmp_path / "history.md").read_text()
        assert "# History" in history
        assert "baseline" in history

    @patch(SCORE_PATCH)
    @patch(HEAD_PATCH, return_value="abc1234567890abcdef1234567890abcdef123456")
    @patch(GIT_PATCH)
    def test_failure_does_not_write_files(self, mock_git, mock_head, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        mock_run_score.return_value = (None, None, 1.0, "script failed")

        establish_baseline(conn, problem_dir=str(tmp_path), config=_make_config())

        assert not (tmp_path / "leaderboard.md").exists()
        assert not (tmp_path / "history.md").exists()


class TestEvaluateProposal:
    """Evaluating a proposal branch: accept, reject, or crash."""

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="improve score")
    @patch(GIT_PATCH)
    def test_accepts_better_score(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        update_incumbent(conn, "base", 100.0)
        mock_run_score.return_value = (50.0, {"cost": 50.0}, 2.0, None)

        evaluate_proposal(
            conn, branch="proposals/agent/test", commit_sha="prop1",
            direction="minimize", problem_dir=str(tmp_path),
            config=_make_config(),
        )

        # Should update incumbent
        inc = get_incumbent(conn)
        assert inc["score"] == 50.0

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="worse attempt")
    @patch(GIT_PATCH)
    def test_rejects_worse_score(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        update_incumbent(conn, "base", 50.0)
        mock_run_score.return_value = (100.0, {"cost": 100.0}, 2.0, None)

        evaluate_proposal(
            conn, branch="proposals/agent/test", commit_sha="prop2",
            direction="minimize", problem_dir=str(tmp_path),
            config=_make_config(),
        )

        # Incumbent unchanged
        inc = get_incumbent(conn)
        assert inc["score"] == 50.0

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="crashed attempt")
    @patch(GIT_PATCH)
    def test_records_crash(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        update_incumbent(conn, "base", 50.0)
        mock_run_score.return_value = (None, None, 0.5, "segfault")

        evaluate_proposal(
            conn, branch="proposals/agent/crash", commit_sha="prop3",
            direction="minimize", problem_dir=str(tmp_path),
            config=_make_config(),
        )

        # Incumbent unchanged
        inc = get_incumbent(conn)
        assert inc["score"] == 50.0

        # Crash recorded
        row = conn.execute(
            "SELECT status, error_message FROM evaluations WHERE commit_sha='prop3'"
        ).fetchone()
        assert row[0] == "crash"
        assert "segfault" in row[1]

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="higher is better")
    @patch(GIT_PATCH)
    def test_maximize_accepts_higher(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        update_incumbent(conn, "base", 50.0)
        mock_run_score.return_value = (100.0, {"accuracy": 100.0}, 2.0, None)

        evaluate_proposal(
            conn, branch="proposals/agent/test", commit_sha="prop4",
            direction="maximize", problem_dir=str(tmp_path),
            config=_make_config(
                score=MagicMock(name="accuracy", timeout=900, direction="maximize"),
            ),
        )

        inc = get_incumbent(conn)
        assert inc["score"] == 100.0

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="accepted improvement")
    @patch(GIT_PATCH)
    def test_accepted_writes_leaderboard_and_history(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        record_evaluation(conn, "base", "main", 100.0, "baseline", "initial", 1.0)
        update_incumbent(conn, "base", 100.0)
        mock_run_score.return_value = (50.0, {"cost": 50.0}, 2.0, None)

        evaluate_proposal(
            conn, branch="proposals/agent/better", commit_sha="prop1",
            direction="minimize", problem_dir=str(tmp_path),
            config=_make_config(),
        )

        leaderboard = (tmp_path / "leaderboard.md").read_text()
        assert "50.0" in leaderboard
        assert "accepted improvement" in leaderboard

        history = (tmp_path / "history.md").read_text()
        assert "accepted" in history
        assert "50.0" in history

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="bad idea")
    @patch(GIT_PATCH)
    def test_rejected_writes_leaderboard_and_history(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        record_evaluation(conn, "base", "main", 50.0, "baseline", "initial", 1.0)
        update_incumbent(conn, "base", 50.0)
        mock_run_score.return_value = (100.0, {"cost": 100.0}, 2.0, None)

        evaluate_proposal(
            conn, branch="proposals/agent/worse", commit_sha="prop2",
            direction="minimize", problem_dir=str(tmp_path),
            config=_make_config(),
        )

        leaderboard = (tmp_path / "leaderboard.md").read_text()
        # Leaderboard only has accepted/baseline, not the rejected proposal
        assert "50.0" in leaderboard

        history = (tmp_path / "history.md").read_text()
        assert "rejected" in history
        assert "100.0" in history
        assert "bad idea" in history

    @patch(SCORE_PATCH)
    @patch(MSG_PATCH, return_value="exploded")
    @patch(GIT_PATCH)
    def test_crash_writes_leaderboard_and_history(self, mock_git, mock_msg, mock_run_score, eval_db, tmp_path):
        conn, db_path = eval_db
        record_evaluation(conn, "base", "main", 50.0, "baseline", "initial", 1.0)
        update_incumbent(conn, "base", 50.0)
        mock_run_score.return_value = (None, None, 0.5, "segfault")

        evaluate_proposal(
            conn, branch="proposals/agent/crash", commit_sha="prop3",
            direction="minimize", problem_dir=str(tmp_path),
            config=_make_config(),
        )

        leaderboard = (tmp_path / "leaderboard.md").read_text()
        assert "# Leaderboard" in leaderboard

        history = (tmp_path / "history.md").read_text()
        assert "crash" in history
        assert "exploded" in history
