"""Tests for autoanything.leaderboard — leaderboard.md rendering.

The leaderboard module reads from the history database and produces
a markdown leaderboard file.
"""

import pytest

from autoanything.leaderboard import export_leaderboard
from autoanything.history import init_db, record_evaluation, update_incumbent


@pytest.fixture
def populated_db(tmp_path):
    """A database with a baseline and several evaluations."""
    db_path = str(tmp_path / "history.db")
    conn = init_db(db_path)
    record_evaluation(conn, "base1", "master", 100.0, "baseline", "initial baseline", 1.0)
    update_incumbent(conn, "base1", 100.0)
    record_evaluation(conn, "acc1", "proposals/a/improve", 80.0, "accepted", "first improvement", 2.0)
    update_incumbent(conn, "acc1", 80.0)
    record_evaluation(conn, "rej1", "proposals/b/worse", 120.0, "rejected", "made it worse", 1.5)
    record_evaluation(conn, "crash1", "proposals/c/bad", None, "crash", "broke everything", 0.5,
                      error_message="segfault")
    record_evaluation(conn, "acc2", "proposals/d/better", 50.0, "accepted", "big win", 3.0)
    update_incumbent(conn, "acc2", 50.0)
    return conn, tmp_path


class TestExportLeaderboard:
    """Leaderboard generation from history."""

    def test_creates_file(self, populated_db):
        conn, tmp_path = populated_db
        out = str(tmp_path / "leaderboard.md")
        export_leaderboard(conn, out, direction="minimize")
        assert (tmp_path / "leaderboard.md").exists()

    def test_contains_header(self, populated_db):
        conn, tmp_path = populated_db
        out = str(tmp_path / "leaderboard.md")
        export_leaderboard(conn, out, direction="minimize")
        content = (tmp_path / "leaderboard.md").read_text()
        assert "# Leaderboard" in content

    def test_top_scores_ordered_for_minimize(self, populated_db):
        conn, tmp_path = populated_db
        out = str(tmp_path / "leaderboard.md")
        export_leaderboard(conn, out, direction="minimize")
        content = (tmp_path / "leaderboard.md").read_text()
        # Best score (50.0) should appear before worse scores
        pos_50 = content.find("50.0")
        pos_80 = content.find("80.0")
        pos_100 = content.find("100.0")
        assert pos_50 < pos_80 < pos_100

    def test_recent_attempts_section(self, populated_db):
        conn, tmp_path = populated_db
        out = str(tmp_path / "leaderboard.md")
        export_leaderboard(conn, out, direction="minimize")
        content = (tmp_path / "leaderboard.md").read_text()
        assert "Recent Attempts" in content

    def test_crash_shown_in_recent(self, populated_db):
        conn, tmp_path = populated_db
        out = str(tmp_path / "leaderboard.md")
        export_leaderboard(conn, out, direction="minimize")
        content = (tmp_path / "leaderboard.md").read_text()
        assert "crash" in content.lower()

    def test_maximize_reverses_order(self, tmp_path):
        db_path = str(tmp_path / "history.db")
        conn = init_db(db_path)
        record_evaluation(conn, "low", "master", 10.0, "baseline", "baseline", 1.0)
        record_evaluation(conn, "high", "proposals/a", 90.0, "accepted", "big score", 1.0)

        out = str(tmp_path / "leaderboard.md")
        export_leaderboard(conn, out, direction="maximize")
        content = (tmp_path / "leaderboard.md").read_text()
        # For maximize, 90.0 should come before 10.0
        pos_90 = content.find("90.0")
        pos_10 = content.find("10.0")
        assert pos_90 < pos_10
        conn.close()
