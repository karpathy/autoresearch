"""Tests for autoanything.history — SQLite evaluation history.

The history module manages the evaluations and incumbent tables.
All SQLite operations are isolated here — no git, no scoring knowledge.
"""

import pytest

from autoanything.history import (
    init_db,
    record_evaluation,
    get_incumbent,
    update_incumbent,
    is_evaluated,
)


class TestInitDb:
    """Database initialization creates required tables."""

    def test_creates_evaluations_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_creates_incumbent_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='incumbent'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_idempotent(self, tmp_path):
        """Calling init_db twice on the same path should not fail."""
        db_path = str(tmp_path / "test.db")
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)
        conn2.close()


class TestIncumbent:
    """Incumbent tracks the current best score and commit."""

    def test_no_incumbent_initially(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        assert get_incumbent(conn) is None
        conn.close()

    def test_set_and_get_incumbent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        update_incumbent(conn, "abc123", 42.5)
        inc = get_incumbent(conn)
        assert inc is not None
        assert inc["commit_sha"] == "abc123"
        assert inc["score"] == 42.5
        conn.close()

    def test_update_incumbent_overwrites(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        update_incumbent(conn, "abc123", 42.5)
        update_incumbent(conn, "def456", 10.0)
        inc = get_incumbent(conn)
        assert inc["commit_sha"] == "def456"
        assert inc["score"] == 10.0
        conn.close()


class TestRecordEvaluation:
    """Recording and querying evaluations."""

    def test_record_and_query(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        record_evaluation(
            conn, commit_sha="abc123", branch="proposals/agent/test",
            score=42.5, status="accepted", description="test run",
            duration=1.5,
        )
        row = conn.execute(
            "SELECT commit_sha, score, status FROM evaluations WHERE commit_sha='abc123'"
        ).fetchone()
        assert row is not None
        assert row[0] == "abc123"
        assert row[1] == 42.5
        assert row[2] == "accepted"
        conn.close()

    def test_record_crash(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        record_evaluation(
            conn, commit_sha="crash1", branch="proposals/agent/bad",
            score=None, status="crash", description="boom",
            duration=0.1, error_message="segfault",
        )
        row = conn.execute(
            "SELECT score, status, error_message FROM evaluations WHERE commit_sha='crash1'"
        ).fetchone()
        assert row[0] is None
        assert row[1] == "crash"
        assert row[2] == "segfault"
        conn.close()

    def test_record_with_metrics(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        record_evaluation(
            conn, commit_sha="met1", branch="proposals/agent/metrics",
            score=10.0, status="accepted", description="with extras",
            duration=2.0, metrics={"cost": 10.0, "iterations": 50},
        )
        row = conn.execute(
            "SELECT metrics_json FROM evaluations WHERE commit_sha='met1'"
        ).fetchone()
        assert row[0] is not None
        import json
        m = json.loads(row[0])
        assert m["cost"] == 10.0
        assert m["iterations"] == 50
        conn.close()


class TestIsEvaluated:
    """is_evaluated checks if a commit SHA has been scored before."""

    def test_not_evaluated(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        assert is_evaluated(conn, "unknown") is False
        conn.close()

    def test_evaluated_after_record(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        record_evaluation(
            conn, commit_sha="abc123", branch="test",
            score=1.0, status="accepted", description="t", duration=0.1,
        )
        assert is_evaluated(conn, "abc123") is True
        conn.close()
