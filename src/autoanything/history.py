"""SQLite evaluation history — all database operations.

Manages the evaluations and incumbent tables. Pure SQLite operations,
no git or scoring knowledge.
"""

import json
import sqlite3
from datetime import datetime, timezone


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the evaluation database if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_sha      TEXT NOT NULL,
            branch          TEXT NOT NULL,
            score           REAL,
            status          TEXT NOT NULL,
            description     TEXT,
            submitted_at    TEXT,
            evaluated_at    TEXT,
            duration_seconds REAL,
            error_message   TEXT,
            metrics_json    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incumbent (
            id              INTEGER PRIMARY KEY CHECK (id = 1),
            commit_sha      TEXT NOT NULL,
            score           REAL NOT NULL,
            promoted_at     TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_incumbent(conn: sqlite3.Connection):
    """Get current incumbent score and commit."""
    row = conn.execute("SELECT commit_sha, score FROM incumbent WHERE id = 1").fetchone()
    if row:
        return {"commit_sha": row[0], "score": row[1]}
    return None


def update_incumbent(conn: sqlite3.Connection, commit_sha: str, score: float):
    """Update the incumbent to a new best."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO incumbent (id, commit_sha, score, promoted_at)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            commit_sha = excluded.commit_sha,
            score = excluded.score,
            promoted_at = excluded.promoted_at
    """, (commit_sha, score, now))
    conn.commit()


def record_evaluation(conn: sqlite3.Connection, commit_sha: str, branch: str,
                      score, status: str, description: str,
                      duration: float, error_message: str = None,
                      metrics: dict = None):
    """Record an evaluation result."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO evaluations (commit_sha, branch, score, status, description,
                                 submitted_at, evaluated_at, duration_seconds,
                                 error_message, metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (commit_sha, branch, score, status, description, now, now, duration,
          error_message, json.dumps(metrics) if metrics else None))
    conn.commit()


def is_evaluated(conn: sqlite3.Connection, commit_sha: str) -> bool:
    """Check if a commit has already been evaluated."""
    row = conn.execute(
        "SELECT 1 FROM evaluations WHERE commit_sha = ?", (commit_sha,)
    ).fetchone()
    return row is not None
