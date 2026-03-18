"""
SQLite-backed experiment memory.

Functions are organized in flow order:
1. Connection + table creation
2. Storage (save / retrieve)
3. Normalizing hyperparameters
4. Cosine similarity retrieval
5. Confidence calculation
6. LLM conflict resolution
7. Accept / Reject verdict flow
"""

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from schema import ExperimentRecord, Verdict, VerdictSnapshot

# ---------------------------------------------------------------------------
# 1. Connection + table creation  (already reviewed)
# ---------------------------------------------------------------------------

def connect(db_path: str = "memory.db") -> sqlite3.Connection:
    """Open (or create) the experiment memory database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,   -- JSON blob
            confidence      REAL,
            llm_used        INTEGER NOT NULL DEFAULT 0,
            last_verdict    TEXT CHECK(last_verdict IN ('ACCEPT', 'REJECT'))
        );

        CREATE TABLE IF NOT EXISTS verdict_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id   TEXT    NOT NULL
                            REFERENCES experiments(id) ON DELETE CASCADE,
            recorded_at     TEXT    NOT NULL,
            verdict         TEXT    NOT NULL CHECK(verdict IN ('ACCEPT', 'REJECT')),
            val_bpb         REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_experiment
            ON verdict_snapshots(experiment_id);
    """)
    conn.commit()


def save_experiment():
    pass