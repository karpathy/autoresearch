"""SQLite database connection and schema for ResearchFS v0."""

import os
import sqlite3
from pathlib import Path

DB_NAME = "researchfs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    url TEXT,
    publish_date TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section TEXT,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS methods (
    method_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    implementation_hint TEXT,
    risks TEXT,
    category TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS method_sources (
    method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    PRIMARY KEY (method_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS experiment_briefs (
    brief_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    objective TEXT,
    constraints TEXT,
    hypotheses TEXT,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_run_id INTEGER,
    commit_hash TEXT NOT NULL,
    val_bpb REAL,
    memory_gb REAL,
    status TEXT NOT NULL,
    description TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_runs (
    agent_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    model TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_run_id INTEGER REFERENCES agent_runs(agent_run_id) ON DELETE CASCADE,
    experiment_id INTEGER REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    query TEXT,
    retrieved_content TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS token_usage (
    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieved_methods (
    event_id INTEGER NOT NULL REFERENCES retrieval_events(event_id) ON DELETE CASCADE,
    method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
    PRIMARY KEY(event_id, method_id)
);
"""


def get_connection() -> sqlite3.Connection:
    """Return a connection to the project SQLite database."""
    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / DB_NAME
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(conn: sqlite3.Connection | None = None) -> None:
    """Create ResearchFS tables if they do not exist."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    initialize_db()
    print(f"Initialized {DB_NAME}")
