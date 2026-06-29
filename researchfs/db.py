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

CREATE VIRTUAL TABLE IF NOT EXISTS methods_fts USING fts5(
    method_id UNINDEXED,
    name,
    description,
    implementation_hint,
    risks,
    category,
    content='methods',
    content_rowid='method_id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    section,
    content,
    content='chunks',
    content_rowid='chunk_id'
);

CREATE TRIGGER IF NOT EXISTS methods_ai AFTER INSERT ON methods BEGIN
    INSERT INTO methods_fts(rowid, method_id, name, description, implementation_hint, risks, category)
    VALUES (new.method_id, new.method_id, new.name, new.description, new.implementation_hint, new.risks, new.category);
END;

CREATE TRIGGER IF NOT EXISTS methods_ad AFTER DELETE ON methods BEGIN
    INSERT INTO methods_fts(methods_fts, rowid, method_id, name, description, implementation_hint, risks, category)
    VALUES ('delete', old.method_id, old.method_id, old.name, old.description, old.implementation_hint, old.risks, old.category);
END;

CREATE TRIGGER IF NOT EXISTS methods_au AFTER UPDATE ON methods BEGIN
    INSERT INTO methods_fts(methods_fts, rowid, method_id, name, description, implementation_hint, risks, category)
    VALUES ('delete', old.method_id, old.method_id, old.name, old.description, old.implementation_hint, old.risks, old.category);
    INSERT INTO methods_fts(rowid, method_id, name, description, implementation_hint, risks, category)
    VALUES (new.method_id, new.method_id, new.name, new.description, new.implementation_hint, new.risks, new.category);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, section, content)
    VALUES (new.chunk_id, new.chunk_id, new.section, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, section, content)
    VALUES ('delete', old.chunk_id, old.chunk_id, old.section, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, section, content)
    VALUES ('delete', old.chunk_id, old.chunk_id, old.section, old.content);
    INSERT INTO chunks_fts(rowid, chunk_id, section, content)
    VALUES (new.chunk_id, new.chunk_id, new.section, new.content);
END;
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a connection to the project SQLite database.

    Args:
        db_path: Optional path to a specific SQLite database file. If omitted,
            the project root database (DB_NAME) is used.
    """
    if db_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        db_path = repo_root / DB_NAME
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_fts_populated(conn: sqlite3.Connection | None = None) -> None:
    """Bulk-populate FTS5 tables from existing rows when they are empty."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        methods_count = conn.execute("SELECT COUNT(*) FROM methods_fts").fetchone()[0]
        if methods_count == 0:
            conn.execute(
                """
                INSERT INTO methods_fts(rowid, method_id, name, description, implementation_hint, risks, category)
                SELECT method_id, method_id, name, description, implementation_hint, risks, category FROM methods
                """
            )
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if chunks_count == 0:
            conn.execute(
                """
                INSERT INTO chunks_fts(rowid, chunk_id, section, content)
                SELECT chunk_id, chunk_id, section, content FROM chunks
                """
            )
        conn.commit()
    finally:
        if close:
            conn.close()


def initialize_db(conn: sqlite3.Connection | None = None) -> None:
    """Create ResearchFS tables if they do not exist."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        conn.executescript(SCHEMA)
        ensure_fts_populated(conn)
        conn.commit()
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    initialize_db()
    print(f"Initialized {DB_NAME}")
