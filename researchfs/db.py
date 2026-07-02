"""SQLite database connection and schema for ResearchFS v0."""

import os
import sqlite3
from pathlib import Path

DB_NAME = "researchfs.db"

DB_PATH_ENV = "RESEARCHFS_DB_PATH"




def _resolve_db_path(db_path: str | Path | None) -> str | Path:
    """Resolve an explicit, configured, or project-root ResearchFS DB path."""
    if db_path is not None:
        return db_path
    configured = os.environ.get(DB_PATH_ENV)
    if configured:
        return configured
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / DB_NAME


def _open_sqlite_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open SQLite, creating parent directories for regular filesystem paths."""
    db_path_str = str(db_path)
    uri = db_path_str.startswith("file:")
    if not uri and db_path_str not in ("", ":memory:"):
        expanded = Path(db_path_str).expanduser()
        expanded.parent.mkdir(parents=True, exist_ok=True)
        db_path_str = str(expanded)
    return sqlite3.connect(db_path_str, uri=uri)


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

CREATE TABLE IF NOT EXISTS paper_extraction_status (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('skipped', 'extracted', 'failed')),
    source_type TEXT,
    external_id TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    method_count INTEGER NOT NULL DEFAULT 0,
    source_char_count INTEGER NOT NULL DEFAULT 0,
    compressed_char_count INTEGER NOT NULL DEFAULT 0,
    compression_ratio REAL,
    confidence_score REAL CHECK (confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)),
    confidence_reason TEXT,
    extractor_model TEXT,
    extractor_prompt_version TEXT,
    error_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS paper_visual_indexes (
    visual_index_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    provider TEXT NOT NULL DEFAULT 'pixelrag',
    status TEXT NOT NULL CHECK (status IN ('planned', 'indexed', 'failed')),
    source TEXT,
    tiles_path TEXT,
    embeddings_path TEXT,
    index_path TEXT,
    error_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, provider)
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
    evidence_text TEXT,
    evidence_start INTEGER,
    evidence_end INTEGER,
    evidence_strategy TEXT,
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

CREATE TABLE IF NOT EXISTS method_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES retrieval_events(event_id) ON DELETE SET NULL,
    method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
    rating REAL NOT NULL CHECK (rating >= -1.0 AND rating <= 1.0),
    label TEXT,
    note TEXT,
    source TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS code_symbols (
    symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    docstring TEXT,
    code TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(file_path, symbol_name, symbol_kind, start_line)
);

CREATE TABLE IF NOT EXISTS method_code_targets (
    method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    rationale TEXT,
    PRIMARY KEY(method_id, file_path, symbol_name, symbol_kind)
);

CREATE TABLE IF NOT EXISTS experiment_methods (
    experiment_id INTEGER NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'retrieved',
    note TEXT,
    PRIMARY KEY(experiment_id, method_id, role)
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id INTEGER PRIMARY KEY AUTOINCREMENT,
    method_id INTEGER REFERENCES methods(method_id) ON DELETE CASCADE,
    experiment_id INTEGER REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    content TEXT NOT NULL,
    outcome TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
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

CREATE VIRTUAL TABLE IF NOT EXISTS code_symbols_fts USING fts5(
    file_path,
    symbol_name,
    symbol_kind,
    docstring,
    code,
    content='code_symbols',
    content_rowid='symbol_id'
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

CREATE TRIGGER IF NOT EXISTS code_symbols_ai AFTER INSERT ON code_symbols BEGIN
    INSERT INTO code_symbols_fts(rowid, file_path, symbol_name, symbol_kind, docstring, code)
    VALUES (new.symbol_id, new.file_path, new.symbol_name, new.symbol_kind, new.docstring, new.code);
END;

CREATE TRIGGER IF NOT EXISTS code_symbols_ad AFTER DELETE ON code_symbols BEGIN
    INSERT INTO code_symbols_fts(code_symbols_fts, rowid, file_path, symbol_name, symbol_kind, docstring, code)
    VALUES ('delete', old.symbol_id, old.file_path, old.symbol_name, old.symbol_kind, old.docstring, old.code);
END;

CREATE TRIGGER IF NOT EXISTS code_symbols_au AFTER UPDATE ON code_symbols BEGIN
    INSERT INTO code_symbols_fts(code_symbols_fts, rowid, file_path, symbol_name, symbol_kind, docstring, code)
    VALUES ('delete', old.symbol_id, old.file_path, old.symbol_name, old.symbol_kind, old.docstring, old.code);
    INSERT INTO code_symbols_fts(rowid, file_path, symbol_name, symbol_kind, docstring, code)
    VALUES (new.symbol_id, new.file_path, new.symbol_name, new.symbol_kind, new.docstring, new.code);
END;

CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_extraction_status_status ON paper_extraction_status(status);
CREATE INDEX IF NOT EXISTS idx_paper_visual_indexes_paper_id ON paper_visual_indexes(paper_id);
CREATE INDEX IF NOT EXISTS idx_method_sources_method_id ON method_sources(method_id);
CREATE INDEX IF NOT EXISTS idx_method_sources_chunk_id ON method_sources(chunk_id);
CREATE INDEX IF NOT EXISTS idx_experiments_agent_run_id ON experiments(agent_run_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_agent_run_id ON retrieval_events(agent_run_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_experiment_id ON retrieval_events(experiment_id);
CREATE INDEX IF NOT EXISTS idx_token_usage_experiment_id ON token_usage(experiment_id);
CREATE INDEX IF NOT EXISTS idx_retrieved_methods_method_id ON retrieved_methods(method_id);
CREATE INDEX IF NOT EXISTS idx_method_feedback_method_id ON method_feedback(method_id);
CREATE INDEX IF NOT EXISTS idx_method_feedback_event_id ON method_feedback(event_id);
CREATE INDEX IF NOT EXISTS idx_code_symbols_file_path ON code_symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_method_code_targets_method_id ON method_code_targets(method_id);
CREATE INDEX IF NOT EXISTS idx_experiment_methods_method_id ON experiment_methods(method_id);
CREATE INDEX IF NOT EXISTS idx_lessons_method_id ON lessons(method_id);
CREATE INDEX IF NOT EXISTS idx_lessons_experiment_id ON lessons(experiment_id);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a connection to the project SQLite database.

    Args:
        db_path: Optional path to a specific SQLite database file. If omitted,
            RESEARCHFS_DB_PATH is honored before falling back to the project
            root database (DB_NAME). ``file:`` URIs are opened with SQLite URI
            mode enabled.
    """
    conn = _open_sqlite_connection(_resolve_db_path(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


FTS_TABLES = ("methods_fts", "chunks_fts", "code_symbols_fts")


def ensure_fts_populated(conn: sqlite3.Connection | None = None) -> None:
    """Rebuild FTS5 indexes from their content tables.

    FTS5 external-content tables can appear non-empty when scanned without
    MATCH even if their internal index was created after the content rows.
    A rebuild is the reliable way to make legacy rows searchable.
    """
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        for table in FTS_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if exists is None:
                continue
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
        conn.commit()
    finally:
        if close:
            conn.close()



def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return column names for an existing SQLite table."""
    return {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})")
    }




def _ensure_paper_extraction_status_columns(conn: sqlite3.Connection) -> None:
    """Add nullable extraction metadata columns to legacy paper_extraction_status tables."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'paper_extraction_status'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "paper_extraction_status")
    metadata_columns = {
        "confidence_score": "REAL",
        "confidence_reason": "TEXT",
        "extractor_model": "TEXT",
        "extractor_prompt_version": "TEXT",
    }
    for column_name, column_type in metadata_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE paper_extraction_status ADD COLUMN {column_name} {column_type}"
            )


def _ensure_paper_extraction_status_external_id(conn: sqlite3.Connection) -> None:
    """Add a nullable external_id provenance column to legacy paper_extraction_status tables."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'paper_extraction_status'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "paper_extraction_status")
    if "external_id" not in existing_columns:
        conn.execute("ALTER TABLE paper_extraction_status ADD COLUMN external_id TEXT")


def _ensure_method_source_evidence_columns(conn: sqlite3.Connection) -> None:
    """Add nullable evidence columns to legacy method_sources tables."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'method_sources'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "method_sources")
    evidence_columns = {
        "evidence_text": "TEXT",
        "evidence_start": "INTEGER",
        "evidence_end": "INTEGER",
        "evidence_strategy": "TEXT",
    }
    for column_name, column_type in evidence_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE method_sources ADD COLUMN {column_name} {column_type}"
            )



def _ensure_method_feedback_columns(conn: sqlite3.Connection) -> None:
    """Add nullable feedback metadata columns to legacy method_feedback tables."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'method_feedback'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "method_feedback")
    if "source" not in existing_columns:
        conn.execute("ALTER TABLE method_feedback ADD COLUMN source TEXT")


def _backfill_method_source_evidence(conn: sqlite3.Connection) -> None:
    """Populate deterministic leading excerpts for legacy method_sources rows."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'method_sources'"
    ).fetchone()
    if table_exists is None:
        return
    existing_columns = _table_columns(conn, "method_sources")
    required = {"evidence_text", "evidence_start", "evidence_end", "evidence_strategy"}
    if not required <= existing_columns:
        return
    rows = conn.execute(
        """
        SELECT ms.method_id, ms.chunk_id, c.content
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE ms.evidence_text IS NULL
        ORDER BY ms.method_id, ms.chunk_id
        """
    ).fetchall()
    for row in rows:
        method_id = row["method_id"] if isinstance(row, sqlite3.Row) else row[0]
        chunk_id = row["chunk_id"] if isinstance(row, sqlite3.Row) else row[1]
        content = (row["content"] if isinstance(row, sqlite3.Row) else row[2]) or ""
        evidence_text = content[:500]
        evidence_end = len(evidence_text)
        evidence_strategy = "migration_leading_excerpt" if evidence_text else "empty_chunk"
        conn.execute(
            """
            UPDATE method_sources
            SET evidence_text = ?,
                evidence_start = ?,
                evidence_end = ?,
                evidence_strategy = ?
            WHERE method_id = ? AND chunk_id = ?
            """,
            (evidence_text, 0, evidence_end, evidence_strategy, method_id, chunk_id),
        )


def migrate_db(conn: sqlite3.Connection | None = None) -> None:
    """Apply idempotent schema migrations for databases created by older ResearchFS versions."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        _ensure_paper_extraction_status_columns(conn)
        _ensure_paper_extraction_status_external_id(conn)
        _ensure_method_source_evidence_columns(conn)
        _ensure_method_feedback_columns(conn)
        _backfill_method_source_evidence(conn)
        conn.commit()
    finally:
        if close:
            conn.close()


def initialize_db(
    conn: sqlite3.Connection | None = None, *, rebuild_fts: bool = False
) -> None:
    """Create ResearchFS tables if they do not exist.

    Args:
        conn: Optional connection to initialize. If omitted, the default
            ResearchFS database is opened.
        rebuild_fts: Explicitly rebuild FTS5 indexes from content tables.
            FTS is also rebuilt automatically when any FTS table is newly
            created, so legacy rows become searchable after migration.
    """
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        existing_fts_tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name IN (?, ?, ?)
                """,
                FTS_TABLES,
            )
        }
        conn.executescript(SCHEMA)
        migrate_db(conn)
        if rebuild_fts or any(table not in existing_fts_tables for table in FTS_TABLES):
            ensure_fts_populated(conn)
        conn.commit()
    finally:
        if close:
            conn.close()


if __name__ == "__main__":
    initialize_db()
    print(f"Initialized {DB_NAME}")
