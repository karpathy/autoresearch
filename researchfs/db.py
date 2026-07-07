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


ARXIV_JOB_TYPES = (
    "sync_metadata",
    "download_pdf",
    "download_source",
    "import_s3_bin",
    "extract_s3_bin",
    "parse_pdf",
    "parse_source",
    "extract_references",
    "chunk_text",
    "embed_chunks",
    "index_bm25",
    "index_vector",
)

_ARXIV_JOB_TYPE_VALUES_SQL = ", ".join(f"'{job_type}'" for job_type in ARXIV_JOB_TYPES)

SCHEMA = f"""
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

CREATE TABLE IF NOT EXISTS paper_code_repositories (
    code_repository_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    repo_url TEXT NOT NULL,
    repo_key TEXT NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    evidence TEXT,
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    selected INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0, 1)),
    discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
    selected_at TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, repo_key)
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

CREATE TABLE IF NOT EXISTS arxiv_papers (
    arxiv_id TEXT PRIMARY KEY,
    latest_version INTEGER NOT NULL DEFAULT 1,
    title TEXT,
    abstract TEXT,
    authors_json TEXT,
    primary_category TEXT,
    categories_json TEXT,
    license TEXT,
    doi TEXT,
    journal_ref TEXT,
    abs_url TEXT,
    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
    oai_datestamp TEXT,
    raw_oai_xml TEXT
);

CREATE TABLE IF NOT EXISTS arxiv_versions (
    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    submitted_at TEXT,
    announced_at TEXT,
    title TEXT,
    abstract TEXT,
    authors_json TEXT,
    categories_json TEXT,
    raw_version_metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (arxiv_id, version),
    FOREIGN KEY (arxiv_id) REFERENCES arxiv_papers(arxiv_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS arxiv_assets (
    asset_key TEXT PRIMARY KEY,
    arxiv_id TEXT NOT NULL,
    version INTEGER,
    asset_type TEXT NOT NULL,
    status TEXT NOT NULL,
    storage_path TEXT,
    checksum TEXT,
    byte_size INTEGER,
    source TEXT,
    error_message TEXT,
    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (arxiv_id) REFERENCES arxiv_papers(arxiv_id) ON DELETE CASCADE,
    CHECK (asset_type IN (
        'metadata',
        'pdf',
        'source',
        'parsed_text',
        'chunks',
        'embedding',
        'bm25_index',
        'vector_index'
    )),
    CHECK (status IN (
        'seen',
        'queued',
        'running',
        'downloaded',
        'imported',
        'parsed',
        'indexed',
        'failed',
        'skipped'
    ))
);

CREATE TABLE IF NOT EXISTS arxiv_chunks (
    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    chunks_asset_key TEXT NOT NULL,
    source_asset_key TEXT,
    source_asset_type TEXT,
    chunks_checksum TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (arxiv_id, version, chunk_index),
    FOREIGN KEY (arxiv_id) REFERENCES arxiv_papers(arxiv_id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS arxiv_jobs (
    job_key TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    arxiv_id TEXT,
    version INTEGER,
    asset_type TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER DEFAULT 100,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 5,
    run_after TEXT DEFAULT CURRENT_TIMESTAMP,
    locked_by TEXT,
    locked_at TEXT,
    payload_json TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    CHECK (job_type IN ({_ARXIV_JOB_TYPE_VALUES_SQL})),
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'skipped'))
);

CREATE TABLE IF NOT EXISTS arxiv_sync_state (
    sync_name TEXT PRIMARY KEY,
    metadata_prefix TEXT NOT NULL DEFAULT 'arXivRaw',
    set_spec TEXT,
    committed_from TEXT,
    last_response_date TEXT,
    pending_response_date TEXT,
    last_resumption_token TEXT,
    status TEXT NOT NULL DEFAULT 'idle',
    records_seen_last_run INTEGER DEFAULT 0,
    records_updated_last_run INTEGER DEFAULT 0,
    records_inserted_last_run INTEGER DEFAULT 0,
    versions_inserted_last_run INTEGER DEFAULT 0,
    jobs_queued_last_run INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    CHECK (status IN ('idle', 'running', 'succeeded', 'failed'))
);

CREATE TABLE IF NOT EXISTS arxiv_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    arxiv_id TEXT,
    version INTEGER,
    asset_type TEXT,
    job_key TEXT,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arxiv_s3_bins (
    bin_key TEXT PRIMARY KEY,
    asset_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER,
    checksum TEXT,
    first_item TEXT,
    last_item TEXT,
    item_count INTEGER,
    status TEXT NOT NULL DEFAULT 'seen',
    storage_path TEXT,
    downloaded_at TEXT,
    imported_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS arxiv_bin_members (
    bin_key TEXT NOT NULL,
    arxiv_id TEXT NOT NULL,
    version INTEGER,
    asset_type TEXT NOT NULL,
    storage_path TEXT,
    checksum TEXT,
    PRIMARY KEY (bin_key, arxiv_id, version, asset_type),
    FOREIGN KEY (bin_key) REFERENCES arxiv_s3_bins(bin_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS arxiv_bloom_filters (
    filter_name TEXT PRIMARY KEY,
    false_positive_rate REAL NOT NULL,
    bit_count INTEGER NOT NULL,
    hash_count INTEGER NOT NULL,
    item_count INTEGER NOT NULL,
    bits BLOB NOT NULL,
    source_table TEXT NOT NULL,
    source_column TEXT NOT NULL,
    rebuilt_at TEXT DEFAULT CURRENT_TIMESTAMP
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

CREATE VIRTUAL TABLE IF NOT EXISTS arxiv_chunks_fts USING fts5(
    arxiv_id UNINDEXED,
    version UNINDEXED,
    chunk_index UNINDEXED,
    text,
    content='arxiv_chunks',
    content_rowid='rowid',
    tokenize='unicode61'
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

CREATE TRIGGER IF NOT EXISTS arxiv_chunks_ai AFTER INSERT ON arxiv_chunks BEGIN
    INSERT INTO arxiv_chunks_fts(rowid, arxiv_id, version, chunk_index, text)
    VALUES (new.rowid, new.arxiv_id, new.version, new.chunk_index, new.text);
END;

CREATE TRIGGER IF NOT EXISTS arxiv_chunks_ad AFTER DELETE ON arxiv_chunks BEGIN
    INSERT INTO arxiv_chunks_fts(arxiv_chunks_fts, rowid, arxiv_id, version, chunk_index, text)
    VALUES ('delete', old.rowid, old.arxiv_id, old.version, old.chunk_index, old.text);
END;

CREATE TRIGGER IF NOT EXISTS arxiv_chunks_au AFTER UPDATE ON arxiv_chunks BEGIN
    INSERT INTO arxiv_chunks_fts(arxiv_chunks_fts, rowid, arxiv_id, version, chunk_index, text)
    VALUES ('delete', old.rowid, old.arxiv_id, old.version, old.chunk_index, old.text);
    INSERT INTO arxiv_chunks_fts(rowid, arxiv_id, version, chunk_index, text)
    VALUES (new.rowid, new.arxiv_id, new.version, new.chunk_index, new.text);
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
CREATE INDEX IF NOT EXISTS idx_arxiv_papers_oai_datestamp ON arxiv_papers(oai_datestamp);
CREATE INDEX IF NOT EXISTS idx_arxiv_papers_primary_category ON arxiv_papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_arxiv_versions_arxiv_id ON arxiv_versions(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_arxiv_assets_arxiv_type_status ON arxiv_assets(arxiv_id, asset_type, status);
CREATE INDEX IF NOT EXISTS idx_arxiv_chunks_asset_key ON arxiv_chunks(chunks_asset_key);
CREATE INDEX IF NOT EXISTS idx_arxiv_chunks_updated_at ON arxiv_chunks(updated_at);
CREATE INDEX IF NOT EXISTS idx_arxiv_jobs_status_run_after ON arxiv_jobs(status, run_after, priority);
CREATE INDEX IF NOT EXISTS idx_arxiv_events_arxiv_id ON arxiv_events(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_arxiv_s3_bins_status ON arxiv_s3_bins(status);
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


FTS_TABLES = ("methods_fts", "chunks_fts", "code_symbols_fts", "arxiv_chunks_fts")


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




def _ensure_arxiv_sync_state_scope_columns(conn: sqlite3.Connection) -> None:
    """Add sync scope columns to legacy arxiv_sync_state tables."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_sync_state'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "arxiv_sync_state")
    scope_columns = {
        "metadata_prefix": "TEXT NOT NULL DEFAULT 'arXivRaw'",
        "set_spec": "TEXT",
    }
    for column_name, column_type in scope_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE arxiv_sync_state ADD COLUMN {column_name} {column_type}"
            )



def _ensure_arxiv_sync_state_counter_columns(conn: sqlite3.Connection) -> None:
    """Add durable last-run insertion counters to legacy arxiv_sync_state tables."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_sync_state'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "arxiv_sync_state")
    counter_columns = {
        "records_inserted_last_run": "INTEGER DEFAULT 0",
        "versions_inserted_last_run": "INTEGER DEFAULT 0",
    }
    for column_name, column_type in counter_columns.items():
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE arxiv_sync_state ADD COLUMN {column_name} {column_type}"
            )



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


def _ensure_paper_code_repository_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes for paper_code_repositories after create or rebuild."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_code_repositories_paper_id "
        "ON paper_code_repositories(paper_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_code_repositories_repo_key "
        "ON paper_code_repositories(repo_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_code_repositories_selected "
        "ON paper_code_repositories(paper_id, selected)"
    )


def _legacy_column_expr(columns: set[str], *names: str, default: str = "NULL") -> str:
    for name in names:
        if name in columns:
            return name
    return default


def _ensure_paper_code_repository_columns(conn: sqlite3.Connection) -> None:
    """Keep paper_code_repositories on the canonical repo_url/repo_key schema."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'paper_code_repositories'"
    ).fetchone()
    if table_exists is None:
        return

    existing_columns = _table_columns(conn, "paper_code_repositories")
    deprecated_aliases = {"repository_url", "repository_key"} & existing_columns
    missing_canonical = {"repo_url", "repo_key"} - existing_columns
    if not deprecated_aliases and not missing_canonical:
        _ensure_paper_code_repository_indexes(conn)
        return

    legacy_table = "_paper_code_repositories_legacy"
    conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")
    conn.execute(f"ALTER TABLE paper_code_repositories RENAME TO {legacy_table}")
    conn.execute(
        """
        CREATE TABLE paper_code_repositories (
            code_repository_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            repo_url TEXT NOT NULL,
            repo_key TEXT NOT NULL,
            owner TEXT NOT NULL,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            evidence TEXT,
            confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
            selected INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0, 1)),
            discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            selected_at TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(paper_id, repo_key)
        )
        """
    )
    legacy_columns = _table_columns(conn, legacy_table)
    code_repository_id = _legacy_column_expr(legacy_columns, "code_repository_id", default="NULL")
    paper_id = _legacy_column_expr(legacy_columns, "paper_id")
    repo_url = _legacy_column_expr(legacy_columns, "repo_url", "repository_url")
    repo_key = _legacy_column_expr(legacy_columns, "repo_key", "repository_key")
    owner = _legacy_column_expr(legacy_columns, "owner", default="''")
    name = _legacy_column_expr(legacy_columns, "name", default="''")
    source = _legacy_column_expr(legacy_columns, "source", default="'migration'")
    evidence = _legacy_column_expr(legacy_columns, "evidence")
    confidence = _legacy_column_expr(legacy_columns, "confidence", default="0.0")
    selected = _legacy_column_expr(legacy_columns, "selected", default="0")
    discovered_at = _legacy_column_expr(legacy_columns, "discovered_at", "created_at", default="CURRENT_TIMESTAMP")
    selected_at = _legacy_column_expr(legacy_columns, "selected_at")
    updated_at = _legacy_column_expr(legacy_columns, "updated_at", default="CURRENT_TIMESTAMP")
    conn.execute(
        f"""
        INSERT OR IGNORE INTO paper_code_repositories (
            code_repository_id, paper_id, repo_url, repo_key, owner, name,
            source, evidence, confidence, selected, discovered_at, selected_at, updated_at
        )
        SELECT
            {code_repository_id},
            {paper_id},
            {repo_url},
            {repo_key},
            {owner},
            {name},
            {source},
            {evidence},
            COALESCE({confidence}, 0.0),
            COALESCE({selected}, 0),
            COALESCE({discovered_at}, CURRENT_TIMESTAMP),
            {selected_at},
            COALESCE({updated_at}, CURRENT_TIMESTAMP)
        FROM {legacy_table}
        WHERE {paper_id} IS NOT NULL
          AND {repo_url} IS NOT NULL
          AND {repo_key} IS NOT NULL
        """
    )
    conn.execute(f"DROP TABLE {legacy_table}")
    _ensure_paper_code_repository_indexes(conn)

def _ensure_arxiv_jobs_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes for arxiv_jobs after create or rebuild."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_arxiv_jobs_status_run_after "
        "ON arxiv_jobs(status, run_after, priority)"
    )


def _ensure_arxiv_jobs_job_type_constraint(conn: sqlite3.Connection) -> None:
    """Rebuild legacy arxiv_jobs tables so job_type is guarded by SQLite."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_jobs'"
    ).fetchone()
    if row is None:
        return
    create_sql = row["sql"] if isinstance(row, sqlite3.Row) else row[0]
    if create_sql and "JOB_TYPE IN" in create_sql.upper():
        return

    existing_columns = _table_columns(conn, "arxiv_jobs")
    new_table = "arxiv_jobs__job_type_migration_new"
    create_new_table_sql = f"""
        CREATE TABLE {new_table} (
            job_key TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            arxiv_id TEXT,
            version INTEGER,
            asset_type TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            priority INTEGER DEFAULT 100,
            attempts INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 5,
            run_after TEXT DEFAULT CURRENT_TIMESTAMP,
            locked_by TEXT,
            locked_at TEXT,
            payload_json TEXT,
            error_message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            CHECK (job_type IN ({_ARXIV_JOB_TYPE_VALUES_SQL})),
            CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'skipped'))
        )
    """
    copy_expressions = {
        "job_key": _legacy_column_expr(existing_columns, "job_key"),
        "job_type": _legacy_column_expr(existing_columns, "job_type"),
        "arxiv_id": _legacy_column_expr(existing_columns, "arxiv_id"),
        "version": _legacy_column_expr(existing_columns, "version"),
        "asset_type": _legacy_column_expr(existing_columns, "asset_type"),
        "status": "COALESCE("
        + _legacy_column_expr(existing_columns, "status", default="'queued'")
        + ", 'queued')",
        "priority": f"COALESCE({_legacy_column_expr(existing_columns, 'priority', default='100')}, 100)",
        "attempts": f"COALESCE({_legacy_column_expr(existing_columns, 'attempts', default='0')}, 0)",
        "max_attempts": f"COALESCE({_legacy_column_expr(existing_columns, 'max_attempts', default='5')}, 5)",
        "run_after": f"COALESCE({_legacy_column_expr(existing_columns, 'run_after', default='CURRENT_TIMESTAMP')}, CURRENT_TIMESTAMP)",
        "locked_by": _legacy_column_expr(existing_columns, "locked_by"),
        "locked_at": _legacy_column_expr(existing_columns, "locked_at"),
        "payload_json": _legacy_column_expr(existing_columns, "payload_json"),
        "error_message": _legacy_column_expr(existing_columns, "error_message"),
        "created_at": f"COALESCE({_legacy_column_expr(existing_columns, 'created_at', default='CURRENT_TIMESTAMP')}, CURRENT_TIMESTAMP)",
        "updated_at": f"COALESCE({_legacy_column_expr(existing_columns, 'updated_at', default='CURRENT_TIMESTAMP')}, CURRENT_TIMESTAMP)",
    }
    columns_sql = ", ".join(copy_expressions)
    select_sql = ", ".join(copy_expressions[column] for column in copy_expressions)

    conn.execute("SAVEPOINT arxiv_jobs_job_type_constraint")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {new_table}")
        conn.execute(create_new_table_sql)
        conn.execute(
            f"""
            INSERT INTO {new_table} ({columns_sql})
            SELECT {select_sql}
            FROM arxiv_jobs
            """
        )
        conn.execute("DROP INDEX IF EXISTS idx_arxiv_jobs_status_run_after")
        conn.execute("DROP TABLE arxiv_jobs")
        conn.execute(f"ALTER TABLE {new_table} RENAME TO arxiv_jobs")
        _ensure_arxiv_jobs_indexes(conn)
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT arxiv_jobs_job_type_constraint")
        conn.execute("RELEASE SAVEPOINT arxiv_jobs_job_type_constraint")
        raise
    else:
        conn.execute("RELEASE SAVEPOINT arxiv_jobs_job_type_constraint")


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
        _ensure_arxiv_jobs_job_type_constraint(conn)
        _ensure_arxiv_sync_state_scope_columns(conn)
        _ensure_arxiv_sync_state_counter_columns(conn)
        _ensure_paper_extraction_status_columns(conn)
        _ensure_paper_extraction_status_external_id(conn)
        _ensure_method_source_evidence_columns(conn)
        _ensure_method_feedback_columns(conn)
        _ensure_paper_code_repository_columns(conn)
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
                  AND name IN (?, ?, ?, ?)
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
