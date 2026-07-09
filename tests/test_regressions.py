import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest


def _memory_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _init_memory_db():
    from researchfs import db

    conn = _memory_conn()
    db.initialize_db(conn)
    return conn


def test_core_manifest_and_layout_exclude_nanochat_runtime():
    """The root install stays lightweight while the fixture owns heavyweight runtime files."""
    import researchfs
    from researchfs import ingest

    manifest = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_block = manifest.split("[project.optional-dependencies]", 1)[0].lower()
    assert 'name = "researchfs-core"' in project_block
    for dependency in (
        "torch",
        "modal",
        "kernels",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "rustbpe",
        "tiktoken",
    ):
        assert dependency not in project_block

    benchmark_root = REPO_ROOT / "benchmarks" / "nanochat"
    for filename in ("train.py", "prepare.py", "modal_loopgraph_benchmark.py"):
        assert not (REPO_ROOT / filename).exists()
        assert (benchmark_root / filename).is_file()
    assert (benchmark_root / "assets" / "benchmark_results.png").is_file()
    assert (benchmark_root / "assets" / "progress.png").is_file()

    benchmark_manifest = (benchmark_root / "pyproject.toml").read_text(encoding="utf-8").lower()
    for dependency in ("torch", "modal", "pyarrow", "rustbpe", "tiktoken"):
        assert dependency in benchmark_manifest

    assert "seed_initial_data" not in researchfs.__all__
    assert "ensure_seed" not in researchfs.__all__
    assert not hasattr(ingest, "seed_initial_data")
    assert not hasattr(ingest, "ensure_seed")


def test_schema_migrations_are_versioned_recorded_and_idempotent():
    """Fresh databases record the immutable migration sequence exactly once."""
    from researchfs import db

    conn = _memory_conn()
    applied = db.migrate_db(conn)

    assert applied == db.MIGRATIONS
    assert db.get_schema_version(conn) == db.CURRENT_SCHEMA_VERSION
    assert db.get_pending_migrations(conn) == ()
    rows = conn.execute(
        "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert [(row["version"], row["name"]) for row in rows] == [
        (migration.version, migration.name) for migration in db.MIGRATIONS
    ]
    assert all(row["applied_at"] for row in rows)

    timestamps = [row["applied_at"] for row in rows]
    assert db.migrate_db(conn) == ()
    assert [
        row["applied_at"]
        for row in conn.execute(
            "SELECT applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()
    ] == timestamps


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1, "wrong_name")], "expected 'baseline_schema'"),
        ([(2, "constrain_arxiv_job_types")], "gaps at version"),
        ([(999, "future_schema")], "newer than supported"),
    ],
)
def test_schema_migration_history_rejects_unsafe_ledgers(rows, message):
    """A drifted, gapped, or future migration ledger fails closed."""
    from researchfs import db

    conn = _memory_conn()
    conn.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.executemany(
        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)", rows
    )

    with pytest.raises(RuntimeError, match=message):
        db.get_pending_migrations(conn)


def test_migrations_cli_reports_checks_and_upgrades_database(tmp_path):
    """The migration CLI distinguishes pending schemas and upgrades them."""
    from researchfs import db
    from researchfs.migrations import main

    db_path = tmp_path / "migration-cli.db"
    stdout = io.StringIO()
    assert main(["status", "--db-path", str(db_path)], stdout=stdout) == 0
    status = json.loads(stdout.getvalue())
    assert status["ok"] is True
    assert status["up_to_date"] is False
    assert status["current_version"] == 0
    assert status["latest_version"] == db.CURRENT_SCHEMA_VERSION
    assert len(status["pending"]) == len(db.MIGRATIONS)

    stdout = io.StringIO()
    assert main(["check", "--db-path", str(db_path)], stdout=stdout) == 1
    assert json.loads(stdout.getvalue())["ok"] is False

    stdout = io.StringIO()
    assert main(["upgrade", "--db-path", str(db_path)], stdout=stdout) == 0
    upgrade = json.loads(stdout.getvalue())
    assert upgrade["from_version"] == 0
    assert upgrade["to_version"] == db.CURRENT_SCHEMA_VERSION
    assert len(upgrade["applied"]) == len(db.MIGRATIONS)

    stdout = io.StringIO()
    assert main(["check", "--db-path", str(db_path)], stdout=stdout) == 0
    assert json.loads(stdout.getvalue()) == {
        "command": "check",
        "current_version": db.CURRENT_SCHEMA_VERSION,
        "latest_version": db.CURRENT_SCHEMA_VERSION,
        "ok": True,
        "pending": [],
        "up_to_date": True,
    }



@pytest.mark.parametrize("invalid_run_id", [True, "7", 7.5, 0, -1])
def test_start_session_rejects_invalid_explicit_run_id_before_side_effects(
    monkeypatch, invalid_run_id
):
    """Explicit run IDs must be positive non-bool integers before any session work."""
    import researchfs.client as client_module

    def fail_start_agent_run(*_args, **_kwargs):
        raise AssertionError("start_agent_run should not run for explicit run_id")

    class FailSession:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("ResearchFSSession should not be constructed")

    monkeypatch.setattr(client_module.rfs, "start_agent_run", fail_start_agent_run)
    monkeypatch.setattr(client_module, "ResearchFSSession", FailSession)

    client = client_module.ResearchFSClient()

    with pytest.raises(ValueError, match="run_id"):
        client.start_session(run_id=invalid_run_id)


@pytest.mark.parametrize("invalid_agent_run_id", [True, "7", 7.5, 0, -1])
def test_researchfs_session_rejects_invalid_agent_run_id_before_assignment(
    invalid_agent_run_id,
):
    """Direct session construction validates agent run IDs before storing them."""
    from researchfs.client import ResearchFSSession

    session = ResearchFSSession.__new__(ResearchFSSession)

    with pytest.raises(ValueError, match="agent_run_id"):
        ResearchFSSession.__init__(
            session,
            client=object(),
            agent_run_id=invalid_agent_run_id,
        )

    assert not hasattr(session, "agent_run_id")


def test_start_session_uses_valid_explicit_run_id_without_creating_run(monkeypatch):
    """A valid explicit run ID attaches the session to that existing agent run."""
    import researchfs.client as client_module

    start_calls = []

    def fail_start_agent_run(*args, **kwargs):
        start_calls.append((args, kwargs))
        raise AssertionError("start_agent_run should not run for explicit run_id")

    monkeypatch.setattr(client_module.rfs, "start_agent_run", fail_start_agent_run)

    session = client_module.ResearchFSClient().start_session(run_id=42)

    assert session.agent_run_id == 42
    assert start_calls == []


def test_start_session_none_run_id_creates_run_and_uses_returned_id(monkeypatch):
    """Omitting run_id preserves the existing behavior of creating a new run."""
    import researchfs.client as client_module

    conn = object()
    calls = []
    client = client_module.ResearchFSClient()

    def get_connection():
        return conn

    def start_agent_run(*, mode, model, conn):
        calls.append({"mode": mode, "model": model, "conn": conn})
        return 314

    client.get_connection = get_connection
    monkeypatch.setattr(client_module.rfs, "start_agent_run", start_agent_run)

    session = client.start_session(mode="test-mode", model="test-model", run_id=None)

    assert session.agent_run_id == 314
    assert calls == [{"mode": "test-mode", "model": "test-model", "conn": conn}]

@pytest.mark.parametrize("field_name", ["chunk_size", "overlap"])
@pytest.mark.parametrize("invalid_value", [True, "2", 2.5])
def test_chunk_text_rejects_non_integer_bounds_before_chunking(field_name, invalid_value):
    """Extractor chunk bounds reject bools and non-integers with the field name."""
    from researchfs.extractor import chunk_text

    kwargs = {"chunk_size": 4, "overlap": 1}
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        chunk_text("alpha beta gamma", **kwargs)


@pytest.mark.parametrize("invalid_window_size", [True, "2", 2.5])
def test_extract_methods_from_chunks_rejects_non_integer_window_size_before_llm(
    invalid_window_size,
):
    """Invalid window sizes fail validation before prompt assembly or LLM calls."""
    from researchfs.extractor import extract_methods_from_chunks

    llm_calls = []

    def fail_llm(_prompt):
        llm_calls.append(_prompt)
        raise AssertionError("llm_fn should not run for invalid window_size")

    with pytest.raises(ValueError, match="window_size"):
        extract_methods_from_chunks(
            {"title": "bounds"},
            ["chunk zero"],
            fail_llm,
            window_size=invalid_window_size,
        )

    assert llm_calls == []


def test_extractor_integer_bounds_still_allow_valid_chunking_and_windowing():
    """Real integers continue to drive chunking and bounded extraction windows."""
    from researchfs.extractor import chunk_text, extract_methods_from_chunks

    chunks = chunk_text("abcdef", chunk_size=3, overlap=0)
    assert chunks == ["abc", "def"]

    prompts = []

    def empty_methods_llm(prompt):
        prompts.append(prompt)
        return '{"methods": []}'

    result = extract_methods_from_chunks(
        {"title": "bounds"},
        chunks,
        empty_methods_llm,
        window_size=1,
    )

    assert result["chunks"] == chunks
    assert result["methods"] == []
    assert len(prompts) == 2


def test_memory_export_rejects_non_positive_paper_ids_before_export(monkeypatch):
    """CLI parsing rejects invalid paper IDs before any export/DB work runs."""
    import researchfs.memory as memory

    export_calls = []

    def fail_export(*_args, **_kwargs):
        export_calls.append((_args, _kwargs))
        raise AssertionError("export should not run for invalid paper IDs")

    monkeypatch.setattr(memory, "_export", fail_export)

    for argv in (
        ["export", "--paper-id", "0"],
        ["export", "--paper-id", "-1"],
        ["export", "--paper-id", "abc"],
        ["export", "--paper-ids", "0"],
        ["export", "--paper-ids", "1,-2"],
        ["export", "--paper-ids", "1,abc"],
        ["export", "--paper-ids", "1,,2"],
    ):
        stderr = io.StringIO()
        assert memory.main(argv, stderr=stderr) == 2
        assert stderr.getvalue()

    assert export_calls == []


def test_memory_export_parser_deduplicates_paper_ids_and_accepts_zero_max_chars(monkeypatch):
    """Valid export arguments preserve ordered paper IDs and allow max_chars=0."""
    import researchfs.memory as memory

    captured = []

    def capture_export(args, _stdout):
        captured.append(args)
        return 0

    monkeypatch.setattr(memory, "_export", capture_export)

    assert memory.main(["export", "--paper-ids", "2,1,2"], stdout=io.StringIO()) == 0
    assert captured[-1].paper_ids == [2, 1]

    assert (
        memory.main(["export", "--all", "--max-chars", "0"], stdout=io.StringIO())
        == 0
    )
    assert captured[-1].all is True
    assert captured[-1].max_chars == 0


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--min-recall", "nan"),
        ("--min-recall", "inf"),
        ("--min-evidence-coverage", "-inf"),
    ],
)
def test_memory_evaluate_rejects_non_finite_thresholds_before_evaluation(
    monkeypatch, option, value
):
    """CLI parsing rejects non-finite metric thresholds before evaluation work."""
    import researchfs.memory as memory

    evaluate_calls = []

    def fail_evaluate(*_args, **_kwargs):
        evaluate_calls.append((_args, _kwargs))
        raise AssertionError("_evaluate should not run for invalid thresholds")

    monkeypatch.setattr(memory, "_evaluate", fail_evaluate)

    stderr = io.StringIO()

    assert (
        memory.main(
            [
                "evaluate",
                "--paper-id",
                "1",
                "--expected-methods",
                "[]",
                option,
                value,
            ],
            stderr=stderr,
        )
        == 2
    )
    assert stderr.getvalue()
    assert evaluate_calls == []


@pytest.mark.parametrize("option", ["--min-recall", "--min-evidence-coverage"])
@pytest.mark.parametrize("value", ["0", "0.5", "1.0"])
def test_memory_evaluate_accepts_finite_thresholds_without_db_work(
    monkeypatch, option, value
):
    """Finite CLI metric thresholds in range reach _evaluate as floats."""
    import researchfs.memory as memory

    captured = []

    def capture_evaluate(args, _stdout):
        captured.append(args)
        return 0

    monkeypatch.setattr(memory, "_evaluate", capture_evaluate)

    stderr = io.StringIO()

    assert (
        memory.main(
            [
                "evaluate",
                "--paper-id",
                "1",
                "--expected-methods",
                "[]",
                option,
                value,
            ],
            stdout=io.StringIO(),
            stderr=stderr,
        )
        == 0
    )
    assert stderr.getvalue() == ""
    assert len(captured) == 1
    attr = "min_recall" if option == "--min-recall" else "min_evidence_coverage"
    assert getattr(captured[0], attr) == pytest.approx(float(value))


def test_initialize_db_skips_existing_fts_rebuild_unless_explicitly_requested(monkeypatch):
    """Default initialization rebuilds FTS only for new FTS tables; explicit repair still rebuilds."""
    from researchfs import db

    conn = _memory_conn()
    rebuild_calls = []

    def spy_ensure_fts_populated(conn_arg=None):
        rebuild_calls.append(conn_arg)

    monkeypatch.setattr(db, "ensure_fts_populated", spy_ensure_fts_populated)

    db.initialize_db(conn)
    db.initialize_db(conn)
    assert rebuild_calls == [conn]

    db.initialize_db(conn, rebuild_fts=True)
    assert rebuild_calls == [conn, conn]


def test_initialize_db_creates_pixelrag_visual_index_table_and_lookup_index():
    """Visual index status has a durable table keyed for paper lookups."""
    from researchfs import db

    conn = _memory_conn()
    db.initialize_db(conn)

    columns = [
        row["name"]
        for row in conn.execute("PRAGMA table_info(paper_visual_indexes)").fetchall()
    ]
    assert columns == [
        "visual_index_id",
        "paper_id",
        "provider",
        "status",
        "source",
        "tiles_path",
        "embeddings_path",
        "index_path",
        "error_text",
        "created_at",
        "updated_at",
    ]

    indexes = {
        row["name"]
        for row in conn.execute("PRAGMA index_list(paper_visual_indexes)").fetchall()
    }
    assert "idx_paper_visual_indexes_paper_id" in indexes


def test_suggest_query_search_false_discards_stale_query_text():
    """A skip-search decision cannot leak an LLM-provided query into the caller's retrieval path."""
    from researchfs.client import ResearchFSSession

    conn = _init_memory_db()
    client = SimpleNamespace(get_connection=lambda: conn)
    session = ResearchFSSession(client, agent_run_id=1)

    decision = session.suggest_query(
        results_content="commit\tval_bpb\n",
        experiment_brief="try one controlled optimizer change",
        llm_fn=lambda _prompt: json.dumps(
            {"search": False, "query": "Muon LR", "reason": "skip this round"}
        ),
    )

    assert decision["search"] is False
    assert decision["query"] == ""
    assert decision["reason"] == "skip this round"


@pytest.mark.parametrize(
    "invalid_max_method_chars",
    [
        pytest.param(True, id="bool"),
        pytest.param("8", id="string"),
        pytest.param(8.0, id="float"),
        pytest.param(-1, id="negative"),
        pytest.param([8], id="list"),
    ],
)
def test_retrieve_context_rejects_invalid_max_method_chars_before_cache_or_search(
    monkeypatch, invalid_max_method_chars
):
    """Invalid method-pack bounds fail before cache lookup or downstream search."""
    import researchfs.client as client_module
    from researchfs.client import ResearchFSSession

    conn = _init_memory_db()
    session = ResearchFSSession(SimpleNamespace(get_connection=lambda: conn), agent_run_id=7)

    class PoisonCache:
        def __contains__(self, _key):
            raise AssertionError("search_cache should not be checked")

        def __getitem__(self, _key):
            raise AssertionError("search_cache should not be read")

        def __setitem__(self, _key, _value):
            raise AssertionError("search_cache should not be written")

    search_calls = []

    def fail_search_methods_and_ids(*_args, **_kwargs):
        search_calls.append((_args, _kwargs))
        raise AssertionError("search_methods_and_ids should not run")

    session.search_cache = PoisonCache()
    monkeypatch.setattr(client_module.rfs, "search_methods_and_ids", fail_search_methods_and_ids)

    with pytest.raises(ValueError, match="max_method_chars"):
        session.retrieve_context("Muon LR", max_method_chars=invalid_max_method_chars)

    assert search_calls == []


def test_retrieve_context_accepts_zero_and_positive_max_method_chars_in_cache_and_search(
    monkeypatch,
):
    """Valid method-pack bounds are forwarded to search and partition cache entries."""
    import researchfs.client as client_module
    from researchfs.client import ResearchFSSession

    conn = _init_memory_db()
    session = ResearchFSSession(SimpleNamespace(get_connection=lambda: conn), agent_run_id=7)
    search_calls = []

    def fake_search_methods_and_ids(query, conn=None, max_chars=None, agent_run_id=None):
        search_calls.append((query, max_chars, agent_run_id, conn is not None))
        return f"pack-{max_chars}-{len(search_calls)}", [max_chars]

    monkeypatch.setattr(client_module.rfs, "search_methods_and_ids", fake_search_methods_and_ids)

    assert session.retrieve_context("Muon LR", max_method_chars=0) == ("pack-0-1", [0])
    assert session.retrieve_context("Muon LR", max_method_chars=0) == ("pack-0-1", [0])
    assert session.retrieve_context("Muon LR", max_method_chars=12) == ("pack-12-2", [12])
    assert session.retrieve_context("Muon LR", max_method_chars=12) == ("pack-12-2", [12])

    assert search_calls == [
        ("Muon LR", 0, 7, True),
        ("Muon LR", 12, 7, True),
    ]


def test_researchfs_session_retrieval_cache_clears_after_outcome_and_retrieval_logs(monkeypatch):
    """Logged outcomes and retrieval events invalidate cached method packs so reuse sees fresh history."""
    import researchfs.client as client_module
    from researchfs.client import ResearchFSSession

    conn = _init_memory_db()
    session = ResearchFSSession(SimpleNamespace(get_connection=lambda: conn), agent_run_id=7)
    search_calls = []

    def fake_search_methods_and_ids(query, conn=None, max_chars=None, agent_run_id=None):
        search_calls.append((query, max_chars, agent_run_id))
        return f"pack-{len(search_calls)}", [len(search_calls)]

    monkeypatch.setattr(client_module.rfs, "search_methods_and_ids", fake_search_methods_and_ids)
    monkeypatch.setattr(client_module.rfs, "log_experiment", lambda **_kwargs: 101)
    monkeypatch.setattr(client_module.rfs, "log_retrieval_event", lambda **_kwargs: 202)

    assert session.retrieve_context("Muon LR") == ("pack-1", [1])
    assert session.retrieve_context("Muon LR") == ("pack-1", [1])
    assert len(search_calls) == 1

    assert session.log_outcome(2.1, "keep", "improved") == 101
    assert session.retrieve_context("Muon LR") == ("pack-2", [2])

    assert session.log_retrieval("agent_search", "Muon LR", "pack", method_ids=[2]) == 202
    assert session.retrieve_context("Muon LR") == ("pack-3", [3])


def _insert_run(conn, mode, model="gpt-4o-mini"):
    cur = conn.cursor()
    cur.execute("INSERT INTO agent_runs (mode, model) VALUES (?, ?)", (mode, model))
    run_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, memory_gb, status, description)
        VALUES (?, 'baseline', 2.500000, 1.0, 'keep', 'baseline')
        """,
        (run_id,),
    )
    conn.commit()
    return run_id


def test_dry_run_duplicate_planning_uses_target_database_state_without_writes():
    """Dry-run ingestion reports a merge when the target DB already has the extracted method."""
    from researchfs.ingest import ingest_paper

    conn = _init_memory_db()
    existing_id = conn.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES ('Existing Gate Scaling', 'old description', 'old hint', 'old risk', 'attention')
        """
    ).lastrowid
    conn.commit()

    meta = {
        "title": "Gate Scaling Note",
        "authors": "Researcher",
        "abstract": "A note about gate scaling.",
        "url": "https://example.invalid/gate-scaling",
        "publish_date": "2026-07-02",
        "content": "Existing Gate Scaling improves attention gates.",
    }

    def llm_fn(_prompt):
        return json.dumps(
            {
                "methods": [
                    {
                        "name": "Existing Gate Scaling",
                        "description": "new description",
                        "implementation_hint": "new hint",
                        "risks": "new risk",
                        "category": "attention",
                        "source_chunk_indices": [0],
                    }
                ]
            }
        )

    result = ingest_paper(meta, llm_fn=llm_fn, conn=conn, dry_run=True)

    assert result["planned"]["methods"] == []
    assert result["planned"]["merged_methods"] == [
        {"name": "Existing Gate Scaling", "merged_into_id": existing_id}
    ]
    assert conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0] == 1


@pytest.mark.parametrize(
    "env_path_value",
    [
        pytest.param(lambda path: str(path), id="plain-path"),
        pytest.param(lambda path: path.as_uri(), id="file-uri"),
    ],
)
def test_dry_run_without_conn_uses_env_database_for_duplicate_planning(
    tmp_path, monkeypatch, env_path_value
):
    """Dry-run without a conn checks the configured target DB and does not write to it."""
    from researchfs import db
    from researchfs.ingest import ingest_paper

    db_path = tmp_path / "researchfs.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    existing_id = conn.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES ('Env Gate Scaling', 'old description', 'old hint', 'old risk', 'attention')
        """
    ).lastrowid
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0] == 1
    conn.close()

    monkeypatch.setenv("RESEARCHFS_DB_PATH", env_path_value(db_path))
    meta = {
        "title": "Gate Scaling Note",
        "authors": "Researcher",
        "abstract": "A note about gate scaling.",
        "url": "https://example.invalid/env-gate-scaling",
        "publish_date": "2026-07-02",
        "content": "Env Gate Scaling improves attention gates.",
    }

    def llm_fn(_prompt):
        return json.dumps(
            {
                "methods": [
                    {
                        "name": "Env Gate Scaling",
                        "description": "new description",
                        "implementation_hint": "new hint",
                        "risks": "new risk",
                        "category": "attention",
                        "source_chunk_indices": [0],
                    }
                ]
            }
        )

    result = ingest_paper(meta, llm_fn=llm_fn, dry_run=True)

    conn = sqlite3.connect(db_path)
    try:
        assert result["planned"]["merged_methods"] == [
            {"name": "Env Gate Scaling", "merged_into_id": existing_id}
        ]
        assert result["planned"]["methods"] == []
        assert conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0] == 1
    finally:
        conn.close()


def test_migrate_db_repairs_legacy_retrieval_and_codegraph_schema():
    """A minimal legacy DB is upgraded enough for retrieval writes and codegraph reads."""
    from researchfs import db
    from researchfs.codegraph import get_code_map, search_code_symbols
    from researchfs.retrieve import (
        link_experiment_methods,
        log_experiment,
        log_lesson,
        log_method_feedback,
    )

    conn = _memory_conn()
    conn.executescript(
        """
        CREATE TABLE papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            publish_date TEXT
        );
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            section TEXT,
            content TEXT NOT NULL
        );
        CREATE TABLE methods (
            method_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            implementation_hint TEXT,
            risks TEXT,
            category TEXT
        );
        CREATE TABLE method_sources (
            method_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            PRIMARY KEY (method_id, chunk_id)
        );
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES (
            'Legacy Migration Paper',
            'Schema Lab',
            'Legacy paper metadata',
            'https://example.invalid/legacy-migration',
            '2026-07-02'
        );
        INSERT INTO chunks (paper_id, section, content)
        VALUES (
            1,
            'legacy',
            'legacymigrationevidencetoken proves old method_sources gain durable evidence excerpts.'
        );
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES (
            'Legacy Migration Method',
            'Use migrated evidence text.',
            'Read upgraded method_sources rows.',
            'Missing ALTER TABLE migrations break retrieval.',
            'memory'
        );
        INSERT INTO method_sources (method_id, chunk_id) VALUES (1, 1);
        """
    )

    db.migrate_db(conn)

    paper_status_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(paper_extraction_status)")
    }
    assert {
        "paper_id",
        "status",
        "source_type",
        "confidence_score",
        "confidence_reason",
        "extractor_model",
        "extractor_prompt_version",
    } <= paper_status_columns
    conn.execute(
        """
        INSERT INTO paper_extraction_status (
            paper_id, status, source_type, chunk_count, method_count,
            source_char_count, compressed_char_count, compression_ratio,
            confidence_score, confidence_reason,
            extractor_model, extractor_prompt_version, error_text
        )
        VALUES (1, 'extracted', 'legacy', 1, 1, 88, 44, 0.5, 0.9, 'ok', 'legacy-model', 'legacy-v1', NULL)
        """
    )

    migrated_source = conn.execute(
        """
        SELECT evidence_text, evidence_start, evidence_end, evidence_strategy
        FROM method_sources
        WHERE method_id = 1 AND chunk_id = 1
        """
    ).fetchone()
    assert migrated_source["evidence_text"].startswith("legacymigrationevidencetoken")
    assert migrated_source["evidence_start"] == 0
    assert migrated_source["evidence_end"] == len(migrated_source["evidence_text"])
    assert migrated_source["evidence_strategy"] == "migration_leading_excerpt"

    method_feedback_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(method_feedback)")
    }
    assert "source" in method_feedback_columns
    feedback_id = log_method_feedback(
        method_id=1,
        rating=0.75,
        label="useful",
        note="Migrated feedback stores its source.",
        source="migration-regression",
        conn=conn,
    )
    stored_feedback = conn.execute(
        "SELECT source, rating FROM method_feedback WHERE feedback_id = ?",
        (feedback_id,),
    ).fetchone()
    assert stored_feedback["source"] == "migration-regression"
    assert stored_feedback["rating"] == 0.75

    experiment_id = log_experiment(
        "legacy-migration",
        val_bpb=1.23,
        memory_gb=4.5,
        status="keep",
        description="Migrated DB accepts experiment linkage.",
        conn=conn,
    )
    link_experiment_methods(
        experiment_id,
        [1],
        role="validated",
        note="legacy schema now has experiment_methods",
        conn=conn,
    )
    lesson_id = log_lesson(
        "Migrated lesson",
        "legacymigrationlessontoken can be persisted after migration.",
        outcome="adopted",
        method_id=1,
        experiment_id=experiment_id,
        conn=conn,
    )
    linked = conn.execute(
        """
        SELECT role, note
        FROM experiment_methods
        WHERE experiment_id = ? AND method_id = ?
        """,
        (experiment_id, 1),
    ).fetchone()
    assert linked["role"] == "validated"
    assert linked["note"] == "legacy schema now has experiment_methods"
    assert conn.execute("SELECT lesson_id FROM lessons WHERE lesson_id = ?", (lesson_id,)).fetchone()

    symbol_id = conn.execute(
        """
        INSERT INTO code_symbols (
            file_path, symbol_name, symbol_kind, start_line, end_line, docstring, code
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy_code.py",
            "legacy_codegraph_reader",
            "function",
            3,
            5,
            "Migrated codegraph docs",
            "def legacy_codegraph_reader():\n    return 'legacycodegraphtoken'",
        ),
    ).lastrowid
    conn.execute(
        """
        INSERT INTO method_code_targets (method_id, file_path, symbol_name, symbol_kind, rationale)
        VALUES (?, ?, ?, ?, ?)
        """,
        (1, "legacy_code.py", "legacy_codegraph_reader", "function", "migration target"),
    )
    conn.commit()

    code_map = get_code_map(conn=conn)
    assert "legacy_code.py" in code_map
    assert "legacy_codegraph_reader" in code_map
    code_search = search_code_symbols("legacycodegraphtoken", conn=conn)
    assert "legacy_codegraph_reader" in code_search
    assert conn.execute(
        "SELECT rowid FROM code_symbols_fts WHERE rowid = ?",
        (symbol_id,),
    ).fetchone()
    assert conn.execute(
        """
        SELECT rationale
        FROM method_code_targets
        WHERE method_id = ? AND file_path = ? AND symbol_name = ?
        """,
        (1, "legacy_code.py", "legacy_codegraph_reader"),
    ).fetchone()["rationale"] == "migration target"



def test_codegraph_public_readers_reject_invalid_max_chars(tmp_path):
    """Public codegraph readers reject non-integer or negative bounds before clipping."""
    from researchfs.codegraph import get_code_map, refresh_code_map, search_code_symbols

    conn = _init_memory_db()
    source = tmp_path / "bounds_target.py"
    source.write_text(
        'def codegraph_bounds_target():\n'
        '    """codegraphboundstoken searchable doc."""\n'
        '    return "codegraphboundstoken"\n'
    )
    refresh_code_map([source], conn=conn)

    for invalid_max_chars in (True, False, "8", 8.0, -1):
        with pytest.raises(ValueError):
            get_code_map(conn=conn, max_chars=invalid_max_chars)
        with pytest.raises(ValueError):
            search_code_symbols("codegraphboundstoken", conn=conn, max_chars=invalid_max_chars)


def test_codegraph_public_readers_keep_zero_and_positive_max_chars_bounds(tmp_path):
    """Zero is a valid empty clip, while positive bounds still cap output length."""
    from researchfs.codegraph import get_code_map, refresh_code_map, search_code_symbols

    conn = _init_memory_db()
    source = tmp_path / "bounded_output_target.py"
    source.write_text(
        'def bounded_output_target():\n'
        '    """boundedoutputtoken searchable doc."""\n'
        '    return "boundedoutputtoken"\n'
    )
    refresh_code_map([source], conn=conn)

    full_map = get_code_map(conn=conn)
    full_search = search_code_symbols("boundedoutputtoken", conn=conn)
    assert "bounded_output_target" in full_map
    assert "bounded_output_target" in full_search

    assert get_code_map(conn=conn, max_chars=0) == ""
    assert search_code_symbols("boundedoutputtoken", conn=conn, max_chars=0) == ""

    map_bound = 10
    search_bound = 12
    bounded_map = get_code_map(conn=conn, max_chars=map_bound)
    bounded_search = search_code_symbols("boundedoutputtoken", conn=conn, max_chars=search_bound)
    assert 0 < len(bounded_map) <= map_bound
    assert 0 < len(bounded_search) <= search_bound


def test_replace_method_code_targets_accepts_explicit_generic_mappings(tmp_path):
    """Core stores caller-selected links without embedding fixture-specific heuristics."""
    from researchfs.codegraph import refresh_code_map, replace_method_code_targets

    conn = _init_memory_db()
    method_id = conn.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES ('Generic evidence method', 'description', 'hint', 'risk', 'workflow')
        """
    ).lastrowid
    source = tmp_path / "workflow.py"
    source.write_text("def validate_evidence():\n    return True\n", encoding="utf-8")
    symbols = refresh_code_map(source, conn=conn)
    symbol = next(row for row in symbols if row["symbol_name"] == "validate_evidence")

    count = replace_method_code_targets(
        [symbol["file_path"]],
        [
            {
                "method_id": method_id,
                "file_path": symbol["file_path"],
                "symbol_name": symbol["symbol_name"],
                "symbol_kind": symbol["symbol_kind"],
                "rationale": "caller-selected evidence validation target",
            }
        ],
        conn=conn,
    )

    assert count == 1
    row = conn.execute(
        "SELECT * FROM method_code_targets WHERE method_id = ?", (method_id,)
    ).fetchone()
    assert row["symbol_name"] == "validate_evidence"
    assert row["rationale"] == "caller-selected evidence validation target"

def test_memory_bundle_import_reconstructs_existing_paper_metadata_without_raw_chunks():
    """Bundle import reuses matching paper/method rows and creates usable compressed memory."""
    from researchfs.bundle import import_paper_memory_bundle
    from researchfs.retrieve import get_paper_memory, search_methods_and_ids

    conn = _init_memory_db()
    paper_id = conn.execute(
        """
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Existing Metadata Bundle Paper",
            "Bundle Import Lab",
            "Only metadata exists before the compressed bundle arrives.",
            "https://example.invalid/existing-metadata-bundle",
            "2026-07-02",
        ),
    ).lastrowid
    existing_method_id = conn.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Existing Metadata Bundle Gate",
            "A pre-existing method shell with no paper evidence yet.",
            "Import should reuse this method id.",
            "Losing provenance would make the bundle unusable.",
            "memory",
        ),
    ).lastrowid
    conn.commit()

    evidence_text = (
        "existingmetadatabundleevidencetoken compact evidence reconstructed from the bundle"
    )
    bundle = {
        "kind": "researchfs.paper_memory_bundle",
        "version": 1,
        "source_ids": {
            "paper_id": 7001,
            "method_ids": [8001],
            "chunk_ids": [9001],
            "lesson_ids": [10001],
        },
        "paper": {
            "source_paper_id": 7001,
            "title": "Existing Metadata Bundle Paper",
            "authors": "Bundle Import Lab",
            "abstract": "Only metadata exists before the compressed bundle arrives.",
            "url": "https://example.invalid/existing-metadata-bundle",
            "publish_date": "2026-07-02",
        },
        "memory": {
            "max_chars": 600,
            "context": "Existing Metadata Bundle Gate uses existingmetadatabundleevidencetoken.",
        },
        "methods": [
            {
                "source_method_id": 8001,
                "name": "Existing Metadata Bundle Gate",
                "description": "Use existingmetadatabundleevidencetoken from imported compressed memory.",
                "implementation_hint": "Retrieve the compact bundle evidence snippet.",
                "risks": "Importing raw chunks would violate the compressed-only contract.",
                "category": "memory",
                "sources": [
                    {
                        "source_method_id": 8001,
                        "source_chunk_id": 9001,
                        "section": "abstract",
                        "evidence_text": evidence_text,
                        "evidence_start": 11,
                        "evidence_end": 11 + len(evidence_text),
                        "evidence_strategy": "exact",
                    }
                ],
            }
        ],
        "lessons": [
            {
                "source_lesson_id": 10001,
                "source_method_id": 8001,
                "label": "Prefer reconstructed bundle memory",
                "content": "existingmetadatabundlelessontoken should retrieve through the reused method.",
                "outcome": "adopted",
                "created_at": "2026-07-02 00:00:00",
            }
        ],
        "extraction_status": {
            "status": "extracted",
            "chunk_count": 1,
            "method_count": 1,
            "source_char_count": 5000,
            "compressed_char_count": 250,
            "compression_ratio": 0.05,
            "confidence_score": 0.93,
            "confidence_reason": "bundle evidence covers the reused method",
            "extractor_model": "bundle-import-regression-model",
            "extractor_prompt_version": "bundle-import-regression-v1",
        },
    }

    result = import_paper_memory_bundle(bundle, conn=conn)

    assert result["duplicate"] is True
    assert result["paper_id"] == paper_id
    assert result["method_ids"] == [existing_method_id]
    assert result["chunk_ids"]
    assert result["lesson_ids"]
    assert result["source_id_map"]["paper_id"] == {"7001": paper_id}
    assert result["source_id_map"]["method_ids"] == {"8001": existing_method_id}
    assert result["source_id_map"]["chunk_ids"]["9001"] == result["chunk_ids"]
    assert result["source_id_map"]["lesson_ids"]["10001"] == result["lesson_ids"][0]
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1

    stored_source = conn.execute(
        """
        SELECT c.paper_id, c.content, ms.evidence_text, ms.evidence_start,
               ms.evidence_end, ms.evidence_strategy
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE ms.method_id = ?
        """,
        (existing_method_id,),
    ).fetchone()
    assert stored_source["paper_id"] == paper_id
    assert stored_source["content"] == evidence_text
    assert stored_source["evidence_text"] == evidence_text
    assert stored_source["evidence_start"] == 11
    assert stored_source["evidence_end"] == 11 + len(evidence_text)
    assert stored_source["evidence_strategy"] == "exact"

    status = conn.execute(
        """
        SELECT status, source_type, method_count, chunk_count, confidence_score,
               confidence_reason, extractor_model, extractor_prompt_version
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    assert dict(status) == {
        "status": "extracted",
        "source_type": "memory_bundle",
        "method_count": 1,
        "chunk_count": 1,
        "confidence_score": 0.93,
        "confidence_reason": "bundle evidence covers the reused method",
        "extractor_model": "bundle-import-regression-model",
        "extractor_prompt_version": "bundle-import-regression-v1",
    }

    context, method_ids = search_methods_and_ids(
        "existingmetadatabundleevidencetoken", conn=conn, max_chars=1000
    )
    assert existing_method_id in method_ids
    assert "Existing Metadata Bundle Gate" in context
    assert "existingmetadatabundleevidencetoken" in context
    lesson_context, lesson_method_ids = search_methods_and_ids(
        "existingmetadatabundlelessontoken", conn=conn, max_chars=1000
    )
    assert existing_method_id in lesson_method_ids
    assert "Prefer reconstructed bundle memory" in lesson_context
    memory = get_paper_memory(paper_id, conn=conn, max_chars=1000)
    assert "Existing Metadata Bundle Gate" in memory["context"]
    assert "existingmetadatabundleevidencetoken" in memory["context"]


def test_arxiv_ingest_persists_status_provenance_overrides():
    """arXiv ingest stores caller provenance labels on the durable extraction status row."""
    from researchfs.api import ResearchFSLabAPI

    conn = _init_memory_db()

    def fake_fetcher(arxiv_id):
        assert arxiv_id == "2401.00001"
        return {
            "title": "arXiv Provenance Override Paper",
            "authors": "arXiv Lab",
            "abstract": "A fetched arXiv paper with override provenance.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2026-07-02",
            "content": "arxivprovenanceevidencetoken shows the fetched source is ingested.",
        }

    api = ResearchFSLabAPI(conn=conn, arxiv_fetcher=fake_fetcher)
    response = api.dispatch(
        "POST",
        "/ingest/arxiv",
        {
            "id": "2401.00001",
            "source_type": "arxiv_latex",
            "extractor_model": "arxiv-regression-model",
            "extractor_prompt_version": "arxiv-regression-prompt-v1",
            "methods": [
                {
                    "name": "arXiv Provenance Gate",
                    "description": "Use arxivprovenanceevidencetoken from fetched arXiv source.",
                    "implementation_hint": "Trust durable extraction status provenance.",
                    "risks": "Dropping overrides hides where compressed memory came from.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )

    assert response.status == 200
    assert response.body["ok"] is True
    status = response.body["extraction_status"]
    assert status["source_type"] == "arxiv_latex"
    assert status["extractor_model"] == "arxiv-regression-model"
    assert status["extractor_prompt_version"] == "arxiv-regression-prompt-v1"

    stored_status = conn.execute(
        """
        SELECT source_type, extractor_model, extractor_prompt_version
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (response.body["paper_id"],),
    ).fetchone()
    assert dict(stored_status) == {
        "source_type": "arxiv_latex",
        "extractor_model": "arxiv-regression-model",
        "extractor_prompt_version": "arxiv-regression-prompt-v1",
    }


@pytest.mark.parametrize(
    "fetch_error,expected_status,expected_code",
    [
        pytest.param(ValueError("not an arXiv id"), 400, "invalid_arxiv_id", id="value-error"),
        pytest.param(RuntimeError("upstream unavailable"), 502, "fetch_failed", id="generic-error"),
    ],
)
def test_arxiv_ingest_fetcher_errors_are_structured_non_500_responses(
    fetch_error, expected_status, expected_code
):
    """Fetcher failures are surfaced as stable API errors instead of internal 500s."""
    from researchfs.api import ResearchFSLabAPI

    conn = _init_memory_db()

    def failing_fetcher(_arxiv_id):
        raise fetch_error

    api = ResearchFSLabAPI(conn=conn, arxiv_fetcher=failing_fetcher)

    response = api.dispatch("POST", "/ingest/arxiv", {"id": "not-used"})

    assert response.status == expected_status
    assert response.status != 500
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == expected_code
    assert response.body["error"]["message"] == str(fetch_error)


@pytest.mark.parametrize(
    "raw_id,expected",
    [
        ("2305.13245", "2305.13245"),
        ("2305.13245v1", "2305.13245"),
        ("arXiv:2305.13245", "2305.13245"),
        ("arxiv:2305.13245v3", "2305.13245"),
        ("https://arxiv.org/abs/2305.13245", "2305.13245"),
        ("https://arxiv.org/abs/2305.13245v2", "2305.13245"),
        ("https://arxiv.org/pdf/2305.13245.pdf", "2305.13245"),
        ("https://arxiv.org/pdf/2305.13245v1.pdf", "2305.13245"),
    ],
)
def test_normalize_arxiv_id_accepts_common_forms(raw_id, expected):
    """normalize_arxiv_id returns the canonical bare id for common inputs."""
    from researchfs.fetcher import normalize_arxiv_id

    assert normalize_arxiv_id(raw_id) == expected


@pytest.mark.parametrize(
    "raw_id",
    ["", "not-an-id", "https://example.com/abs/2305.13245", "1234.123"],
)
def test_normalize_arxiv_id_rejects_invalid_identifiers(raw_id):
    """normalize_arxiv_id raises ValueError for malformed or unsupported ids."""
    from researchfs.fetcher import normalize_arxiv_id

    with pytest.raises(ValueError, match="Invalid arXiv identifier"):
        normalize_arxiv_id(raw_id)


_ATOM_ENTRY = """\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>  GQA: Training Generalized Multi-Query Transformer Models  </title>
    <author><name>Joshua Ainslie</name></author>
    <author><name>James Lee-Thorp</name></author>
    <summary>Multi-query attention speeds up decoder inference.</summary>
    <published>2023-05-22T00:00:00Z</published>
    <id>https://arxiv.org/abs/2305.13245</id>
  </entry>
</feed>
"""


def test_fetch_arxiv_metadata_api_parses_atom_feed(monkeypatch):
    """fetch_arxiv_metadata_api extracts title, authors, abstract, date, and url from Atom XML."""
    import requests
    from researchfs.fetcher import fetch_arxiv_metadata_api

    class FakeResponse:
        content = _ATOM_ENTRY.encode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda _url, **kwargs: FakeResponse())

    meta = fetch_arxiv_metadata_api("2305.13245")
    assert meta["title"] == "GQA: Training Generalized Multi-Query Transformer Models"
    assert meta["authors"] == "Joshua Ainslie, James Lee-Thorp"
    assert meta["abstract"] == "Multi-query attention speeds up decoder inference."
    assert meta["publish_date"] == "2023-05-22"
    assert meta["url"] == "https://arxiv.org/abs/2305.13245"


@pytest.mark.parametrize(
    ("raw_proxy", "expected_proxy"),
    [
        ("http://proxy-url.example:821", "http://proxy-url.example:821"),
        ("proxy-host.example:822", "http://proxy-host.example:822"),
        (
            "sticky-user:sticky-pass@sticky-host.example:823",
            "http://sticky-user:sticky-pass@sticky-host.example:823",
        ),
        (
            "four-field-host.example:824:four-user:four-pass",
            "http://four-user:four-pass@four-field-host.example:824",
        ),
    ],
)
def test_scraping_proxy_normalization_accepts_common_formats(raw_proxy, expected_proxy):
    """Reusable proxy normalization accepts the formats used by arXiv proxy lists."""
    from researchfs.scraping import normalize_proxy_url

    assert normalize_proxy_url(raw_proxy) == expected_proxy


def test_scraping_proxy_lists_skip_blank_comment_and_invalid_entries(tmp_path):
    """Proxy-list loading keeps only safe fake proxy entries."""
    from researchfs.scraping import load_proxy_list

    proxy_list = tmp_path / "scraping-proxies.txt"
    proxy_list.write_text(
        "\n".join(
            [
                "",
                "   ",
                "# comment only",
                "proxy-one.example:801",
                "://invalid",
                "proxy-two.example:not-a-port",
                "proxy-three.example:803:user:pass",
            ]
        ),
        encoding="utf-8",
    )

    assert load_proxy_list(proxy_list) == [
        {"http": "http://proxy-one.example:801", "https": "http://proxy-one.example:801"},
        {
            "http": "http://user:pass@proxy-three.example:803",
            "https": "http://user:pass@proxy-three.example:803",
        },
    ]


def test_scraping_engine_proxy_list_round_robins_successful_calls(
    monkeypatch, tmp_path
):
    """Successful GET calls advance the process-level round-robin start proxy."""
    import requests
    from researchfs.scraping import ScrapingEngine

    proxy_list = tmp_path / "scraping-proxies.txt"
    proxy_list.write_text(
        "http://proxy-one.example:801\nhttp://proxy-two.example:802\n",
        encoding="utf-8",
    )
    calls = []

    class FakeResponse:
        status_code = 200

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.delenv("SCRAPING_TEST_PROXY_URL", raising=False)
    monkeypatch.setenv("SCRAPING_TEST_PROXY_LIST_PATH", str(proxy_list))
    monkeypatch.setattr(requests, "get", fake_get)

    engine = ScrapingEngine(
        proxy_url_env="SCRAPING_TEST_PROXY_URL",
        proxy_list_path_env="SCRAPING_TEST_PROXY_LIST_PATH",
        retries_env="SCRAPING_TEST_RETRIES",
        retry_delay_env="SCRAPING_TEST_RETRY_DELAY_SECONDS",
        retry_status_codes={429},
    )

    engine.get("https://example.test/one", timeout=5)
    engine.get("https://example.test/two", timeout=5)

    assert [call["proxies"] for call in calls] == [
        {"http": "http://proxy-one.example:801", "https": "http://proxy-one.example:801"},
        {"http": "http://proxy-two.example:802", "https": "http://proxy-two.example:802"},
    ]


def test_scraping_engine_retry_statuses_rotate_proxies_within_call(
    monkeypatch, tmp_path
):
    """Retryable HTTP statuses advance through the proxy list during one call."""
    import requests
    from researchfs.scraping import ScrapingEngine

    proxy_list = tmp_path / "scraping-proxies.txt"
    proxy_list.write_text(
        "http://proxy-one.example:801\nhttp://proxy-two.example:802\n",
        encoding="utf-8",
    )
    statuses = [429, 429, 200]
    calls = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(statuses.pop(0))

    monkeypatch.delenv("SCRAPING_TEST_PROXY_URL", raising=False)
    monkeypatch.setenv("SCRAPING_TEST_PROXY_LIST_PATH", str(proxy_list))
    monkeypatch.setenv("SCRAPING_TEST_RETRIES", "3")
    monkeypatch.setenv("SCRAPING_TEST_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr(requests, "get", fake_get)

    engine = ScrapingEngine(
        proxy_url_env="SCRAPING_TEST_PROXY_URL",
        proxy_list_path_env="SCRAPING_TEST_PROXY_LIST_PATH",
        retries_env="SCRAPING_TEST_RETRIES",
        retry_delay_env="SCRAPING_TEST_RETRY_DELAY_SECONDS",
        retry_status_codes={429},
    )

    response = engine.get("https://example.test/paper", timeout=5)

    assert response.status_code == 200
    assert [call["proxies"] for call in calls] == [
        {"http": "http://proxy-one.example:801", "https": "http://proxy-one.example:801"},
        {"http": "http://proxy-two.example:802", "https": "http://proxy-two.example:802"},
        {"http": "http://proxy-one.example:801", "https": "http://proxy-one.example:801"},
    ]


def test_scraping_engine_request_exceptions_retry_and_raise_final_exception(
    monkeypatch, tmp_path
):
    """Request exceptions retry the next proxy and exhaust by raising the last error."""
    import requests
    from researchfs.scraping import ScrapingEngine

    proxy_list = tmp_path / "scraping-proxies.txt"
    proxy_list.write_text(
        "http://proxy-one.example:801\nhttp://proxy-two.example:802\n",
        encoding="utf-8",
    )
    calls = []
    errors = [
        requests.exceptions.ProxyError("first fake proxy failed"),
        requests.exceptions.ConnectTimeout("second fake proxy failed"),
    ]

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        raise errors[len(calls) - 1]

    monkeypatch.delenv("SCRAPING_TEST_PROXY_URL", raising=False)
    monkeypatch.setenv("SCRAPING_TEST_PROXY_LIST_PATH", str(proxy_list))
    monkeypatch.setenv("SCRAPING_TEST_RETRIES", "2")
    monkeypatch.setenv("SCRAPING_TEST_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr(requests, "get", fake_get)

    engine = ScrapingEngine(
        proxy_url_env="SCRAPING_TEST_PROXY_URL",
        proxy_list_path_env="SCRAPING_TEST_PROXY_LIST_PATH",
        retries_env="SCRAPING_TEST_RETRIES",
        retry_delay_env="SCRAPING_TEST_RETRY_DELAY_SECONDS",
        retry_status_codes={429},
    )

    with pytest.raises(requests.exceptions.ConnectTimeout, match="second fake proxy"):
        engine.get("https://example.test/paper", timeout=5)

    assert [call["proxies"] for call in calls] == [
        {"http": "http://proxy-one.example:801", "https": "http://proxy-one.example:801"},
        {"http": "http://proxy-two.example:802", "https": "http://proxy-two.example:802"},
    ]


def test_scraping_engine_explicit_proxies_kwarg_overrides_env_proxies(
    monkeypatch, tmp_path
):
    """Caller-provided requests proxies are never replaced by environment proxies."""
    import requests
    from researchfs.scraping import ScrapingEngine

    proxy_list = tmp_path / "scraping-proxies.txt"
    proxy_list.write_text("http://list-proxy.example:801\n", encoding="utf-8")
    explicit_proxies = {
        "http": "http://explicit-proxy.example:811",
        "https": "http://explicit-proxy.example:811",
    }
    calls = []

    class FakeResponse:
        status_code = 200

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setenv("SCRAPING_TEST_PROXY_URL", "http://env-proxy.example:812")
    monkeypatch.setenv("SCRAPING_TEST_PROXY_LIST_PATH", str(proxy_list))
    monkeypatch.setattr(requests, "get", fake_get)

    engine = ScrapingEngine(
        proxy_url_env="SCRAPING_TEST_PROXY_URL",
        proxy_list_path_env="SCRAPING_TEST_PROXY_LIST_PATH",
        retries_env="SCRAPING_TEST_RETRIES",
        retry_delay_env="SCRAPING_TEST_RETRY_DELAY_SECONDS",
        retry_status_codes={429},
    )

    engine.get("https://example.test/paper", timeout=5, proxies=explicit_proxies)

    assert calls == [{"timeout": 5, "proxies": explicit_proxies}]


def test_fetch_arxiv_metadata_api_uses_proxy_and_retries_rate_limits(monkeypatch):
    """arXiv metadata fetches keep the legacy single-proxy retry behavior."""
    import requests
    from researchfs.fetcher import fetch_arxiv_metadata_api

    proxy_url = "http://user:pass@proxy.example:823"
    calls = []
    statuses = [429, 200]

    class FakeResponse:
        content = _ATOM_ENTRY.encode()

        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            pass

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(statuses.pop(0))

    monkeypatch.setenv("RESEARCHFS_ARXIV_PROXY_URL", proxy_url)
    monkeypatch.setenv("RESEARCHFS_ARXIV_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr(requests, "get", fake_get)

    meta = fetch_arxiv_metadata_api("2305.13245")

    assert meta["title"] == "GQA: Training Generalized Multi-Query Transformer Models"
    assert len(calls) == 2
    assert all(call["proxies"] == {"http": proxy_url, "https": proxy_url} for call in calls)


def test_arxiv_proxy_list_env_uses_scraping_engine_formats(monkeypatch, tmp_path):
    """The arXiv fetcher remains compatible with reusable proxy-list normalization."""
    import itertools
    import requests
    import researchfs.fetcher as fetcher

    proxy_list = tmp_path / "arxiv-proxies.txt"
    proxy_list.write_text(
        "\n".join(
            [
                "# fake proxies only",
                "proxy-one.example:801",
                "proxy-two.example:802:user:pass",
            ]
        ),
        encoding="utf-8",
    )
    calls = []
    statuses = [429, 200]

    class FakeResponse:
        content = _ATOM_ENTRY.encode()

        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            pass

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(statuses.pop(0))

    monkeypatch.delenv("RESEARCHFS_ARXIV_PROXY_URL", raising=False)
    monkeypatch.setenv("RESEARCHFS_ARXIV_PROXY_LIST_PATH", str(proxy_list))
    monkeypatch.setattr(fetcher, "_ARXIV_PROXY_ROTATION_COUNTER", itertools.count())
    monkeypatch.setenv("RESEARCHFS_ARXIV_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr(requests, "get", fake_get)

    meta = fetcher.fetch_arxiv_metadata_api("2305.13245")

    assert meta["title"] == "GQA: Training Generalized Multi-Query Transformer Models"
    assert [call["proxies"] for call in calls] == [
        {"http": "http://proxy-one.example:801", "https": "http://proxy-one.example:801"},
        {
            "http": "http://user:pass@proxy-two.example:802",
            "https": "http://user:pass@proxy-two.example:802",
        },
    ]


def test_arxiv_single_proxy_env_overrides_proxy_list(monkeypatch, tmp_path):
    """The legacy single-proxy env var remains authoritative when a list is also configured."""
    import requests
    from researchfs.fetcher import fetch_arxiv_metadata_api

    proxy_list = tmp_path / "ignored-arxiv-proxies.txt"
    proxy_list.write_text("http://list-proxy.example:811\n", encoding="utf-8")
    single_proxy = "http://single-proxy.example:812"
    calls = []
    statuses = [429, 200]

    class FakeResponse:
        content = _ATOM_ENTRY.encode()

        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            pass

    def fake_get(_url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(statuses.pop(0))

    monkeypatch.setenv("RESEARCHFS_ARXIV_PROXY_URL", single_proxy)
    monkeypatch.setenv("RESEARCHFS_ARXIV_PROXY_LIST_PATH", str(proxy_list))
    monkeypatch.setenv("RESEARCHFS_ARXIV_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setattr(requests, "get", fake_get)

    fetch_arxiv_metadata_api("2305.13245")

    assert [call["proxies"] for call in calls] == [
        {"http": single_proxy, "https": single_proxy},
        {"http": single_proxy, "https": single_proxy},
    ]


def test_cli_arxiv_proxy_list_sets_env_before_fetcher(capsys, monkeypatch, tmp_path):
    """The ingest CLI forwards --arxiv-proxy-list via env before fetching arXiv."""
    import sys
    import researchfs.ingest as ingest_module

    proxy_list = tmp_path / "cli-arxiv-proxies.txt"
    proxy_list.write_text("http://cli-proxy.example:831\n", encoding="utf-8")
    observed = {}

    def fake_fetcher(arxiv_id):
        observed["arxiv_id"] = arxiv_id
        observed["proxy_list_env"] = os.environ.get("RESEARCHFS_ARXIV_PROXY_LIST_PATH")
        return {
            "title": "CLI Proxy Paper",
            "authors": "CLI Author",
            "abstract": "Abstract.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2023-05-22",
            "content": "content for cli proxy paper",
            "external_id": arxiv_id,
        }

    monkeypatch.setattr(ingest_module, "fetch_arxiv_source", fake_fetcher)

    original_argv = sys.argv
    try:
        sys.argv = [
            "researchfs.ingest",
            "--arxiv",
            "2305.13245",
            "--arxiv-proxy-list",
            str(proxy_list),
            "--dry-run",
        ]
        ingest_module.main()
    finally:
        sys.argv = original_argv

    assert observed == {
        "arxiv_id": "2305.13245",
        "proxy_list_env": str(proxy_list),
    }
    assert "DRY RUN" in capsys.readouterr().out


def test_fetch_arxiv_source_pdf_fallback(monkeypatch, tmp_path):
    """fetch_arxiv_source falls back to PDF extraction when LaTeX source is unavailable."""
    import requests
    from researchfs.fetcher import fetch_arxiv_source

    class FakeResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError("not found")

    def fake_requests_get(url, **_kwargs):
        if "/src/" in url:
            return FakeResponse(b"not a tarball", status_code=404)
        if "/pdf/" in url:
            return FakeResponse(b"fake pdf bytes")
        if "/api/query" in url:
            return FakeResponse(_ATOM_ENTRY.encode())
        raise ValueError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_requests_get)

    def fake_extract_pdf_text(path):
        return "PDF fallback extracted text."

    monkeypatch.setattr("researchfs.fetcher.extract_pdf_text", fake_extract_pdf_text)

    meta = fetch_arxiv_source("2305.13245")
    assert meta["content"] == "PDF fallback extracted text."
    assert meta["external_id"] == "2305.13245"


_ATOM_FEED = """\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2305.13245</id>
    <title>Paper One</title>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2002.05202</id>
    <title>Paper Two</title>
  </entry>
</feed>
"""


def test_fetch_arxiv_query_parses_search_results(monkeypatch):
    """fetch_arxiv_query returns normalized ids from an Atom search feed."""
    import requests
    from researchfs.fetcher import fetch_arxiv_query

    class FakeResponse:
        content = _ATOM_FEED.encode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda _url, **kwargs: FakeResponse())

    ids = fetch_arxiv_query("all:transformers", max_results=2)
    assert ids == ["2305.13245", "2002.05202"]


def test_ingest_arxiv_ids_batch_persists_papers_and_external_ids(monkeypatch):
    """ingest_arxiv_ids stores multiple papers and persists external_id on extraction status."""
    from researchfs.ingest import ingest_arxiv_ids

    conn = _init_memory_db()

    fetched = {}

    def fake_fetcher(arxiv_id):
        fetched[arxiv_id] = True
        return {
            "title": f"Paper {arxiv_id}",
            "authors": "Test Author",
            "abstract": "An abstract.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2023-05-22",
            "content": f"content for {arxiv_id}",
            "external_id": arxiv_id,
        }

    import researchfs.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "fetch_arxiv_source", fake_fetcher)
    results = ingest_arxiv_ids(["2305.13245", "2002.05202"], conn=conn)

    assert len(results) == 2
    assert all(r["status"] == "skipped" for r in results)
    assert all(r["error"] is None for r in results)
    assert set(fetched) == {"2305.13245", "2002.05202"}

    for arxiv_id, expected_title in [("2305.13245", "Paper 2305.13245"), ("2002.05202", "Paper 2002.05202")]:
        row = conn.execute(
            "SELECT p.title, e.external_id FROM papers p JOIN paper_extraction_status e ON p.paper_id = e.paper_id WHERE e.external_id = ?",
            (arxiv_id,),
        ).fetchone()
        assert row is not None
        assert row["title"] == expected_title
        assert row["external_id"] == arxiv_id


def test_ingest_arxiv_query_uses_search_and_ingests(monkeypatch):
    """ingest_arxiv_query searches arXiv and ingests the returned ids."""
    import researchfs.ingest as ingest_module
    from researchfs.ingest import ingest_arxiv_query

    conn = _init_memory_db()

    monkeypatch.setattr(
        ingest_module,
        "fetch_arxiv_query",
        lambda query, max_results: ["2305.13245"],
    )
    monkeypatch.setattr(
        ingest_module,
        "fetch_arxiv_source",
        lambda arxiv_id: {
            "title": "Search Result Paper",
            "authors": "Searcher",
            "abstract": "Found via query.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2023-05-22",
            "content": "search result content",
            "external_id": arxiv_id,
        },
    )
    results = ingest_arxiv_query("all:transformers", conn=conn, max_results=1)

    assert len(results) == 1
    assert results[0]["arxiv_id"] == "2305.13245"
    assert results[0]["status"] == "skipped"
    assert results[0]["error"] is None


def test_cli_arxiv_list_dry_run_prints_batch_summary(capsys, monkeypatch):
    """The CLI --arxiv-list --dry-run prints a summary for each planned paper."""
    import sys
    import researchfs.ingest as ingest_module

    def fake_fetcher(arxiv_id):
        return {
            "title": f"Paper {arxiv_id}",
            "authors": "CLI Author",
            "abstract": "Abstract.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2023-05-22",
            "content": f"content for {arxiv_id}",
            "external_id": arxiv_id,
        }

    monkeypatch.setattr(ingest_module, "fetch_arxiv_source", fake_fetcher)

    original_argv = sys.argv
    try:
        sys.argv = [
            "researchfs.ingest",
            "--arxiv-list",
            "2305.13245,2002.05202",
            "--dry-run",
        ]
        ingest_module.main()
    finally:
        sys.argv = original_argv

    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "2305.13245" in captured.out
    assert "2002.05202" in captured.out


def test_arxiv_external_id_persisted_on_extraction_status():
    """arXiv ingestion stores the normalized arXiv id as external_id on extraction_status."""
    from researchfs.api import ResearchFSLabAPI

    conn = _init_memory_db()

    def fake_fetcher(_arxiv_id):
        normalized_id = "2305.13245"
        return {
            "title": "External ID Paper",
            "authors": "External",
            "abstract": "Abstract.",
            "url": f"https://arxiv.org/abs/{normalized_id}",
            "publish_date": "2023-05-22",
            "content": "external id content",
            "external_id": normalized_id,
        }

    api = ResearchFSLabAPI(conn=conn, arxiv_fetcher=fake_fetcher)
    response = api.dispatch("POST", "/ingest/arxiv", {"id": "arXiv:2305.13245v1"})

    assert response.status == 200
    assert response.body["extraction_status"]["external_id"] == "2305.13245"

    stored = conn.execute(
        "SELECT external_id FROM paper_extraction_status WHERE paper_id = ?",
        (response.body["paper_id"],),
    ).fetchone()
    assert stored["external_id"] == "2305.13245"



def test_plan_pixelrag_visual_index_uses_arxiv_pdf_and_separate_artifact_paths(tmp_path):
    """Planning maps an arXiv id to its PDF and non-colliding per-paper artifact dirs."""
    from researchfs.visual import plan_pixelrag_visual_index

    plan = plan_pixelrag_visual_index(
        "planned-2305.13245",
        {"external_id": "2305.13245"},
        output_root=tmp_path,
    )

    expected_paper_dir = tmp_path / "paper-planned-2305.13245"
    assert plan["status"] == "planned"
    assert plan["source"] == "https://arxiv.org/pdf/2305.13245.pdf"
    assert Path(plan["tiles_path"]) == expected_paper_dir / "tiles"
    assert Path(plan["embeddings_path"]) == expected_paper_dir / "embeddings"
    assert Path(plan["index_path"]) == expected_paper_dir / "index"
    assert len({plan["tiles_path"], plan["embeddings_path"], plan["index_path"]}) == 3


def test_ingest_pixelrag_visual_index_success_stores_row_and_local_config(tmp_path):
    """Successful visual indexing stores the row and feeds PixelRAG a local-source config."""
    from researchfs.ingest import ingest_paper

    conn = _init_memory_db()
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-1.4 local visual source\n")
    calls = []

    def stub_runner(command, *, cwd=None):
        calls.append({"command": list(command), "cwd": Path(cwd)})
        return SimpleNamespace(returncode=0)

    result = ingest_paper(
        {
            "title": "PixelRAG Local Visual Paper",
            "authors": "Visual Author",
            "abstract": "Visual indexing should use a local source.",
            "url": source_path.as_uri(),
            "publish_date": "2026-07-02",
            "content": "pixelragvisualsuccess evidence text for chunking.",
            "visual_source_path": str(source_path),
        },
        conn=conn,
        visual_index=True,
        visual_index_root=tmp_path,
        visual_index_runner=stub_runner,
    )

    paper_id = result["paper_id"]
    expected_paper_dir = tmp_path / f"paper-{paper_id}"
    assert conn.execute("SELECT COUNT(*) FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()[0] == 1
    assert result["visual_index"]["status"] == "indexed"
    row = conn.execute(
        """
        SELECT paper_id, provider, status, source, tiles_path, embeddings_path, index_path, error_text
        FROM paper_visual_indexes
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    assert dict(row) == {
        "paper_id": paper_id,
        "provider": "pixelrag",
        "status": "indexed",
        "source": str(source_path),
        "tiles_path": str(expected_paper_dir / "tiles"),
        "embeddings_path": str(expected_paper_dir / "embeddings"),
        "index_path": str(expected_paper_dir / "index"),
        "error_text": None,
    }
    assert calls == [{"command": ["pixelrag", "index", "build"], "cwd": expected_paper_dir}]

    config = (expected_paper_dir / "pixelrag.yaml").read_text(encoding="utf-8")
    assert "source:\n  type: local\n" in config
    assert "type: url" not in config


def test_ingest_pixelrag_visual_index_failure_still_stores_paper_and_guidance(tmp_path):
    """Missing PixelRAG is recorded as failed visual indexing without rolling back ingestion."""
    from researchfs.ingest import ingest_paper

    conn = _init_memory_db()
    source_path = tmp_path / "source.txt"
    source_path.write_text("local visual source", encoding="utf-8")

    def missing_pixelrag(_command, *, cwd=None):
        raise FileNotFoundError("pixelrag")

    result = ingest_paper(
        {
            "title": "PixelRAG Missing Tool Paper",
            "authors": "Visual Author",
            "abstract": "Ingestion should survive missing optional visual tooling.",
            "url": source_path.as_uri(),
            "publish_date": "2026-07-02",
            "content": "pixelragvisualfailure evidence text for chunking.",
            "visual_source_path": str(source_path),
        },
        conn=conn,
        visual_index=True,
        visual_index_root=tmp_path,
        visual_index_runner=missing_pixelrag,
    )

    paper_id = result["paper_id"]
    assert conn.execute("SELECT COUNT(*) FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()[0] == 1
    assert result["visual_index"]["status"] == "failed"
    assert "Install PixelRAG separately" in result["visual_index"]["error_text"]
    assert "pixelrag" in result["visual_index"]["error_text"]
    stored = conn.execute(
        "SELECT status, error_text FROM paper_visual_indexes WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    assert stored["status"] == "failed"
    assert "Install PixelRAG separately" in stored["error_text"]


def test_cli_arxiv_dry_run_visual_index_plans_without_pixelrag_runner(
    capsys, monkeypatch, tmp_path
):
    """Dry-run arXiv visual indexing reports a plan without invoking PixelRAG."""
    import sys
    import researchfs.ingest as ingest_module

    def fake_fetcher(arxiv_id):
        assert arxiv_id == "2305.13245"
        return {
            "title": "Dry Run Visual Paper",
            "authors": "CLI Author",
            "abstract": "Abstract.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2023-05-22",
            "content": "dry run visual planning content",
            "external_id": arxiv_id,
        }

    monkeypatch.setattr(ingest_module, "fetch_arxiv_source", fake_fetcher)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "researchfs.ingest",
            "--arxiv",
            "2305.13245",
            "--dry-run",
            "--visual-index",
            "--visual-index-root",
            str(tmp_path),
        ],
    )

    ingest_module.main()

    captured = capsys.readouterr()
    assert "Visual index: planned" in captured.out


def test_api_ingest_pixelrag_visual_index_uses_runner_and_returns_indexed(tmp_path):
    """POST /ingest forwards local visual-source requests into optional PixelRAG indexing."""
    from researchfs.api import ResearchFSLabAPI

    conn = _init_memory_db()
    source_path = tmp_path / "api-source.pdf"
    source_path.write_bytes(b"%PDF-1.4 local api visual source\n")
    calls = []

    def stub_runner(command, *, cwd=None):
        calls.append({"command": list(command), "cwd": Path(cwd)})
        return SimpleNamespace(returncode=0)

    api = ResearchFSLabAPI(conn=conn, visual_index_runner=stub_runner)
    response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "API PixelRAG Visual Paper",
            "authors": "API Author",
            "abstract": "API visual indexing should use the supplied local source.",
            "publish_date": "2026-07-02",
            "content": "apipixelragvisualsuccess evidence text for chunking.",
            "visual_index": True,
            "visual_index_root": str(tmp_path),
            "visual_source_path": str(source_path),
        },
    )

    assert response.status == 200
    assert response.body["visual_index"]["status"] == "indexed"
    assert calls == [
        {
            "command": ["pixelrag", "index", "build"],
            "cwd": tmp_path / f"paper-{response.body['paper_id']}",
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"visual_index": "true"}, id="visual-index-string"),
        pytest.param({"visual_index": 1}, id="visual-index-integer"),
        pytest.param({"visual_index_root": 42}, id="visual-index-root-integer"),
        pytest.param({"visual_index_root": False}, id="visual-index-root-bool"),
    ],
)
def test_api_ingest_rejects_invalid_visual_index_fields(payload):
    """POST /ingest rejects malformed visual-index controls before ingestion work."""
    from researchfs.api import ResearchFSLabAPI

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch(
        "POST",
        "/ingest",
        {
            "title": "Invalid Visual Index Request",
            "authors": "API Author",
            "abstract": "Invalid visual controls should be rejected.",
            "content": "invalid visual index request body",
            **payload,
        },
    )

    assert response.status == 400
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == "invalid_field"

@pytest.mark.parametrize(
    "route,payload,patch_target",
    [
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "max_chars": True},
            "search_methods_and_ids",
            id="retrieve-max-chars-bool",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "max_chars": "120"},
            "search_methods_and_ids",
            id="retrieve-max-chars-string",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "max_chars": 12.5},
            "search_methods_and_ids",
            id="retrieve-max-chars-float",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "max_chars": -1},
            "search_methods_and_ids",
            id="retrieve-max-chars-negative",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "agent_run_id": False},
            "search_methods_and_ids",
            id="retrieve-agent-run-id-bool",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "agent_run_id": "7"},
            "search_methods_and_ids",
            id="retrieve-agent-run-id-string",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "agent_run_id": 7.5},
            "search_methods_and_ids",
            id="retrieve-agent-run-id-float",
        ),
        pytest.param(
            "/retrieve",
            {"query": "optimizer", "agent_run_id": -7},
            "search_methods_and_ids",
            id="retrieve-agent-run-id-negative",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "method_ids": [True]},
            "log_experiment",
            id="experiment-method-ids-bool",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "method_ids": ["1"]},
            "log_experiment",
            id="experiment-method-ids-string",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "method_ids": [1.5]},
            "log_experiment",
            id="experiment-method-ids-float",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "method_ids": [-1]},
            "log_experiment",
            id="experiment-method-ids-negative",
        ),
        pytest.param(
            "/feedback",
            {"method_id": True, "rating": 1.0},
            "log_method_feedback",
            id="feedback-method-id-bool",
        ),
        pytest.param(
            "/feedback",
            {"method_id": "1", "rating": 1.0},
            "log_method_feedback",
            id="feedback-method-id-string",
        ),
        pytest.param(
            "/feedback",
            {"method_id": 1.5, "rating": 1.0},
            "log_method_feedback",
            id="feedback-method-id-float",
        ),
        pytest.param(
            "/feedback",
            {"method_id": -1, "rating": 1.0},
            "log_method_feedback",
            id="feedback-method-id-negative",
        ),
        pytest.param(
            "/feedback",
            {"method_id": 0, "rating": 1.0},
            "log_method_feedback",
            id="feedback-method-id-zero",
        ),
    ],
)
def test_api_json_integer_validation_rejects_invalid_fields_before_deeper_work(
    monkeypatch, route, payload, patch_target
):
    """Invalid JSON integer fields return 400 before dispatch reaches ResearchFS work."""
    import researchfs.api as api_module
    from researchfs.api import ResearchFSLabAPI

    def fail_if_dispatched(*_args, **_kwargs):
        raise AssertionError(f"{patch_target} should not be called for invalid JSON integers")

    monkeypatch.setattr(api_module.rfs, patch_target, fail_if_dispatched)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch("POST", route, payload)

    assert response.status == 400
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == "invalid_field"



@pytest.mark.parametrize(
    "route,handler_name",
    [
        pytest.param("/paper/0", "_handle_paper_detail", id="paper-detail-zero"),
        pytest.param("/paper/0/memory", "_handle_paper_memory", id="paper-memory-zero"),
        pytest.param(
            "/paper/0/memory/export",
            "_handle_paper_memory_export",
            id="paper-memory-export-zero",
        ),
        pytest.param("/method/0", "_handle_method_detail", id="method-detail-zero"),
        pytest.param(
            "/memory/export?paper_ids=0",
            "_handle_corpus_memory_export",
            id="corpus-export-zero-paper-id",
        ),
    ],
)
def test_api_url_ids_reject_zero_before_handlers(monkeypatch, route, handler_name):
    """URL paper/method identifiers must be positive before DB or export handlers run."""
    from researchfs.api import ResearchFSLabAPI

    handler_calls = []

    def fail_if_dispatched(self, *_args, **_kwargs):
        handler_calls.append((_args, _kwargs))
        raise AssertionError(f"{handler_name} should not run for zero URL identifiers")

    monkeypatch.setattr(ResearchFSLabAPI, handler_name, fail_if_dispatched)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch("GET", route)

    assert response.status == 400
    assert response.body["ok"] is False
    assert handler_calls == []


def test_api_url_positive_ids_and_zero_max_chars_still_dispatch(monkeypatch):
    """Positive decimal URL/query IDs parse normally, and max_chars=0 remains valid."""
    from researchfs.api import LabAPIResponse, ResearchFSLabAPI

    calls = []

    def capture_paper_detail(self, paper_id):
        calls.append(("paper", paper_id))
        return LabAPIResponse(200, {"ok": True, "paper_id": paper_id})

    def capture_paper_memory(self, paper_id, max_chars):
        calls.append(("paper_memory", paper_id, max_chars))
        return LabAPIResponse(
            200, {"ok": True, "paper_id": paper_id, "max_chars": max_chars}
        )

    def capture_paper_memory_export(self, paper_id, max_chars):
        calls.append(("paper_memory_export", paper_id, max_chars))
        return LabAPIResponse(
            200, {"ok": True, "bundle": {"paper_id": paper_id, "max_chars": max_chars}}
        )

    def capture_method_detail(self, method_id):
        calls.append(("method", method_id))
        return LabAPIResponse(200, {"ok": True, "method_id": method_id})

    def capture_corpus_memory_export(self, paper_ids, max_chars):
        calls.append(("corpus_memory_export", paper_ids, max_chars))
        return LabAPIResponse(
            200, {"ok": True, "bundle": {"paper_ids": paper_ids, "max_chars": max_chars}}
        )

    monkeypatch.setattr(ResearchFSLabAPI, "_handle_paper_detail", capture_paper_detail)
    monkeypatch.setattr(ResearchFSLabAPI, "_handle_paper_memory", capture_paper_memory)
    monkeypatch.setattr(
        ResearchFSLabAPI, "_handle_paper_memory_export", capture_paper_memory_export
    )
    monkeypatch.setattr(ResearchFSLabAPI, "_handle_method_detail", capture_method_detail)
    monkeypatch.setattr(
        ResearchFSLabAPI, "_handle_corpus_memory_export", capture_corpus_memory_export
    )

    api = ResearchFSLabAPI(conn=_init_memory_db())

    assert api.dispatch("GET", "/paper/12").status == 200
    assert api.dispatch("GET", "/paper/13/memory?max_chars=0").status == 200
    assert api.dispatch("GET", "/paper/14/memory/export?max_chars=0").status == 200
    assert api.dispatch("GET", "/method/15").status == 200
    assert (
        api.dispatch("GET", "/memory/export?paper_ids=16&paper_ids=17,16&max_chars=0").status
        == 200
    )

    assert calls == [
        ("paper", 12),
        ("paper_memory", 13, 0),
        ("paper_memory_export", 14, 0),
        ("method", 15),
        ("corpus_memory_export", [16, 17], 0),
    ]


@pytest.mark.parametrize(
    "route,payload,patch_target",
    [
        pytest.param(
            "/feedback",
            {"method_id": 1, "rating": "1.0"},
            "log_method_feedback",
            id="feedback-rating-string",
        ),
        pytest.param(
            "/feedback",
            {"method_id": 1, "rating": float("nan")},
            "log_method_feedback",
            id="feedback-rating-nan",
        ),
        pytest.param(
            "/feedback",
            {"method_id": 1, "rating": float("inf")},
            "log_method_feedback",
            id="feedback-rating-inf",
        ),
        pytest.param(
            "/feedback",
            {"method_id": 1, "rating": float("-inf")},
            "log_method_feedback",
            id="feedback-rating-neg-inf",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "val_bpb": "1.23"},
            "log_experiment",
            id="experiment-val-bpb-string",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "memory_gb": "4.5"},
            "log_experiment",
            id="experiment-memory-gb-string",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "val_bpb": float("nan")},
            "log_experiment",
            id="experiment-val-bpb-nan",
        ),
        pytest.param(
            "/experiment",
            {"commit_hash": "abc123", "status": "keep", "memory_gb": float("inf")},
            "log_experiment",
            id="experiment-memory-gb-inf",
        ),
    ],
)
def test_api_json_float_validation_rejects_invalid_fields_before_deeper_work(
    monkeypatch, route, payload, patch_target
):
    """Invalid JSON number fields return 400 before dispatch reaches ResearchFS work."""
    import researchfs.api as api_module
    from researchfs.api import ResearchFSLabAPI

    def fail_if_dispatched(*_args, **_kwargs):
        raise AssertionError(f"{patch_target} should not be called for invalid JSON numbers")

    monkeypatch.setattr(api_module.rfs, patch_target, fail_if_dispatched)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch(
        "POST", route, json.dumps(payload)
    )

    assert response.status == 400
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == "invalid_field"


@pytest.mark.parametrize(
    "payload,expected_rating",
    [
        pytest.param({"method_id": 1, "rating": 1}, 1.0, id="int"),
        pytest.param({"method_id": 1, "rating": 0.25}, 0.25, id="float"),
    ],
)
def test_api_feedback_json_number_validation_accepts_numeric_ratings(
    monkeypatch, payload, expected_rating
):
    """Valid JSON int and float ratings dispatch to feedback logging without coercing strings."""
    import researchfs.api as api_module
    from researchfs.api import ResearchFSLabAPI

    calls = []

    def log_method_feedback(**kwargs):
        calls.append(kwargs)
        return 987

    monkeypatch.setattr(api_module.rfs, "log_method_feedback", log_method_feedback)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch(
        "POST", "/feedback", json.dumps(payload)
    )

    assert response.status == 200
    assert response.body == {"ok": True, "feedback_id": 987}
    assert len(calls) == 1
    assert calls[0]["rating"] == expected_rating


@pytest.mark.parametrize(
    "payload,expected_val_bpb,expected_memory_gb",
    [
        pytest.param(
            {"commit_hash": "abc123", "status": "keep", "val_bpb": 1, "memory_gb": 4.5},
            1.0,
            4.5,
            id="int-val-float-memory",
        ),
        pytest.param(
            {"commit_hash": "abc123", "status": "keep", "val_bpb": 1.23, "memory_gb": 4},
            1.23,
            4.0,
            id="float-val-int-memory",
        ),
    ],
)
def test_api_experiment_json_number_validation_accepts_numeric_metrics(
    monkeypatch, payload, expected_val_bpb, expected_memory_gb
):
    """Valid JSON int and float metrics dispatch to experiment logging without string coercion."""
    import researchfs.api as api_module
    from researchfs.api import ResearchFSLabAPI

    calls = []

    def log_experiment(commit_hash, val_bpb, memory_gb, status, description, **kwargs):
        calls.append(
            {
                "commit_hash": commit_hash,
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status,
                "description": description,
                **kwargs,
            }
        )
        return 654

    monkeypatch.setattr(api_module.rfs, "log_experiment", log_experiment)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch(
        "POST", "/experiment", json.dumps(payload)
    )

    assert response.status == 200
    assert response.body == {"ok": True, "experiment_id": 654}
    assert len(calls) == 1
    assert calls[0]["val_bpb"] == expected_val_bpb
    assert calls[0]["memory_gb"] == expected_memory_gb

@pytest.mark.parametrize(
    "paper_id",
    [
        pytest.param(True, id="bool"),
        pytest.param("1", id="string"),
        pytest.param(1.5, id="float"),
        pytest.param(-1, id="negative"),
    ],
)
def test_evaluate_extraction_json_paper_id_validation_rejects_invalid_fields_before_deeper_work(
    monkeypatch, paper_id
):
    """Invalid required JSON IDs return 400 before extraction evaluation runs."""
    import researchfs.api as api_module
    from researchfs.api import ResearchFSLabAPI

    def fail_if_evaluated(*_args, **_kwargs):
        raise AssertionError("evaluate_extraction should not be called for invalid paper_id")

    monkeypatch.setattr(api_module, "evaluate_extraction", fail_if_evaluated)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch(
        "POST",
        "/evaluate/extraction",
        {"paper_id": paper_id},
    )

    assert response.status == 400
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == "invalid_field"

def _insert_searchable_method(
    conn,
    name,
    token,
    *,
    category="memory",
    description_prefix="Regression retrieval method",
):
    method_id = conn.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            name,
            f"{description_prefix} keyed by {token}.",
            f"Use the {token} boundary path.",
            "Returning a method outside the bounded context misleads callers.",
            category,
        ),
    ).lastrowid
    conn.commit()
    return method_id


def _insert_paper_memory_card(conn):
    paper_id = conn.execute(
        """
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Paper Memory Bounds Paper",
            "Bounds Lab",
            "A paper with compressed memory for accessor bounds tests.",
            "https://example.invalid/paper-memory-bounds",
            "2026-07-02",
        ),
    ).lastrowid
    chunk_id = conn.execute(
        """
        INSERT INTO chunks (paper_id, section, content)
        VALUES (?, ?, ?)
        """,
        (
            paper_id,
            "abstract",
            "papermemoryboundsevidencetoken links the method to this paper.",
        ),
    ).lastrowid
    method_id = conn.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Paper Memory Bounds Method",
            "Use papermemoryboundsevidencetoken for bounded paper memory.",
            "Keep bounded context and provenance metadata aligned.",
            "Returning hidden IDs would mislead callers.",
            "memory",
        ),
    ).lastrowid
    conn.execute(
        """
        INSERT INTO method_sources (
            method_id, chunk_id, evidence_text, evidence_start, evidence_end,
            evidence_strategy
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            method_id,
            chunk_id,
            "papermemoryboundsevidencetoken links the method to this paper.",
            0,
            61,
            "exact",
        ),
    )
    conn.execute(
        """
        INSERT INTO paper_extraction_status (
            paper_id, status, source_type, chunk_count, method_count,
            source_char_count, compressed_char_count, compression_ratio,
            confidence_score, confidence_reason, extractor_model,
            extractor_prompt_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            "extracted",
            "memory_bundle",
            1,
            1,
            600,
            120,
            0.2,
            0.9,
            "bounds regression confidence",
            "bounds-regression-model",
            "bounds-regression-v1",
        ),
    )
    conn.commit()
    return paper_id, method_id, chunk_id



@pytest.mark.parametrize(
    "argv",
    [
        ["--k", "0"],
        ["--max-reader-tokens", "0"],
        ["--max-retries", "-1"],
        ["--min-score", "nan"],
        ["--overflow-policy", "bogus"],
    ],
)
def test_memory_ask_parser_rejects_invalid_routing_flags_before_work(monkeypatch, argv):
    """Invalid ask routing flags fail in argparse before opening the runner."""
    import researchfs.memory as memory

    ask_calls = []

    def fail_if_called(*_args, **_kwargs):
        ask_calls.append((_args, _kwargs))
        raise AssertionError("_ask should not run for invalid ask parser flags")

    monkeypatch.setattr(memory, "_ask", fail_if_called)

    stderr = io.StringIO()
    code = memory.main(
        ["ask", "--query", "q", *argv],
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert ask_calls == []


def test_memory_ask_parser_passes_typed_values_to_runner(monkeypatch):
    """Valid ask CLI fields reach the runner with parsed Python values."""
    import researchfs.memory as memory

    captured = []

    def capture_ask(args, _stdout):
        captured.append(args)
        return 0

    monkeypatch.setattr(memory, "_ask", capture_ask)

    code = memory.main(
        [
            "ask",
            "--query",
            "q",
            "--max-chars",
            "0",
            "--max-papers",
            "2",
            "--source-type",
            "paper",
            "--source-type",
            "lab_note",
            "--include-scores",
            "--log",
            "--k",
            "1",
            "--max-reader-tokens",
            "64",
            "--min-score",
            "0.5",
            "--overflow-policy",
            "truncate",
            "--max-retries",
            "0",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    assert len(captured) == 1
    args = captured[0]
    assert args.query == "q"
    assert args.max_chars == 0
    assert args.max_papers == 2
    assert args.source_type == ["paper", "lab_note"]
    assert args.include_scores is True
    assert args.log is True
    assert args.k == 1
    assert args.max_reader_tokens == 64
    assert args.min_score == pytest.approx(0.5)
    assert args.overflow_policy == "truncate"
    assert args.max_retries == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "max_papers": 0},
        {"query": "q", "k": 0},
        {"query": "q", "max_reader_tokens": 0},
        {"query": "q", "max_retries": -1},
        {"query": "q", "min_score": float("nan")},
        {"query": "q", "overflow_policy": "bogus"},
    ],
)
def test_api_ask_rejects_invalid_fields_before_retrieval(monkeypatch, payload):
    """Invalid /ask fields return 400 before any retrieval work starts."""
    import researchfs.api as api_module
    from researchfs.api import ResearchFSLabAPI

    retrieval_calls = []

    def fail_if_retrieved(*_args, **_kwargs):
        retrieval_calls.append((_args, _kwargs))
        raise AssertionError("get_research_brief should not run for invalid /ask fields")

    monkeypatch.setattr(api_module.rfs, "get_research_brief", fail_if_retrieved)

    response = ResearchFSLabAPI(conn=_init_memory_db()).dispatch("POST", "/ask", payload)

    assert response.status == 400
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == "invalid_field"
    assert retrieval_calls == []


def test_api_ask_dispatch_returns_routed_result_and_logs_when_requested():
    """POST /ask returns the routed ask schema, logs events, and is listed in metadata."""
    from researchfs.api import ResearchFSLabAPI

    conn = _init_memory_db()
    _insert_paper_memory_card(conn)

    api = ResearchFSLabAPI(conn=conn)
    response = api.dispatch(
        "POST",
        "/ask",
        {"query": "papermemoryboundsevidencetoken", "log": True, "k": 1, "max_retries": 0},
    )

    assert response.status == 200
    body = response.body
    assert body["ok"] is True
    assert body["context"].startswith("# Selected context")
    assert "papermemoryboundsevidencetoken" in body["context"]
    assert body["selected_ids"]
    assert len(body["attempts"]) == 1
    assert body["event_id"] is not None
    route_paths = {route["path"] for route in api.dispatch("GET", "/").body["routes"]}
    assert "/ask" in route_paths


def test_lab_client_ask_success_and_error():
    """Lab client ask passes through success bodies and preserves API error fields."""
    from researchfs.api import LabAPIResponse
    from researchfs.lab_client import ResearchFSLabClient, ResearchFSLabClientError

    conn = _init_memory_db()
    _insert_paper_memory_card(conn)

    body = ResearchFSLabClient(conn=conn).ask(
        "papermemoryboundsevidencetoken",
        k=1,
        max_retries=0,
    )
    assert body["ok"] is True
    assert body["selected_ids"]

    class ErrorAPI:
        def dispatch(self, _method, _path, _body=None):
            return LabAPIResponse(
                400,
                {"ok": False, "error": {"code": "invalid_field", "message": "bad"}},
            )

    with pytest.raises(ResearchFSLabClientError) as exc_info:
        ResearchFSLabClient(api=ErrorAPI()).ask("bad")

    assert exc_info.value.status == 400
    assert exc_info.value.code == "invalid_field"
    assert exc_info.value.message == "bad"
    assert exc_info.value.body == {
        "ok": False,
        "error": {"code": "invalid_field", "message": "bad"},
    }


def test_session_routed_ask_uses_existing_agent_run():
    """Session routed ask logs to the existing run and never creates a replacement run."""
    from researchfs.client import ResearchFSSession

    conn = _init_memory_db()
    _insert_paper_memory_card(conn)
    run_id = _insert_run(conn, "existing-routed-ask-run")

    session = ResearchFSSession(SimpleNamespace(get_connection=lambda: conn), agent_run_id=run_id)
    result = session.routed_ask(
        "papermemoryboundsevidencetoken",
        log=True,
        k=1,
        max_retries=0,
    )

    assert result["ok"] is True
    event_run_ids = [
        row["agent_run_id"]
        for row in conn.execute(
            "SELECT agent_run_id FROM retrieval_events ORDER BY event_id"
        )
    ]
    assert event_run_ids
    assert set(event_run_ids) == {run_id}
    assert conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 1


def test_researchfs_top_level_ask_exports_are_lazy():
    """Top-level ask exports resolve to the first-class ask module implementations."""
    import researchfs
    import researchfs.ask as ask_module
    from researchfs import citation_verifier, default_reader, run_routed_ask

    assert "run_routed_ask" in researchfs.__all__
    assert run_routed_ask is ask_module.run_routed_ask
    assert default_reader is ask_module.default_reader
    assert citation_verifier is ask_module.citation_verifier



@pytest.mark.parametrize(
    "argv",
    [
        ["--agent-run-id", "0"],
        ["--experiment-id", "0"],
        ["--since-event-id", "0"],
        ["--limit", "0"],
        ["--max-candidate-chars", "-1"],
    ],
)
def test_memory_export_routing_traces_parser_rejects_invalid_filters_before_work(
    monkeypatch, argv
):
    """Invalid export-routing-traces filters fail in argparse before export work starts."""
    import researchfs.memory as memory

    export_calls = []

    def fail_if_called(*_args, **_kwargs):
        export_calls.append((_args, _kwargs))
        raise AssertionError("_export_routing_traces should not run for invalid filters")

    monkeypatch.setattr(memory, "_export_routing_traces", fail_if_called)

    code = memory.main(
        ["export-routing-traces", *argv],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 2
    assert export_calls == []


def test_memory_export_routing_traces_stdout_jsonl(monkeypatch, tmp_path):
    """Stdout mode emits only trace JSONL rows, with no trailing summary object."""
    import researchfs.memory as memory

    traces = [
        {"trace_id": "trace-one", "label": "candidate_not_selected"},
        {"trace_id": "trace-two", "label": "cited_supported"},
    ]
    export_calls = []

    def fake_export_routing_traces(conn=None, **kwargs):
        export_calls.append({"conn": conn, **kwargs})
        return {
            "ok": True,
            "schema_version": 1,
            "trace_count": len(traces),
            "event_count": 1,
            "skipped_event_count": 0,
            "skipped_events": [],
            "traces": traces,
        }

    monkeypatch.setattr(memory, "export_routing_traces", fake_export_routing_traces)

    stdout = io.StringIO()
    code = memory.main(
        ["export-routing-traces", "--db-path", str(tmp_path / "memory.db")],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    lines = stdout.getvalue().splitlines()
    assert code == 0
    assert [json.loads(line) for line in lines] == traces
    assert not any(json.loads(line).get("action") == "export-routing-traces" for line in lines)
    assert export_calls[0]["include_candidate_text"] is True
    assert export_calls[0]["max_candidate_chars"] == 2000


def test_memory_export_routing_traces_output_file_summary(monkeypatch, tmp_path):
    """Output-file mode writes JSONL rows to disk and prints a compact summary to stdout."""
    import researchfs.memory as memory

    traces = [
        {"trace_id": "trace-one", "label": "candidate_not_selected"},
        {"trace_id": "trace-two", "label": "cited_supported"},
    ]
    export_calls = []

    def fake_export_routing_traces(conn=None, **kwargs):
        export_calls.append({"conn": conn, **kwargs})
        return {
            "ok": True,
            "schema_version": 1,
            "trace_count": len(traces),
            "event_count": 1,
            "skipped_event_count": 0,
            "skipped_events": [],
            "traces": traces,
        }

    monkeypatch.setattr(memory, "export_routing_traces", fake_export_routing_traces)

    output_path = tmp_path / "traces.jsonl"
    stdout = io.StringIO()
    code = memory.main(
        [
            "export-routing-traces",
            "--db-path",
            str(tmp_path / "memory.db"),
            "--output",
            str(output_path),
            "--no-candidate-text",
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    file_lines = output_path.read_text(encoding="utf-8").splitlines()
    summary = json.loads(stdout.getvalue())
    assert code == 0
    assert [json.loads(line) for line in file_lines] == traces
    assert summary["ok"] is True
    assert summary["action"] == "export-routing-traces"
    assert summary["output"] == str(output_path)
    assert summary["trace_count"] == 2
    assert summary["event_count"] == 1
    assert summary["skipped_event_count"] == 0
    assert summary["schema_version"] == 1
    assert export_calls[0]["include_candidate_text"] is False


def test_researchfs_top_level_routing_trace_exports_are_lazy():
    """Top-level routing trace exports resolve to the routing_traces implementations."""
    import researchfs
    import researchfs.routing_traces as routing_traces_module
    from researchfs import (
        export_routing_traces,
        iter_routing_traces,
        write_routing_traces_jsonl,
    )

    assert "export_routing_traces" in researchfs.__all__
    assert "iter_routing_traces" in researchfs.__all__
    assert "write_routing_traces_jsonl" in researchfs.__all__
    assert export_routing_traces is routing_traces_module.export_routing_traces
    assert iter_routing_traces is routing_traces_module.iter_routing_traces
    assert write_routing_traces_jsonl is routing_traces_module.write_routing_traces_jsonl

def test_lab_api_index_and_health_keep_local_only_boundary():
    """The API index and health route keep the loopback-only product boundary visible."""
    from researchfs.api import ResearchFSLabAPI

    api = ResearchFSLabAPI(conn=_init_memory_db())
    index = api.dispatch("GET", "/")
    health = api.dispatch("GET", "/health")

    assert index.status == 200
    assert health.status == 200
    assert index.body["local_only"] is True
    assert health.body["local_only"] is True
    assert "127.0.0.1" in index.body["caveat"]

def _search_result(search_fn, query, *, conn, max_chars, agent_run_id=None):
    result = search_fn(query, conn=conn, max_chars=max_chars, agent_run_id=agent_run_id)
    if len(result) == 2:
        context, method_ids = result
        return context, method_ids, None
    context, method_ids, score_rows = result
    return context, method_ids, score_rows


@pytest.mark.parametrize(
    "search_fn_name",
    ["search_methods_and_ids", "search_methods_with_scores"],
)
def test_method_search_max_chars_bounds_omitted_markers_and_returned_ids(search_fn_name):
    """Bounded method retrieval cannot leak omitted methods through either text or IDs."""
    from researchfs import retrieve

    conn = _init_memory_db()
    token = "boundedomittedmarkertoken"
    included_id = _insert_searchable_method(
        conn,
        "A bounded retrieval method",
        token,
        description_prefix="Short enough to fit before the omitted marker",
    )
    search_fn = getattr(retrieve, search_fn_name)
    included_context, included_ids, _scores = _search_result(
        search_fn, token, conn=conn, max_chars=None
    )
    assert included_ids == [included_id]

    omitted_id = _insert_searchable_method(
        conn,
        "Z bounded retrieval method omitted",
        token,
        description_prefix="This second method should be outside the bounded context",
    )
    max_chars = len(included_context) + 5

    context, method_ids, score_rows = _search_result(
        search_fn, token, conn=conn, max_chars=max_chars
    )

    assert len(context) <= max_chars
    assert included_id in method_ids
    assert "A bounded retrieval method" in context
    assert omitted_id not in method_ids
    assert "Z bounded retrieval method omitted" not in context
    if score_rows is not None:
        assert [row["method_id"] for row in score_rows] == method_ids


@pytest.mark.parametrize(
    "search_fn_name",
    ["search_methods_and_ids", "search_methods_with_scores"],
)
def test_method_search_max_chars_bounds_optional_general_method_section(search_fn_name):
    """General-method appendices obey the same strict max_chars and ID inclusion contract."""
    from researchfs import retrieve

    conn = _init_memory_db()
    token = "boundedgeneralsectiontoken"
    direct_id = _insert_searchable_method(
        conn,
        "Direct bounded retrieval method",
        token,
        category="memory",
    )
    search_fn = getattr(retrieve, search_fn_name)
    direct_context, direct_ids, _scores = _search_result(
        search_fn, token, conn=conn, max_chars=None
    )
    assert direct_ids == [direct_id]

    general_id = _insert_searchable_method(
        conn,
        "General optimizer should not leak past budget",
        "unmatchedgeneralmethodtoken",
        category="optimizer",
    )
    max_chars = len(direct_context) + 8

    context, method_ids, score_rows = _search_result(
        search_fn, token, conn=conn, max_chars=max_chars
    )

    assert len(context) <= max_chars
    assert direct_id in method_ids
    assert "Direct bounded retrieval method" in context
    assert general_id not in method_ids
    assert "General optimizer should not leak past budget" not in context
    if score_rows is not None:
        assert [row["method_id"] for row in score_rows] == method_ids


def test_method_search_tiny_max_chars_never_returns_unrendered_method_ids():
    """A budget too small to render any method returns no method IDs."""
    from researchfs.retrieve import search_methods_and_ids

    conn = _init_memory_db()
    token = "tinybudgetretrievaltoken"
    omitted_id = _insert_searchable_method(
        conn,
        "Tiny budget omitted method",
        token,
    )

    context, method_ids = search_methods_and_ids(token, conn=conn, max_chars=12)

    assert len(context) <= 12
    assert method_ids == []
    assert omitted_id not in method_ids
    assert "Tiny budget omitted method" not in context


@pytest.mark.parametrize(
    "search_fn_name",
    ["search_methods", "search_methods_and_ids", "search_methods_with_scores"],
)
@pytest.mark.parametrize("bad_max_chars", [True, False, "0", -1])
def test_method_search_accessors_reject_invalid_max_chars(search_fn_name, bad_max_chars):
    """Public method search accessors reject non-integer and negative budgets."""
    from researchfs import retrieve

    conn = _init_memory_db()
    _insert_searchable_method(conn, "Invalid max chars search method", "invalidmaxcharstoken")
    search_fn = getattr(retrieve, search_fn_name)

    with pytest.raises(ValueError, match="max_chars"):
        search_fn("invalidmaxcharstoken", conn=conn, max_chars=bad_max_chars)


@pytest.mark.parametrize(
    "search_fn_name",
    ["search_methods", "search_methods_and_ids", "search_methods_with_scores"],
)
def test_method_search_accessors_accept_zero_max_chars(search_fn_name):
    """A zero budget remains valid and returns only budget-compliant output."""
    from researchfs import retrieve

    conn = _init_memory_db()
    _insert_searchable_method(conn, "Zero max chars search method", "zeromaxcharstoken")
    search_fn = getattr(retrieve, search_fn_name)

    result = search_fn("zeromaxcharstoken", conn=conn, max_chars=0)

    if search_fn_name == "search_methods":
        assert result == ""
    elif search_fn_name == "search_methods_and_ids":
        context, method_ids = result
        assert context == ""
        assert method_ids == []
    else:
        context, method_ids, score_rows = result
        assert context == ""
        assert method_ids == []
        assert score_rows == []


@pytest.mark.parametrize(
    "search_fn_name",
    ["search_methods_and_ids", "search_methods_with_scores"],
)
@pytest.mark.parametrize("bad_agent_run_id", [True, False, "1", 0, -1, 999_999])
def test_method_search_id_helpers_reject_invalid_agent_run_id(
    search_fn_name, bad_agent_run_id
):
    """ID-returning search helpers reject malformed and nonexistent run IDs before scoring."""
    from researchfs import retrieve

    conn = _init_memory_db()
    _insert_searchable_method(conn, "Invalid run scoped search method", "invalidrunsearchtoken")
    search_fn = getattr(retrieve, search_fn_name)

    with pytest.raises(ValueError, match="agent_run_id"):
        search_fn("invalidrunsearchtoken", conn=conn, agent_run_id=bad_agent_run_id)


@pytest.mark.parametrize(
    "search_fn_name",
    ["search_methods_and_ids", "search_methods_with_scores"],
)
def test_method_search_id_helpers_accept_existing_agent_run_id_and_score_with_it(
    search_fn_name, monkeypatch
):
    """Existing run IDs are accepted and passed through to method scoring."""
    from researchfs import retrieve

    conn = _init_memory_db()
    run_id = retrieve.start_agent_run("loopgraph", "search-bounds-model", conn)
    method_id = _insert_searchable_method(
        conn,
        "Scoped Search Bounds Method",
        "scopedsearchboundstoken",
    )
    search_fn = getattr(retrieve, search_fn_name)
    seen_run_ids = []
    original_score_methods = retrieve._score_methods

    def capture_score_methods(*args, agent_run_id=None, **kwargs):
        seen_run_ids.append(agent_run_id)
        return original_score_methods(*args, agent_run_id=agent_run_id, **kwargs)

    monkeypatch.setattr(retrieve, "_score_methods", capture_score_methods)

    context, method_ids, score_rows = _search_result(
        search_fn,
        "scopedsearchboundstoken",
        conn=conn,
        max_chars=None,
        agent_run_id=run_id,
    )

    assert "Scoped Search Bounds Method" in context
    assert method_ids == [method_id]
    assert seen_run_ids == [run_id]
    if score_rows is not None:
        assert [row["method_id"] for row in score_rows] == method_ids


@pytest.mark.parametrize("bad_paper_id", [True, False, "1", 1.5, 0, -1])
def test_get_paper_memory_rejects_invalid_paper_id_before_lookup(bad_paper_id):
    """Public paper memory accessor accepts only positive integer paper IDs."""
    from researchfs.retrieve import get_paper_memory

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="paper_id"):
        get_paper_memory(bad_paper_id, conn=conn)


@pytest.mark.parametrize("bad_max_chars", [True, False, "0", -1])
def test_get_paper_memory_rejects_invalid_max_chars(bad_max_chars):
    """Paper memory context bounds reject non-integer and negative budgets."""
    from researchfs.retrieve import get_paper_memory

    conn = _init_memory_db()
    paper_id, _method_id, _chunk_id = _insert_paper_memory_card(conn)

    with pytest.raises(ValueError, match="max_chars"):
        get_paper_memory(paper_id, conn=conn, max_chars=bad_max_chars)


def test_get_paper_memory_zero_max_chars_preserves_metadata_with_empty_context():
    """A zero budget truncates only context, not the paper memory identifiers."""
    from researchfs.retrieve import get_paper_memory

    conn = _init_memory_db()
    paper_id, method_id, chunk_id = _insert_paper_memory_card(conn)

    memory = get_paper_memory(paper_id, conn=conn, max_chars=0)

    assert memory["paper_id"] == paper_id
    assert memory["max_chars"] == 0
    assert memory["context"] == ""
    assert memory["method_ids"] == [method_id]
    assert memory["chunk_ids"] == [chunk_id]
    assert memory["provenance_sources"] == [
        {
            "method_id": method_id,
            "chunk_id": chunk_id,
            "section": "abstract",
            "evidence_text": "papermemoryboundsevidencetoken links the method to this paper.",
            "evidence_start": 0,
            "evidence_end": 61,
            "evidence_strategy": "exact",
        }
    ]
    assert memory["extraction_status"]["paper_id"] == paper_id
    assert memory["confidence_score"] == 0.9


@pytest.mark.parametrize("bad_max_chars", [True, False, "0", 1.5, -1])
def test_get_research_brief_rejects_invalid_max_chars(bad_max_chars):
    """Research briefs reject malformed and negative context budgets before retrieval."""
    from researchfs.retrieve import get_research_brief

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="max_chars"):
        get_research_brief("briefboundstoken", conn=conn, max_chars=bad_max_chars)


@pytest.mark.parametrize("bad_max_papers", [True, False, "1", 1.5, 0, -1])
def test_get_research_brief_rejects_invalid_max_papers(bad_max_papers):
    """Research briefs accept only positive integer paper limits."""
    from researchfs.retrieve import get_research_brief

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="max_papers"):
        get_research_brief("briefboundstoken", conn=conn, max_papers=bad_max_papers)


def test_get_research_brief_zero_max_chars_is_valid_empty_bounded_context():
    """A zero budget remains valid while preserving structured brief metadata."""
    from researchfs.retrieve import get_research_brief

    conn = _init_memory_db()
    _paper_id, method_id, _chunk_id = _insert_paper_memory_card(conn)

    brief = get_research_brief("papermemoryboundsevidencetoken", conn=conn, max_chars=0)

    assert brief["ok"] is True
    assert brief["query"] == "papermemoryboundsevidencetoken"
    assert brief["max_chars"] == 0
    assert brief["context"] == ""
    assert method_id in brief["method_ids"]


def test_get_research_brief_positive_max_papers_selects_bounded_paper_memory():
    """A positive paper limit still returns the matching deterministic paper brief."""
    from researchfs.retrieve import get_research_brief

    conn = _init_memory_db()
    paper_id, method_id, chunk_id = _insert_paper_memory_card(conn)

    brief = get_research_brief(
        "papermemoryboundsevidencetoken",
        conn=conn,
        max_chars=2000,
        max_papers=1,
    )

    assert brief["ok"] is True
    assert brief["max_papers"] == 1
    assert brief["paper_ids"] == [paper_id]
    assert method_id in brief["method_ids"]
    assert chunk_id in brief["chunk_ids"]
    assert "Paper memory card 1" in brief["context"]


def test_get_paper_memory_missing_positive_paper_id_still_raises_lookup_error():
    """Positive but nonexistent paper IDs keep the documented not-found behavior."""
    from researchfs.retrieve import get_paper_memory

    conn = _init_memory_db()

    with pytest.raises(LookupError, match="No paper found"):
        get_paper_memory(999_999, conn=conn)



@pytest.mark.parametrize("bad_paper_id", [True, False, "1", 1.5, 0, -1])
def test_evaluate_extraction_rejects_invalid_paper_id_before_loading_rows(
    monkeypatch, bad_paper_id
):
    """Extraction evaluator validates paper_id shape before querying paper rows."""
    from researchfs import evaluation

    conn = _init_memory_db()

    def fail_if_loaded(_conn, _paper_id):
        raise AssertionError("paper extraction rows should not be loaded")

    monkeypatch.setattr(evaluation, "_load_paper_extractions", fail_if_loaded)

    with pytest.raises(ValueError, match="paper_id"):
        evaluation.evaluate_extraction(conn, bad_paper_id, ["Any Method"])


def test_evaluate_extraction_accepts_valid_positive_paper_id_fixture():
    """Valid positive paper IDs keep the existing deterministic match behavior."""
    from researchfs.evaluation import evaluate_extraction

    conn = _init_memory_db()
    paper_id, _method_id, _chunk_id = _insert_paper_memory_card(conn)

    result = evaluate_extraction(conn, paper_id, ["Paper Memory Bounds Method"])

    assert result["paper_id"] == paper_id
    assert result["expected_count"] == 1
    assert result["extracted_count"] == 1
    assert result["matched_count"] == 1
    assert result["missing_methods"] == []
    assert result["extra_methods"] == []
    assert result["passed"] is True

@pytest.mark.parametrize("bad_paper_id", [True, False, "1", 1.5, 0, -1])
def test_export_paper_memory_bundle_rejects_invalid_paper_id_before_lookup(bad_paper_id):
    """Paper bundle export accepts only positive integer paper IDs."""
    from researchfs.bundle import export_paper_memory_bundle

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="paper IDs"):
        export_paper_memory_bundle(bad_paper_id, conn=conn)


@pytest.mark.parametrize("bad_paper_id", [True, False, "1", 1.5, 0, -1])
def test_export_corpus_memory_bundle_rejects_invalid_explicit_paper_ids(bad_paper_id):
    """Explicit corpus bundle paper IDs reject malformed values before export."""
    from researchfs.bundle import export_corpus_memory_bundle

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="paper IDs"):
        export_corpus_memory_bundle([bad_paper_id], conn=conn)


def test_export_corpus_memory_bundle_deduplicates_positive_explicit_paper_ids():
    """Duplicate positive IDs are exported once while preserving valid bundle output."""
    from researchfs.bundle import export_corpus_memory_bundle

    conn = _init_memory_db()
    paper_id, method_id, _chunk_id = _insert_paper_memory_card(conn)

    bundle = export_corpus_memory_bundle([paper_id, paper_id], conn=conn, max_chars=1000)

    assert bundle["metadata"]["paper_count"] == 1
    assert len(bundle["papers"]) == 1
    paper_bundle = bundle["papers"][0]
    assert paper_bundle["source_ids"]["paper_id"] == paper_id
    assert paper_bundle["source_ids"]["method_ids"] == [method_id]

@pytest.mark.parametrize("bad_method_id", [True, False, "1", 1.5, 0, -1])
def test_get_method_history_rejects_invalid_method_id_before_lookup(bad_method_id):
    """Public method history accessor accepts only positive integer method IDs."""
    from researchfs.retrieve import get_method_history

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="method_id"):
        get_method_history(bad_method_id, conn=conn)


@pytest.mark.parametrize("bad_max_items", [True, False, "0", -1])
def test_get_method_history_rejects_invalid_max_items(bad_max_items):
    """Method history item bounds reject non-integer and negative limits."""
    from researchfs.retrieve import get_method_history

    conn = _init_memory_db()
    method_id = _insert_searchable_method(
        conn,
        "Valid History Bounds Method",
        "validhistoryboundstoken",
    )

    with pytest.raises(ValueError, match="max_items"):
        get_method_history(method_id, conn=conn, max_items=bad_max_items)


def test_get_method_history_zero_max_items_is_valid_history_string():
    """A zero item limit still renders the method card without recent items."""
    from researchfs.retrieve import get_method_history

    conn = _init_memory_db()
    method_id = _insert_searchable_method(
        conn,
        "Zero Item History Method",
        "zerohistoryitemstoken",
    )

    history = get_method_history(method_id, conn=conn, max_items=0)

    assert isinstance(history, str)
    assert f"# Method History: Zero Item History Method (method_id={method_id})" in history
    assert "Recent experiments:\n- None linked." in history
    assert "Lessons:\n- None recorded." in history



@pytest.mark.parametrize("bad_max_items", [True, False, "0", -1])
def test_recommend_next_experiments_rejects_invalid_max_items(bad_max_items):
    """Recommendation item bounds reject non-integer and negative limits."""
    from researchfs.retrieve import recommend_next_experiments

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="max_items"):
        recommend_next_experiments("", conn=conn, max_items=bad_max_items)


def test_recommend_next_experiments_zero_max_items_is_valid_empty_recommendation():
    """A zero item limit is valid and returns the no-methods-requested recommendation."""
    from researchfs.retrieve import recommend_next_experiments

    conn = _init_memory_db()

    recommendation = recommend_next_experiments("", conn=conn, max_items=0)

    assert recommendation == "# Recommended Next Experiments\n\nNo methods requested for query: ''"


@pytest.mark.parametrize(
    "accessor_name",
    ["recommend_next_experiments", "get_graph_context"],
)
@pytest.mark.parametrize("bad_agent_run_id", [True, False, "1", 0, -1, 999_999])
def test_graph_accessors_reject_invalid_agent_run_id(accessor_name, bad_agent_run_id):
    """Graph accessors reject malformed and nonexistent run IDs before composing context."""
    from researchfs import retrieve

    conn = _init_memory_db()
    accessor = getattr(retrieve, accessor_name)

    with pytest.raises(ValueError, match="agent_run_id"):
        accessor("", conn=conn, agent_run_id=bad_agent_run_id)


def test_graph_accessors_accept_existing_agent_run_id_and_score_with_it(monkeypatch):
    """Graph accessors pass an existing run ID through to recommendation scoring."""
    from researchfs import retrieve

    conn = _init_memory_db()
    run_id = retrieve.start_agent_run("loopgraph", "graph-bounds-model", conn)
    _insert_searchable_method(conn, "Scoped Graph Bounds Method", "scopedgraphboundstoken")
    seen_run_ids = []
    original_score_methods = retrieve._score_methods

    def capture_score_methods(*args, agent_run_id=None, **kwargs):
        seen_run_ids.append(agent_run_id)
        return original_score_methods(*args, agent_run_id=agent_run_id, **kwargs)

    monkeypatch.setattr(retrieve, "_score_methods", capture_score_methods)

    recommendation = retrieve.recommend_next_experiments(
        "scopedgraphboundstoken",
        conn=conn,
        agent_run_id=run_id,
    )
    graph_context = retrieve.get_graph_context(
        "scopedgraphboundstoken",
        conn=conn,
        agent_run_id=run_id,
        max_chars=4000,
    )

    assert "Scoped Graph Bounds Method" in recommendation
    assert "Scoped Graph Bounds Method" in graph_context
    assert seen_run_ids == [run_id, run_id]


@pytest.mark.parametrize("bad_max_chars", [True, False, "0", -1])
def test_get_graph_context_rejects_invalid_max_chars(bad_max_chars):
    """Graph context bounds reject non-integer and negative budgets."""
    from researchfs.retrieve import get_graph_context

    conn = _init_memory_db()

    with pytest.raises(ValueError, match="max_chars"):
        get_graph_context("", conn=conn, max_chars=bad_max_chars)


def test_get_graph_context_zero_max_chars_is_valid_empty_context():
    """A zero character budget is valid and returns an empty graph context."""
    from researchfs.retrieve import get_graph_context

    conn = _init_memory_db()

    assert get_graph_context("", conn=conn, max_chars=0) == ""

def test_get_method_history_missing_positive_method_id_returns_history_text():
    """Positive but nonexistent method IDs keep the documented not-found text."""
    from researchfs.retrieve import get_method_history

    conn = _init_memory_db()

    assert get_method_history(999_999, conn=conn) == (
        "# Method History\n\nNo method found for method_id=999999."
    )

def test_link_experiment_methods_validates_public_ids_before_sqlite_errors():
    """Experiment-method logging rejects invalid public IDs with ValueError, not SQLite errors."""
    from researchfs.retrieve import link_experiment_methods, log_experiment

    conn = _init_memory_db()
    method_id = _insert_searchable_method(conn, "Valid linked method", "validlinktoken")
    experiment_id = log_experiment(
        "valid-link-experiment",
        val_bpb=1.23,
        memory_gb=4.5,
        status="keep",
        description="valid linkage target",
        conn=conn,
    )

    invalid_cases = [
        (True, [method_id], "experiment_id"),
        (experiment_id, [True], "method_id"),
        (999_999, [method_id], "experiment_id"),
        (experiment_id, [999_999], "method_id"),
        (experiment_id, [], "method_ids"),
    ]

    for bad_experiment_id, bad_method_ids, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            link_experiment_methods(
                bad_experiment_id,
                bad_method_ids,
                role="tested",
                note="invalid cases should not reach SQLite integrity handling",
                conn=conn,
            )

    assert conn.execute("SELECT COUNT(*) FROM experiment_methods").fetchone()[0] == 0


def test_log_experiment_validates_inputs_before_persisting():
    """Experiment logging rejects invalid public fields before inserting rows."""
    from researchfs.retrieve import log_experiment

    conn = _init_memory_db()
    agent_run_id = conn.execute(
        "INSERT INTO agent_runs (mode, model) VALUES (?, ?)",
        ("loopgraph", "experiment-validation-model"),
    ).lastrowid
    conn.commit()

    invalid_cases = [
        (
            {
                "commit_hash": "",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
            },
            "commit_hash",
        ),
        (
            {
                "commit_hash": "   ",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
            },
            "commit_hash",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "",
                "description": "valid description",
            },
            "status",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "adopt",
                "description": "valid description",
            },
            "status",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": " \t\n ",
            },
            "description",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": True,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
            },
            "val_bpb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": -0.01,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
            },
            "val_bpb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": float("nan"),
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
            },
            "val_bpb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": float("inf"),
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
            },
            "val_bpb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": True,
                "status": "keep",
                "description": "valid description",
            },
            "memory_gb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": -0.01,
                "status": "keep",
                "description": "valid description",
            },
            "memory_gb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": float("nan"),
                "status": "keep",
                "description": "valid description",
            },
            "memory_gb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": float("inf"),
                "status": "keep",
                "description": "valid description",
            },
            "memory_gb",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
                "agent_run_id": True,
            },
            "agent_run_id",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
                "agent_run_id": "1",
            },
            "agent_run_id",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
                "agent_run_id": 0,
            },
            "agent_run_id",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
                "agent_run_id": -1,
            },
            "agent_run_id",
        ),
        (
            {
                "commit_hash": "valid-commit",
                "val_bpb": 1.23,
                "memory_gb": 4.5,
                "status": "keep",
                "description": "valid description",
                "agent_run_id": 999_999,
            },
            "agent_run_id",
        ),
    ]

    for kwargs, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            log_experiment(conn=conn, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0

    experiment_id = log_experiment(
        "valid-experiment-commit",
        val_bpb=0.0,
        memory_gb=0.0,
        status="keep",
        description="valid experiment persists after rejected inputs",
        agent_run_id=agent_run_id,
        conn=conn,
    )

    stored = conn.execute(
        """
        SELECT agent_run_id, commit_hash, val_bpb, memory_gb, status, description
        FROM experiments
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    assert dict(stored) == {
        "agent_run_id": agent_run_id,
        "commit_hash": "valid-experiment-commit",
        "val_bpb": 0.0,
        "memory_gb": 0.0,
        "status": "keep",
        "description": "valid experiment persists after rejected inputs",
    }


def test_update_experiment_validates_inputs_before_mutating_rows():
    """Experiment updates reject invalid public fields before changing stored metrics."""
    from researchfs.retrieve import log_experiment, update_experiment

    conn = _init_memory_db()
    experiment_id = log_experiment(
        "update-validation-commit",
        val_bpb=1.23,
        memory_gb=4.5,
        status="keep",
        description="original update target",
        conn=conn,
    )
    original = dict(
        conn.execute(
            """
            SELECT commit_hash, val_bpb, memory_gb, status, description
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
    )

    invalid_cases = [
        ({"experiment_id": True, "val_bpb": 2.0}, "experiment_id"),
        ({"experiment_id": "1", "val_bpb": 2.0}, "experiment_id"),
        ({"experiment_id": 0, "val_bpb": 2.0}, "experiment_id"),
        ({"experiment_id": -1, "val_bpb": 2.0}, "experiment_id"),
        ({"experiment_id": 999_999, "val_bpb": 2.0}, "experiment_id"),
        ({"experiment_id": experiment_id, "val_bpb": True}, "val_bpb"),
        ({"experiment_id": experiment_id, "val_bpb": -0.01}, "val_bpb"),
        ({"experiment_id": experiment_id, "val_bpb": float("nan")}, "val_bpb"),
        ({"experiment_id": experiment_id, "val_bpb": float("inf")}, "val_bpb"),
        ({"experiment_id": experiment_id, "memory_gb": True}, "memory_gb"),
        ({"experiment_id": experiment_id, "memory_gb": -0.01}, "memory_gb"),
        ({"experiment_id": experiment_id, "memory_gb": float("nan")}, "memory_gb"),
        ({"experiment_id": experiment_id, "memory_gb": float("inf")}, "memory_gb"),
        ({"experiment_id": experiment_id, "val_bpb": 2.0, "status": ""}, "status"),
        ({"experiment_id": experiment_id, "status": "adopt"}, "status"),
        (
            {
                "experiment_id": experiment_id,
                "memory_gb": 6.0,
                "description": " \t\n ",
            },
            "description",
        ),
    ]

    for kwargs, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            update_experiment(conn=conn, **kwargs)
        current = dict(
            conn.execute(
                """
                SELECT commit_hash, val_bpb, memory_gb, status, description
                FROM experiments
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()
        )
        assert current == original
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1

    update_experiment(
        experiment_id,
        val_bpb=0.98,
        memory_gb=3.25,
        status="discard",
        description="valid update applied after rejected inputs",
        conn=conn,
    )

    updated = dict(
        conn.execute(
            """
            SELECT commit_hash, val_bpb, memory_gb, status, description
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
    )
    assert updated == {
        "commit_hash": "update-validation-commit",
        "val_bpb": 0.98,
        "memory_gb": 3.25,
        "status": "discard",
        "description": "valid update applied after rejected inputs",
    }


def test_log_lesson_validates_ids_and_required_text_before_persisting():
    """Lesson logging rejects invalid IDs and blank label/content before inserting rows."""
    from researchfs.retrieve import log_experiment, log_lesson

    conn = _init_memory_db()
    method_id = _insert_searchable_method(conn, "Valid lesson method", "validlessontoken")
    experiment_id = log_experiment(
        "valid-lesson-experiment",
        val_bpb=1.23,
        memory_gb=4.5,
        status="keep",
        description="valid lesson target",
        conn=conn,
    )

    invalid_cases = [
        (
            {"label": "Valid label", "content": "Valid content", "method_id": True},
            "method_id",
        ),
        (
            {"label": "Valid label", "content": "Valid content", "experiment_id": True},
            "experiment_id",
        ),
        (
            {"label": "Valid label", "content": "Valid content", "method_id": 999_999},
            "method_id",
        ),
        (
            {"label": "Valid label", "content": "Valid content", "experiment_id": 999_999},
            "experiment_id",
        ),
        (
            {"label": "", "content": "Valid content", "method_id": method_id},
            "label",
        ),
        (
            {"label": "   ", "content": "Valid content", "method_id": method_id},
            "label",
        ),
        (
            {"label": "Valid label", "content": "", "experiment_id": experiment_id},
            "content",
        ),
        (
            {"label": "Valid label", "content": " \t\n ", "experiment_id": experiment_id},
            "content",
        ),
    ]

    for kwargs, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            log_lesson(outcome="discard", conn=conn, **kwargs)

    assert conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 0


def test_log_retrieval_event_validates_public_ids_before_persisting():
    """Retrieval logging rejects invalid public IDs before creating events."""
    from researchfs.retrieve import log_experiment, log_retrieval_event

    conn = _init_memory_db()
    agent_run_id = conn.execute(
        "INSERT INTO agent_runs (mode, model) VALUES (?, ?)",
        ("loopgraph", "validation-model"),
    ).lastrowid
    experiment_id = log_experiment(
        "valid-retrieval-event-experiment",
        val_bpb=1.23,
        memory_gb=4.5,
        status="keep",
        description="valid retrieval event target",
        agent_run_id=agent_run_id,
        conn=conn,
    )
    method_id = _insert_searchable_method(
        conn,
        "Valid retrieval event method",
        "validretrievaleventtoken",
    )

    invalid_cases = [
        ({"agent_run_id": True}, "agent_run_id"),
        ({"agent_run_id": "1"}, "agent_run_id"),
        ({"agent_run_id": 0}, "agent_run_id"),
        ({"agent_run_id": -1}, "agent_run_id"),
        ({"agent_run_id": 999_999}, "agent_run_id"),
        ({"experiment_id": True}, "experiment_id"),
        ({"experiment_id": "1"}, "experiment_id"),
        ({"experiment_id": 0}, "experiment_id"),
        ({"experiment_id": -1}, "experiment_id"),
        ({"experiment_id": 999_999}, "experiment_id"),
        ({"method_ids": [True]}, "method_id"),
        ({"method_ids": ["1"]}, "method_id"),
        ({"method_ids": [0]}, "method_id"),
        ({"method_ids": [-1]}, "method_id"),
        ({"method_ids": [999_999]}, "method_ids"),
    ]

    for overrides, expected_message in invalid_cases:
        kwargs = {
            "agent_run_id": agent_run_id,
            "experiment_id": experiment_id,
            "method_ids": [method_id],
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=expected_message):
            log_retrieval_event(
                "agent_search",
                "validretrievaleventtoken",
                "retrieved context",
                conn=conn,
                **kwargs,
            )
        assert conn.execute("SELECT COUNT(*) FROM retrieval_events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM retrieved_methods").fetchone()[0] == 0


def test_log_retrieval_event_persists_no_link_and_valid_method_links():
    """None/empty method IDs create unlinked events; valid IDs create retrieved-method links."""
    from researchfs.retrieve import log_retrieval_event

    conn = _init_memory_db()
    first_method_id = _insert_searchable_method(
        conn,
        "First linked retrieval event method",
        "firstretrievallinktoken",
    )
    second_method_id = _insert_searchable_method(
        conn,
        "Second linked retrieval event method",
        "secondretrievallinktoken",
    )

    none_event_id = log_retrieval_event(
        "agent_search",
        "nolink-none",
        "retrieved context without method IDs",
        conn=conn,
        method_ids=None,
    )
    empty_event_id = log_retrieval_event(
        "agent_search_reuse",
        "nolink-empty",
        "retrieved context with empty method IDs",
        conn=conn,
        method_ids=[],
    )
    linked_event_id = log_retrieval_event(
        "agent_search",
        "linked-methods",
        "retrieved context with valid methods",
        conn=conn,
        method_ids=[first_method_id, second_method_id],
    )

    event_rows = conn.execute(
        """
        SELECT event_id, query
        FROM retrieval_events
        ORDER BY event_id
        """
    ).fetchall()
    assert [dict(row) for row in event_rows] == [
        {"event_id": none_event_id, "query": "nolink-none"},
        {"event_id": empty_event_id, "query": "nolink-empty"},
        {"event_id": linked_event_id, "query": "linked-methods"},
    ]

    linked_rows = conn.execute(
        """
        SELECT event_id, method_id
        FROM retrieved_methods
        ORDER BY event_id, method_id
        """
    ).fetchall()
    assert [tuple(row) for row in linked_rows] == [
        (linked_event_id, first_method_id),
        (linked_event_id, second_method_id),
    ]


def test_log_method_feedback_validates_public_ids_before_persisting():
    """Feedback logging rejects bool, non-integer, and non-positive IDs before inserts."""
    from researchfs.retrieve import log_method_feedback, log_retrieval_event

    conn = _init_memory_db()
    method_id = _insert_searchable_method(conn, "Valid feedback method", "validfeedbacktoken")
    event_id = log_retrieval_event(
        "agent_search",
        "validfeedbacktoken",
        "retrieved context",
        conn=conn,
        method_ids=[method_id],
    )

    invalid_cases = [
        ({"method_id": True, "event_id": event_id}, "method_id"),
        ({"method_id": "1", "event_id": event_id}, "method_id"),
        ({"method_id": 0, "event_id": event_id}, "method_id"),
        ({"method_id": -1, "event_id": event_id}, "method_id"),
        ({"method_id": method_id, "event_id": True}, "event_id"),
        ({"method_id": method_id, "event_id": "1"}, "event_id"),
        ({"method_id": method_id, "event_id": 0}, "event_id"),
        ({"method_id": method_id, "event_id": -1}, "event_id"),
    ]

    for ids, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            log_method_feedback(
                rating=0.5,
                label="invalid-id-regression",
                conn=conn,
                **ids,
            )
        assert conn.execute("SELECT COUNT(*) FROM method_feedback").fetchone()[0] == 0

    feedback_id = log_method_feedback(
        method_id=method_id,
        event_id=event_id,
        rating=0.5,
        label="valid-id-control",
        conn=conn,
    )
    stored = conn.execute(
        """
        SELECT event_id, method_id, rating, label
        FROM method_feedback
        WHERE feedback_id = ?
        """,
        (feedback_id,),
    ).fetchone()
    assert dict(stored) == {
        "event_id": event_id,
        "method_id": method_id,
        "rating": 0.5,
        "label": "valid-id-control",
    }



def test_get_method_feedback_summary_validates_public_method_id():
    """Feedback summaries reject bool, non-integer, and non-positive method IDs before queries."""
    from researchfs.retrieve import get_method_feedback_summary

    conn = _init_memory_db()

    for invalid_method_id in (True, False, "1", 1.5, 0, -1):
        with pytest.raises(ValueError, match="method_id"):
            get_method_feedback_summary(invalid_method_id, conn=conn)


def test_get_method_feedback_summary_keeps_zero_shape_for_missing_positive_method():
    """Unknown positive method IDs still return the established zero-count aggregate shape."""
    from researchfs.retrieve import get_method_feedback_summary

    conn = _init_memory_db()

    assert get_method_feedback_summary(12345, conn=conn) == {
        "method_id": 12345,
        "feedback_count": 0,
        "feedback_avg_rating": 0.0,
        "positive_feedback_count": 0,
        "negative_feedback_count": 0,
        "neutral_feedback_count": 0,
        "last_feedback_at": None,
    }


def test_get_method_feedback_summary_reports_existing_method_aggregates():
    """Feedback summaries preserve aggregate counts and average for a valid method."""
    from researchfs.retrieve import get_method_feedback_summary, log_method_feedback

    conn = _init_memory_db()
    method_id = _insert_searchable_method(
        conn, "Feedback summary method", "feedbacksummarytoken"
    )
    other_method_id = _insert_searchable_method(
        conn, "Other feedback summary method", "otherfeedbacksummarytoken"
    )

    log_method_feedback(
        method_id=method_id,
        rating=1.0,
        label="positive-summary-regression",
        conn=conn,
    )
    log_method_feedback(
        method_id=method_id,
        rating=0.0,
        label="neutral-summary-regression",
        conn=conn,
    )
    log_method_feedback(
        method_id=method_id,
        rating=-0.5,
        label="negative-summary-regression",
        conn=conn,
    )
    log_method_feedback(
        method_id=other_method_id,
        rating=1.0,
        label="other-method-regression",
        conn=conn,
    )

    summary = get_method_feedback_summary(method_id, conn=conn)

    assert summary["method_id"] == method_id
    assert summary["feedback_count"] == 3
    assert summary["feedback_avg_rating"] == pytest.approx(1.0 / 6.0)
    assert summary["positive_feedback_count"] == 1
    assert summary["negative_feedback_count"] == 1
    assert summary["neutral_feedback_count"] == 1
    assert summary["last_feedback_at"] is not None

def test_log_token_usage_validates_numeric_fields_and_experiment_id_before_persisting():
    """Token usage logging rejects invalid numeric values and experiment IDs before inserts."""
    import math

    from researchfs.retrieve import log_experiment, log_token_usage

    conn = _init_memory_db()
    experiment_id = log_experiment(
        "valid-token-usage-experiment",
        val_bpb=1.23,
        memory_gb=4.5,
        status="keep",
        description="valid token usage target",
        conn=conn,
    )

    invalid_cases = [
        ({"prompt_tokens": True}, "prompt_tokens"),
        ({"prompt_tokens": "12"}, "prompt_tokens"),
        ({"prompt_tokens": 12.5}, "prompt_tokens"),
        ({"prompt_tokens": -1}, "prompt_tokens"),
        ({"completion_tokens": False}, "completion_tokens"),
        ({"completion_tokens": "34"}, "completion_tokens"),
        ({"completion_tokens": 34.5}, "completion_tokens"),
        ({"completion_tokens": -1}, "completion_tokens"),
        ({"cost": True}, "cost"),
        ({"cost": "0.05"}, "cost"),
        ({"cost": -0.01}, "cost"),
        ({"cost": math.nan}, "cost"),
        ({"cost": math.inf}, "cost"),
        ({"cost": -math.inf}, "cost"),
        ({"experiment_id": True}, "experiment_id"),
        ({"experiment_id": 999_999}, "experiment_id"),
    ]

    for overrides, expected_message in invalid_cases:
        kwargs = {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "cost": 0.05,
            "experiment_id": experiment_id,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=expected_message):
            log_token_usage(conn=conn, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0] == 0

    usage_id = log_token_usage(
        prompt_tokens=0,
        completion_tokens=0,
        cost=None,
        experiment_id=None,
        conn=conn,
    )
    stored = conn.execute(
        """
        SELECT experiment_id, prompt_tokens, completion_tokens, cost
        FROM token_usage
        WHERE usage_id = ?
        """,
        (usage_id,),
    ).fetchone()
    assert dict(stored) == {
        "experiment_id": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": None,
    }


def test_start_agent_run_validates_mode_and_model_before_insert():
    """Agent run creation rejects blank lifecycle text and stores valid mode/model pairs."""
    from researchfs.retrieve import start_agent_run

    conn = _init_memory_db()

    invalid_cases = [
        ({"mode": ""}, "mode"),
        ({"mode": "   "}, "mode"),
        ({"mode": "baseline", "model": ""}, "model"),
        ({"mode": "loopgraph", "model": "   "}, "model"),
    ]

    for kwargs, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            start_agent_run(conn=conn, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0] == 0

    valid_cases = [
        ("baseline", None),
        ("baseline", "gpt-4o-mini"),
        ("loopgraph", None),
        ("loopgraph", "gpt-4o-mini"),
        ("custom-agent", "claude-sonnet"),
    ]

    run_ids = [
        start_agent_run(mode=mode, model=model, conn=conn)
        for mode, model in valid_cases
    ]

    rows = conn.execute(
        """
        SELECT agent_run_id, mode, model, finished_at
        FROM agent_runs
        ORDER BY agent_run_id
        """
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "agent_run_id": run_id,
            "mode": mode,
            "model": model,
            "finished_at": None,
        }
        for run_id, (mode, model) in zip(run_ids, valid_cases)
    ]


def test_finish_agent_run_validates_id_without_mutating_other_runs():
    """Run completion rejects invalid IDs without finishing another row, then finishes the target run."""
    from researchfs.retrieve import finish_agent_run, start_agent_run

    conn = _init_memory_db()
    run_id = start_agent_run("loopgraph", "gpt-4o-mini", conn)

    invalid_cases = [
        (True, "agent_run_id"),
        ("1", "agent_run_id"),
        (0, "agent_run_id"),
        (-1, "agent_run_id"),
        (999_999, "agent_run_id"),
    ]

    for invalid_id, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            finish_agent_run(invalid_id, conn)
        rows = conn.execute(
            """
            SELECT agent_run_id, mode, model, finished_at
            FROM agent_runs
            ORDER BY agent_run_id
            """
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {
                "agent_run_id": run_id,
                "mode": "loopgraph",
                "model": "gpt-4o-mini",
                "finished_at": None,
            }
        ]

    finish_agent_run(run_id, conn)
    finished_at = conn.execute(
        "SELECT finished_at FROM agent_runs WHERE agent_run_id = ?",
        (run_id,),
    ).fetchone()["finished_at"]
    assert finished_at is not None


def _insert_code_repo_paper(
    conn,
    *,
    title="GPC: Generalized Predictive Control for Agile Humanoid Whole-Body Control",
    abstract="ProtoMotions provides a deterministic clue for code repository discovery.",
    url="https://arxiv.org/abs/2606.29148",
    content="ProtoMotions clue text for repository enrichment.",
):
    paper_id = conn.execute(
        """
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            title,
            "GPC Authors",
            abstract,
            url,
            "2026-06-29",
        ),
    ).lastrowid
    chunk_id = conn.execute(
        "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
        (paper_id, "extracted", content),
    ).lastrowid
    conn.commit()
    return paper_id, chunk_id


def _paper_code_repo_rows(conn, paper_id):
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                paper_id,
                repo_url AS repository_url,
                repo_key AS repository_key,
                owner,
                name,
                source,
                evidence,
                confidence,
                selected,
                selected_at
            FROM paper_code_repositories
            WHERE paper_id = ?
            ORDER BY selected DESC, confidence DESC, repo_key
            """,
            (paper_id,),
        ).fetchall()
    ]


def test_code_repo_schema_creation_and_legacy_migration_are_idempotent():
    """Schema initialization and migration create the durable paper/repository link table."""
    from researchfs import db

    conn = _memory_conn()
    db.initialize_db(conn)

    columns = {
        row["name"]: row["type"]
        for row in conn.execute("PRAGMA table_info(paper_code_repositories)")
    }
    assert {
        "code_repository_id",
        "paper_id",
        "repo_url",
        "repo_key",
        "owner",
        "name",
        "source",
        "evidence",
        "confidence",
        "selected",
        "discovered_at",
        "selected_at",
        "updated_at",
    } <= set(columns)
    assert columns["paper_id"].upper() == "INTEGER"
    assert columns["repo_key"].upper() == "TEXT"

    foreign_keys = [
        dict(row) for row in conn.execute("PRAGMA foreign_key_list(paper_code_repositories)")
    ]
    assert any(fk["table"] == "papers" and fk["from"] == "paper_id" for fk in foreign_keys)

    legacy = _memory_conn()
    legacy.executescript(
        """
        CREATE TABLE papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            publish_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES ('Legacy Code Repo Paper', 'Legacy Lab', 'legacy abstract', 'https://example.invalid/legacy', '2026-07-02');
        """
    )
    db.migrate_db(legacy)
    db.migrate_db(legacy)

    migrated_columns = {
        row["name"] for row in legacy.execute("PRAGMA table_info(paper_code_repositories)")
    }
    assert set(columns) <= migrated_columns
    legacy.execute(
        """
        INSERT INTO paper_code_repositories (
            paper_id,
            repo_url,
            repo_key,
            owner,
            name,
            source,
            evidence,
            confidence,
            selected,
            selected_at
        )
        VALUES (1, ?, ?, ?, ?, 'migration-test', 'legacy insert proves FK target exists', 0.5, 1, CURRENT_TIMESTAMP)
        """,
        (
            "https://github.com/nvlabs/protomotions",
            "nvlabs/protomotions",
            "nvlabs",
            "protomotions",
        ),
    )
    with pytest.raises(sqlite3.IntegrityError):
        legacy.execute(
            """
            INSERT INTO paper_code_repositories (
                paper_id,
                repo_url,
                repo_key,
                owner,
                name,
                source,
                evidence,
                confidence,
                selected
            )
            VALUES (999, 'https://github.com/nvlabs/protomotions', 'nvlabs/protomotions', 'nvlabs', 'protomotions', 'fk-test', 'bad paper', 0.1, 0)
            """
        )



def test_code_repo_migration_rebuilds_alias_only_legacy_table():
    """Alias-only prototype tables are rebuilt before repo_key indexes/upserts run."""
    from researchfs import db

    legacy = _memory_conn()
    legacy.executescript(
        """
        CREATE TABLE papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            publish_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES ('Alias Legacy Paper', 'Legacy Lab', 'legacy abstract', 'https://example.invalid/alias', '2026-07-02');
        CREATE TABLE paper_code_repositories (
            code_repository_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            repository_url TEXT NOT NULL,
            repository_key TEXT NOT NULL,
            owner TEXT NOT NULL,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            evidence TEXT,
            confidence REAL NOT NULL DEFAULT 0.0,
            selected INTEGER NOT NULL DEFAULT 0,
            discovered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            selected_at TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(paper_id, repository_key)
        );
        INSERT INTO paper_code_repositories (
            paper_id, repository_url, repository_key, owner, name, source,
            evidence, confidence, selected
        )
        VALUES (
            1,
            'https://github.com/nvlabs/protomotions',
            'nvlabs/protomotions',
            'nvlabs',
            'protomotions',
            'legacy-alias',
            'legacy alias evidence',
            0.6,
            1
        );
        """
    )

    db.migrate_db(legacy)
    db.migrate_db(legacy)

    columns = {
        row["name"] for row in legacy.execute("PRAGMA table_info(paper_code_repositories)")
    }
    assert "repo_url" in columns
    assert "repo_key" in columns
    assert "repository_url" not in columns
    assert "repository_key" not in columns
    row = legacy.execute(
        """
        SELECT repo_url, repo_key, source, evidence, confidence, selected
        FROM paper_code_repositories
        WHERE paper_id = 1
        """
    ).fetchone()
    assert dict(row) == {
        "repo_url": "https://github.com/nvlabs/protomotions",
        "repo_key": "nvlabs/protomotions",
        "source": "legacy-alias",
        "evidence": "legacy alias evidence",
        "confidence": 0.6,
        "selected": 1,
    }

@pytest.mark.parametrize(
    "raw",
    [
        "https://github.com/NVlabs/ProtoMotions",
        "github.com/NVlabs/ProtoMotions",
        "http://github.com/NVlabs/ProtoMotions",
        "https://www.github.com/NVlabs/ProtoMotions/",
        "HTTPS://GITHUB.COM/NVLABS/PROTOMOTIONS.git",
        "https://github.com/NVlabs/ProtoMotions/tree/main/protomotions",
        "https://github.com/NVlabs/ProtoMotions/blob/main/README.md#quick-start",
        "https://github.com/NVlabs/ProtoMotions?tab=readme-ov-file",
    ],
)
def test_code_repo_url_normalization_accepts_github_root_variants(raw):
    """GitHub repository normalization follows the eval contract for harmless URL variants."""
    from researchfs.code_repos import normalize_github_repo_url

    assert normalize_github_repo_url(raw) == "https://github.com/nvlabs/protomotions"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "https://example.com/NVlabs/ProtoMotions",
        "https://github.com/NVlabs",
        "https://gist.github.com/NVlabs/ProtoMotions",
        "https://github.com//ProtoMotions",
    ],
)
def test_code_repo_url_normalization_rejects_non_repo_urls(raw):
    """Only github.com owner/repo URLs become repository keys."""
    from researchfs.code_repos import normalize_github_repo_url

    assert normalize_github_repo_url(raw) is None


def test_code_repo_url_extraction_deduplicates_and_ignores_non_github_hosts():
    """Stored text extraction returns deterministic unique GitHub repository keys in encounter order."""
    from researchfs.code_repos import extract_github_repo_urls

    text = (
        "Official code: https://github.com/NVlabs/ProtoMotions/tree/main. "
        "Duplicate spelling: HTTPS://github.com/NVLABS/PROTOMOTIONS.git; "
        "sibling: https://github.com/NVlabs/SOMA-X/blob/main/README.md "
        "docs-only host: https://protomotions.github.io/."
    )

    assert extract_github_repo_urls(text) == [
        "https://github.com/nvlabs/protomotions",
        "https://github.com/nvlabs/soma-x",
    ]


def test_code_repo_enrichment_stores_explicit_github_url_from_chunks_without_search():
    """Deterministic enrichment links GitHub URLs already present in stored paper chunks."""
    from researchfs.code_repos import enrich_paper_code_repositories

    conn = _init_memory_db()
    paper_id, _chunk_id = _insert_code_repo_paper(
        conn,
        content=(
            "The authors publish the official implementation at "
            "https://github.com/NVlabs/ProtoMotions/tree/main?tab=readme-ov-file."
        ),
    )

    result = enrich_paper_code_repositories(conn, paper_id)
    rows = _paper_code_repo_rows(conn, paper_id)

    assert len(rows) == 1
    assert rows[0]["repository_url"] == "https://github.com/nvlabs/protomotions"
    assert rows[0]["repository_key"] == "nvlabs/protomotions"
    assert rows[0]["owner"] == "nvlabs"
    assert rows[0]["name"] == "protomotions"
    assert rows[0]["selected"] == 1
    assert rows[0]["selected_at"] is not None
    assert "github.com/NVlabs/ProtoMotions" in rows[0]["evidence"]
    assert result["selected"]["repository_url"] == "https://github.com/nvlabs/protomotions"


def test_code_repo_enrichment_uses_injected_searcher_for_gpc_protomotions_clue():
    """Opt-in injected search discovers and selects ProtoMotions without any network dependency."""
    from researchfs.code_repos import enrich_paper_code_repositories

    conn = _init_memory_db()
    paper_id, _chunk_id = _insert_code_repo_paper(
        conn,
        title="GPC: Generalized Predictive Control for Agile Humanoid Whole-Body Control",
        abstract="A GPC paper whose artifacts are associated with ProtoMotions.",
        content="ProtoMotions is the implementation clue; no GitHub URL is stored here.",
    )
    queries = []

    def stub_searcher(query, *_args, **_kwargs):
        if isinstance(query, (list, tuple)):
            query_text = " ".join(str(part) for part in query)
        else:
            query_text = str(query)
        queries.append(query_text)
        if "protomotions" not in query_text.lower():
            return []
        return [
            {
                "url": "https://github.com/NVlabs/ProtoMotions/tree/main",
                "repository_url": "https://github.com/NVlabs/ProtoMotions/tree/main",
                "evidence": f"stub search matched: {query_text}",
                "confidence": 0.97,
            }
        ]

    result = enrich_paper_code_repositories(conn, paper_id, searcher=stub_searcher)
    rows = _paper_code_repo_rows(conn, paper_id)

    assert queries
    assert any("ProtoMotions" in query for query in queries)
    assert len(rows) == 1
    assert rows[0]["repository_url"] == "https://github.com/nvlabs/protomotions"
    assert rows[0]["repository_key"] == "nvlabs/protomotions"
    assert rows[0]["source"] == "search"
    assert rows[0]["confidence"] == pytest.approx(0.97)
    assert rows[0]["selected"] == 1
    assert result["selected"]["repository_url"] == "https://github.com/nvlabs/protomotions"


def test_paper_code_repo_memory_surfaces_selected_repository():
    """Paper memory includes selected linked code repositories in both data and compressed context."""
    from researchfs.retrieve import get_paper_memory

    conn = _init_memory_db()
    paper_id, _chunk_id = _insert_code_repo_paper(conn)
    conn.execute(
        """
        INSERT INTO paper_code_repositories (
            paper_id,
            repo_url,
            repo_key,
            owner,
            name,
            source,
            evidence,
            confidence,
            selected,
            selected_at
        )
        VALUES (?, ?, ?, ?, ?, 'test-selected', 'selected repo evidence', 0.99, 1, CURRENT_TIMESTAMP)
        """,
        (
            paper_id,
            "https://github.com/nvlabs/protomotions",
            "nvlabs/protomotions",
            "nvlabs",
            "protomotions",
        ),
    )
    conn.commit()

    memory = get_paper_memory(paper_id, conn=conn, max_chars=2000)

    [repo] = memory["code_repositories"]
    assert {
        "repository_url": repo["repository_url"],
        "repository_key": repo["repository_key"],
        "owner": repo["owner"],
        "name": repo["name"],
        "source": repo["source"],
        "evidence": repo["evidence"],
        "confidence": repo["confidence"],
        "selected": repo["selected"],
    } == {
        "repository_url": "https://github.com/nvlabs/protomotions",
        "repository_key": "nvlabs/protomotions",
        "owner": "nvlabs",
        "name": "protomotions",
        "source": "test-selected",
        "evidence": "selected repo evidence",
        "confidence": 0.99,
        "selected": 1,
    }
    assert "## Code repositories" in memory["context"]
    assert "https://github.com/nvlabs/protomotions" in memory["context"]
    assert "selected repo evidence" in memory["context"]


def test_code_repo_ingest_paper_enriches_explicit_urls_by_default_and_can_disable():
    """Direct ingestion links explicit GitHub URLs by default; enrich_code=False disables it."""
    from researchfs.ingest import ingest_paper

    conn = _init_memory_db()
    default_result = ingest_paper(
        {
            "title": "Default Ingest Code Repo Paper",
            "authors": "Code Repo Lab",
            "abstract": "Default ingestion should enrich explicit code repositories.",
            "url": "https://arxiv.org/abs/2606.29148",
            "publish_date": "2026-07-02",
            "content": "Official code: https://github.com/NVlabs/ProtoMotions/tree/main.",
        },
        conn=conn,
    )

    rows = _paper_code_repo_rows(conn, default_result["paper_id"])
    assert len(rows) == 1
    assert rows[0]["repository_url"] == "https://github.com/nvlabs/protomotions"
    assert rows[0]["selected"] == 1
    assert default_result["code_repositories"][0]["repository_url"] == "https://github.com/nvlabs/protomotions"

    disabled_result = ingest_paper(
        {
            "title": "Disabled Ingest Code Repo Paper",
            "authors": "Code Repo Lab",
            "abstract": "Explicit disable should suppress code repository enrichment.",
            "url": "https://arxiv.org/abs/2606.29149",
            "publish_date": "2026-07-02",
            "content": "Official code: https://github.com/NVlabs/ProtoMotions/tree/main.",
        },
        conn=conn,
        enrich_code=False,
    )

    assert _paper_code_repo_rows(conn, disabled_result["paper_id"]) == []
    assert "code_repositories" not in disabled_result

    explicit_true_result = ingest_paper(
        {
            "title": "Explicit True Ingest Code Repo Paper",
            "authors": "Code Repo Lab",
            "abstract": "Explicit affirmative remains accepted for backward compatibility.",
            "url": "https://arxiv.org/abs/2606.29150",
            "publish_date": "2026-07-02",
            "content": "Official code: https://github.com/NVlabs/ProtoMotions/tree/main.",
        },
        conn=conn,
        enrich_code=True,
    )

    explicit_true_rows = _paper_code_repo_rows(conn, explicit_true_result["paper_id"])
    assert len(explicit_true_rows) == 1
    assert explicit_true_rows[0]["repository_url"] == "https://github.com/nvlabs/protomotions"
    assert explicit_true_rows[0]["selected"] == 1
    assert explicit_true_result["code_repositories"][0]["repository_url"] == "https://github.com/nvlabs/protomotions"


def test_code_repo_ingest_cli_enriches_by_default_and_no_enrich_code_disables(tmp_path, monkeypatch):
    """CLI file ingestion enriches by default; --no-enrich-code disables; --enrich-code remains accepted."""
    import socket

    import researchfs.ingest as ingest_module

    def fail_network(*_args, **_kwargs):
        raise AssertionError("code repository regression tests must not open network connections")

    search_queries = []

    def stub_default_searcher(query, *_args, **_kwargs):
        query_text = " ".join(str(part) for part in query) if isinstance(query, (list, tuple)) else str(query)
        search_queries.append(query_text)
        return [
            {
                "url": "https://github.com/NVlabs/ProtoMotions",
                "evidence": "CLI default searcher hit",
                "confidence": 0.94,
            }
        ]

    db_path = tmp_path / "researchfs.sqlite"
    default_source = tmp_path / "default-code-repo-paper.md"
    default_source.write_text(
        "# Default Code Repo Paper\n\nOfficial code: https://github.com/NVlabs/ProtoMotions/tree/main.",
        encoding="utf-8",
    )
    disabled_source = tmp_path / "disabled-code-repo-paper.md"
    disabled_source.write_text(
        "# Disabled Code Repo Paper\n\nhttps://github.com/NVlabs/ProtoMotions should not enrich.",
        encoding="utf-8",
    )
    search_source = tmp_path / "search-code-repo-paper.md"
    search_source.write_text(
        "# Search Code Repo Paper\n\nGPC uses the ProtoMotions artifact without embedding a GitHub URL.",
        encoding="utf-8",
    )
    monkeypatch.setenv("RESEARCHFS_DB_PATH", str(db_path))
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(ingest_module, "_default_github_code_searcher", stub_default_searcher)

    monkeypatch.setattr(
        sys,
        "argv",
        ["researchfs.ingest", "--file", str(default_source)],
    )
    ingest_module.main()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT repo_url AS repository_url, repo_key AS repository_key, selected
            FROM paper_code_repositories
            ORDER BY code_repository_id
            """
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {
                "repository_url": "https://github.com/nvlabs/protomotions",
                "repository_key": "nvlabs/protomotions",
                "selected": 1,
            }
        ]
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        ["researchfs.ingest", "--file", str(disabled_source), "--no-enrich-code"],
    )
    ingest_module.main()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        assert conn.execute("SELECT COUNT(*) FROM paper_code_repositories").fetchone()[0] == 1
    finally:
        conn.close()

    monkeypatch.setattr(
        sys,
        "argv",
        ["researchfs.ingest", "--file", str(search_source), "--enrich-code"],
    )
    ingest_module.main()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT repo_url AS repository_url, repo_key AS repository_key, selected
            FROM paper_code_repositories
            ORDER BY code_repository_id
            """
        ).fetchall()
        assert search_queries
        assert any("ProtoMotions" in query for query in search_queries)
        assert [dict(row) for row in rows] == [
            {
                "repository_url": "https://github.com/nvlabs/protomotions",
                "repository_key": "nvlabs/protomotions",
                "selected": 1,
            },
            {
                "repository_url": "https://github.com/nvlabs/protomotions",
                "repository_key": "nvlabs/protomotions",
                "selected": 1,
            },
        ]
    finally:
        conn.close()


def test_paper_code_repo_api_and_client_enrich_code_defaults_with_searcher_and_explicit_urls():
    """API/client ingestion defaults to code enrichment for injected searchers and explicit URLs."""
    import inspect

    from researchfs.api import ResearchFSLabAPI
    from researchfs.lab_client import ResearchFSLabClient

    if "code_repo_searcher" not in inspect.signature(ResearchFSLabAPI).parameters:
        pytest.skip("API code repository enrichment route is not implemented")
    assert "code_repo_searcher" in inspect.signature(ResearchFSLabClient).parameters

    conn = _init_memory_db()
    queries = []

    def stub_searcher(query, *_args, **_kwargs):
        query_text = " ".join(str(part) for part in query) if isinstance(query, (list, tuple)) else str(query)
        queries.append(query_text)
        return [
            {
                "url": "https://github.com/NVlabs/ProtoMotions",
                "evidence": "API injected searcher hit",
                "confidence": 0.93,
            }
        ]

    client = ResearchFSLabClient(conn=conn, code_repo_searcher=stub_searcher)
    response = client.ingest(
        {
            "title": "API GPC ProtoMotions Paper",
            "authors": "API Code Repo Lab",
            "abstract": "GPC uses a ProtoMotions clue without embedding a GitHub URL.",
            "url": "https://arxiv.org/abs/2606.29148",
            "publish_date": "2026-07-02",
            "content": "ProtoMotions appears in stored paper text so search queries can be derived deterministically.",
        },
    )

    rows = _paper_code_repo_rows(conn, response["paper_id"])
    assert queries
    assert rows[0]["repository_url"] == "https://github.com/nvlabs/protomotions"
    assert rows[0]["selected"] == 1
    assert response["code_repositories"][0]["repository_url"] == "https://github.com/nvlabs/protomotions"

    explicit_response = client.ingest(
        {
            "title": "API Explicit Code Repo Paper",
            "authors": "API Code Repo Lab",
            "abstract": "Explicit GitHub URLs should enrich without requiring a search hit.",
            "url": "https://arxiv.org/abs/2606.29149",
            "publish_date": "2026-07-02",
            "content": "Official code: https://github.com/NVlabs/ProtoMotions/tree/main.",
        },
    )

    explicit_rows = _paper_code_repo_rows(conn, explicit_response["paper_id"])
    assert explicit_rows[0]["repository_url"] == "https://github.com/nvlabs/protomotions"
    assert explicit_rows[0]["selected"] == 1
    assert explicit_response["code_repositories"][0]["repository_url"] == "https://github.com/nvlabs/protomotions"

    disabled_response = client.ingest(
        {
            "title": "API Disabled Code Repo Paper",
            "authors": "API Code Repo Lab",
            "abstract": "Explicit disable should suppress enrichment.",
            "url": "https://arxiv.org/abs/2606.29150",
            "publish_date": "2026-07-02",
            "content": "Official code: https://github.com/NVlabs/ProtoMotions/tree/main.",
        },
        enrich_code=False,
    )

    assert _paper_code_repo_rows(conn, disabled_response["paper_id"]) == []
    assert "code_repositories" not in disabled_response

    memory = client.get_paper_memory(response["paper_id"], max_chars=2000)
    assert memory["code_repositories"][0]["repository_key"] == "nvlabs/protomotions"
    assert "https://github.com/nvlabs/protomotions" in memory["context"]

def _paper_code_repo_passed(result):
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        return result.get("passed")
    return getattr(result, "passed", None)


def test_paper_code_repo_prompt_includes_gpc_metadata_without_gold_url():
    """The agent prompt names the paper but does not leak the answer URL."""
    from evals.paper_code_repo import (
        GPC_PAPER_FIXTURE,
        GOLD_CODE_REPO_URL,
        build_paper_code_repo_prompt,
    )

    prompt = build_paper_code_repo_prompt(GPC_PAPER_FIXTURE)

    assert "2606.29148" in prompt
    assert "GPC" in prompt
    assert "github.com/NVlabs/ProtoMotions".lower() not in prompt.lower()
    assert GOLD_CODE_REPO_URL.lower() not in prompt.lower()


@pytest.mark.parametrize(
    "response",
    [
        "https://github.com/NVlabs/ProtoMotions",
        "github.com/NVlabs/ProtoMotions",
        "http://github.com/NVlabs/ProtoMotions",
        "https://www.github.com/NVlabs/ProtoMotions",
        "https://github.com/NVlabs/ProtoMotions/",
        "https://github.com/NVlabs/ProtoMotions.git",
        "The official code is HTTPS://GitHub.com/NVLABS/PROTOMOTIONS.git",
        "https://github.com/NVlabs/ProtoMotions/tree/main",
        "https://github.com/NVlabs/ProtoMotions/blob/main/README.md#quick-start",
        "https://github.com/NVlabs/ProtoMotions?tab=readme-ov-file",
    ],
)
def test_paper_code_repo_scorer_accepts_normalized_gold_url_variants(response):
    """URL normalization accepts harmless spelling variants of the gold repo."""
    from evals.paper_code_repo import score_paper_code_repo_response

    result = score_paper_code_repo_response(response)

    assert _paper_code_repo_passed(result) is True


@pytest.mark.parametrize(
    "response",
    [
        "The project page is https://protomotions.github.io/",
        "The code is https://github.com/NVlabs/SOMA-X",
        "See both https://protomotions.github.io/ and https://github.com/NVlabs/SOMA-X",
    ],
)
def test_paper_code_repo_scorer_rejects_docs_only_and_sibling_repos(response):
    """Nearby paper resources are not accepted as the official GitHub code repo."""
    from evals.paper_code_repo import score_paper_code_repo_response

    result = score_paper_code_repo_response(response)

    assert _paper_code_repo_passed(result) is False


def test_run_paper_code_repo_eval_calls_injected_agent_and_passes_gold_response():
    """The optional runner delegates to the provided agent callable without network use."""
    from evals.paper_code_repo import (
        GOLD_CODE_REPO_URL,
        run_paper_code_repo_eval,
    )

    prompts = []

    def agent(prompt):
        prompts.append(prompt)
        assert "2606.29148" in prompt
        assert GOLD_CODE_REPO_URL.lower() not in prompt.lower()
        return f"The official code repository is {GOLD_CODE_REPO_URL}"

    result = run_paper_code_repo_eval(agent)

    assert len(prompts) == 1
    assert _paper_code_repo_passed(result) is True


def test_offline_paper_code_repo_default_suite_passes_expected_scenarios():
    """The deterministic offline suite covers the scorer's required pass/fail cases."""
    from evals.paper_code_repo import (
        DEFAULT_PAPER_CODE_REPO_OFFLINE_SCENARIOS,
        run_offline_paper_code_repo_eval_suite,
    )

    suite = run_offline_paper_code_repo_eval_suite()
    scenario_names = {scenario.name.replace("_", "-") for scenario in DEFAULT_PAPER_CODE_REPO_OFFLINE_SCENARIOS}

    assert suite["passed"] is True
    assert suite["counts"] == {
        "total": len(DEFAULT_PAPER_CODE_REPO_OFFLINE_SCENARIOS),
        "matched": len(DEFAULT_PAPER_CODE_REPO_OFFLINE_SCENARIOS),
        "failed": 0,
    }
    assert {
        "correct-root",
        "correct-subpath",
        "docs-only",
        "sibling-repo",
        "ambiguous-correct-plus-wrong",
    } <= scenario_names
    assert {result["name"].replace("_", "-") for result in suite["results"]} == scenario_names
    assert all(result["matched"] is True for result in suite["results"])


def test_offline_paper_code_repo_suite_needs_no_network_or_agent(monkeypatch):
    """Synthetic offline scenarios run without a paper fetch, GitHub call, or agent callable."""
    import socket

    import evals.paper_code_repo as paper_code_repo

    def fail_agent_runner(*_args, **_kwargs):
        raise AssertionError("offline suite should not require an agent callable")

    def fail_network(*_args, **_kwargs):
        raise AssertionError("offline suite should not open network connections")

    monkeypatch.setattr(paper_code_repo, "run_paper_code_repo_eval", fail_agent_runner)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    suite = paper_code_repo.run_offline_paper_code_repo_eval_suite()

    assert suite["passed"] is True


def test_offline_paper_code_repo_suite_reports_misconfigured_scenario_failure():
    """A scenario with the wrong expected outcome is reported as a harness failure."""
    from evals.paper_code_repo import (
        GOLD_CODE_REPO_URL,
        PaperCodeRepoOfflineScenario,
        run_offline_paper_code_repo_eval_suite,
    )

    suite = run_offline_paper_code_repo_eval_suite(
        scenarios=(
            PaperCodeRepoOfflineScenario(
                name="misconfigured_correct_root",
                response=f"The official code repository is {GOLD_CODE_REPO_URL}",
                expected_passed=False,
                description="Intentionally wrong expected outcome.",
            ),
        )
    )

    assert suite["passed"] is False
    assert suite["counts"] == {"total": 1, "matched": 0, "failed": 1}
    assert suite["results"][0]["matched"] is False
    assert suite["results"][0]["actual_passed"] is True
    assert suite["failures"] == [
        {
            "name": "misconfigured_correct_root",
            "expected_passed": False,
            "actual_passed": True,
            "score": suite["results"][0]["score"],
        }
    ]


PAPERSTREAM_TABLES = {
    "arxiv_papers",
    "arxiv_versions",
    "arxiv_assets",
    "arxiv_chunks",
    "arxiv_jobs",
    "arxiv_sync_state",
    "arxiv_events",
    "arxiv_s3_bins",
    "arxiv_bin_members",
    "arxiv_bloom_filters",
}

PAPERSTREAM_INDEXES = {
    "idx_arxiv_papers_oai_datestamp",
    "idx_arxiv_papers_primary_category",
    "idx_arxiv_versions_arxiv_id",
    "idx_arxiv_assets_arxiv_type_status",
    "idx_arxiv_jobs_status_run_after",
    "idx_arxiv_events_arxiv_id",
    "idx_arxiv_s3_bins_status",
}


def _paperstream_conn():
    from researchfs import db

    conn = _memory_conn()
    db.initialize_db(conn)
    return conn


def _insert_paperstream_paper(
    conn, arxiv_id="2606.29148", *, version=1, category="cs.AI", categories=None
):
    conn.execute(
        """
        INSERT INTO arxiv_papers (
            arxiv_id,
            latest_version,
            title,
            abstract,
            authors_json,
            primary_category,
            categories_json,
            abs_url,
            oai_datestamp,
            raw_oai_xml
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            latest_version = excluded.latest_version,
            title = excluded.title,
            primary_category = excluded.primary_category
        """,
        (
            arxiv_id,
            version,
            f"Paper {arxiv_id}",
            "A deterministic PaperStream test paper.",
            json.dumps(["Ada Lovelace"]),
            category,
            json.dumps(categories or [category]),
            f"https://arxiv.org/abs/{arxiv_id}",
            "2026-07-03",
            f"<record><id>{arxiv_id}</id></record>",
        ),
    )
    conn.execute(
        """
        INSERT INTO arxiv_versions (
            arxiv_id,
            version,
            submitted_at,
            announced_at,
            title,
            abstract,
            authors_json,
            categories_json,
            raw_version_metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id, version) DO UPDATE SET
            title = excluded.title,
            raw_version_metadata_json = excluded.raw_version_metadata_json
        """,
        (
            arxiv_id,
            version,
            "2026-07-01T00:00:00Z",
            "2026-07-02",
            f"Paper {arxiv_id}",
            "A deterministic PaperStream test paper.",
            json.dumps(["Ada Lovelace"]),
            json.dumps(categories or [category]),
            json.dumps({"version": version}),
        ),
    )


_PAPERSTREAM_OAI_PAGE = """\
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-07-03T12:00:00Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXivRaw" set="cs">https://export.arxiv.org/oai2</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2606.29148</identifier>
        <datestamp>2026-07-02</datestamp>
        <setSpec>cs</setSpec>
        <setSpec>cs.AI</setSpec>
      </header>
      <metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
          <id>2606.29148</id>
          <submitter>Ada Lovelace</submitter>
          <versions>
            <version>
              <version>v1</version>
              <date>Mon, 01 Jun 2026 00:00:00 GMT</date>
            </version>
            <version>
              <version>v2</version>
              <date>Tue, 02 Jun 2026 00:00:00 GMT</date>
            </version>
          </versions>
          <title>Metadata Only PaperStream Core</title>
          <authors>Ada Lovelace, Alan Turing</authors>
          <categories>cs.AI cs.LG</categories>
          <license>http://creativecommons.org/licenses/by/4.0/</license>
          <doi>10.48550/arXiv.2606.29148</doi>
          <journal-ref>Journal of Local Fixtures 1 (2026)</journal-ref>
          <abstract>PaperStream keeps metadata synchronization deterministic and local.</abstract>
        </arXivRaw>
      </metadata>
    </record>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:2606.00000</identifier>
        <datestamp>2026-07-02</datestamp>
        <setSpec>cs</setSpec>
      </header>
    </record>
    <resumptionToken cursor="0" completeListSize="2">cursor-token-1</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""


_PAPERSTREAM_OAI_FINAL_PAGE = _PAPERSTREAM_OAI_PAGE.replace(
    "<responseDate>2026-07-03T12:00:00Z</responseDate>",
    "<responseDate>2026-07-04T12:00:00Z</responseDate>",
).replace(
    '<resumptionToken cursor="0" completeListSize="2">cursor-token-1</resumptionToken>',
    '<resumptionToken cursor="2" completeListSize="2" />',
)


_PAPERSTREAM_DELETED_OAI_PAGE = """\
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-07-05T12:00:00Z</responseDate>
  <request verb="ListRecords" metadataPrefix="arXivRaw" set="cs">https://export.arxiv.org/oai2</request>
  <ListRecords>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:2606.29148</identifier>
        <datestamp>2026-07-05</datestamp>
        <setSpec>cs</setSpec>
        <setSpec>cs.AI</setSpec>
      </header>
    </record>
  </ListRecords>
</OAI-PMH>
"""


def _paperstream_row_count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def _insert_paperstream_s3_bin(
    conn,
    *,
    asset_type="pdf",
    filename="pdf/arXiv_pdf_2606_001.tar",
    **fields,
):
    from researchfs.paperstream import import_arxiv_s3_manifest, s3_bin_key

    record = {"asset_type": asset_type, "filename": filename, **fields}
    import_arxiv_s3_manifest(conn, [record])
    return s3_bin_key(asset_type, filename)

def _write_paperstream_tar(path, members):
    import tarfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            if isinstance(data, str):
                data = data.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return path


def _mark_paperstream_s3_bin_downloaded(conn, bin_key, storage_path):
    import hashlib

    data = Path(storage_path).read_bytes()
    conn.execute(
        """
        UPDATE arxiv_s3_bins
        SET status = 'downloaded',
            storage_path = ?,
            size_bytes = ?,
            checksum = ?,
            downloaded_at = ?
        WHERE bin_key = ?
        """,
        (
            str(storage_path),
            len(data),
            "sha256:" + hashlib.sha256(data).hexdigest(),
            "2026-07-04T00:00:00Z",
            bin_key,
        ),
    )


def _enqueue_paperstream_extract_s3_bin_job(
    conn,
    bin_key,
    *,
    source="s3_bin_import",
    run_after="2026-07-03T00:00:00Z",
):
    row = conn.execute(
        "SELECT asset_type FROM arxiv_s3_bins WHERE bin_key = ?",
        (bin_key,),
    ).fetchone()
    assert row is not None
    job_key = f"extract_s3_bin:{bin_key}"
    conn.execute(
        """
        INSERT INTO arxiv_jobs (
            job_key, job_type, asset_type, priority, max_attempts, run_after, payload_json
        )
        VALUES (?, 'extract_s3_bin', ?, 100, 5, ?, ?)
        """,
        (
            job_key,
            row["asset_type"],
            run_after,
            json.dumps({"source": source, "bin_key": bin_key}),
        ),
    )
    return job_key


def _sha256_text(data):
    import hashlib

    return "sha256:" + hashlib.sha256(data).hexdigest()


def _paperstream_event_types(conn):
    return [
        row["event_type"]
        for row in conn.execute("SELECT event_type FROM arxiv_events ORDER BY event_id")
    ]


def _paperstream_db_counts(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "arxiv_sync_state",
                "arxiv_papers",
                "arxiv_versions",
                "arxiv_assets",
                "arxiv_jobs",
                "arxiv_events",
            )
        }
    finally:
        conn.close()

def _insert_paperstream_sync_state(
    conn,
    *,
    sync_name="default",
    status="succeeded",
    committed_from=None,
    last_response_date=None,
    pending_response_date=None,
    last_resumption_token=None,
    records_seen_last_run=0,
    records_updated_last_run=0,
    records_inserted_last_run=0,
    versions_inserted_last_run=0,
    jobs_queued_last_run=0,
    error_message=None,
    started_at=None,
    finished_at=None,
    updated_at=None,
):
    conn.execute(
        """
        INSERT INTO arxiv_sync_state (
            sync_name,
            committed_from,
            last_response_date,
            pending_response_date,
            last_resumption_token,
            status,
            records_seen_last_run,
            records_updated_last_run,
            records_inserted_last_run,
            versions_inserted_last_run,
            jobs_queued_last_run,
            error_message,
            started_at,
            finished_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sync_name,
            committed_from,
            last_response_date,
            pending_response_date,
            last_resumption_token,
            status,
            records_seen_last_run,
            records_updated_last_run,
            records_inserted_last_run,
            versions_inserted_last_run,
            jobs_queued_last_run,
            error_message,
            started_at,
            finished_at,
            updated_at,
        ),
    )




def test_paperstream_iter_oai_metadata_pages_uses_cursor_urls_and_rate_limit():
    """Injected OAI requests follow resumption-token rules and rate-limit each page."""
    from urllib.parse import parse_qs, urlparse

    from researchfs.paperstream import iter_oai_metadata_pages

    requested_urls = []
    calls = []
    responses = [
        SimpleNamespace(read=lambda: _PAPERSTREAM_OAI_PAGE.encode("utf-8")),
        SimpleNamespace(text=_PAPERSTREAM_OAI_FINAL_PAGE),
    ]

    def request_fn(url):
        requested_url = getattr(url, "full_url", str(url))
        calls.append(f"request:{requested_url}")
        requested_urls.append(requested_url)
        return responses.pop(0)

    def sleep_fn(seconds):
        calls.append(f"sleep:{seconds}")

    pages = list(
        iter_oai_metadata_pages(
            metadata_prefix="arXivRaw",
            set_spec="cs",
            from_date="2026-07-01",
            until_date="2026-07-02",
            base_url="https://example.test/oai2",
            max_pages=2,
            request_fn=request_fn,
            sleep_fn=sleep_fn,
            rate_limit_seconds=1.25,
        )
    )

    assert "cursor-token-1" in pages[0]
    assert "2026-07-04T12:00:00Z" in pages[1]
    assert len(requested_urls) == 2
    assert calls == [
        "sleep:1.25",
        f"request:{requested_urls[0]}",
        "sleep:1.25",
        f"request:{requested_urls[1]}",
    ]

    first_url = urlparse(requested_urls[0])
    assert (first_url.scheme, first_url.netloc, first_url.path) == ("https", "example.test", "/oai2")
    assert parse_qs(first_url.query) == {
        "verb": ["ListRecords"],
        "metadataPrefix": ["arXivRaw"],
        "set": ["cs"],
        "from": ["2026-07-01"],
        "until": ["2026-07-02"],
    }

    second_query = parse_qs(urlparse(requested_urls[1]).query)
    assert second_query == {"verb": ["ListRecords"], "resumptionToken": ["cursor-token-1"]}
    assert "metadataPrefix" not in second_query
    assert "set" not in second_query


def test_paperstream_iter_oai_metadata_pages_stops_before_request_when_rate_limiter_fails():
    """OAI rate-limit failures abort before an external request or yielded page."""
    from researchfs.paperstream import iter_oai_metadata_pages

    requested_urls = []
    sleeps = []

    def request_fn(url):
        requested_urls.append(getattr(url, "full_url", str(url)))
        raise AssertionError("OAI request must not run after a rate limiter failure")

    def sleep_fn(seconds):
        sleeps.append(seconds)
        raise TimeoutError("local limiter unavailable")

    pages = iter_oai_metadata_pages(
        metadata_prefix="arXivRaw",
        set_spec="cs",
        from_date="2026-07-01",
        until_date="2026-07-02",
        base_url="https://example.test/oai2",
        max_pages=1,
        request_fn=request_fn,
        sleep_fn=sleep_fn,
        rate_limit_seconds=1.25,
    )

    with pytest.raises(RuntimeError, match="rate limiter failed before requesting"):
        next(pages)

    assert sleeps == [1.25]
    assert requested_urls == []


def test_paperstream_iter_oai_metadata_pages_uses_default_oai_base_url():
    """Omitted OAI base_url requests the PRD default arXiv OAI-PMH endpoint."""
    from urllib.parse import parse_qs, urlparse

    from researchfs.paperstream import _DEFAULT_OAI_BASE_URL, iter_oai_metadata_pages

    requested_urls = []

    def request_fn(url):
        requested_url = getattr(url, "full_url", str(url))
        requested_urls.append(requested_url)
        return _PAPERSTREAM_OAI_FINAL_PAGE

    pages = list(
        iter_oai_metadata_pages(
            metadata_prefix="arXivRaw",
            set_spec="cs",
            from_date="2026-07-01",
            until_date="2026-07-02",
            max_pages=1,
            request_fn=request_fn,
            rate_limit_seconds=0,
        )
    )

    assert pages == [_PAPERSTREAM_OAI_FINAL_PAGE]
    assert _DEFAULT_OAI_BASE_URL == "https://oaipmh.arxiv.org/oai"
    assert len(requested_urls) == 1
    first_url = urlparse(requested_urls[0])
    assert (first_url.scheme, first_url.netloc, first_url.path) == (
        "https",
        "oaipmh.arxiv.org",
        "/oai",
    )
    assert parse_qs(first_url.query) == {
        "verb": ["ListRecords"],
        "metadataPrefix": ["arXivRaw"],
        "set": ["cs"],
        "from": ["2026-07-01"],
        "until": ["2026-07-02"],
    }

def test_paperstream_cli_live_pages_default_from_committed_cursor(monkeypatch):
    """Live metadata sync defaults its OAI from-date to the committed sync cursor."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    conn.execute(
        """
        INSERT INTO arxiv_sync_state (
            sync_name, committed_from, last_response_date, status
        )
        VALUES ('default', '2026-07-03T12:00:00Z', '2026-07-03T12:00:00Z', 'succeeded')
        """
    )
    captured = {}

    def fake_iter_pages(**kwargs):
        captured.update(kwargs)
        return ["<OAI-PMH />"]

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fake_iter_pages)
    args = SimpleNamespace(
        input=None,
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date=None,
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=False,
    )

    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH />"]
    assert captured["from_date"] == "2026-07-03T12:00:00Z"


def test_paperstream_cli_live_pages_failed_state_restarts_from_committed_cursor(monkeypatch):
    """Failed live sync state retries from the durable committed cursor, not transient OAI cursors."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="default",
        status="failed",
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
        pending_response_date="2026-07-03T12:00:00Z",
        last_resumption_token="expired-token",
    )
    captured = {}

    def fake_iter_pages(**kwargs):
        captured.update(kwargs)
        return ["<OAI-PMH />"]

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fake_iter_pages)
    args = SimpleNamespace(
        input=None,
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date=None,
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=False,
    )

    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH />"]
    assert captured["from_date"] == "2026-07-02T12:00:00Z"
    assert captured["from_date"] != "2026-07-03T12:00:00Z"
    assert captured["from_date"] != "expired-token"
    assert "initial_resumption_token" not in captured


def test_paperstream_cli_live_pages_failed_state_without_committed_cursor_starts_open(monkeypatch):
    """Failed live sync state without a committed cursor does not resume from pending state."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="default",
        status="failed",
        committed_from=None,
        last_response_date=None,
        pending_response_date="2026-07-03T12:00:00Z",
        last_resumption_token="expired-token",
    )
    captured = {}

    def fake_iter_pages(**kwargs):
        captured.update(kwargs)
        return ["<OAI-PMH />"]

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fake_iter_pages)
    args = SimpleNamespace(
        input=None,
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date=None,
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=False,
    )

    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH />"]
    assert captured["from_date"] is None
    assert captured["from_date"] != "2026-07-03T12:00:00Z"
    assert captured["from_date"] != "expired-token"
    assert "initial_resumption_token" not in captured


def test_paperstream_cli_live_pages_no_catch_up_preserves_committed_cursor(monkeypatch):
    """No-catch-up does not bypass an existing committed sync cursor."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="default",
        status="succeeded",
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )
    captured = {}

    def fake_iter_pages(**kwargs):
        captured.update(kwargs)
        return ["<OAI-PMH />"]

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fake_iter_pages)
    args = SimpleNamespace(
        input=None,
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date=None,
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=True,
    )

    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH />"]
    assert captured["from_date"] == "2026-07-02T12:00:00Z"


def test_paperstream_cli_live_pages_no_catch_up_without_cursor_starts_today(monkeypatch):
    """No-catch-up starts at current UTC date only when no explicit or committed cursor exists."""
    from datetime import datetime, timezone

    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="default",
        status="failed",
        committed_from=None,
        last_response_date=None,
        pending_response_date="2026-07-03T12:00:00Z",
        last_resumption_token="expired-token",
    )
    captured = {}

    def fake_iter_pages(**kwargs):
        captured.update(kwargs)
        return ["<OAI-PMH />"]

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fake_iter_pages)
    args = SimpleNamespace(
        input=None,
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date=None,
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=True,
    )

    before = datetime.now(timezone.utc).date().isoformat()
    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH />"]
    after = datetime.now(timezone.utc).date().isoformat()
    assert before <= captured["from_date"] <= after
    assert captured["from_date"] != "2026-07-03T12:00:00Z"
    assert captured["from_date"] != "expired-token"


def test_paperstream_cli_live_pages_no_catch_up_preserves_explicit_from_date(monkeypatch):
    """Explicit --from remains authoritative when --no-catch-up is also supplied."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="default",
        status="succeeded",
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )
    captured = {}

    def fake_iter_pages(**kwargs):
        captured.update(kwargs)
        return ["<OAI-PMH />"]

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fake_iter_pages)
    args = SimpleNamespace(
        input=None,
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date="2026-07-04",
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=True,
    )

    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH />"]
    assert captured["from_date"] == "2026-07-04"


def test_paperstream_cli_fixture_input_ignores_no_catch_up(monkeypatch, tmp_path):
    """Fixture-backed metadata sync reads only the local input page even with no-catch-up."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="default",
        status="succeeded",
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )
    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text("<OAI-PMH>fixture</OAI-PMH>", encoding="utf-8")

    def fail_iter_pages(**_kwargs):
        raise AssertionError("fixture --input must not call live OAI page iterator")

    monkeypatch.setattr(paperstream, "iter_oai_metadata_pages", fail_iter_pages)
    args = SimpleNamespace(
        input=str(fixture_path),
        metadata_prefix="arXivRaw",
        set_spec=None,
        from_date=None,
        until_date=None,
        max_pages=1,
        rate_limit_seconds=0.0,
        sync_name="default",
        no_catch_up=True,
    )

    assert list(paperstream._metadata_sync_pages(args, conn)) == ["<OAI-PMH>fixture</OAI-PMH>"]

def test_paperstream_parse_oai_list_records_extracts_cursor_and_raw_records():
    """OAI ListRecords parsing preserves cursor metadata and per-record raw XML."""
    from researchfs.paperstream import parse_oai_list_records

    parsed = parse_oai_list_records(_PAPERSTREAM_OAI_PAGE.encode("utf-8"))

    assert parsed["response_date"] == "2026-07-03T12:00:00Z"
    assert parsed["resumption_token"] == "cursor-token-1"
    assert len(parsed["records"]) == 2

    active = parsed["records"][0]
    assert active["identifier"] == "oai:arXiv.org:2606.29148"
    assert active["datestamp"] == "2026-07-02"
    assert active["set_specs"] == ["cs", "cs.AI"]
    assert active.get("deleted") in (None, False)
    assert "record" in active["raw_xml"]
    assert "Metadata Only PaperStream Core" in active["raw_xml"]

    deleted = parsed["records"][1]
    assert deleted["identifier"] == "oai:arXiv.org:2606.00000"
    assert deleted["deleted"] is True
    assert deleted["set_specs"] == ["cs"]


def test_paperstream_parse_oai_list_records_rejects_error_response():
    """OAI-PMH error pages fail with structured code/message context."""
    from researchfs.paperstream import parse_oai_list_records

    xml_text = """\
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-07-03T12:00:00Z</responseDate>
  <request verb="ListRecords">https://example.test/oai</request>
  <error code="badResumptionToken">Expired cursor token</error>
</OAI-PMH>
"""

    with pytest.raises(ValueError) as exc_info:
        parse_oai_list_records(xml_text)

    message = str(exc_info.value)
    assert "OAI-PMH error response" in message
    assert "code=badResumptionToken" in message
    assert "message=Expired cursor token" in message


@pytest.mark.parametrize(
    "replacement",
    ["", "<responseDate>   </responseDate>"],
    ids=("missing", "blank"),
)
def test_paperstream_parse_oai_list_records_rejects_missing_response_date(replacement):
    """OAI-PMH pages without a non-blank responseDate are not valid ListRecords pages."""
    from researchfs.paperstream import parse_oai_list_records

    xml_text = _PAPERSTREAM_OAI_FINAL_PAGE.replace(
        "<responseDate>2026-07-04T12:00:00Z</responseDate>",
        replacement,
    )

    with pytest.raises(ValueError, match="missing responseDate"):
        parse_oai_list_records(xml_text)


def test_paperstream_parse_oai_list_records_rejects_missing_list_records():
    """OAI-PMH pages without a ListRecords element fail before sync can commit them."""
    from researchfs.paperstream import parse_oai_list_records

    xml_text = """\
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-07-03T12:00:00Z</responseDate>
  <request verb="ListRecords">https://example.test/oai</request>
</OAI-PMH>
"""

    with pytest.raises(ValueError, match="missing ListRecords"):
        parse_oai_list_records(xml_text)

def test_paperstream_parse_arxiv_raw_record_extracts_paper_and_version_fields():
    """arXivRaw metadata becomes normalized paper fields with raw XML preserved."""
    from researchfs.paperstream import parse_arxiv_raw_record, parse_oai_list_records

    oai_record = parse_oai_list_records(_PAPERSTREAM_OAI_PAGE)["records"][0]
    parsed = parse_arxiv_raw_record(oai_record)

    assert parsed["arxiv_id"] == "2606.29148"
    assert parsed["latest_version"] == 2
    assert parsed["title"] == "Metadata Only PaperStream Core"
    assert parsed["abstract"] == "PaperStream keeps metadata synchronization deterministic and local."
    assert parsed["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert parsed["categories"] == ["cs.AI", "cs.LG"]
    assert parsed["primary_category"] == "cs.AI"
    assert parsed["license"] == "http://creativecommons.org/licenses/by/4.0/"
    assert parsed["doi"] == "10.48550/arXiv.2606.29148"
    assert parsed["journal_ref"] == "Journal of Local Fixtures 1 (2026)"
    assert parsed["abs_url"] == "https://arxiv.org/abs/2606.29148"
    assert parsed["oai_datestamp"] == "2026-07-02"
    assert "Metadata Only PaperStream Core" in parsed["raw_oai_xml"]
    assert [version["version"] for version in parsed["versions"]] == [1, 2]
    assert parsed["versions"][0]["submitted_at"] == "Mon, 01 Jun 2026 00:00:00 GMT"
    assert parsed["versions"][1]["submitted_at"] == "Tue, 02 Jun 2026 00:00:00 GMT"


def test_paperstream_metadata_upsert_creates_rows_events_and_is_idempotent():
    """Metadata upsert creates durable paper/version/asset state without duplicate rows."""
    from researchfs.paperstream import (
        asset_key,
        parse_arxiv_raw_record,
        parse_oai_list_records,
        upsert_paperstream_metadata_record,
    )

    conn = _paperstream_conn()
    parsed = parse_arxiv_raw_record(parse_oai_list_records(_PAPERSTREAM_OAI_PAGE)["records"][0])

    result = upsert_paperstream_metadata_record(conn, parsed)
    second_result = upsert_paperstream_metadata_record(conn, parsed)

    assert result["arxiv_id"] == second_result["arxiv_id"] == "2606.29148"
    assert result["latest_version"] == second_result["latest_version"] == 2
    assert result["versions_inserted"] == 2
    assert result["versions_changed"] == 0
    assert second_result["versions_inserted"] == 0
    assert second_result["versions_changed"] == 0
    assert _paperstream_row_count(conn, "arxiv_papers") == 1
    assert _paperstream_row_count(conn, "arxiv_versions") == 2
    assert _paperstream_row_count(conn, "arxiv_assets") == 1

    paper = conn.execute("SELECT * FROM arxiv_papers WHERE arxiv_id = ?", ("2606.29148",)).fetchone()
    assert paper["latest_version"] == 2
    assert paper["title"] == "Metadata Only PaperStream Core"
    assert json.loads(paper["authors_json"]) == ["Ada Lovelace", "Alan Turing"]
    assert json.loads(paper["categories_json"]) == ["cs.AI", "cs.LG"]
    assert paper["primary_category"] == "cs.AI"
    assert "Metadata Only PaperStream Core" in paper["raw_oai_xml"]

    version_rows = conn.execute(
        "SELECT version, submitted_at FROM arxiv_versions WHERE arxiv_id = ? ORDER BY version",
        ("2606.29148",),
    ).fetchall()
    assert [(row["version"], row["submitted_at"]) for row in version_rows] == [
        (1, "Mon, 01 Jun 2026 00:00:00 GMT"),
        (2, "Tue, 02 Jun 2026 00:00:00 GMT"),
    ]

    metadata = conn.execute("SELECT * FROM arxiv_assets").fetchone()
    assert metadata["asset_key"] == asset_key("2606.29148", 2, "metadata")
    assert metadata["asset_type"] == "metadata"
    assert metadata["source"] == "oai"
    assert _paperstream_row_count(conn, "arxiv_events") >= 1


def test_paperstream_metadata_upsert_update_event_lists_changed_fields_without_values():
    """Paper update events name changed normalized fields without old/new metadata values."""
    import copy

    from researchfs.paperstream import (
        parse_arxiv_raw_record,
        parse_oai_list_records,
        upsert_paperstream_metadata_record,
    )

    conn = _paperstream_conn()
    parsed = parse_arxiv_raw_record(parse_oai_list_records(_PAPERSTREAM_OAI_PAGE)["records"][0])
    upsert_paperstream_metadata_record(conn, parsed)

    updated = copy.deepcopy(parsed)
    updated["categories"] = ["cs.CV", "cs.AI"]
    updated["primary_category"] = "cs.CV"
    updated["license"] = "https://example.invalid/license/v2"
    updated["doi"] = "10.48550/arXiv.2606.29148.v2"
    updated["journal_ref"] = "Journal of Changed Fixtures 2 (2026)"
    updated["oai_datestamp"] = "2026-07-04"

    result = upsert_paperstream_metadata_record(conn, updated)

    assert result["changed"] is True
    assert result["changed_fields"] == [
        "primary_category",
        "categories_json",
        "license",
        "doi",
        "journal_ref",
        "oai_datestamp",
    ]
    event = conn.execute(
        "SELECT payload_json FROM arxiv_events WHERE event_type = 'paper_updated'"
    ).fetchone()
    assert event is not None
    payload = json.loads(event["payload_json"])
    assert payload == {
        "latest_version": 2,
        "changed": True,
        "changed_fields": result["changed_fields"],
    }
    payload_json = event["payload_json"]
    assert "cs.CV" not in payload_json
    assert "https://example.invalid/license/v2" not in payload_json
    assert "10.48550/arXiv.2606.29148.v2" not in payload_json
    assert "Journal of Changed Fixtures 2 (2026)" not in payload_json


def test_paperstream_metadata_upsert_versionless_update_preserves_latest_version():
    """Versionless metadata updates refresh paper fields without inventing version rows."""
    from researchfs.paperstream import arxiv_asset_key, upsert_paperstream_metadata_record

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.AI")
    before_versions = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM arxiv_versions WHERE arxiv_id = ? ORDER BY version",
            ("2606.29148",),
        ).fetchall()
    ]

    result = upsert_paperstream_metadata_record(
        conn,
        {
            "arxiv_id": "2606.29148",
            "title": "Updated Metadata-Only Title",
            "abstract": "Updated paper-level metadata without a version payload.",
            "authors": ["Grace Hopper"],
            "categories": ["cs.LG", "cs.AI"],
            "primary_category": "cs.LG",
            "license": "http://creativecommons.org/licenses/by/4.0/",
            "doi": "10.48550/arXiv.2606.29148",
            "journal_ref": "Updated Journal Ref",
            "abs_url": "https://arxiv.org/abs/2606.29148",
            "oai_datestamp": "2026-07-04",
            "raw_oai_xml": "<record><id>2606.29148</id><title>Updated</title></record>",
        },
    )

    assert result["latest_version"] == 2
    assert result["changed"] is True
    assert result["versions_seen"] == 0
    assert result["versions_changed"] == 0
    assert result["asset_key"] == arxiv_asset_key("2606.29148", 2, "metadata")

    paper = conn.execute("SELECT * FROM arxiv_papers WHERE arxiv_id = ?", ("2606.29148",)).fetchone()
    assert paper["latest_version"] == 2
    assert paper["title"] == "Updated Metadata-Only Title"
    assert paper["primary_category"] == "cs.LG"
    assert json.loads(paper["categories_json"]) == ["cs.LG", "cs.AI"]
    assert paper["license"] == "http://creativecommons.org/licenses/by/4.0/"
    assert paper["oai_datestamp"] == "2026-07-04"
    after_versions = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM arxiv_versions WHERE arxiv_id = ? ORDER BY version",
            ("2606.29148",),
        ).fetchall()
    ]
    assert after_versions == before_versions


def test_paperstream_metadata_upsert_ignores_stale_or_undated_existing_updates():
    """Older or undated existing-paper metadata cannot clobber fresher OAI state."""
    from researchfs.paperstream import arxiv_asset_key, upsert_paperstream_metadata_record

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.AI")

    stale_result = upsert_paperstream_metadata_record(
        conn,
        {
            "arxiv_id": "2606.29148",
            "latest_version": 3,
            "title": "Stale Metadata",
            "abstract": "This older OAI record should not win.",
            "authors": ["Stale Author"],
            "categories": ["stat.ML"],
            "primary_category": "stat.ML",
            "license": "stale-license",
            "abs_url": "https://arxiv.org/abs/2606.29148",
            "oai_datestamp": "2026-07-02",
            "raw_oai_xml": "<record><id>stale</id></record>",
            "versions": [{"version": 3, "title": "Stale Version"}],
        },
    )
    undated_result = upsert_paperstream_metadata_record(
        conn,
        {
            "arxiv_id": "2606.29148",
            "title": "Undated Metadata",
            "abstract": "This missing-datestamp record should not win.",
            "authors": ["Undated Author"],
            "categories": ["math.OC"],
            "primary_category": "math.OC",
            "license": "undated-license",
            "abs_url": "https://arxiv.org/abs/2606.29148",
            "raw_oai_xml": "<record><id>undated</id></record>",
        },
    )

    assert stale_result["changed"] is False
    assert stale_result["latest_version"] == 2
    assert stale_result["versions_changed"] == 0
    assert stale_result["changed_fields"] == []
    assert stale_result["asset_key"] == arxiv_asset_key("2606.29148", 2, "metadata")
    assert undated_result["changed"] is False
    assert undated_result["latest_version"] == 2
    assert undated_result["versions_seen"] == 0
    assert undated_result["changed_fields"] == []
    assert undated_result["versions_changed"] == 0

    paper = conn.execute("SELECT * FROM arxiv_papers WHERE arxiv_id = ?", ("2606.29148",)).fetchone()
    assert paper["latest_version"] == 2
    assert paper["title"] == "Paper 2606.29148"
    assert paper["primary_category"] == "cs.AI"
    assert paper["license"] is None
    assert paper["oai_datestamp"] == "2026-07-03"
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_versions WHERE arxiv_id = ?",
        ("2606.29148",),
    ).fetchone()[0] == 1
    assert _paperstream_row_count(conn, "arxiv_assets") == 0
    assert _paperstream_row_count(conn, "arxiv_events") == 0

def test_paperstream_enqueue_missing_arxiv_assets_creates_jobs_and_is_idempotent():
    """Safe queue activation creates missing download assets/jobs once from metadata rows."""
    from researchfs.paperstream import asset_key, enqueue_missing_arxiv_assets, job_key

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.AI")

    result = enqueue_missing_arxiv_assets(
        conn,
        asset_types=("pdf", "source"),
        priority=7,
        max_attempts=3,
        now="2026-07-04T00:00:00Z",
    )

    expected_asset_keys = [
        asset_key("2606.29148", 2, "pdf"),
        asset_key("2606.29148", 2, "source"),
    ]
    expected_job_keys = [
        job_key("download_pdf", "2606.29148", 2),
        job_key("download_source", "2606.29148", 2),
    ]
    expected_job_payloads = [
        {
            "source": "metadata_queue",
            "asset_key": expected_asset_keys[0],
            "intended_source": "arxiv_pdf",
            "download_url": "https://arxiv.org/pdf/2606.29148.pdf",
        },
        {
            "source": "metadata_queue",
            "asset_key": expected_asset_keys[1],
            "intended_source": "arxiv_source",
            "download_url": "https://arxiv.org/src/2606.29148",
        },
    ]
    assert result["ok"] is True
    assert result["asset_types"] == ["pdf", "source"]
    assert result["papers_scanned"] == 1
    assert result["assets_created"] == 2
    assert result["jobs_queued"] == 2
    assert result["asset_keys"] == expected_asset_keys
    assert result["job_keys"] == expected_job_keys

    asset_rows = conn.execute(
        """
        SELECT asset_key, asset_type, status, source, storage_path, checksum, byte_size
        FROM arxiv_assets
        WHERE asset_type IN ('pdf', 'source')
        ORDER BY asset_type
        """
    ).fetchall()
    assert [row["asset_key"] for row in asset_rows] == expected_asset_keys
    assert [
        (row["asset_type"], row["status"], row["source"])
        for row in asset_rows
    ] == [
        ("pdf", "queued", "metadata_queue"),
        ("source", "queued", "metadata_queue"),
    ]
    assert all(row["storage_path"] is None for row in asset_rows)
    assert all(row["checksum"] is None for row in asset_rows)
    assert all(row["byte_size"] is None for row in asset_rows)

    job_rows = conn.execute(
        """
        SELECT job_key, job_type, asset_type, status, priority, max_attempts, run_after, payload_json
        FROM arxiv_jobs
        ORDER BY job_type
        """
    ).fetchall()
    assert [row["job_key"] for row in job_rows] == expected_job_keys
    assert [
        (row["job_type"], row["asset_type"], row["status"], row["priority"], row["max_attempts"])
        for row in job_rows
    ] == [
        ("download_pdf", "pdf", "queued", 7, 3),
        ("download_source", "source", "queued", 7, 3),
    ]
    assert all(row["run_after"] == "2026-07-04T00:00:00Z" for row in job_rows)
    assert [json.loads(row["payload_json"]) for row in job_rows] == expected_job_payloads
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_seen'"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_queued'"
    ).fetchone()[0] == 2
    job_event_rows = conn.execute(
        """
        SELECT asset_type, payload_json
        FROM arxiv_events
        WHERE event_type = 'job_queued'
        ORDER BY asset_type
        """
    ).fetchall()
    assert [json.loads(row["payload_json"]) for row in job_event_rows] == expected_job_payloads

    rerun = enqueue_missing_arxiv_assets(
        conn,
        asset_types=("pdf", "source"),
        priority=1,
        max_attempts=1,
        now="2026-07-05T00:00:00Z",
    )

    assert rerun["assets_created"] == 0
    assert rerun["jobs_queued"] == 0
    assert rerun["asset_keys"] == []
    assert rerun["job_keys"] == []
    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs").fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type IN ('asset_seen', 'job_queued')"
    ).fetchone()[0] == 4


def test_paperstream_enqueue_missing_arxiv_assets_queues_job_for_preexisting_asset():
    """Existing queued asset rows still receive missing download jobs."""
    from researchfs.paperstream import enqueue_missing_arxiv_assets, job_key, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.AI")
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="queued",
        source="metadata_queue",
    )

    result = enqueue_missing_arxiv_assets(conn, asset_types=("pdf",), priority=9)

    assert result["assets_created"] == 0
    assert result["jobs_queued"] == 1
    assert result["asset_keys"] == []
    assert result["job_keys"] == [job_key("download_pdf", "2606.29148", 1)]
    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 1
    job = conn.execute(
        "SELECT job_type, asset_type, priority, payload_json FROM arxiv_jobs"
    ).fetchone()
    assert {
        "job_type": job["job_type"],
        "asset_type": job["asset_type"],
        "priority": job["priority"],
    } == {"job_type": "download_pdf", "asset_type": "pdf", "priority": 9}
    assert json.loads(job["payload_json"]) == {
        "source": "metadata_queue",
        "asset_key": "2606.29148:v1:pdf",
        "intended_source": "arxiv_pdf",
        "download_url": "https://arxiv.org/pdf/2606.29148.pdf",
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_seen'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_queued'"
    ).fetchone()[0] == 1


def test_paperstream_enqueue_missing_arxiv_assets_filters_category_and_limit():
    """Category matching includes JSON categories and limit bounds queued asset work."""
    from researchfs.paperstream import enqueue_missing_arxiv_assets

    conn = _paperstream_conn()
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=1,
        category="cs.LG",
        categories=["cs.LG", "cs.AI"],
    )
    _insert_paperstream_paper(conn, "2606.29149", version=1, category="stat.ML")

    result = enqueue_missing_arxiv_assets(
        conn,
        asset_types=("pdf",),
        category="cs.AI",
        limit=1,
        now="2026-07-04T00:00:00Z",
    )

    assert result["ok"] is True
    assert result["category"] == "cs.AI"
    assert result["papers_scanned"] == 1
    assert result["assets_created"] == 1
    assert result["jobs_queued"] == 1
    matched_assets = conn.execute(
        "SELECT arxiv_id, asset_type, status FROM arxiv_assets WHERE asset_type = 'pdf'"
    ).fetchall()
    assert [(row["arxiv_id"], row["asset_type"], row["status"]) for row in matched_assets] == [
        ("2606.29148", "pdf", "queued")
    ]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE arxiv_id = '2606.29149'"
    ).fetchone()[0] == 0


def test_paperstream_enqueue_missing_arxiv_assets_rejects_invalid_asset_type():
    """The activation helper only accepts downloadable pdf/source asset types."""
    from researchfs.paperstream import enqueue_missing_arxiv_assets

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)

    with pytest.raises(ValueError, match="asset_types"):
        enqueue_missing_arxiv_assets(conn, asset_types=("metadata",))

    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0


def _assert_paperstream_public_payload_has_no_local_asset_fields(payload):
    encoded = json.dumps(payload, sort_keys=True)
    assert "storage_path" not in encoded
    assert "checksum" not in encoded
    assert "file://" not in encoded
    assert "/private/" not in encoded
    assert "/private/object-store" not in encoded


_PAPERSTREAM_ARXIV_ATTRIBUTION = [
    "Data from arXiv.org.",
    "This project is not endorsed by arXiv.",
    "Users should retrieve e-print content from arXiv unless redistribution is licensed.",
]


def _assert_paperstream_payload_has_arxiv_attribution(payload):
    assert payload["attribution"] == _PAPERSTREAM_ARXIV_ATTRIBUTION


def test_paperstream_state_helper_returns_latest_metadata_assets_jobs_and_links():
    """Paper state exposes latest-version public state without local object details."""
    from researchfs.paperstream import (
        enqueue_arxiv_job,
        get_arxiv_paper_state,
        upsert_arxiv_asset,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=1,
        category="cs.AI",
        categories=["cs.AI"],
    )
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=2,
        category="cs.LG",
        categories=["cs.LG", "cs.AI"],
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="downloaded",
        storage_path="/private/object-store/old.pdf",
        checksum="old-checksum",
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path="/private/object-store/latest.pdf",
        checksum="latest-checksum",
        byte_size=1234,
        source="arxiv",
    )
    upsert_arxiv_asset(conn, "2606.29148", 2, "source", status="queued", source="metadata_queue")
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "bm25_index",
        status="indexed",
        storage_path="/private/object-store/latest-bm25.sqlite",
        checksum="bm25-checksum",
        byte_size=4321,
        source="index_bm25",
    )
    enqueue_arxiv_job(conn, "download_pdf", "2606.29148", 1, "pdf", priority=1)
    enqueue_arxiv_job(conn, "download_source", "2606.29148", 2, "source", priority=4)
    parse_key = enqueue_arxiv_job(
        conn,
        "parse_pdf",
        "2606.29148",
        2,
        "pdf",
        priority=2,
        payload={"private": "/private/object-store/latest.pdf"},
    )
    conn.execute(
        "UPDATE arxiv_jobs SET status = 'failed', error_message = 'parse failed' WHERE job_key = ?",
        (parse_key,),
    )

    state = get_arxiv_paper_state(conn, "2606.29148")

    assert state["arxiv_id"] == "2606.29148"
    assert state["latest_version"] == 2
    assert state["title"] == "Paper 2606.29148"
    assert state["abstract"] == "A deterministic PaperStream test paper."
    assert state["primary_category"] == "cs.LG"
    assert state["categories"] == ["cs.LG", "cs.AI"]
    assert state["links"] == {
        "arxiv_abs": "https://arxiv.org/abs/2606.29148",
        "arxiv_pdf": "https://arxiv.org/pdf/2606.29148.pdf",
        "arxiv_source": "https://arxiv.org/e-print/2606.29148",
    }
    _assert_paperstream_payload_has_arxiv_attribution(state)
    assert state["assets"]["pdf"]["status"] == "downloaded"
    assert state["assets"]["pdf"]["byte_size"] == 1234
    assert state["assets"]["source"]["status"] == "queued"
    assert state["assets"]["bm25_index"]["status"] == "indexed"
    assert state["assets"]["bm25_index"]["byte_size"] == 4321
    assert "metadata" not in state["assets"]
    assert state["jobs"]["total"] == 2
    assert state["jobs"]["by_status"] == {"failed": 1, "queued": 1}
    assert state["jobs"]["by_type"] == {"download_source": 1, "parse_pdf": 1}
    _assert_paperstream_public_payload_has_no_local_asset_fields(state)
    assert get_arxiv_paper_state(conn, "2606.00000") is None


def test_paperstream_missing_assets_filters_latest_category_and_existing_ready_or_queued():
    """Missing-assets inspection reports only latest-version papers lacking queued/ready assets."""
    from researchfs.paperstream import list_missing_arxiv_assets, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=1,
        category="cs.AI",
        categories=["cs.AI"],
    )
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=2,
        category="cs.LG",
        categories=["cs.LG", "cs.AI"],
    )
    _insert_paperstream_paper(conn, "2606.29149", version=1, category="cs.AI", categories=["cs.AI"])
    _insert_paperstream_paper(conn, "2606.29150", version=1, category="cs.AI", categories=["cs.AI"])
    _insert_paperstream_paper(conn, "2606.29151", version=1, category="stat.ML", categories=["stat.ML"])
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="downloaded",
        storage_path="/private/old-version.pdf",
        checksum="old-version-checksum",
    )
    upsert_arxiv_asset(conn, "2606.29149", 1, "pdf", status="queued", source="metadata_queue")
    upsert_arxiv_asset(
        conn,
        "2606.29150",
        1,
        "pdf",
        status="downloaded",
        storage_path="/private/ready.pdf",
        checksum="ready-checksum",
    )
    upsert_arxiv_asset(conn, "2606.29151", 1, "pdf", status="failed")

    missing_pdf = list_missing_arxiv_assets(conn, asset_type="pdf", category="cs.AI", limit=10)
    missing_source = list_missing_arxiv_assets(conn, asset_type="source", category="cs.AI", limit=2)

    assert [row["arxiv_id"] for row in missing_pdf] == ["2606.29148"]
    assert missing_pdf[0]["latest_version"] == 2
    assert missing_pdf[0]["asset_type"] == "pdf"
    assert missing_pdf[0]["primary_category"] == "cs.LG"
    assert missing_pdf[0]["categories"] == ["cs.LG", "cs.AI"]
    assert missing_pdf[0]["links"] == {
        "arxiv_abs": "https://arxiv.org/abs/2606.29148",
        "arxiv_pdf": "https://arxiv.org/pdf/2606.29148.pdf",
        "arxiv_source": "https://arxiv.org/e-print/2606.29148",
    }
    _assert_paperstream_payload_has_arxiv_attribution(missing_pdf[0])
    assert [row["arxiv_id"] for row in missing_source] == ["2606.29148", "2606.29149"]
    assert all(row["asset_type"] == "source" for row in missing_source)
    assert all(row["attribution"] == _PAPERSTREAM_ARXIV_ATTRIBUTION for row in missing_source)
    _assert_paperstream_public_payload_has_no_local_asset_fields(missing_pdf)
    _assert_paperstream_public_payload_has_no_local_asset_fields(missing_source)


def test_paperstream_jobs_helper_filters_status_job_types_limit_and_omits_payload_json():
    """Jobs inspection returns a bounded stable queue view without payload_json."""
    from researchfs.paperstream import enqueue_arxiv_job, list_arxiv_jobs

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    failed_download = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29148",
        1,
        "pdf",
        priority=5,
        run_after="2026-07-04T00:00:00Z",
        payload={"secret": "/private/pdf"},
    )
    failed_parse = enqueue_arxiv_job(
        conn,
        "parse_pdf",
        "2606.29148",
        1,
        "pdf",
        priority=3,
        run_after="2026-07-03T00:00:00Z",
        payload={"secret": "/private/parse"},
    )
    queued_download = enqueue_arxiv_job(conn, "download_source", "2606.29149", 1, "source", priority=1)
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 2 WHERE job_key = ?", (failed_download,))
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 1 WHERE job_key = ?", (failed_parse,))
    conn.execute("UPDATE arxiv_jobs SET status = 'queued' WHERE job_key = ?", (queued_download,))

    failed_jobs = list_arxiv_jobs(
        conn,
        status="failed",
        job_types=("download_pdf", "parse_pdf"),
        limit=1,
    )

    assert [row["job_key"] for row in failed_jobs] == [failed_parse]
    assert failed_jobs[0]["status"] == "failed"
    assert failed_jobs[0]["job_type"] == "parse_pdf"
    assert failed_jobs[0]["attempts"] == 1
    assert "payload_json" not in failed_jobs[0]
    assert "payload" not in failed_jobs[0]
    _assert_paperstream_public_payload_has_no_local_asset_fields(failed_jobs)
    queued_jobs = list_arxiv_jobs(conn, status="queued", limit=10)
    assert [row["job_key"] for row in queued_jobs] == [queued_download]


@pytest.mark.parametrize(
    ("helper_name", "kwargs", "message"),
    [
        ("list_missing_arxiv_assets", {"asset_type": "metadata"}, "asset_type"),
        ("list_missing_arxiv_assets", {"asset_type": "pdf", "limit": -1}, "limit"),
        ("list_missing_arxiv_assets", {"asset_type": "pdf", "limit": "over_cap"}, "limit"),
        ("list_arxiv_jobs", {"status": "all"}, "status"),
        ("list_arxiv_jobs", {"job_types": ("delete_everything",)}, "job_types|job_type"),
        ("list_arxiv_jobs", {"limit": 0}, "limit"),
        ("list_arxiv_jobs", {"limit": "over_cap"}, "limit"),
    ],
)
def test_paperstream_state_missing_and_jobs_invalid_args_fail_before_broad_data(
    helper_name, kwargs, message
):
    """Inspection helpers validate filters and caps before touching broad PaperStream data."""
    import researchfs.paperstream as paperstream

    class ExecuteForbidden:
        def execute(self, *_args, **_kwargs):
            raise AssertionError(f"{helper_name} should validate before sqlite execution")

    if kwargs.get("limit") == "over_cap":
        kwargs = {**kwargs, "limit": paperstream._PAPERSTREAM_INSPECTION_MAX_LIMIT + 1}

    with pytest.raises(ValueError, match=message):
        getattr(paperstream, helper_name)(ExecuteForbidden(), **kwargs)


def test_paperstream_state_missing_and_jobs_accept_limit_at_cap_on_empty_db():
    """Inspection caps are inclusive and preserve public JSON-ready payloads."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    limit = paperstream._PAPERSTREAM_INSPECTION_MAX_LIMIT

    missing_assets = paperstream.list_missing_arxiv_assets(conn, asset_type="pdf", limit=limit)
    jobs = paperstream.list_arxiv_jobs(conn, limit=limit)

    assert missing_assets == []
    assert jobs == []
    _assert_paperstream_public_payload_has_no_local_asset_fields(missing_assets)
    _assert_paperstream_public_payload_has_no_local_asset_fields(jobs)


def test_paperstream_api_arxiv_paper_route_returns_state_and_structured_errors():
    """GET /arxiv/papers/{arxiv_id} exposes helper state and structured 400/404 errors."""
    from researchfs.api import ResearchFSLabAPI
    from researchfs.paperstream import enqueue_arxiv_job, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.AI")
    upsert_arxiv_asset(conn, "2606.29148", 1, "pdf", status="downloaded", byte_size=1234)
    enqueue_arxiv_job(conn, "download_source", "2606.29148", 1, "source", priority=4)
    api = ResearchFSLabAPI(conn=conn)

    response = api.dispatch("GET", "/arxiv/papers/2606.29148")

    assert response.status == 200
    assert response.body["ok"] is True
    assert response.body["arxiv_id"] == "2606.29148"
    assert response.body["latest_version"] == 1
    assert response.body["assets"]["pdf"]["status"] == "downloaded"
    assert response.body["jobs"]["by_type"] == {"download_source": 1}
    _assert_paperstream_payload_has_arxiv_attribution(response.body)
    _assert_paperstream_public_payload_has_no_local_asset_fields(response.body)

    missing = api.dispatch("GET", "/arxiv/papers/2606.00000")
    assert missing.status == 404
    assert missing.body["ok"] is False
    assert missing.body["error"]["code"] == "not_found"

    invalid = api.dispatch("GET", "/arxiv/papers/not-an-arxiv-id")
    assert invalid.status == 400
    assert invalid.body["ok"] is False
    assert invalid.body["error"]["code"] == "invalid_arxiv_id"


def test_paperstream_client_missing_assets_route_respects_filters_limit_and_validation():
    """GET /arxiv/assets/missing filters by asset_type/category/limit and rejects blank asset_type."""
    from researchfs.lab_client import ResearchFSLabClient
    from researchfs.paperstream import upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.AI", categories=["cs.AI"])
    _insert_paperstream_paper(conn, "2606.29149", version=1, category="cs.AI", categories=["cs.AI"])
    _insert_paperstream_paper(conn, "2606.29150", version=1, category="stat.ML", categories=["stat.ML"])
    upsert_arxiv_asset(conn, "2606.29149", 1, "pdf", status="queued")
    client = ResearchFSLabClient(conn=conn)

    response = client.list_missing_arxiv_assets(asset_type="pdf", category="cs.AI", limit=1)

    assert response["ok"] is True
    assert [row["arxiv_id"] for row in response["assets"]] == ["2606.29148"]
    assert response["assets"][0]["asset_type"] == "pdf"
    assert response["assets"][0]["primary_category"] == "cs.AI"
    _assert_paperstream_payload_has_arxiv_attribution(response["assets"][0])
    _assert_paperstream_public_payload_has_no_local_asset_fields(response)

    blank = client._api.dispatch("GET", "/arxiv/assets/missing?asset_type=&category=cs.AI&limit=1")
    assert blank.status == 400
    assert blank.body["ok"] is False
    assert blank.body["error"]["code"] == "invalid_field"

    invalid = client._api.dispatch("GET", "/arxiv/assets/missing?asset_type=metadata")
    assert invalid.status == 400
    assert invalid.body["ok"] is False
    assert invalid.body["error"]["code"] == "invalid_field"


def test_paperstream_client_jobs_route_respects_filters_limit_and_validation():
    """GET /arxiv/jobs filters by status/job_types/limit and rejects invalid query values."""
    from researchfs.lab_client import ResearchFSLabClient
    from researchfs.paperstream import enqueue_arxiv_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    failed_download = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29148",
        1,
        "pdf",
        priority=5,
        run_after="2026-07-04T00:00:00Z",
    )
    failed_source = enqueue_arxiv_job(
        conn,
        "download_source",
        "2606.29149",
        1,
        "source",
        priority=3,
        run_after="2026-07-03T00:00:00Z",
    )
    queued_parse = enqueue_arxiv_job(conn, "parse_pdf", "2606.29148", 1, "pdf", priority=1)
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 2 WHERE job_key = ?", (failed_download,))
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 1 WHERE job_key = ?", (failed_source,))
    conn.execute("UPDATE arxiv_jobs SET status = 'queued' WHERE job_key = ?", (queued_parse,))
    client = ResearchFSLabClient(conn=conn)

    response = client.list_arxiv_jobs(
        status="failed",
        job_types=("download_pdf", "download_source"),
        limit=1,
    )

    assert response["ok"] is True
    assert [row["job_key"] for row in response["jobs"]] == [failed_source]
    assert response["jobs"][0]["status"] == "failed"
    assert response["jobs"][0]["job_type"] == "download_source"
    assert "payload_json" not in response["jobs"][0]
    _assert_paperstream_public_payload_has_no_local_asset_fields(response)

    for route in (
        "/arxiv/jobs?status=",
        "/arxiv/jobs?status=all",
        "/arxiv/jobs?job_types=delete_everything",
        "/arxiv/jobs?job_types=download_pdf,",
    ):
        error = client._api.dispatch("GET", route)
        assert error.status == 400
        assert error.body["ok"] is False
        assert error.body["error"]["code"] == "invalid_field"


def test_paperstream_api_sync_status_matches_helper_fields_and_inserted_counters():
    """GET /arxiv/sync/status returns get_paperstream_status fields, including inserted counters."""
    from researchfs.api import ResearchFSLabAPI
    from researchfs.lab_client import ResearchFSLabClient
    from researchfs.paperstream import enqueue_arxiv_job, get_paperstream_status, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    upsert_arxiv_asset(conn, "2606.29148", 1, "pdf", status="failed")
    enqueue_arxiv_job(conn, "download_pdf", "2606.29148", 1, "pdf", priority=1)
    failed_job = enqueue_arxiv_job(conn, "parse_pdf", "2606.29148", 1, "pdf", priority=2)
    conn.execute("UPDATE arxiv_jobs SET status = 'failed' WHERE job_key = ?", (failed_job,))
    _insert_paperstream_sync_state(
        conn,
        status="succeeded",
        committed_from="2026-07-03T12:00:00Z",
        last_response_date="2026-07-03T12:00:00Z",
        records_seen_last_run=15,
        records_updated_last_run=6,
        records_inserted_last_run=4,
        versions_inserted_last_run=9,
        jobs_queued_last_run=2,
        finished_at="2026-07-03T12:00:01Z",
        updated_at="2026-07-03T12:00:01Z",
    )
    expected = get_paperstream_status(conn)
    response = ResearchFSLabAPI(conn=conn).dispatch("GET", "/arxiv/sync/status")

    assert response.status == 200
    assert response.body == {"ok": True, **expected}
    assert response.body["records_inserted_last_run"] == 4
    assert response.body["versions_inserted_last_run"] == 9
    assert response.body["sync"]["records_inserted_last_run"] == 4
    assert response.body["sync"]["versions_inserted_last_run"] == 9
    assert response.body["queued_jobs"] == 1
    assert response.body["failed_jobs"] == 1
    assert response.body["assets_failed"] == 1
    assert response.body["oldest_queued_job"] is not None
    assert response.body["jobs"]["by_type"] == {"download_pdf": 1, "parse_pdf": 1}
    assert ResearchFSLabClient(conn=conn).paperstream_sync_status() == response.body


def test_paperstream_api_index_lists_prd_arxiv_paths_without_paperstream_aliases():
    """GET / lists the exact PRD §19 /arxiv paths and no /paperstream aliases."""
    from researchfs.api import ResearchFSLabAPI

    response = ResearchFSLabAPI(conn=_paperstream_conn()).dispatch("GET", "/")

    assert response.status == 200
    route_paths = {route["path"] for route in response.body["routes"]}
    assert {
        "/arxiv/papers/{arxiv_id}",
        "/arxiv/assets/missing",
        "/arxiv/jobs",
        "/arxiv/sync/status",
        "/arxiv/search",
        "/arxiv/papers/{arxiv_id}/chunks",
    }.issubset(route_paths)
    assert not any(path.startswith("/paperstream") for path in route_paths)


def test_paperstream_api_search_route_returns_public_results_and_validates_bool(tmp_path):
    """GET /arxiv/search exposes bounded public PaperStream search with strict query bools."""
    from researchfs.api import ResearchFSLabAPI
    from researchfs.paperstream import upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        title="Neuralquokka API Search Fixture",
        abstract="This paper should be found by the API search route.",
        authors=["Ada Lovelace"],
        category="cs.IR",
        categories=["cs.IR"],
    )
    private_path = tmp_path / "private-api-search.pdf"
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="downloaded",
        storage_path=private_path,
        checksum="sha256:api-search-private",
        byte_size=42,
        source="download_pdf",
    )
    api = ResearchFSLabAPI(conn=conn)

    response = api.dispatch(
        "GET",
        "/arxiv/search?query=neuralquokka&include_chunks=false&limit=3&max_chunk_chars=64",
    )

    assert response.status == 200
    assert response.body["ok"] is True
    assert response.body["query"] == "neuralquokka"
    assert response.body["chunks"] == []
    assert [paper["arxiv_id"] for paper in response.body["papers"]] == ["2606.29148"]
    _assert_paperstream_payload_has_arxiv_attribution(response.body)
    _assert_paperstream_public_payload_has_no_local_asset_fields(response.body)
    assert str(private_path) not in json.dumps(response.body, sort_keys=True)

    for route in (
        "/arxiv/search?query=neuralquokka&include_chunks=",
        "/arxiv/search?query=neuralquokka&include_chunks=definitely",
    ):
        invalid = api.dispatch("GET", route)
        assert invalid.status == 400
        assert invalid.body["ok"] is False
        assert invalid.body["error"]["code"] == "invalid_field"


def test_paperstream_api_paper_chunks_route_returns_bounds_and_structured_errors(tmp_path):
    """GET /arxiv/papers/{id}/chunks is routed before paper-state lookup and stays public."""
    from researchfs.api import ResearchFSLabAPI
    from researchfs.paperstream import materialize_paperstream_chunks

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.IR")
    old_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["old private chunk should not be returned"],
    )
    latest_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=2,
        chunks=["latest alpha chunk", "latest beta chunk"],
    )
    assert materialize_paperstream_chunks(conn, limit=10)["ok"] is True
    latest_chunks_path.unlink()
    api = ResearchFSLabAPI(conn=conn)

    response = api.dispatch(
        "GET",
        "/arxiv/papers/2606.29148/chunks?limit=1&offset=1&max_chars=4",
    )

    assert response.status == 200
    assert response.body["arxiv_id"] == "2606.29148"
    assert response.body["version"] == 2
    assert response.body["total"] == 2
    assert response.body["chunks"] == [
        {
            "chunk_index": 1,
            "text": "late",
            "source_asset_type": "pdf",
            "chunks_asset_key": "2606.29148:v2:chunks",
            "chunks_checksum": response.body["chunks"][0]["chunks_checksum"],
        }
    ]
    assert response.body["chunks"][0]["chunks_checksum"].startswith("sha256:")
    _assert_paperstream_payload_has_arxiv_attribution(response.body)
    _assert_paperstream_payload_has_no_storage_path_leak(response.body, old_chunks_path, latest_chunks_path, tmp_path)

    missing = api.dispatch("GET", "/arxiv/papers/2606.00000/chunks")
    assert missing.status == 404
    assert missing.body["error"]["code"] == "not_found"

    invalid = api.dispatch("GET", "/arxiv/papers/not-an-arxiv-id/chunks")
    assert invalid.status == 400
    assert invalid.body["error"]["code"] == "invalid_arxiv_id"


def test_paperstream_client_search_and_chunks_wrappers_use_arxiv_routes(tmp_path):
    """ResearchFSLabClient exposes PaperStream search and chunk retrieval routes."""
    from researchfs.lab_client import ResearchFSLabClient
    from researchfs.paperstream import materialize_paperstream_chunks

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        title="Clientroute Search Fixture",
        abstract="Clientroute should match through the SDK wrapper.",
        authors=["Grace Hopper"],
        category="cs.IR",
        categories=["cs.IR"],
    )
    chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["clientroute chunk payload"],
    )
    assert materialize_paperstream_chunks(conn, limit=10)["ok"] is True
    chunks_path.unlink()
    client = ResearchFSLabClient(conn=conn)

    search = client.search_arxiv_papers("clientroute", limit=5, include_chunks=False)
    chunks = client.get_arxiv_paper_chunks("2606.29148", limit=1, max_chars=11)

    assert search["ok"] is True
    assert [paper["arxiv_id"] for paper in search["papers"]] == ["2606.29148"]
    assert search["chunks"] == []
    assert chunks["arxiv_id"] == "2606.29148"
    assert chunks["chunks"][0]["text"] == "clientroute"
    _assert_paperstream_payload_has_arxiv_attribution(search)
    _assert_paperstream_payload_has_arxiv_attribution(chunks)
    _assert_paperstream_public_payload_has_no_local_asset_fields(search)
    _assert_paperstream_payload_has_no_storage_path_leak(chunks, chunks_path, tmp_path)


def test_paperstream_cli_state_missing_and_jobs_print_expected_json(tmp_path):
    """State inspection CLI commands print JSON matching helper contracts."""
    from researchfs import db
    from researchfs.paperstream import enqueue_arxiv_job, upsert_arxiv_asset

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=2,
        category="cs.LG",
        categories=["cs.LG", "cs.AI"],
    )
    _insert_paperstream_paper(conn, "2606.29149", version=1, category="cs.AI", categories=["cs.AI"])
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path="/private/downloaded.pdf",
        checksum="downloaded-checksum",
    )
    failed_key = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29148",
        2,
        "pdf",
        payload={"secret": "/private/downloaded.pdf"},
    )
    enqueue_arxiv_job(conn, "download_source", "2606.29149", 1, "source")
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 1 WHERE job_key = ?", (failed_key,))
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    paper_state = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "paper-state",
            "2606.29148",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    missing_assets = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "missing-assets",
            "--asset-type",
            "source",
            "--category",
            "cs.AI",
            "--limit",
            "100",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    jobs = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "jobs",
            "--status",
            "failed",
            "--job-types",
            "download_pdf,parse_pdf",
            "--limit",
            "100",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    state_payload = json.loads(paper_state.stdout)
    missing_payload = json.loads(missing_assets.stdout)
    jobs_payload = json.loads(jobs.stdout)
    assert state_payload["arxiv_id"] == "2606.29148"
    assert state_payload["assets"]["pdf"]["status"] == "downloaded"
    assert state_payload["links"]["arxiv_source"] == "https://arxiv.org/e-print/2606.29148"
    assert [row["arxiv_id"] for row in missing_payload] == ["2606.29148", "2606.29149"]
    assert all(row["asset_type"] == "source" for row in missing_payload)
    assert [row["job_key"] for row in jobs_payload] == [failed_key]
    assert "payload_json" not in jobs_payload[0]
    _assert_paperstream_public_payload_has_no_local_asset_fields(state_payload)
    _assert_paperstream_public_payload_has_no_local_asset_fields(missing_payload)
    _assert_paperstream_public_payload_has_no_local_asset_fields(jobs_payload)


@pytest.mark.parametrize(
    "argv",
    [
        ["missing-assets", "--asset-type", "metadata"],
        ["missing-assets", "--asset-type", "pdf", "--limit", "-1"],
        ["jobs", "--status", "not-a-status"],
        ["jobs", "--job-types", "not_a_job"],
        ["jobs", "--limit", "0"],
    ],
)
def test_paperstream_cli_state_missing_and_jobs_invalid_args_fail_before_broad_data(
    tmp_path, argv
):
    """Invalid inspection CLI filters fail without printing broad PaperStream data."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    rejected = subprocess.run(
        [sys.executable, "-m", "researchfs.paperstream", "--db-path", str(db_path), *argv],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode != 0
    assert "2606.29148" not in rejected.stdout


def test_paperstream_cli_state_missing_and_jobs_reject_limits_above_cap_before_broad_data(
    tmp_path,
):
    """Inspection CLI limit caps are enforced without printing broad PaperStream data."""
    from researchfs import db
    import researchfs.paperstream as paperstream

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}
    too_large = str(paperstream._PAPERSTREAM_INSPECTION_MAX_LIMIT + 1)

    for argv in (
        ["missing-assets", "--asset-type", "pdf", "--limit", too_large],
        ["jobs", "--limit", too_large],
    ):
        rejected = subprocess.run(
            [sys.executable, "-m", "researchfs.paperstream", "--db-path", str(db_path), *argv],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert rejected.returncode != 0
        assert "2606.29148" not in rejected.stdout

def test_paperstream_cli_enqueue_assets_writes_queue_rows_and_json_without_downloads(tmp_path):
    """CLI enqueue-assets commits queued download work without fetching or storing files."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=2,
        category="cs.LG",
        categories=["cs.LG", "cs.AI"],
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "enqueue-assets",
            "--asset-types",
            "pdf,source",
            "--category",
            "cs.AI",
            "--limit",
            "100",
            "--priority",
            "7",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["category"] == "cs.AI"
    assert result["asset_types"] == ["pdf", "source"]
    assert result["assets_created"] == 2
    assert result["jobs_queued"] == 2
    assert result["asset_keys"] == [
        "2606.29148:v2:pdf",
        "2606.29148:v2:source",
    ]
    assert result["job_keys"] == [
        "download_pdf:2606.29148:v2",
        "download_source:2606.29148:v2",
    ]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        asset_rows = conn.execute(
            """
            SELECT asset_type, status, source, storage_path, checksum, byte_size
            FROM arxiv_assets
            WHERE asset_type IN ('pdf', 'source')
            ORDER BY asset_type
            """
        ).fetchall()
        assert [
            (row["asset_type"], row["status"], row["source"])
            for row in asset_rows
        ] == [
            ("pdf", "queued", "metadata_queue"),
            ("source", "queued", "metadata_queue"),
        ]
        assert all(row["storage_path"] is None for row in asset_rows)
        assert all(row["checksum"] is None for row in asset_rows)
        assert all(row["byte_size"] is None for row in asset_rows)
        job_rows = conn.execute(
            """
            SELECT job_type, asset_type, status, priority
            FROM arxiv_jobs
            ORDER BY job_type
            """
        ).fetchall()
        assert [
            (row["job_type"], row["asset_type"], row["status"], row["priority"])
            for row in job_rows
        ] == [
            ("download_pdf", "pdf", "queued", 7),
            ("download_source", "source", "queued", 7),
        ]
    finally:
        conn.close()


def test_paperstream_sync_default_run_persists_metadata_only_without_download_work():
    """Default metadata sync writes paper/version/metadata state without download work."""
    from researchfs.paperstream import get_paperstream_status, sync_paperstream_metadata

    conn = _paperstream_conn()

    result = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert result["ok"] is True
    assert result["records_seen"] == 2
    assert result["records_deleted"] == 1
    assert result["records_updated"] == 0
    assert result["records_inserted"] == 1
    assert result["versions_inserted"] == 2
    assert state["status"] == "succeeded"
    assert state["records_seen_last_run"] == 2
    assert state["records_updated_last_run"] == 0
    assert state["records_inserted_last_run"] == 1
    assert state["versions_inserted_last_run"] == 2
    assert state["metadata_prefix"] == "arXivRaw"
    assert state["set_spec"] is None
    status = get_paperstream_status(conn)
    assert status["metadata_prefix"] == "arXivRaw"
    assert status["set_spec"] is None
    assert status["sync"]["metadata_prefix"] == "arXivRaw"
    assert status["sync"]["set_spec"] is None
    page_payload = json.loads(
        conn.execute(
            "SELECT payload_json FROM arxiv_events WHERE event_type = 'sync_page_committed'"
        ).fetchone()["payload_json"]
    )
    assert page_payload["records_inserted"] == 1
    assert page_payload["versions_inserted"] == 2
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2

    assert result["jobs_queued"] == 0
    asset_rows = conn.execute(
        """
        SELECT asset_type, status, source, storage_path, checksum, byte_size
        FROM arxiv_assets
        ORDER BY asset_type
        """
    ).fetchall()
    assert [
        (row["asset_type"], row["status"], row["source"])
        for row in asset_rows
    ] == [("metadata", "seen", "oai")]
    assert all(row["storage_path"] is None for row in asset_rows)
    assert all(row["checksum"] is None for row in asset_rows)
    assert all(row["byte_size"] is None for row in asset_rows)
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_assets WHERE asset_type IN ('pdf', 'source')"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type IN ('download_pdf', 'download_source')"
    ).fetchone()[0] == 0


def test_paperstream_sync_scope_persists_and_reports_normalized_values():
    """Metadata sync scope is normalized, persisted, and exposed through status."""
    from researchfs.paperstream import get_paperstream_status, sync_paperstream_metadata

    conn = _paperstream_conn()
    sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        metadata_prefix="  arXivCustom  ",
        set_spec="  cs.AI  ",
    )

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    status = get_paperstream_status(conn)
    sync_started_payload = json.loads(
        conn.execute(
            """
            SELECT payload_json
            FROM arxiv_events
            WHERE event_type = 'sync_started'
            ORDER BY event_id DESC
            LIMIT 1
            """
        ).fetchone()["payload_json"]
    )
    assert result["ok"] is True
    assert result["metadata_prefix"] == "arXivCustom"
    assert result["set_spec"] == "cs.AI"
    assert state["metadata_prefix"] == "arXivCustom"
    assert state["set_spec"] == "cs.AI"
    assert status["metadata_prefix"] == "arXivCustom"
    assert status["set_spec"] == "cs.AI"
    assert status["sync"]["metadata_prefix"] == "arXivCustom"
    assert status["sync"]["set_spec"] == "cs.AI"
    assert sync_started_payload["metadata_prefix"] == "arXivCustom"
    assert sync_started_payload["set_spec"] == "cs.AI"


@pytest.mark.parametrize(
    ("scope_kwargs", "message"),
    [
        ({"metadata_prefix": None}, "metadata_prefix must not be empty"),
        ({"metadata_prefix": "   "}, "metadata_prefix must not be empty"),
        ({"set_spec": "   "}, "set_spec must not be empty"),
    ],
)
def test_paperstream_sync_scope_rejects_blank_values_without_mutation(scope_kwargs, message):
    """Blank sync scope values fail before writing sync state or audit events."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    with pytest.raises(ValueError, match=message):
        sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], **scope_kwargs)

    assert conn.execute("SELECT COUNT(*) FROM arxiv_sync_state").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0

def test_paperstream_sync_rebuilds_missing_metadata_bloom_filter_on_startup():
    """Non-dry-run metadata sync rebuilds the version-key Bloom filter before pages."""
    from researchfs.paperstream import load_bloom_filter, sync_paperstream_metadata

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)

    assert load_bloom_filter(conn, "seen_metadata_versions") is None

    result = sync_paperstream_metadata(
        conn,
        [{"records": [], "response_date": "2026-07-04T12:00:00Z"}],
        sync_name="default",
    )

    assert result["ok"] is True
    assert result["metadata_bloom_filter_loaded"] == 0
    assert result["metadata_bloom_filter_rebuilt"] == 1
    assert result["metadata_bloom_filter_refreshed"] == 0
    assert result["metadata_bloom_prechecks"] == 0
    loaded = load_bloom_filter(conn, "seen_metadata_versions")
    assert loaded is not None
    assert loaded.item_count == 1
    assert "2606.29148:v1" in loaded


def test_paperstream_sync_bloom_false_positive_does_not_skip_missing_version_insert():
    """A probable Bloom hit is checked in SQLite and cannot suppress the upsert path."""
    from researchfs.paperstream import (
        BloomFilter,
        load_bloom_filter,
        save_bloom_filter,
        sync_paperstream_metadata,
    )

    conn = _paperstream_conn()
    false_positive_filter = BloomFilter.from_bytes(
        b"\xff",
        bit_count=8,
        hash_count=1,
        item_count=999,
        false_positive_rate=0.01,
    )
    save_bloom_filter(
        conn,
        "seen_metadata_versions",
        false_positive_filter,
        source_table="arxiv_versions",
        source_column="version",
    )

    result = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")

    assert result["ok"] is True
    assert result["metadata_bloom_filter_loaded"] == 1
    assert result["metadata_bloom_filter_rebuilt"] == 0
    assert result["metadata_bloom_prechecks"] == 2
    assert result["metadata_bloom_hits"] == 2
    assert result["metadata_bloom_sqlite_checks"] == 2
    assert result["metadata_bloom_confirmed"] == 0
    assert result["metadata_bloom_false_positives"] == 2
    assert result["versions_inserted"] == 2
    assert result["metadata_bloom_filter_refreshed"] == 1
    sync_succeeded = conn.execute(
        "SELECT payload_json FROM arxiv_events WHERE event_type = 'sync_succeeded'"
    ).fetchone()
    assert json.loads(sync_succeeded["payload_json"])["metadata_bloom_filter_refreshed"] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2
    refreshed = load_bloom_filter(conn, "seen_metadata_versions")
    assert refreshed is not None
    assert refreshed.item_count == 2
    assert "2606.29148:v1" in refreshed
    assert "2606.29148:v2" in refreshed



def test_paperstream_sync_bad_oai_page_records_failure_without_advancing_cursor():
    """Invalid OAI pages fail sync without moving the durable cursor or committing records."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    first = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")
    bad_page = (
        _PAPERSTREAM_OAI_FINAL_PAGE.replace(
            "<responseDate>2026-07-04T12:00:00Z</responseDate>",
            "<responseDate>   </responseDate>",
        )
        .replace("oai:arXiv.org:2606.29148", "oai:arXiv.org:2606.99999")
        .replace("<id>2606.29148</id>", "<id>2606.99999</id>")
        .replace("Metadata Only PaperStream Core", "Bad Cursor Page Must Not Commit")
        .replace("10.48550/arXiv.2606.29148", "10.48550/arXiv.2606.99999")
    )
    event_count_after_success = conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0]

    assert first["ok"] is True
    assert first["committed_from"] == "2026-07-04T12:00:00Z"
    with pytest.raises(ValueError, match="missing responseDate"):
        sync_paperstream_metadata(conn, [bad_page], sync_name="default")

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert state["status"] == "failed"
    assert state["committed_from"] == "2026-07-04T12:00:00Z"
    assert state["last_response_date"] == "2026-07-04T12:00:00Z"
    assert state["pending_response_date"] is None
    assert "missing responseDate" in state["error_message"]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_papers WHERE arxiv_id = '2606.99999'"
    ).fetchone()[0] == 0
    new_event_rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT event_type, payload_json
            FROM arxiv_events
            WHERE event_id > ?
            ORDER BY event_id
            """,
            (event_count_after_success,),
        ).fetchall()
    ]
    new_events = [row["event_type"] for row in new_event_rows]
    assert new_events == ["sync_started", "sync_failed"]
    failure_payload = json.loads(new_event_rows[-1]["payload_json"])
    assert failure_payload["sync_name"] == "default"
    assert "missing responseDate" in failure_payload["error"]
    assert "cursor_committed" not in new_events


def test_paperstream_sync_failure_does_not_persist_refreshed_metadata_bloom_filter():
    """Failed syncs may keep committed page rows but not a refreshed version Bloom filter."""
    from researchfs.paperstream import load_bloom_filter, sync_paperstream_metadata

    conn = _paperstream_conn()

    def fail_after_page():
        raise RuntimeError("fixture page unavailable")

    with pytest.raises(RuntimeError, match="fixture page unavailable"):
        sync_paperstream_metadata(
            conn,
            [_PAPERSTREAM_OAI_FINAL_PAGE, fail_after_page],
            sync_name="default",
        )

    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2
    startup_filter = load_bloom_filter(conn, "seen_metadata_versions")
    assert startup_filter is not None
    assert startup_filter.item_count == 0
    assert "2606.29148:v1" not in startup_filter
    assert "2606.29148:v2" not in startup_filter

def test_paperstream_sync_emits_prd_events_and_does_not_duplicate_queue_audits():
    """Successful metadata sync emits required audit events without duplicate queue logs."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    def event_rows():
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT event_type, arxiv_id, version, asset_type, job_key, payload_json
                FROM arxiv_events
                ORDER BY event_id
                """
            ).fetchall()
        ]

    def event_counts(events):
        counts = {}
        for event in events:
            counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1
        return counts

    first = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets=("pdf", "source"),
    )
    first_events = event_rows()

    assert first["ok"] is True
    assert first["jobs_queued"] == 2
    assert event_counts(first_events) == {
        "sync_started": 1,
        "paper_seen": 1,
        "version_seen": 2,
        "asset_seen": 3,
        "job_queued": 2,
        "paper_deleted": 1,
        "sync_page_committed": 1,
        "sync_succeeded": 1,
        "cursor_committed": 1,
    }

    sync_started = [event for event in first_events if event["event_type"] == "sync_started"]
    assert json.loads(sync_started[0]["payload_json"]) == {
        "sync_name": "default",
        "committed_from": None,
        "metadata_prefix": "arXivRaw",
        "set_spec": None,
    }
    page_committed = [
        event for event in first_events if event["event_type"] == "sync_page_committed"
    ]
    assert json.loads(page_committed[0]["payload_json"]) == {
        "sync_name": "default",
        "page": 1,
        "response_date": "2026-07-04T12:00:00Z",
        "resumption_token": None,
        "records_seen": 2,
        "records_updated": 0,
        "records_inserted": 1,
        "records_deleted": 1,
        "versions_inserted": 2,
        "assets_seen": 1,
        "jobs_queued": 2,
    }
    sync_succeeded = [event for event in first_events if event["event_type"] == "sync_succeeded"]
    sync_succeeded_payload = json.loads(sync_succeeded[0]["payload_json"])
    assert sync_succeeded_payload["sync_name"] == "default"
    assert sync_succeeded_payload["committed_from"] == "2026-07-04T12:00:00Z"
    assert sync_succeeded_payload["records_seen"] == 2
    assert sync_succeeded_payload["records_inserted"] == 1
    assert sync_succeeded_payload["versions_inserted"] == 2
    assert sync_succeeded_payload["jobs_queued"] == 2
    cursor_committed = [
        event for event in first_events if event["event_type"] == "cursor_committed"
    ]
    assert json.loads(cursor_committed[0]["payload_json"]) == {
        "sync_name": "default",
        "committed_from": "2026-07-04T12:00:00Z",
        "last_response_date": "2026-07-04T12:00:00Z",
        "last_resumption_token": None,
    }

    paper_seen = [event for event in first_events if event["event_type"] == "paper_seen"]
    assert [(event["arxiv_id"], event["version"]) for event in paper_seen] == [
        ("2606.29148", 2)
    ]
    assert json.loads(paper_seen[0]["payload_json"]) == {
        "latest_version": 2,
        "changed": True,
    }
    version_seen = [event for event in first_events if event["event_type"] == "version_seen"]
    assert [
        (event["arxiv_id"], event["version"], json.loads(event["payload_json"])["changed"])
        for event in version_seen
    ] == [
        ("2606.29148", 1, True),
        ("2606.29148", 2, True),
    ]
    asset_seen = [event for event in first_events if event["event_type"] == "asset_seen"]
    assert [
        (
            event["arxiv_id"],
            event["version"],
            event["asset_type"],
            json.loads(event["payload_json"])["asset_key"],
            json.loads(event["payload_json"]).get("source"),
        )
        for event in asset_seen
    ] == [
        ("2606.29148", 2, "metadata", "2606.29148:v2:metadata", None),
        ("2606.29148", 2, "pdf", "2606.29148:v2:pdf", "metadata_sync"),
        ("2606.29148", 2, "source", "2606.29148:v2:source", "metadata_sync"),
    ]
    job_queued = [event for event in first_events if event["event_type"] == "job_queued"]
    assert [
        (event["arxiv_id"], event["version"], event["asset_type"], event["job_key"])
        for event in job_queued
    ] == [
        ("2606.29148", 2, "pdf", "download_pdf:2606.29148:v2"),
        ("2606.29148", 2, "source", "download_source:2606.29148:v2"),
    ]
    expected_job_payloads = [
        {
            "source": "metadata_sync",
            "asset_key": "2606.29148:v2:pdf",
            "intended_source": "arxiv_pdf",
            "download_url": "https://arxiv.org/pdf/2606.29148.pdf",
        },
        {
            "source": "metadata_sync",
            "asset_key": "2606.29148:v2:source",
            "intended_source": "arxiv_source",
            "download_url": "https://arxiv.org/src/2606.29148",
        },
    ]
    assert [json.loads(event["payload_json"]) for event in job_queued] == expected_job_payloads
    job_rows = conn.execute(
        """
        SELECT payload_json
        FROM arxiv_jobs
        WHERE job_type IN ('download_pdf', 'download_source')
        ORDER BY job_type
        """
    ).fetchall()
    assert [json.loads(row["payload_json"]) for row in job_rows] == expected_job_payloads

    second = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets=("pdf", "source"),
    )
    second_events = event_rows()[len(first_events):]

    assert second["ok"] is True
    assert second["jobs_queued"] == 0
    assert "job_queued" not in {event["event_type"] for event in second_events}
    assert [
        event["asset_type"]
        for event in second_events
        if event["event_type"] == "asset_seen"
    ] == ["metadata"]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type IN ('download_pdf', 'download_source')"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_queued'"
    ).fetchone()[0] == 2


def test_paperstream_sync_enqueue_assets_pdf_string_queues_only_pdf_asset_jobs():
    """Direct metadata sync can queue only PDF follow-up work."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets="pdf",
    )

    assert result["ok"] is True
    assert result["jobs_queued"] == 1
    asset_rows = conn.execute(
        """
        SELECT asset_type, status, source
        FROM arxiv_assets
        WHERE asset_type IN ('pdf', 'source')
        ORDER BY asset_type
        """
    ).fetchall()
    assert [(row["asset_type"], row["status"], row["source"]) for row in asset_rows] == [
        ("pdf", "queued", "metadata_sync"),
    ]
    job_rows = conn.execute(
        """
        SELECT job_type, asset_type, status
        FROM arxiv_jobs
        ORDER BY job_type
        """
    ).fetchall()
    assert [(row["job_type"], row["asset_type"], row["status"]) for row in job_rows] == [
        ("download_pdf", "pdf", "queued"),
    ]


def test_paperstream_sync_enqueue_assets_source_tuple_queues_only_source_asset_jobs():
    """Direct metadata sync can queue only source follow-up work."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets=("source",),
    )

    assert result["ok"] is True
    assert result["jobs_queued"] == 1
    asset_rows = conn.execute(
        """
        SELECT asset_type, status, source
        FROM arxiv_assets
        WHERE asset_type IN ('pdf', 'source')
        ORDER BY asset_type
        """
    ).fetchall()
    assert [(row["asset_type"], row["status"], row["source"]) for row in asset_rows] == [
        ("source", "queued", "metadata_sync"),
    ]
    job_rows = conn.execute(
        """
        SELECT job_type, asset_type, status
        FROM arxiv_jobs
        ORDER BY job_type
        """
    ).fetchall()
    assert [(row["job_type"], row["asset_type"], row["status"]) for row in job_rows] == [
        ("download_source", "source", "queued"),
    ]


@pytest.mark.parametrize(
    ("asset_type", "ready_status", "job_type"),
    [
        ("pdf", "downloaded", "download_pdf"),
        ("pdf", "imported", "download_pdf"),
        ("pdf", "parsed", "download_pdf"),
        ("pdf", "indexed", "download_pdf"),
        ("source", "downloaded", "download_source"),
        ("source", "imported", "download_source"),
        ("source", "parsed", "download_source"),
        ("source", "indexed", "download_source"),
    ],
)
def test_paperstream_sync_enqueue_assets_preserves_ready_assets_without_download_jobs(
    asset_type,
    ready_status,
    job_type,
):
    """Sync-time enqueue skips ready PDF/source assets instead of re-queuing them."""
    from researchfs.paperstream import sync_paperstream_metadata, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    conn.execute(
        "UPDATE arxiv_papers SET oai_datestamp = '2026-07-01' WHERE arxiv_id = '2606.29148'"
    )
    ready_asset_key = upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        asset_type,
        status=ready_status,
        storage_path=f"/object-store/{asset_type}/2606.29148v2",
        checksum=f"sha256:{asset_type}-{ready_status}",
        byte_size=1234,
        source="paperstream_ready_fixture",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets=True,
    )

    assert result["ok"] is True
    assert result["jobs_queued"] == 1
    ready_asset = conn.execute(
        """
        SELECT status, source, storage_path, checksum, byte_size
        FROM arxiv_assets
        WHERE asset_key = ?
        """,
        (ready_asset_key,),
    ).fetchone()
    assert dict(ready_asset) == {
        "status": ready_status,
        "source": "paperstream_ready_fixture",
        "storage_path": f"/object-store/{asset_type}/2606.29148v2",
        "checksum": f"sha256:{asset_type}-{ready_status}",
        "byte_size": 1234,
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = ?",
        (job_type,),
    ).fetchone()[0] == 0

def test_paperstream_sync_enqueue_assets_existing_queued_asset_job_payload_is_enriched():
    """Existing non-ready asset rows still queue download jobs with replayable payloads."""
    from researchfs.paperstream import sync_paperstream_metadata, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    conn.execute(
        "UPDATE arxiv_papers SET oai_datestamp = '2026-07-01' WHERE arxiv_id = '2606.29148'"
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "source",
        status="failed",
        source="metadata_sync",
        error_message="previous transient failure",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets=("source",),
    )

    assert result["ok"] is True
    assert result["jobs_queued"] == 1
    job = conn.execute(
        "SELECT job_type, asset_type, payload_json FROM arxiv_jobs"
    ).fetchone()
    assert (job["job_type"], job["asset_type"]) == ("download_source", "source")
    expected_payload = {
        "source": "metadata_sync",
        "asset_key": "2606.29148:v2:source",
        "intended_source": "arxiv_source",
        "download_url": "https://arxiv.org/src/2606.29148",
    }
    assert json.loads(job["payload_json"]) == expected_payload
    event = conn.execute(
        """
        SELECT payload_json
        FROM arxiv_events
        WHERE event_type = 'job_queued'
        """
    ).fetchone()
    assert json.loads(event["payload_json"]) == expected_payload


def test_paperstream_sync_enqueue_assets_pdf_string_skips_downloaded_pdf_without_job():
    """PDF-only sync enqueue preserves a downloaded PDF and reports no queued work."""
    from researchfs.paperstream import sync_paperstream_metadata, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    conn.execute(
        "UPDATE arxiv_papers SET oai_datestamp = '2026-07-01' WHERE arxiv_id = '2606.29148'"
    )
    pdf_asset_key = upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path="/object-store/pdf/2606.29148v2.pdf",
        checksum="sha256:downloaded-pdf",
        byte_size=4321,
        source="paperstream_ready_fixture",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets="pdf",
    )

    assert result["ok"] is True
    assert result["jobs_queued"] == 0
    ready_pdf = conn.execute(
        """
        SELECT status, source, storage_path, checksum, byte_size
        FROM arxiv_assets
        WHERE asset_key = ?
        """,
        (pdf_asset_key,),
    ).fetchone()
    assert dict(ready_pdf) == {
        "status": "downloaded",
        "source": "paperstream_ready_fixture",
        "storage_path": "/object-store/pdf/2606.29148v2.pdf",
        "checksum": "sha256:downloaded-pdf",
        "byte_size": 4321,
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'download_pdf'"
    ).fetchone()[0] == 0


def test_paperstream_sync_enqueue_assets_skips_downloaded_pdf_but_queues_missing_source():
    """Selecting both downloadable assets skips a ready PDF while queuing missing source."""
    from researchfs.paperstream import sync_paperstream_metadata, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    conn.execute(
        "UPDATE arxiv_papers SET oai_datestamp = '2026-07-01' WHERE arxiv_id = '2606.29148'"
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path="/object-store/pdf/2606.29148v2.pdf",
        checksum="sha256:downloaded-pdf",
        byte_size=4321,
        source="paperstream_ready_fixture",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        enqueue_assets=("pdf", "source"),
    )

    assert result["ok"] is True
    assert result["jobs_queued"] == 1
    asset_rows = conn.execute(
        """
        SELECT asset_type, status, source, storage_path
        FROM arxiv_assets
        WHERE asset_type IN ('pdf', 'source')
        ORDER BY asset_type
        """
    ).fetchall()
    assert [(row["asset_type"], row["status"], row["source"], row["storage_path"]) for row in asset_rows] == [
        ("pdf", "downloaded", "paperstream_ready_fixture", "/object-store/pdf/2606.29148v2.pdf"),
        ("source", "queued", "metadata_sync", None),
    ]
    job_rows = conn.execute(
        """
        SELECT job_key, job_type, asset_type, status
        FROM arxiv_jobs
        ORDER BY job_type
        """
    ).fetchall()
    assert [(row["job_key"], row["job_type"], row["asset_type"], row["status"]) for row in job_rows] == [
        ("download_source:2606.29148:v2", "download_source", "source", "queued"),
    ]


def test_paperstream_sync_enqueue_assets_dry_run_counts_only_missing_download_jobs():
    """Dry-run planning counts missing source work without inflating for a ready PDF."""
    from researchfs.paperstream import sync_paperstream_metadata, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    conn.execute(
        "UPDATE arxiv_papers SET oai_datestamp = '2026-07-01' WHERE arxiv_id = '2606.29148'"
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path="/object-store/pdf/2606.29148v2.pdf",
        checksum="sha256:downloaded-pdf",
        byte_size=4321,
        source="paperstream_ready_fixture",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        dry_run=True,
        enqueue_assets=("pdf", "source"),
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["jobs_queued"] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs").fetchone()[0] == 0


def test_paperstream_sync_idempotent_rerun_persists_zero_inserted_counters():
    """Repeated metadata sync preserves update semantics while reporting no new inserts."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    first = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")
    second = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert first["records_updated"] == 0
    assert first["records_inserted"] == 1
    assert first["versions_inserted"] == 2
    assert second["ok"] is True
    assert second["records_seen"] == 2
    assert second["records_deleted"] == 1
    assert second["records_updated"] == 0
    assert second["records_inserted"] == 0
    assert second["versions_inserted"] == 0
    assert state["status"] == "succeeded"
    assert state["records_seen_last_run"] == 2
    assert state["records_updated_last_run"] == 0
    assert state["records_inserted_last_run"] == 0
    assert state["versions_inserted_last_run"] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2


def test_paperstream_sync_existing_metadata_change_counts_updated_not_inserted():
    """Changed existing paper metadata increments updates without insert counters."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")
    updated_page = _PAPERSTREAM_OAI_FINAL_PAGE.replace(
        "<responseDate>2026-07-04T12:00:00Z</responseDate>",
        "<responseDate>2026-07-05T12:00:00Z</responseDate>",
    ).replace(
        "<datestamp>2026-07-02</datestamp>",
        "<datestamp>2026-07-05</datestamp>",
        1,
    ).replace(
        "<title>Metadata Only PaperStream Core</title>",
        "<title>Metadata Only PaperStream Core Revised</title>",
    )

    result = sync_paperstream_metadata(conn, [updated_page], sync_name="default")
    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()

    assert result["records_inserted"] == 0
    assert result["records_updated"] == 1
    assert result["versions_inserted"] == 0
    assert state["records_inserted_last_run"] == 0
    assert state["records_updated_last_run"] == 1
    assert state["versions_inserted_last_run"] == 0


def test_paperstream_sync_dry_run_reports_inserted_counters_without_mutation():
    """Dry-run metadata sync computes inserted counters without writing state or events."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["records_seen"] == 2
    assert result["records_deleted"] == 1
    assert result["records_updated"] == 0
    assert result["records_inserted"] == 1
    assert result["versions_inserted"] == 2
    assert result["metadata_bloom_filter_loaded"] == 0
    assert result["metadata_bloom_filter_rebuilt"] == 0
    assert result["metadata_bloom_filter_refreshed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_sync_state").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_bloom_filters").fetchone()[0] == 0

def test_paperstream_sync_commits_successful_cursor_but_preserves_it_on_failure():
    """A failed metadata sync records failure without advancing the committed cursor."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    first = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")
    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    committed_after_success = state["committed_from"]
    assert first["ok"] is True
    assert first["dry_run"] is False
    assert committed_after_success == "2026-07-04T12:00:00Z"
    assert state["last_response_date"] == "2026-07-04T12:00:00Z"
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'"
    ).fetchone()[0] == 1

    def fail_after_page():
        raise RuntimeError("fixture page unavailable")

    with pytest.raises(RuntimeError, match="fixture page unavailable"):
        sync_paperstream_metadata(
            conn,
            [_PAPERSTREAM_OAI_FINAL_PAGE, fail_after_page],
            sync_name="default",
        )

    failed_state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert failed_state["status"] == "failed"
    assert failed_state["committed_from"] == committed_after_success
    assert "fixture page unavailable" in failed_state["error_message"]


def test_paperstream_sync_malformed_oai_page_preserves_committed_cursor_after_success():
    """Malformed OAI pages fail observably without promoting or clobbering the cursor."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    first = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_FINAL_PAGE], sync_name="default")
    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    committed_after_success = state["committed_from"]
    last_response_after_success = state["last_response_date"]
    cursor_commits_after_success = conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'"
    ).fetchone()[0]

    assert first["ok"] is True
    assert committed_after_success == "2026-07-04T12:00:00Z"
    assert last_response_after_success == "2026-07-04T12:00:00Z"
    assert cursor_commits_after_success == 1

    with pytest.raises(ValueError, match="OAI-PMH XML"):
        sync_paperstream_metadata(
            conn,
            ["<OAI-PMH><ListRecords><record></ListRecords></OAI-PMH>"],
            sync_name="default",
        )

    failed_state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    failure_events = conn.execute(
        """
        SELECT payload_json
        FROM arxiv_events
        WHERE event_type = 'sync_failed'
        ORDER BY event_id
        """
    ).fetchall()
    assert failed_state["status"] == "failed"
    assert failed_state["committed_from"] == committed_after_success
    assert failed_state["last_response_date"] == last_response_after_success
    assert "OAI-PMH XML" in failed_state["error_message"]
    assert len(failure_events) == 1
    assert "OAI-PMH XML" in json.loads(failure_events[0]["payload_json"])["error"]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'"
    ).fetchone()[0] == cursor_commits_after_success


def test_paperstream_sync_malformed_second_page_rolls_back_page_only():
    """A parser failure rolls back the failed page while preserving prior committed pages."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    second_page = {
        "response_date": "2026-07-05T12:00:00Z",
        "records": [
            {
                "arxiv_id": "2606.29149",
                "latest_version": 1,
                "title": "Rolled Back Paper",
                "abstract": "This row should be undone with its failed page.",
                "authors": ["Rollback Author"],
                "categories": ["cs.AI"],
                "primary_category": "cs.AI",
                "abs_url": "https://arxiv.org/abs/2606.29149",
                "oai_datestamp": "2026-07-05",
                "raw_oai_xml": "<record><id>2606.29149</id></record>",
                "versions": [{"version": 1, "submitted_at": "Wed, 03 Jun 2026 00:00:00 GMT"}],
            },
            {"raw_xml": "<record><metadata><arXivRaw><id>2606.99999</id></arXivRaw>"},
        ],
    }

    with pytest.raises(ValueError, match="OAI record XML"):
        sync_paperstream_metadata(
            conn,
            [_PAPERSTREAM_OAI_PAGE, second_page],
            sync_name="default",
        )

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    event_types = _paperstream_event_types(conn)
    assert state["status"] == "failed"
    assert state["committed_from"] is None
    assert state["last_response_date"] is None
    assert state["pending_response_date"] == "2026-07-05T12:00:00Z"
    assert state["last_resumption_token"] is None
    assert "OAI record XML" in state["error_message"]
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers WHERE arxiv_id = '2606.29148'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions WHERE arxiv_id = '2606.29148'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers WHERE arxiv_id = '2606.29149'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions WHERE arxiv_id = '2606.29149'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets WHERE arxiv_id = '2606.29149'").fetchone()[0] == 0
    assert event_types.count("sync_page_committed") == 1
    assert event_types.count("sync_failed") == 1
    assert "cursor_committed" not in event_types


def test_paperstream_sync_dry_run_malformed_page_does_not_mutate_state_or_events():
    """Dry-run parser failures may raise but must not write sync state or audit events."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    with pytest.raises(ValueError, match="OAI-PMH XML"):
        sync_paperstream_metadata(
            conn,
            ["<OAI-PMH><ListRecords><record></ListRecords></OAI-PMH>"],
            sync_name="default",
            dry_run=True,
        )

    assert conn.execute("SELECT COUNT(*) FROM arxiv_sync_state").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0

def test_paperstream_sync_partial_token_preserves_committed_cursor_and_pending_state():
    """A non-exhausted final page is a committed partial run, not a cursor promotion."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )

    result = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_PAGE], sync_name="default")

    assert result["ok"] is False
    assert result["partial"] is True
    assert result["dry_run"] is False
    assert result["committed_from"] == "2026-07-02T12:00:00Z"
    assert result["last_response_date"] == "2026-07-02T12:00:00Z"
    assert result["pages_seen"] == 1
    assert result["pages_committed"] == 1
    assert result["records_seen"] == 2
    assert result["records_updated"] == 0

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert state["status"] == "failed"
    assert state["committed_from"] == "2026-07-02T12:00:00Z"
    assert state["last_response_date"] == "2026-07-02T12:00:00Z"
    assert state["pending_response_date"] == "2026-07-03T12:00:00Z"
    assert state["last_resumption_token"] == "cursor-token-1"
    assert state["records_seen_last_run"] == 2
    assert state["records_updated_last_run"] == 0
    assert "resumption" in state["error_message"].lower()
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'sync_page_committed'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'").fetchone()[0] == 0


def test_paperstream_sync_exhausted_final_page_promotes_cursor_after_token_page():
    """A token page followed by an exhausted page succeeds and commits the final cursor."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_PAGE, _PAPERSTREAM_OAI_FINAL_PAGE],
        sync_name="default",
    )

    assert result["ok"] is True
    assert result.get("partial") is not True
    assert result["committed_from"] == "2026-07-04T12:00:00Z"
    assert result["last_response_date"] == "2026-07-04T12:00:00Z"
    assert result["pages_seen"] == 2
    assert result["pages_committed"] == 2
    assert result["records_seen"] == 4
    assert result["records_updated"] == 0

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert state["status"] == "succeeded"
    assert state["committed_from"] == "2026-07-04T12:00:00Z"
    assert state["last_response_date"] == "2026-07-04T12:00:00Z"
    assert state["pending_response_date"] is None
    assert state["last_resumption_token"] is None
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'").fetchone()[0] == 1


def test_paperstream_sync_max_records_cut_returns_partial_without_cursor_promotion():
    """A max-records cut reports partial failure while preserving visible pending progress."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )

    result = sync_paperstream_metadata(
        conn,
        [_PAPERSTREAM_OAI_PAGE],
        sync_name="default",
        max_records=1,
    )

    assert result["ok"] is False
    assert result["partial"] is True
    assert result["committed_from"] == "2026-07-02T12:00:00Z"
    assert result["last_response_date"] == "2026-07-02T12:00:00Z"
    assert result["pages_seen"] == 1
    assert result["pages_committed"] == 0
    assert result["records_seen"] == 1
    assert result["records_updated"] == 0

    state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
    assert state["status"] == "failed"
    assert state["committed_from"] == "2026-07-02T12:00:00Z"
    assert state["last_response_date"] == "2026-07-02T12:00:00Z"
    assert state["pending_response_date"] == "2026-07-03T12:00:00Z"
    assert state["last_resumption_token"] == "cursor-token-1"
    assert state["records_seen_last_run"] == 1
    assert state["records_updated_last_run"] == 0
    assert "max_records" in state["error_message"]
    assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'").fetchone()[0] == 0


def test_paperstream_sync_deleted_oai_record_logs_audit_without_mutating_existing_rows():
    """A deleted OAI marker is audited without deleting or rewriting local paper state."""
    from researchfs.paperstream import sync_paperstream_metadata, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(
        conn,
        "2606.29148",
        version=2,
        category="cs.AI",
        categories=["cs.AI", "cs.LG"],
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path="/paperstream/cache/2606.29148v2.pdf",
        checksum="sha256:fixture",
        byte_size=123,
        source="fixture",
    )
    before_paper = dict(
        conn.execute(
            """
            SELECT arxiv_id, latest_version, title, abstract, authors_json,
                   primary_category, categories_json, abs_url, oai_datestamp, raw_oai_xml
            FROM arxiv_papers
            WHERE arxiv_id = '2606.29148'
            """
        ).fetchone()
    )
    before_versions = [
        dict(row)
        for row in conn.execute(
            """
            SELECT arxiv_id, version, submitted_at, announced_at, title, abstract,
                   authors_json, categories_json, raw_version_metadata_json
            FROM arxiv_versions
            WHERE arxiv_id = '2606.29148'
            ORDER BY version
            """
        ).fetchall()
    ]
    before_assets = [
        dict(row)
        for row in conn.execute(
            """
            SELECT asset_key, arxiv_id, version, asset_type, status, storage_path,
                   checksum, byte_size, source, error_message
            FROM arxiv_assets
            WHERE arxiv_id = '2606.29148'
            ORDER BY asset_key
            """
        ).fetchall()
    ]

    result = sync_paperstream_metadata(conn, [_PAPERSTREAM_DELETED_OAI_PAGE])

    assert result["ok"] is True
    assert result["records_seen"] == 1
    assert result["records_deleted"] == 1
    assert result["records_updated"] == 0
    assert result["assets_seen"] == 0
    assert dict(
        conn.execute(
            """
            SELECT arxiv_id, latest_version, title, abstract, authors_json,
                   primary_category, categories_json, abs_url, oai_datestamp, raw_oai_xml
            FROM arxiv_papers
            WHERE arxiv_id = '2606.29148'
            """
        ).fetchone()
    ) == before_paper
    assert [
        dict(row)
        for row in conn.execute(
            """
            SELECT arxiv_id, version, submitted_at, announced_at, title, abstract,
                   authors_json, categories_json, raw_version_metadata_json
            FROM arxiv_versions
            WHERE arxiv_id = '2606.29148'
            ORDER BY version
            """
        ).fetchall()
    ] == before_versions
    assert [
        dict(row)
        for row in conn.execute(
            """
            SELECT asset_key, arxiv_id, version, asset_type, status, storage_path,
                   checksum, byte_size, source, error_message
            FROM arxiv_assets
            WHERE arxiv_id = '2606.29148'
            ORDER BY asset_key
            """
        ).fetchall()
    ] == before_assets

    events = conn.execute(
        """
        SELECT event_type, arxiv_id, payload_json
        FROM arxiv_events
        WHERE event_type = 'paper_deleted'
        ORDER BY event_id
        """
    ).fetchall()
    assert len(events) == 1
    assert events[0]["arxiv_id"] == "2606.29148"
    payload = json.loads(events[0]["payload_json"])
    assert payload == {
        "identifier": "oai:arXiv.org:2606.29148",
        "arxiv_id": "2606.29148",
        "oai_datestamp": "2026-07-05",
        "set_specs": ["cs", "cs.AI"],
    }
    assert "raw_xml" not in payload
    assert "raw_oai_xml" not in payload


def test_paperstream_sync_rolls_back_deleted_oai_event_when_page_fails():
    """A page-level rollback removes deleted-record audit events from the failed page."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()
    failed_page = {
        "response_date": "2026-07-05T12:00:00Z",
        "records": [
            {
                "deleted": True,
                "identifier": "oai:arXiv.org:2606.29148",
                "datestamp": "2026-07-05",
                "set_specs": ["cs"],
            },
            {
                "raw_xml": (
                    "<record><header><identifier>oai:arXiv.org:not-valid</identifier>"
                    "<datestamp>2026-07-05</datestamp></header><metadata></metadata></record>"
                )
            },
        ],
    }

    with pytest.raises(ValueError, match="invalid arxiv_id"):
        sync_paperstream_metadata(conn, [failed_page])

    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'paper_deleted'"
    ).fetchone()[0] == 0


def test_paperstream_sync_dry_run_deleted_oai_record_logs_no_event():
    """Dry-run metadata sync reports deleted records without writing events or rows."""
    from researchfs.paperstream import sync_paperstream_metadata

    conn = _paperstream_conn()

    result = sync_paperstream_metadata(conn, [_PAPERSTREAM_OAI_PAGE], dry_run=True)

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["records_seen"] == 2
    assert result["records_deleted"] == 1
    assert _paperstream_row_count(conn, "arxiv_sync_state") == 0
    assert _paperstream_row_count(conn, "arxiv_papers") == 0
    assert _paperstream_row_count(conn, "arxiv_versions") == 0
    assert _paperstream_row_count(conn, "arxiv_assets") == 0
    assert _paperstream_row_count(conn, "arxiv_events") == 0



def test_paperstream_s3_manifest_import_writes_bins_events_and_idempotent_updates():
    """S3 manifest imports write bin rows, suppress unchanged events, and update deltas only."""
    from researchfs.paperstream import import_arxiv_s3_manifest, s3_bin_key

    conn = _paperstream_conn()
    filename = "pdf/arXiv_pdf_2601_001.tar"
    expected_key = s3_bin_key("pdf", filename)
    record = {
        "asset_type": "pdf",
        "filename": filename,
        "size_bytes": 1024,
        "checksum": "sha256:initial",
        "first_item": "2601.00001",
        "last_item": "2601.99999",
        "item_count": 42,
        "status": "seen",
        "storage_path": "s3://arxiv/pdf/arXiv_pdf_2601_001.tar",
    }

    first = import_arxiv_s3_manifest(conn, [record])
    unchanged = import_arxiv_s3_manifest(conn, [dict(record)])
    changed_record = {
        "asset_type": "pdf",
        "filename": filename,
        "size_bytes": 2048,
        "status": "imported",
    }
    changed = import_arxiv_s3_manifest(conn, [changed_record])
    partial_refresh = import_arxiv_s3_manifest(
        conn,
        [
            {
                "asset_type": "pdf",
                "filename": filename,
                "size_bytes": 4096,
            }
        ],
    )

    assert first["parser_version"] == 1
    assert first["inserted"] == 1
    assert first["updated"] == 0
    assert first["unchanged"] == 0
    assert first["rows"][0]["bin_key"] == expected_key
    assert unchanged["inserted"] == 0
    assert unchanged["updated"] == 0
    assert unchanged["unchanged"] == 1
    assert changed["inserted"] == 0
    assert changed["updated"] == 1
    assert changed["unchanged"] == 0
    assert partial_refresh["inserted"] == 0
    assert partial_refresh["updated"] == 1
    assert partial_refresh["unchanged"] == 0
    assert "status" not in partial_refresh["rows"][0]["changed_fields"]

    row = conn.execute("SELECT * FROM arxiv_s3_bins WHERE bin_key = ?", (expected_key,)).fetchone()
    assert row["asset_type"] == "pdf"
    assert row["filename"] == filename
    assert row["size_bytes"] == 4096
    assert row["status"] == "imported"
    assert row["checksum"] == "sha256:initial"
    assert row["first_item"] == "2601.00001"
    assert row["last_item"] == "2601.99999"
    assert row["item_count"] == 42
    assert row["storage_path"] == "s3://arxiv/pdf/arXiv_pdf_2601_001.tar"

    event_types = [
        row["event_type"]
        for row in conn.execute("SELECT event_type FROM arxiv_events ORDER BY event_id")
    ]
    assert event_types == ["s3_bin_seen", "s3_bin_updated", "s3_bin_updated"]


def test_paperstream_s3_manifest_dry_run_reports_changes_without_mutation():
    """Dry-run classifies inserts/updates but leaves bin rows and events unchanged."""
    from researchfs.paperstream import import_arxiv_s3_manifest, s3_bin_key

    conn = _paperstream_conn()
    existing_filename = "src/arXiv_src_2601_001.tar"
    import_arxiv_s3_manifest(
        conn,
        [
            {
                "asset_type": "source",
                "filename": existing_filename,
                "size_bytes": 4096,
                "checksum": "sha256:before",
                "status": "seen",
            }
        ],
    )
    existing_key = s3_bin_key("source", existing_filename)
    before_events = _paperstream_row_count(conn, "arxiv_events")

    result = import_arxiv_s3_manifest(
        conn,
        [
            {
                "asset_type": "source",
                "filename": existing_filename,
                "size_bytes": 8192,
                "status": "downloaded",
            },
            {
                "asset_type": "pdf",
                "filename": "pdf/arXiv_pdf_2601_002.tar",
                "size_bytes": 512,
            },
        ],
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["parser_version"] == 1
    assert result["inserted"] == 1
    assert result["updated"] == 1
    assert result["unchanged"] == 0
    assert {row["action"] for row in result["rows"]} == {"inserted", "updated"}
    row = conn.execute("SELECT * FROM arxiv_s3_bins WHERE bin_key = ?", (existing_key,)).fetchone()
    assert row["size_bytes"] == 4096
    assert row["checksum"] == "sha256:before"
    assert row["status"] == "seen"
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM arxiv_s3_bins WHERE filename = ?",
            ("pdf/arXiv_pdf_2601_002.tar",),
        ).fetchone()[0]
        == 0
    )
    assert _paperstream_row_count(conn, "arxiv_events") == before_events


@pytest.mark.parametrize(
    "bad_record, message",
    [
        ({"asset_type": "metadata", "filename": "pdf/arXiv_pdf_2601_001.tar"}, "asset_type"),
        ({"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_001.tar", "status": "queued"}, "status"),
        ({"asset_type": "pdf", "filename": "/tmp/arXiv_pdf_2601_001.tar"}, "filename"),
        ({"asset_type": "pdf", "filename": "../arXiv_pdf_2601_001.tar"}, "filename"),
        ({"asset_type": "pdf", "filename": r"pdf\arXiv_pdf_2601_001.tar"}, "filename"),
        ({"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_001.tar", "size_bytes": -1}, "size_bytes"),
        ({"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_001.tar", "size_bytes": True}, "size_bytes"),
    ],
)
def test_paperstream_s3_manifest_invalid_records_fail_before_mutation_and_log_event(bad_record, message):
    """Invalid S3 manifests reject the whole batch and audit direct API failures."""
    from researchfs.paperstream import import_arxiv_s3_manifest

    conn = _paperstream_conn()
    valid_record = {
        "asset_type": "pdf",
        "filename": "pdf/arXiv_pdf_2601_000.tar",
        "size_bytes": 1,
    }
    bad_record = dict(bad_record)
    bad_record["raw_manifest_body"] = "RAW_MANIFEST_BODY_SHOULD_NOT_APPEAR"

    with pytest.raises((TypeError, ValueError), match=message):
        import_arxiv_s3_manifest(conn, [valid_record, bad_record])

    assert _paperstream_row_count(conn, "arxiv_s3_bins") == 0
    events = conn.execute(
        "SELECT event_type, payload_json FROM arxiv_events ORDER BY event_id"
    ).fetchall()
    assert [row["event_type"] for row in events] == ["s3_manifest_import_failed"]
    payload = json.loads(events[0]["payload_json"])
    assert message in payload["error"]
    assert payload["parser_version"] == 1
    assert payload["record_index"] == 2
    assert "RAW_MANIFEST_BODY_SHOULD_NOT_APPEAR" not in events[0]["payload_json"]


def test_paperstream_s3_manifest_dry_run_invalid_records_do_not_mutate_or_log():
    """Dry-run validation failures leave bins and audit events untouched."""
    from researchfs.paperstream import import_arxiv_s3_manifest

    conn = _paperstream_conn()
    with pytest.raises(ValueError, match="status"):
        import_arxiv_s3_manifest(
            conn,
            [
                {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_000.tar", "size_bytes": 1},
                {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_001.tar", "status": "queued"},
            ],
            dry_run=True,
        )

    assert _paperstream_row_count(conn, "arxiv_s3_bins") == 0
    assert _paperstream_row_count(conn, "arxiv_events") == 0


def test_paperstream_s3_member_import_requires_existing_bin_before_mutation():
    """Bin-member imports reject unknown bins before writing members, assets, or events."""
    from researchfs.paperstream import upsert_arxiv_bin_member

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)

    with pytest.raises(ValueError, match="bin"):
        upsert_arxiv_bin_member(
            conn,
            {
                "bin_key": "pdf:missing.tar",
                "arxiv_id": "2606.29148",
                "version": 2,
                "asset_type": "pdf",
                "storage_path": "cache/2606.29148v2.pdf",
                "checksum": "sha256:member",
            },
        )

    assert _paperstream_row_count(conn, "arxiv_bin_members") == 0
    assert _paperstream_row_count(conn, "arxiv_assets") == 0
    assert _paperstream_event_types(conn) == []


def test_paperstream_s3_member_import_writes_assets_events_and_idempotent_updates():
    """Storage-backed members upsert member/asset rows and emit only real change events."""
    from researchfs.paperstream import asset_key, import_arxiv_bin_members

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    bin_key = _insert_paperstream_s3_bin(
        conn,
        filename="pdf/arXiv_pdf_2606_001.tar",
        checksum="sha256:bin",
    )
    record = {
        "bin_key": bin_key,
        "arxiv_id": "2606.29148",
        "version": 2,
        "asset_type": "pdf",
        "storage_path": "cache/pdf/2606.29148v2.pdf",
        "checksum": "sha256:member-v1",
    }

    first = import_arxiv_bin_members(conn, [record])
    unchanged = import_arxiv_bin_members(conn, [dict(record)])
    changed_record = {
        **record,
        "storage_path": "cache/pdf/2606.29148v2-repacked.pdf",
        "checksum": "sha256:member-v2",
    }
    changed = import_arxiv_bin_members(conn, [changed_record])
    partial_refresh = import_arxiv_bin_members(
        conn,
        [
            {
                "bin_key": bin_key,
                "arxiv_id": "2606.29148",
                "version": 2,
                "asset_type": "pdf",
            }
        ],
    )

    assert first["inserted"] == 1
    assert first["updated"] == 0
    assert first["unchanged"] == 0
    assert unchanged["inserted"] == 0
    assert unchanged["updated"] == 0
    assert unchanged["unchanged"] == 1
    assert changed["inserted"] == 0
    assert changed["updated"] == 1
    assert changed["unchanged"] == 0
    assert set(changed["rows"][0]["changed_fields"]) == {"storage_path", "checksum"}
    assert partial_refresh["inserted"] == 0
    assert partial_refresh["updated"] == 0
    assert partial_refresh["unchanged"] == 1
    assert partial_refresh["rows"][0]["changed_fields"] == []

    member = conn.execute(
        """
        SELECT * FROM arxiv_bin_members
        WHERE bin_key = ? AND arxiv_id = ? AND version = ? AND asset_type = ?
        """,
        (bin_key, "2606.29148", 2, "pdf"),
    ).fetchone()
    assert member["storage_path"] == "cache/pdf/2606.29148v2-repacked.pdf"
    assert member["checksum"] == "sha256:member-v2"

    asset = conn.execute(
        "SELECT * FROM arxiv_assets WHERE asset_key = ?",
        (asset_key("2606.29148", 2, "pdf"),),
    ).fetchone()
    assert asset["status"] == "imported"
    assert asset["source"] == "s3_bin_member"
    assert asset["storage_path"] == "cache/pdf/2606.29148v2-repacked.pdf"
    assert asset["checksum"] == "sha256:member-v2"

    assert _paperstream_event_types(conn) == [
        "s3_bin_seen",
        "s3_bin_member_seen",
        "asset_imported",
        "s3_bin_member_updated",
        "asset_imported",
    ]


def test_paperstream_s3_member_without_storage_path_does_not_create_imported_asset():
    """A member row without a local path is not treated as an imported asset."""
    from researchfs.paperstream import import_arxiv_bin_members

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="source",
        filename="src/arXiv_src_2606_001.tar",
    )

    result = import_arxiv_bin_members(
        conn,
        [
            {
                "bin_key": bin_key,
                "arxiv_id": "2606.29148",
                "version": 1,
                "asset_type": "source",
                "checksum": "sha256:source-member",
            }
        ],
    )

    assert result["inserted"] == 1
    assert _paperstream_row_count(conn, "arxiv_bin_members") == 1
    assert _paperstream_row_count(conn, "arxiv_assets") == 0
    assert _paperstream_event_types(conn) == ["s3_bin_seen", "s3_bin_member_seen"]


def test_paperstream_s3_member_dry_run_reports_changes_without_mutation():
    """Dry-run member import classifies inserts/updates without changing rows, assets, or events."""
    from researchfs.paperstream import import_arxiv_bin_members

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    bin_key = _insert_paperstream_s3_bin(conn)
    import_arxiv_bin_members(
        conn,
        [
            {
                "bin_key": bin_key,
                "arxiv_id": "2606.29148",
                "version": 1,
                "asset_type": "pdf",
                "storage_path": "cache/pdf/2606.29148v1.pdf",
                "checksum": "sha256:before",
            }
        ],
    )
    before_events = _paperstream_event_types(conn)

    result = import_arxiv_bin_members(
        conn,
        [
            {
                "bin_key": bin_key,
                "arxiv_id": "2606.29148",
                "version": 1,
                "asset_type": "pdf",
                "storage_path": "cache/pdf/2606.29148v1-updated.pdf",
                "checksum": "sha256:after",
            },
            {
                "bin_key": bin_key,
                "arxiv_id": "2606.29149",
                "version": 1,
                "asset_type": "pdf",
                "storage_path": "cache/pdf/2606.29149v1.pdf",
                "checksum": "sha256:new",
            },
        ],
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["inserted"] == 1
    assert result["updated"] == 1
    assert result["unchanged"] == 0
    assert {row["action"] for row in result["rows"]} == {"inserted", "updated"}
    assert _paperstream_row_count(conn, "arxiv_bin_members") == 1
    stored_member = conn.execute(
        "SELECT storage_path, checksum FROM arxiv_bin_members WHERE arxiv_id = ?",
        ("2606.29148",),
    ).fetchone()
    assert stored_member["storage_path"] == "cache/pdf/2606.29148v1.pdf"
    assert stored_member["checksum"] == "sha256:before"
    assert _paperstream_row_count(conn, "arxiv_assets") == 1
    assert _paperstream_event_types(conn) == before_events


@pytest.mark.parametrize(
    "bad_record, message",
    [
        (
            {
                "bin_key": "pdf:missing.tar",
                "arxiv_id": "2606.29149",
                "version": 1,
                "asset_type": "pdf",
            },
            "bin",
        ),
        (
            {
                "bin_key": "BIN_KEY",
                "arxiv_id": "../2606.29149",
                "version": 1,
                "asset_type": "pdf",
            },
            "arxiv_id",
        ),
        (
            {
                "bin_key": "BIN_KEY",
                "arxiv_id": "2606.29149",
                "version": 0,
                "asset_type": "pdf",
            },
            "version",
        ),
        (
            {
                "bin_key": "BIN_KEY",
                "arxiv_id": "2606.29149",
                "version": 1,
                "asset_type": "metadata",
            },
            "asset_type",
        ),
        (
            {
                "bin_key": "BIN_KEY",
                "arxiv_id": "2606.29149",
                "version": 1,
                "asset_type": "pdf",
                "checksum": "",
            },
            "checksum",
        ),
    ],
)
def test_paperstream_s3_member_invalid_records_fail_before_mutation(bad_record, message):
    """Invalid member batches leave earlier valid members, imported assets, and events untouched."""
    from researchfs.paperstream import import_arxiv_bin_members

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    bin_key = _insert_paperstream_s3_bin(conn)
    valid_record = {
        "bin_key": bin_key,
        "arxiv_id": "2606.29148",
        "version": 1,
        "asset_type": "pdf",
        "storage_path": "cache/pdf/2606.29148v1.pdf",
        "checksum": "sha256:valid",
    }
    if bad_record.get("bin_key") == "BIN_KEY":
        bad_record = {**bad_record, "bin_key": bin_key}
    before_events = _paperstream_event_types(conn)

    with pytest.raises((TypeError, ValueError), match=message):
        import_arxiv_bin_members(conn, [valid_record, bad_record])

    assert _paperstream_row_count(conn, "arxiv_bin_members") == 0
    assert _paperstream_row_count(conn, "arxiv_assets") == 0
    assert _paperstream_event_types(conn) == before_events


def test_paperstream_s3_member_cli_accepts_json_array_object_jsonl_and_bin_key(tmp_path):
    """The member CLI accepts array, object/default-bin, and JSONL fixture shapes."""
    from researchfs import db
    from researchfs.paperstream import import_arxiv_s3_manifest, s3_bin_key

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    _insert_paperstream_paper(conn, "2606.29150", version=1)
    _insert_paperstream_paper(conn, "2606.29151", version=1)
    bins = [
        {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2606_001.tar"},
        {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2606_002.tar"},
        {"asset_type": "source", "filename": "src/arXiv_src_2606_001.tar"},
    ]
    import_arxiv_s3_manifest(conn, bins)
    array_bin_key = s3_bin_key("pdf", "pdf/arXiv_pdf_2606_001.tar")
    object_bin_key = s3_bin_key("pdf", "pdf/arXiv_pdf_2606_002.tar")
    jsonl_bin_key = s3_bin_key("source", "src/arXiv_src_2606_001.tar")
    conn.commit()
    conn.close()
    fixtures = [
        (
            "members-array.json",
            [
                {
                    "bin_key": array_bin_key,
                    "arxiv_id": "2606.29148",
                    "version": 1,
                    "asset_type": "pdf",
                }
            ],
            [],
            1,
        ),
        (
            "members-object.json",
            {
                "bin_key": object_bin_key,
                "members": [
                    {"arxiv_id": "2606.29149", "version": 1, "asset_type": "pdf"},
                ],
            },
            [],
            1,
        ),
        (
            "members.jsonl",
            [
                {"arxiv_id": "2606.29150", "version": 1, "asset_type": "source"},
                {"arxiv_id": "2606.29151", "version": 1, "asset_type": "source"},
            ],
            ["--bin-key", jsonl_bin_key],
            2,
        ),
    ]
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    for fixture_name, payload, extra_args, expected_inserted in fixtures:
        fixture_path = tmp_path / fixture_name
        if fixture_name.endswith(".jsonl"):
            fixture_path.write_text(
                "\n".join(json.dumps(record) for record in payload) + "\n",
                encoding="utf-8",
            )
        else:
            fixture_path.write_text(json.dumps(payload), encoding="utf-8")
        imported = subprocess.run(
            [
                sys.executable,
                "-m",
                "researchfs.paperstream",
                "--db-path",
                str(db_path),
                "import-s3-members",
                "--input",
                str(fixture_path),
                *extra_args,
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert imported.returncode == 0, imported.stderr
        result = json.loads(imported.stdout)
        assert result["inserted"] == expected_inserted
        assert result["updated"] == 0
        assert result["unchanged"] == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT bin_key, arxiv_id, version, asset_type
            FROM arxiv_bin_members
            ORDER BY arxiv_id
            """
        ).fetchall()
    finally:
        conn.close()
    assert [(row["bin_key"], row["arxiv_id"], row["version"], row["asset_type"]) for row in rows] == [
        (array_bin_key, "2606.29148", 1, "pdf"),
        (object_bin_key, "2606.29149", 1, "pdf"),
        (jsonl_bin_key, "2606.29150", 1, "source"),
        (jsonl_bin_key, "2606.29151", 1, "source"),
    ]


def test_paperstream_s3_member_status_counts_by_asset_type():
    """PaperStream status includes arXiv bin-member totals grouped by type."""
    from researchfs.paperstream import get_paperstream_status, import_arxiv_bin_members

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    pdf_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_001.tar",
    )
    source_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="source",
        filename="src/arXiv_src_2606_001.tar",
    )
    import_arxiv_bin_members(
        conn,
        [
            {"bin_key": pdf_bin_key, "arxiv_id": "2606.29148", "version": 1, "asset_type": "pdf"},
            {"bin_key": pdf_bin_key, "arxiv_id": "2606.29149", "version": 1, "asset_type": "pdf"},
            {
                "bin_key": source_bin_key,
                "arxiv_id": "2606.29148",
                "version": 1,
                "asset_type": "source",
            },
        ],
    )

    status = get_paperstream_status(conn)

    assert status["arxiv_bin_members"]["total"] == 3
    assert status["arxiv_bin_members"]["by_type"] == {"pdf": 2, "source": 1}


def test_paperstream_extract_s3_bin_job_imports_matching_assets_and_queues_parse(tmp_path):
    """Downloaded S3 bins extract only same-type members into deterministic assets."""
    from researchfs.paperstream import (
        arxiv_storage_path,
        s3_bin_storage_path,
        work_next_arxiv_s3_bin_extract_job,
    )

    object_root = tmp_path / "objects"
    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    pdf_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_001.tar",
    )
    source_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="source",
        filename="src/arXiv_src_2606_001.tar",
    )
    pdf_bytes = b"%PDF-1.4\npaper 2606.29148v2\n"
    source_bytes = b"source tarball bytes for 2606.29149v1"
    pdf_bin_path = _write_paperstream_tar(
        s3_bin_storage_path(object_root, "pdf", "pdf/arXiv_pdf_2606_001.tar"),
        {
            "2606.29148v2.pdf": pdf_bytes,
            "src/2606.29149v1.tar.gz": b"wrong type for a pdf bin",
        },
    )
    source_bin_path = _write_paperstream_tar(
        s3_bin_storage_path(object_root, "source", "src/arXiv_src_2606_001.tar"),
        {
            "2606.29148v2.pdf": b"wrong type for a source bin",
            "src/2606.29149v1.tar.gz": source_bytes,
        },
    )
    _mark_paperstream_s3_bin_downloaded(conn, pdf_bin_key, pdf_bin_path)
    _mark_paperstream_s3_bin_downloaded(conn, source_bin_key, source_bin_path)
    pdf_job_key = _enqueue_paperstream_extract_s3_bin_job(conn, pdf_bin_key)
    source_job_key = _enqueue_paperstream_extract_s3_bin_job(conn, source_bin_key)

    pdf_result = work_next_arxiv_s3_bin_extract_job(
        conn,
        worker_id="extract-pdf-worker",
        object_root=object_root,
        now="2026-07-04T00:00:01Z",
    )
    source_result = work_next_arxiv_s3_bin_extract_job(
        conn,
        worker_id="extract-source-worker",
        object_root=object_root,
        now="2026-07-04T00:00:02Z",
    )

    expected_pdf_path = Path(arxiv_storage_path(object_root, "2606.29148", 2, "pdf"))
    expected_source_path = Path(arxiv_storage_path(object_root, "2606.29149", 1, "source"))
    assert pdf_result["worked"] is True
    assert pdf_result["ok"] is True
    assert source_result["worked"] is True
    assert source_result["ok"] is True
    assert expected_pdf_path.read_bytes() == pdf_bytes
    assert expected_source_path.read_bytes() == source_bytes

    members = conn.execute(
        """
        SELECT arxiv_id, version, asset_type, storage_path, checksum
        FROM arxiv_bin_members
        ORDER BY asset_type, arxiv_id
        """
    ).fetchall()
    assert [dict(row) for row in members] == [
        {
            "arxiv_id": "2606.29148",
            "version": 2,
            "asset_type": "pdf",
            "storage_path": str(expected_pdf_path),
            "checksum": _sha256_text(pdf_bytes),
        },
        {
            "arxiv_id": "2606.29149",
            "version": 1,
            "asset_type": "source",
            "storage_path": str(expected_source_path),
            "checksum": _sha256_text(source_bytes),
        },
    ]
    assets = conn.execute(
        """
        SELECT asset_key, status, storage_path, checksum, byte_size, source, error_message
        FROM arxiv_assets
        WHERE asset_type IN ('pdf', 'source')
        ORDER BY asset_type, arxiv_id
        """
    ).fetchall()
    assert [dict(row) for row in assets] == [
        {
            "asset_key": "2606.29148:v2:pdf",
            "status": "imported",
            "storage_path": str(expected_pdf_path),
            "checksum": _sha256_text(pdf_bytes),
            "byte_size": len(pdf_bytes),
            "source": "s3_bin_member",
            "error_message": None,
        },
        {
            "asset_key": "2606.29149:v1:source",
            "status": "imported",
            "storage_path": str(expected_source_path),
            "checksum": _sha256_text(source_bytes),
            "byte_size": len(source_bytes),
            "source": "s3_bin_member",
            "error_message": None,
        },
    ]
    parse_jobs = conn.execute(
        """
        SELECT job_key, job_type, arxiv_id, version, asset_type, status
        FROM arxiv_jobs
        WHERE job_type LIKE 'parse_%'
        ORDER BY job_type
        """
    ).fetchall()
    assert [dict(row) for row in parse_jobs] == [
        {
            "job_key": "parse_pdf:2606.29148:v2",
            "job_type": "parse_pdf",
            "arxiv_id": "2606.29148",
            "version": 2,
            "asset_type": "pdf",
            "status": "queued",
        },
        {
            "job_key": "parse_source:2606.29149:v1",
            "job_type": "parse_source",
            "arxiv_id": "2606.29149",
            "version": 1,
            "asset_type": "source",
            "status": "queued",
        },
    ]
    extract_jobs = conn.execute(
        """
        SELECT job_key, status, locked_by
        FROM arxiv_jobs
        WHERE job_type = 'extract_s3_bin'
        ORDER BY job_key
        """
    ).fetchall()
    assert [dict(row) for row in extract_jobs] == [
        {"job_key": pdf_job_key, "status": "succeeded", "locked_by": None},
        {"job_key": source_job_key, "status": "succeeded", "locked_by": None},
    ]
    bins = conn.execute(
        """
        SELECT bin_key, status, imported_at
        FROM arxiv_s3_bins
        WHERE bin_key IN (?, ?)
        ORDER BY asset_type
        """,
        (pdf_bin_key, source_bin_key),
    ).fetchall()
    assert [(row["bin_key"], row["status"]) for row in bins] == [
        (pdf_bin_key, "imported"),
        (source_bin_key, "imported"),
    ]
    assert all(row["imported_at"] for row in bins)


def test_paperstream_extract_s3_bin_records_missing_metadata_members_without_assets(tmp_path):
    """Recognized members missing metadata are tracked without asset or parse rows."""
    from researchfs.paperstream import s3_bin_storage_path, work_next_arxiv_s3_bin_extract_job

    object_root = tmp_path / "objects"
    conn = _paperstream_conn()
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_002.tar",
    )
    bin_path = _write_paperstream_tar(
        s3_bin_storage_path(object_root, "pdf", "pdf/arXiv_pdf_2606_002.tar"),
        {"2606.29150v1.pdf": b"%PDF missing metadata\n"},
    )
    _mark_paperstream_s3_bin_downloaded(conn, bin_key, bin_path)
    _enqueue_paperstream_extract_s3_bin_job(conn, bin_key)

    result = work_next_arxiv_s3_bin_extract_job(
        conn,
        worker_id="extract-missing-metadata-worker",
        object_root=object_root,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    member = conn.execute(
        "SELECT arxiv_id, version, asset_type FROM arxiv_bin_members"
    ).fetchone()
    assert dict(member) == {"arxiv_id": "2606.29150", "version": 1, "asset_type": "pdf"}
    assert _paperstream_row_count(conn, "arxiv_assets") == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type LIKE 'parse_%'"
    ).fetchone()[0] == 0


def test_paperstream_extract_s3_bin_rejects_path_traversal_members(tmp_path):
    """Unsafe tar member names must not escape object_root during extraction."""
    from researchfs.paperstream import (
        arxiv_storage_path,
        s3_bin_storage_path,
        work_next_arxiv_s3_bin_extract_job,
    )

    object_root = tmp_path / "sandbox" / "objects"
    escaped_relative = tmp_path / "evil.pdf"
    escaped_absolute = tmp_path / "absolute-evil.pdf"
    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_003.tar",
    )
    safe_bytes = b"%PDF safe member\n"
    bin_path = _write_paperstream_tar(
        s3_bin_storage_path(object_root, "pdf", "pdf/arXiv_pdf_2606_003.tar"),
        {
            "2606.29148v1.pdf": safe_bytes,
            "../../evil.pdf": b"escaped relative",
            str(escaped_absolute): b"escaped absolute",
        },
    )
    _mark_paperstream_s3_bin_downloaded(conn, bin_key, bin_path)
    _enqueue_paperstream_extract_s3_bin_job(conn, bin_key)

    result = work_next_arxiv_s3_bin_extract_job(
        conn,
        worker_id="extract-traversal-worker",
        object_root=object_root,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert Path(arxiv_storage_path(object_root, "2606.29148", 1, "pdf")).read_bytes() == safe_bytes
    assert not escaped_relative.exists()
    assert not escaped_absolute.exists()
    members = conn.execute(
        "SELECT arxiv_id, version, asset_type FROM arxiv_bin_members ORDER BY arxiv_id"
    ).fetchall()
    assert [dict(row) for row in members] == [
        {"arxiv_id": "2606.29148", "version": 1, "asset_type": "pdf"}
    ]


def test_paperstream_extract_s3_bin_empty_tar_fails_without_parse_jobs(tmp_path):
    """An extract job with no recognized members fails durably and queues no parse work."""
    from researchfs.paperstream import s3_bin_storage_path, work_next_arxiv_s3_bin_extract_job

    object_root = tmp_path / "objects"
    conn = _paperstream_conn()
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_004.tar",
    )
    bin_path = _write_paperstream_tar(
        s3_bin_storage_path(object_root, "pdf", "pdf/arXiv_pdf_2606_004.tar"),
        {},
    )
    _mark_paperstream_s3_bin_downloaded(conn, bin_key, bin_path)
    job_key = _enqueue_paperstream_extract_s3_bin_job(conn, bin_key)

    result = work_next_arxiv_s3_bin_extract_job(
        conn,
        worker_id="extract-empty-worker",
        object_root=object_root,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is False
    job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (job_key,),
    ).fetchone()
    assert job["status"] == "failed"
    assert job["locked_by"] is None
    assert job["error_message"]
    bin_row = conn.execute(
        "SELECT status FROM arxiv_s3_bins WHERE bin_key = ?",
        (bin_key,),
    ).fetchone()
    assert bin_row["status"] == "failed"
    assert _paperstream_row_count(conn, "arxiv_bin_members") == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type LIKE 'parse_%'"
    ).fetchone()[0] == 0

def test_paperstream_s3_manifest_cli_accepts_json_array_object_and_jsonl(tmp_path):
    """The manifest CLI accepts every offline fixture shape and writes bin rows only."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    fixtures = [
        (
            "array.json",
            [
                {
                    "asset_type": "pdf",
                    "filename": "pdf/arXiv_pdf_2601_001.tar",
                    "size_bytes": 100,
                }
            ],
        ),
        (
            "object.json",
            {
                "bins": [
                    {
                        "asset_type": "source",
                        "filename": "src/arXiv_src_2601_001.tar",
                        "size_bytes": 200,
                    }
                ]
            },
        ),
        (
            "bins.jsonl",
            [
                {
                    "asset_type": "pdf",
                    "filename": "pdf/arXiv_pdf_2601_002.tar",
                    "size_bytes": 300,
                },
                {
                    "asset_type": "source",
                    "filename": "src/arXiv_src_2601_002.tar",
                    "size_bytes": 400,
                },
            ],
        ),
    ]
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    for fixture_name, payload in fixtures:
        fixture_path = tmp_path / fixture_name
        if fixture_name.endswith(".jsonl"):
            fixture_path.write_text(
                "\n".join(json.dumps(record) for record in payload) + "\n",
                encoding="utf-8",
            )
        else:
            fixture_path.write_text(json.dumps(payload), encoding="utf-8")
        imported = subprocess.run(
            [
                sys.executable,
                "-m",
                "researchfs.paperstream",
                "--db-path",
                str(db_path),
                "import-s3-manifest",
                "--input",
                str(fixture_path),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert imported.returncode == 0, imported.stderr
        result = json.loads(imported.stdout)
        assert result["inserted"] == len(payload if isinstance(payload, list) else payload["bins"])
        assert result["updated"] == 0
        assert result["unchanged"] == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT asset_type, filename, size_bytes, status FROM arxiv_s3_bins ORDER BY filename"
        ).fetchall()
        events = conn.execute(
            "SELECT event_type FROM arxiv_events ORDER BY event_id"
        ).fetchall()
    finally:
        conn.close()
    assert [(row["asset_type"], row["filename"], row["size_bytes"], row["status"]) for row in rows] == [
        ("pdf", "pdf/arXiv_pdf_2601_001.tar", 100, "seen"),
        ("pdf", "pdf/arXiv_pdf_2601_002.tar", 300, "seen"),
        ("source", "src/arXiv_src_2601_001.tar", 200, "seen"),
        ("source", "src/arXiv_src_2601_002.tar", 400, "seen"),
    ]
    assert [row["event_type"] for row in events] == ["s3_bin_seen"] * 4


@pytest.mark.parametrize(
    "fixture_name,payload,error_fragment",
    [
        (
            "bad-record.json",
            [
                {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_000.tar", "size_bytes": 1},
                {
                    "asset_type": "pdf",
                    "filename": "pdf/arXiv_pdf_2601_001.tar",
                    "status": "queued",
                    "raw_manifest_body": "RAW_MANIFEST_BODY_SHOULD_NOT_APPEAR",
                },
            ],
            "status",
        ),
        (
            "bad-jsonl.jsonl",
            '{"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_000.tar", "size_bytes": 1}\n{"asset_type":',
            "invalid JSONL at line 2",
        ),
        (
            "bad-shape.json",
            {"bins": "not-a-list"},
            "JSON array",
        ),
        (
            "missing-input.json",
            None,
            "No such file",
        ),
    ],
)
def test_paperstream_s3_manifest_cli_invalid_inputs_log_one_failure_event_without_bins(
    tmp_path, fixture_name, payload, error_fragment
):
    """CLI manifest failures roll back data rows and persist one durable failure event."""
    from researchfs import db

    db_path = tmp_path / f"{fixture_name}.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    fixture_path = tmp_path / fixture_name
    if payload is None:
        pass
    elif isinstance(payload, str):
        fixture_path.write_text(payload, encoding="utf-8")
    else:
        fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    imported = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "import-s3-manifest",
            "--input",
            str(fixture_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "RESEARCHFS_DB_PATH": str(db_path)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert imported.returncode != 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        assert _paperstream_row_count(conn, "arxiv_s3_bins") == 0
        events = conn.execute(
            "SELECT event_type, payload_json FROM arxiv_events ORDER BY event_id"
        ).fetchall()
    finally:
        conn.close()
    assert [row["event_type"] for row in events] == ["s3_manifest_import_failed"]
    payload_json = events[0]["payload_json"]
    event_payload = json.loads(payload_json)
    assert event_payload["parser_version"] == 1
    assert error_fragment in event_payload["error"]
    if fixture_name == "bad-record.json":
        assert event_payload["record_index"] == 2
    assert str(fixture_path) not in payload_json
    assert "RAW_MANIFEST_BODY_SHOULD_NOT_APPEAR" not in payload_json


def test_paperstream_s3_manifest_cli_dry_run_invalid_input_does_not_log_or_mutate(tmp_path):
    """Dry-run manifest CLI failures keep rollback/no-mutation semantics."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    fixture_path = tmp_path / "bad-dry-run.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_000.tar", "size_bytes": 1},
                {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_001.tar", "status": "queued"},
            ]
        ),
        encoding="utf-8",
    )

    imported = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "import-s3-manifest",
            "--input",
            str(fixture_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "RESEARCHFS_DB_PATH": str(db_path)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert imported.returncode != 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        assert _paperstream_row_count(conn, "arxiv_s3_bins") == 0
        assert _paperstream_row_count(conn, "arxiv_events") == 0
    finally:
        conn.close()


def test_paperstream_s3_status_counts_by_asset_type_and_status():
    """PaperStream status includes S3 bin totals grouped like other local state."""
    from researchfs.paperstream import get_paperstream_status, import_arxiv_s3_manifest

    conn = _paperstream_conn()
    import_arxiv_s3_manifest(
        conn,
        [
            {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_001.tar", "status": "seen"},
            {"asset_type": "pdf", "filename": "pdf/arXiv_pdf_2601_002.tar", "status": "imported"},
            {"asset_type": "source", "filename": "src/arXiv_src_2601_001.tar", "status": "seen"},
        ],
    )

    status = get_paperstream_status(conn)

    assert status["arxiv_s3_bins"]["total"] == 3
    assert status["arxiv_s3_bins"]["by_type"] == {"pdf": 2, "source": 1}
    assert status["arxiv_s3_bins"]["by_status"] == {"imported": 1, "seen": 2}

def test_paperstream_cli_status_rebuild_bloom_and_sync_metadata_dry_run_use_local_fixture(
    tmp_path,
):
    """Safe PaperStream CLI commands operate on local DB/fixture paths without network."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    conn.commit()
    conn.close()

    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    status = subprocess.run(
        [sys.executable, "-m", "researchfs.paperstream", "--db-path", str(db_path), "status"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "papers" in status.stdout

    rebuild = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "rebuild-bloom",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"filter_name"' in rebuild.stdout

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "dry" in dry_run.stdout.lower()

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM arxiv_sync_state").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 1
    finally:
        conn.close()


def test_paperstream_cli_rebuild_bloom_filter_alias_uses_metadata_version_keys(tmp_path):
    """PRD-shaped rebuild-bloom --filter persists version-key metadata Bloom state."""
    from researchfs import db
    from researchfs.paperstream import load_bloom_filter, upsert_arxiv_asset

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    asset_key_value = upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29148",
        version=2,
        asset_type="metadata",
        status="seen",
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    rebuild = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "rebuild-bloom",
            "--filter",
            "seen_metadata_versions",
            "--false-positive-rate",
            "0.02",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(rebuild.stdout)
    assert output["filter_name"] == "seen_metadata_versions"
    assert output["source_table"] == "arxiv_versions"
    assert output["source_column"] == "version"
    assert output["item_count"] == 2
    assert output["false_positive_rate"] == pytest.approx(0.02)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT source_table, source_column, item_count, false_positive_rate
            FROM arxiv_bloom_filters
            WHERE filter_name = 'seen_metadata_versions'
            """
        ).fetchone()
        assert dict(row) == {
            "source_table": "arxiv_versions",
            "source_column": "version",
            "item_count": 2,
            "false_positive_rate": pytest.approx(0.02),
        }
        loaded = load_bloom_filter(conn, "seen_metadata_versions")
        assert loaded is not None
        assert "2606.29148:v1" in loaded
        assert "2606.29148:v2" in loaded
        assert loaded.item_count == 2
        assert loaded.false_positive_rate == pytest.approx(0.02)
        assert conn.execute(
            """
            SELECT source_table, source_column
            FROM arxiv_bloom_filters
            WHERE filter_name = 'asset_key'
            """
        ).fetchone() is None
        assert asset_key_value == "2606.29148:v2:metadata"
    finally:
        conn.close()


def test_paperstream_cli_rebuild_bloom_filter_name_and_default_keep_asset_keys(tmp_path):
    """Existing default and --filter-name paths still rebuild from arxiv_assets.asset_key."""
    from researchfs import db
    from researchfs.paperstream import load_bloom_filter, upsert_arxiv_asset

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    asset_key_value = upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29148",
        version=1,
        asset_type="metadata",
        status="seen",
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    default_rebuild = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "rebuild-bloom",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    named_rebuild = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "rebuild-bloom",
            "--filter-name",
            "metadata_assets",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(default_rebuild.stdout)["source_table"] == "arxiv_assets"
    named_output = json.loads(named_rebuild.stdout)
    assert named_output["filter_name"] == "metadata_assets"
    assert named_output["source_table"] == "arxiv_assets"
    assert named_output["source_column"] == "asset_key"
    assert named_output["item_count"] == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = {
            row["filter_name"]: dict(row)
            for row in conn.execute(
                """
                SELECT filter_name, source_table, source_column, item_count
                FROM arxiv_bloom_filters
                WHERE filter_name IN ('asset_key', 'metadata_assets')
                """
            )
        }
        assert rows["asset_key"]["source_table"] == "arxiv_assets"
        assert rows["asset_key"]["source_column"] == "asset_key"
        assert rows["asset_key"]["item_count"] == 1
        assert rows["metadata_assets"]["source_table"] == "arxiv_assets"
        assert rows["metadata_assets"]["source_column"] == "asset_key"
        assert rows["metadata_assets"]["item_count"] == 1
        for filter_name in ("asset_key", "metadata_assets"):
            loaded = load_bloom_filter(conn, filter_name)
            assert loaded is not None
            assert asset_key_value in loaded
            assert loaded.item_count == 1
    finally:
        conn.close()


def test_paperstream_cli_sync_metadata_local_fixture_commits_partial_metadata_without_cursor_promotion(
    tmp_path,
):
    """Bounded fixture sync writes metadata and pending state without promoting cursor."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_sync_state(
        conn,
        committed_from="2026-07-02T12:00:00Z",
        last_response_date="2026-07-02T12:00:00Z",
    )
    conn.commit()
    conn.close()
    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    sync = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--max-pages",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(sync.stdout)
    assert result["ok"] is False
    assert result["partial"] is True
    assert result["dry_run"] is False
    assert result["committed_from"] == "2026-07-02T12:00:00Z"
    assert result["last_response_date"] == "2026-07-02T12:00:00Z"
    assert result["pages_seen"] == 1
    assert result["records_seen"] == 2
    assert result["records_updated"] == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        state = conn.execute("SELECT * FROM arxiv_sync_state WHERE sync_name = 'default'").fetchone()
        assert state["status"] == "failed"
        assert state["committed_from"] == "2026-07-02T12:00:00Z"
        assert state["last_response_date"] == "2026-07-02T12:00:00Z"
        assert state["pending_response_date"] == "2026-07-03T12:00:00Z"
        assert state["last_resumption_token"] == "cursor-token-1"
        assert conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM arxiv_versions").fetchone()[0] == 2
        asset_rows = conn.execute(
            "SELECT asset_type, status, source FROM arxiv_assets ORDER BY asset_type"
        ).fetchall()
        assert [(row["asset_type"], row["status"], row["source"]) for row in asset_rows] == [
            ("metadata", "seen", "oai")
        ]
        assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'cursor_committed'").fetchone()[0] == 0
    finally:
        conn.close()


def test_paperstream_cli_sync_metadata_non_dry_run_requires_max_pages(tmp_path):
    """Non-dry-run sync refuses unbounded harvests before mutating local state."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert "max-pages" in rejected.stderr
    assert _paperstream_db_counts(db_path) == {
        "arxiv_sync_state": 0,
        "arxiv_papers": 0,
        "arxiv_versions": 0,
        "arxiv_assets": 0,
        "arxiv_jobs": 0,
        "arxiv_events": 0,
    }


def test_paperstream_cli_sync_metadata_dry_run_does_not_mutate_local_db(tmp_path):
    """Fixture-backed CLI dry-run remains a parse-only operation."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_PAGE, encoding="utf-8")
    before_counts = _paperstream_db_counts(db_path)
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--dry-run",
            "--max-pages",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(dry_run.stdout)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["pages_seen"] == 1
    assert result["records_seen"] == 2
    assert _paperstream_db_counts(db_path) == before_counts

def test_paperstream_cli_sync_metadata_default_commits_metadata_only_without_object_files(
    tmp_path,
):
    """Default sync-metadata commits metadata rows without queued downloads or object files."""
    from researchfs import db
    from researchfs.paperstream import get_paperstream_status

    db_path = tmp_path / "paperstream.db"
    fixture_path = tmp_path / "oai-final-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_FINAL_PAGE, encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    allowed_temp_names = {
        fixture_path.name,
        db_path.name,
        f"{db_path.name}-journal",
        f"{db_path.name}-shm",
        f"{db_path.name}-wal",
    }
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--max-pages",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["jobs_queued"] == 0
    assert [path.name for path in tmp_path.iterdir() if path.name not in allowed_temp_names] == []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        asset_rows = conn.execute(
            """
            SELECT asset_type, status, source, storage_path, checksum, byte_size
            FROM arxiv_assets
            ORDER BY asset_type
            """
        ).fetchall()
        assert [
            (row["asset_type"], row["status"], row["source"])
            for row in asset_rows
        ] == [("metadata", "seen", "oai")]
        assert all(row["storage_path"] is None for row in asset_rows)
        assert all(row["checksum"] is None for row in asset_rows)
        assert all(row["byte_size"] is None for row in asset_rows)
        assert conn.execute(
            "SELECT COUNT(*) FROM arxiv_assets WHERE asset_type IN ('pdf', 'source')"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type IN ('download_pdf', 'download_source')"
        ).fetchone()[0] == 0
        state = conn.execute(
            "SELECT metadata_prefix, set_spec FROM arxiv_sync_state WHERE sync_name = 'default'"
        ).fetchone()
        assert dict(state) == {"metadata_prefix": "arXivRaw", "set_spec": None}
        status = get_paperstream_status(conn)
        assert status["metadata_prefix"] == "arXivRaw"
        assert status["set_spec"] is None
        assert status["sync"]["metadata_prefix"] == "arXivRaw"
        assert status["sync"]["set_spec"] is None
    finally:
        conn.close()

def test_paperstream_cli_sync_metadata_fixture_persists_normalized_scope(tmp_path):
    """CLI fixture sync records normalized non-default OAI scope without network."""
    from researchfs import db
    from researchfs.paperstream import get_paperstream_status

    db_path = tmp_path / "paperstream.db"
    fixture_path = tmp_path / "oai-final-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_FINAL_PAGE, encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--max-pages",
            "1",
            "--metadata-prefix",
            "  arXivCustom  ",
            "--set",
            "  cs.AI  ",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["metadata_prefix"] == "arXivCustom"
    assert result["set_spec"] == "cs.AI"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        state = conn.execute(
            "SELECT metadata_prefix, set_spec FROM arxiv_sync_state WHERE sync_name = 'default'"
        ).fetchone()
        status = get_paperstream_status(conn)
        sync_started_payload = json.loads(
            conn.execute(
                """
                SELECT payload_json
                FROM arxiv_events
                WHERE event_type = 'sync_started'
                ORDER BY event_id DESC
                LIMIT 1
                """
            ).fetchone()["payload_json"]
        )
        assert dict(state) == {"metadata_prefix": "arXivCustom", "set_spec": "cs.AI"}
        assert status["metadata_prefix"] == "arXivCustom"
        assert status["set_spec"] == "cs.AI"
        assert status["sync"]["metadata_prefix"] == "arXivCustom"
        assert status["sync"]["set_spec"] == "cs.AI"
        assert sync_started_payload["metadata_prefix"] == "arXivCustom"
        assert sync_started_payload["set_spec"] == "cs.AI"
    finally:
        conn.close()


def test_paperstream_cli_sync_metadata_enqueue_assets_pdf_commits_only_pdf_queue_rows(tmp_path):
    """CLI sync-metadata --enqueue-assets pdf commits PDF queue work without source jobs."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    fixture_path = tmp_path / "oai-final-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_FINAL_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--max-pages",
            "1",
            "--enqueue-assets",
            "pdf",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["jobs_queued"] == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        asset_rows = conn.execute(
            """
            SELECT asset_type, status, source
            FROM arxiv_assets
            WHERE asset_type IN ('pdf', 'source')
            ORDER BY asset_type
            """
        ).fetchall()
        assert [(row["asset_type"], row["status"], row["source"]) for row in asset_rows] == [
            ("pdf", "queued", "metadata_sync"),
        ]
        job_rows = conn.execute(
            """
            SELECT job_type, asset_type, status
            FROM arxiv_jobs
            ORDER BY job_type
            """
        ).fetchall()
        assert [(row["job_type"], row["asset_type"], row["status"]) for row in job_rows] == [
            ("download_pdf", "pdf", "queued"),
        ]
    finally:
        conn.close()


def test_paperstream_cli_sync_metadata_enqueue_assets_without_value_commits_both_queue_rows(tmp_path):
    """CLI sync-metadata --enqueue-assets without a value remains PDF plus source."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    fixture_path = tmp_path / "oai-final-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_FINAL_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--max-pages",
            "1",
            "--enqueue-assets",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["jobs_queued"] == 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        asset_rows = conn.execute(
            """
            SELECT asset_type, status, source
            FROM arxiv_assets
            WHERE asset_type IN ('pdf', 'source')
            ORDER BY asset_type
            """
        ).fetchall()
        assert [(row["asset_type"], row["status"], row["source"]) for row in asset_rows] == [
            ("pdf", "queued", "metadata_sync"),
            ("source", "queued", "metadata_sync"),
        ]
        job_rows = conn.execute(
            """
            SELECT job_type, asset_type, status
            FROM arxiv_jobs
            ORDER BY job_type
            """
        ).fetchall()
        assert [(row["job_type"], row["asset_type"], row["status"]) for row in job_rows] == [
            ("download_pdf", "pdf", "queued"),
            ("download_source", "source", "queued"),
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("asset_value", ["metadata", ""])
def test_paperstream_cli_sync_metadata_rejects_invalid_enqueue_assets_value_before_mutation(
    tmp_path,
    asset_value,
):
    """Invalid or empty sync-metadata asset type values fail before DB mutation."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    before_counts = _paperstream_db_counts(db_path)

    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_FINAL_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--max-pages",
            "1",
            "--enqueue-assets",
            asset_value,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert "enqueue-assets" in rejected.stderr
    assert "pdf" in rejected.stderr
    assert "source" in rejected.stderr
    assert _paperstream_db_counts(db_path) == before_counts


@pytest.mark.parametrize("enqueue_args", [[], ["pdf"]])
def test_paperstream_cli_sync_metadata_rejects_metadata_only_with_enqueue_assets(
    tmp_path,
    enqueue_args,
):
    """Metadata-only and enqueue-assets are contradictory and fail before DB mutation."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    before_counts = _paperstream_db_counts(db_path)

    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--metadata-only",
            "--enqueue-assets",
            *enqueue_args,
            "--max-pages",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert "metadata-only" in rejected.stderr
    assert "enqueue-assets" in rejected.stderr
    assert _paperstream_db_counts(db_path) == before_counts


def test_paperstream_cli_sync_metadata_accepts_metadata_only_fixture_dry_run(tmp_path):
    """Metadata-only is an explicit no-op safety flag for bounded fixture dry-runs."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()
    before_counts = _paperstream_db_counts(db_path)

    fixture_path = tmp_path / "oai-page.xml"
    fixture_path.write_text(_PAPERSTREAM_OAI_PAGE, encoding="utf-8")
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    dry_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "sync-metadata",
            "--input",
            str(fixture_path),
            "--dry-run",
            "--metadata-only",
            "--max-pages",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(dry_run.stdout)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["pages_seen"] == 1
    assert result["records_seen"] == 2
    assert _paperstream_db_counts(db_path) == before_counts

def test_paperstream_initialize_db_creates_required_tables_and_indexes():
    """PaperStream schema objects are available on a fresh initialized database."""
    from researchfs import db

    conn = _memory_conn()
    db.initialize_db(conn)

    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    indexes = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }

    assert PAPERSTREAM_TABLES <= tables
    assert PAPERSTREAM_INDEXES <= indexes

    asset_checks = [
        row["sql"]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_assets'"
        )
    ][0]
    job_checks = [
        row["sql"]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_jobs'"
        )
    ][0]
    assert "asset_type IN" in asset_checks
    assert "status IN" in asset_checks
    assert "status IN" in job_checks
    assert "job_type IN" in job_checks
    assert "embed_chunks" in job_checks


@pytest.mark.parametrize(
    ("case", "statement", "params"),
    [
        (
            "arxiv_assets.asset_type",
            """
            INSERT INTO arxiv_assets (
                asset_key, arxiv_id, version, asset_type, status, source
            )
            VALUES (?, '2606.29148', 1, ?, 'queued', 'schema-test')
            """,
            ("2606.29148:v1:ebook", "ebook"),
        ),
        (
            "arxiv_assets.status",
            """
            INSERT INTO arxiv_assets (
                asset_key, arxiv_id, version, asset_type, status, source
            )
            VALUES (?, '2606.29148', 1, 'pdf', ?, 'schema-test')
            """,
            ("2606.29148:v1:pdf", "archived"),
        ),
        (
            "arxiv_jobs.status",
            """
            INSERT INTO arxiv_jobs (
                job_key, job_type, asset_type, status, payload_json
            )
            VALUES ('schema-test-job', 'download_pdf', 'pdf', ?, '{}')
            """,
            ("paused",),
        ),
        (
            "arxiv_jobs.job_type",
            """
            INSERT INTO arxiv_jobs (
                job_key, job_type, asset_type, status, payload_json
            )
            VALUES ('schema-test-job-type', ?, 'pdf', 'queued', '{}')
            """,
            ("delete_everything",),
        ),
        (
            "arxiv_sync_state.status",
            """
            INSERT INTO arxiv_sync_state (sync_name, status)
            VALUES ('default', ?)
            """,
            ("paused",),
        ),
    ],
    ids=[
        "asset_type",
        "asset_status",
        "job_status",
        "job_type",
        "sync_state_status",
    ],
)
def test_paperstream_schema_rejects_invalid_statuses_and_asset_type(
    case, statement, params
):
    """PaperStream enum-like schema fields reject unknown values before app validators."""
    from researchfs import db

    conn = _memory_conn()
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        conn.execute(statement, params)


@pytest.mark.parametrize(
    ("case", "statement", "params"),
    [
        (
            "arxiv_versions",
            """
            INSERT INTO arxiv_versions (
                arxiv_id, version, submitted_at, announced_at, title
            )
            VALUES (?, 1, '2026-07-01T00:00:00Z', '2026-07-02', 'Missing paper')
            """,
            ("2606.99999",),
        ),
        (
            "arxiv_assets",
            """
            INSERT INTO arxiv_assets (
                asset_key, arxiv_id, version, asset_type, status, source
            )
            VALUES (?, ?, 1, 'metadata', 'seen', 'schema-test')
            """,
            ("2606.99999:v1:metadata", "2606.99999"),
        ),
    ],
    ids=[
        "version_parent",
        "asset_parent",
    ],
)
def test_paperstream_schema_rejects_missing_paper_references_when_foreign_keys_enabled(
    case, statement, params
):
    """Version and asset rows cannot outlive their parent paper when SQLite FKs are on."""
    from researchfs import db

    conn = _memory_conn()
    conn.execute("PRAGMA foreign_keys = ON")
    db.initialize_db(conn)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        conn.execute(statement, params)


def test_paperstream_migrate_db_upgrades_legacy_database_idempotently():
    """A legacy DB without PaperStream objects can be migrated repeatedly."""
    from researchfs import db

    legacy = _memory_conn()
    legacy.execute(
        "CREATE TABLE papers (paper_id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
    )
    legacy.execute("INSERT INTO papers (title) VALUES ('legacy paper')")

    db.migrate_db(legacy)
    db.migrate_db(legacy)

    tables = {
        row["name"]
        for row in legacy.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    indexes = {
        row["name"]
        for row in legacy.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }

    assert PAPERSTREAM_TABLES <= tables
    assert PAPERSTREAM_INDEXES <= indexes
    assert legacy.execute("SELECT title FROM papers").fetchone()["title"] == "legacy paper"


def test_paperstream_migrate_db_adds_arxiv_job_type_constraint_idempotently():
    """Legacy arxiv_jobs rows survive the CHECK rebuild and gain DB-level validation."""
    from researchfs import db

    legacy = _memory_conn()
    legacy.execute(
        """
        CREATE TABLE arxiv_jobs (
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
            CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'skipped'))
        )
        """
    )
    legacy.execute(
        """
        CREATE INDEX idx_arxiv_jobs_status_run_after
        ON arxiv_jobs(status, run_after, priority)
        """
    )
    legacy.executemany(
        """
        INSERT INTO arxiv_jobs (
            job_key, job_type, arxiv_id, version, asset_type, status, priority,
            attempts, max_attempts, run_after, locked_by, locked_at, payload_json,
            error_message, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "legacy-download",
                "download_pdf",
                "2606.29148",
                1,
                "pdf",
                "failed",
                10,
                2,
                5,
                "2026-07-01T00:00:00Z",
                "worker-a",
                "2026-07-01T00:00:01Z",
                '{"source":"legacy"}',
                "network timeout",
                "2026-07-01T00:00:00Z",
                "2026-07-01T00:00:02Z",
            ),
            (
                "legacy-embed",
                "embed_chunks",
                "2606.29149",
                2,
                "chunks",
                "queued",
                30,
                0,
                3,
                "2026-07-02T00:00:00Z",
                None,
                None,
                '{"planned":true}',
                None,
                "2026-07-02T00:00:00Z",
                "2026-07-02T00:00:00Z",
            ),
        ],
    )

    db.migrate_db(legacy)
    db.migrate_db(legacy)

    create_sql = legacy.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_jobs'"
    ).fetchone()["sql"]
    rows = [
        dict(row)
        for row in legacy.execute(
            """
            SELECT job_key, job_type, arxiv_id, version, asset_type, status, priority,
                   attempts, max_attempts, run_after, locked_by, locked_at, payload_json,
                   error_message, created_at, updated_at
            FROM arxiv_jobs
            ORDER BY job_key
            """
        )
    ]
    indexes = {
        row["name"]
        for row in legacy.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'arxiv_jobs'"
        )
    }

    assert "job_type IN" in create_sql
    assert "embed_chunks" in create_sql
    assert "idx_arxiv_jobs_status_run_after" in indexes
    assert rows == [
        {
            "job_key": "legacy-download",
            "job_type": "download_pdf",
            "arxiv_id": "2606.29148",
            "version": 1,
            "asset_type": "pdf",
            "status": "failed",
            "priority": 10,
            "attempts": 2,
            "max_attempts": 5,
            "run_after": "2026-07-01T00:00:00Z",
            "locked_by": "worker-a",
            "locked_at": "2026-07-01T00:00:01Z",
            "payload_json": '{"source":"legacy"}',
            "error_message": "network timeout",
            "created_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-01T00:00:02Z",
        },
        {
            "job_key": "legacy-embed",
            "job_type": "embed_chunks",
            "arxiv_id": "2606.29149",
            "version": 2,
            "asset_type": "chunks",
            "status": "queued",
            "priority": 30,
            "attempts": 0,
            "max_attempts": 3,
            "run_after": "2026-07-02T00:00:00Z",
            "locked_by": None,
            "locked_at": None,
            "payload_json": '{"planned":true}',
            "error_message": None,
            "created_at": "2026-07-02T00:00:00Z",
            "updated_at": "2026-07-02T00:00:00Z",
        },
    ]

    legacy.execute(
        """
        INSERT INTO arxiv_jobs (job_key, job_type, asset_type, payload_json)
        VALUES ('legacy-index-vector', 'index_vector', 'chunks', '{}')
        """
    )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        legacy.execute(
            """
            INSERT INTO arxiv_jobs (job_key, job_type, asset_type, payload_json)
            VALUES ('legacy-invalid', 'delete_everything', 'pdf', '{}')
            """
        )



def test_paperstream_migrate_db_rejects_invalid_legacy_arxiv_job_types():
    """Legacy arxiv_jobs rows with unknown durable job types block the rebuild."""
    from researchfs import db

    legacy = _memory_conn()
    legacy.execute(
        """
        CREATE TABLE arxiv_jobs (
            job_key TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            run_after TEXT DEFAULT CURRENT_TIMESTAMP,
            priority INTEGER DEFAULT 100
        )
        """
    )
    legacy.execute(
        "INSERT INTO arxiv_jobs (job_key, job_type) VALUES ('bad-legacy', 'delete_everything')"
    )

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.migrate_db(legacy)

    row = legacy.execute(
        "SELECT job_key, job_type FROM arxiv_jobs WHERE job_key = 'bad-legacy'"
    ).fetchone()
    create_sql = legacy.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'arxiv_jobs'"
    ).fetchone()["sql"]
    assert dict(row) == {"job_key": "bad-legacy", "job_type": "delete_everything"}
    assert "job_type IN" not in create_sql

@pytest.mark.parametrize("migration_fn", ["migrate_db", "initialize_db"])
def test_paperstream_migration_adds_scope_and_counter_columns_to_legacy_sync_state(migration_fn):
    """Legacy sync state tables gain writable scope and inserted-counter columns."""
    from researchfs import db
    from researchfs.paperstream import get_paperstream_status

    legacy = _memory_conn()
    legacy.execute(
        """
        CREATE TABLE arxiv_sync_state (
            sync_name TEXT PRIMARY KEY,
            committed_from TEXT,
            last_response_date TEXT,
            pending_response_date TEXT,
            last_resumption_token TEXT,
            status TEXT NOT NULL DEFAULT 'idle',
            records_seen_last_run INTEGER DEFAULT 0,
            records_updated_last_run INTEGER DEFAULT 0,
            jobs_queued_last_run INTEGER DEFAULT 0,
            error_message TEXT,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            CHECK (status IN ('idle', 'running', 'succeeded', 'failed'))
        )
        """
    )
    legacy.execute(
        """
        INSERT INTO arxiv_sync_state (
            sync_name, status, records_seen_last_run, records_updated_last_run
        )
        VALUES ('legacy', 'succeeded', 8, 5)
        """
    )

    getattr(db, migration_fn)(legacy)
    getattr(db, migration_fn)(legacy)

    columns = {
        row["name"] for row in legacy.execute("PRAGMA table_info(arxiv_sync_state)")
    }
    assert {
        "metadata_prefix",
        "set_spec",
        "records_inserted_last_run",
        "versions_inserted_last_run",
    } <= columns
    legacy.execute(
        """
        UPDATE arxiv_sync_state
        SET records_inserted_last_run = ?,
            versions_inserted_last_run = ?
        WHERE sync_name = ?
        """,
        (3, 7, "legacy"),
    )
    row = legacy.execute(
        """
        SELECT sync_name, metadata_prefix, set_spec, status,
               records_seen_last_run, records_updated_last_run,
               records_inserted_last_run, versions_inserted_last_run
        FROM arxiv_sync_state
        WHERE sync_name = 'legacy'
        """
    ).fetchone()
    assert dict(row) == {
        "sync_name": "legacy",
        "metadata_prefix": "arXivRaw",
        "set_spec": None,
        "status": "succeeded",
        "records_seen_last_run": 8,
        "records_updated_last_run": 5,
        "records_inserted_last_run": 3,
        "versions_inserted_last_run": 7,
    }
    status = get_paperstream_status(legacy)
    assert status["records_inserted_last_run"] == 3
    assert status["versions_inserted_last_run"] == 7
    assert status["sync"]["records_inserted_last_run"] == 3
    assert status["sync"]["versions_inserted_last_run"] == 7
    assert status["metadata_prefix"] == "arXivRaw"
    assert status["set_spec"] is None
    assert status["sync"]["metadata_prefix"] == "arXivRaw"
    assert status["sync"]["set_spec"] is None



def test_paperstream_key_and_path_helpers_handle_modern_and_old_arxiv_ids(tmp_path):
    """Canonical keys and object paths are deterministic for both arXiv ID styles."""
    from researchfs.paperstream import asset_key, job_key, object_path

    assert asset_key("2606.29148", 1, "metadata") == "2606.29148:v1:metadata"
    assert job_key("download_pdf", "2606.29148", 1) == "download_pdf:2606.29148:v1"
    assert asset_key("hep-th/9901001", 2, "pdf") == "hep-th/9901001:v2:pdf"
    assert job_key("parse_source", "hep-th/9901001", 2) == "parse_source:hep-th/9901001:v2"

    modern_asset_paths = {
        "metadata": tmp_path / "arxiv" / "metadata" / "2606" / "29148" / "v1.json",
        "pdf": tmp_path / "arxiv" / "pdf" / "2606" / "29148" / "v1.pdf",
        "source": tmp_path / "arxiv" / "source" / "2606" / "29148" / "v1.tar.gz",
        "parsed_text": tmp_path / "arxiv" / "text" / "2606" / "29148" / "v1.json",
        "chunks": tmp_path / "arxiv" / "chunks" / "2606" / "29148" / "v1.json",
        "embedding": tmp_path / "arxiv" / "embeddings" / "2606" / "29148" / "v1.json",
        "bm25_index": tmp_path / "arxiv" / "bm25" / "2606" / "29148" / "v1.sqlite",
        "vector_index": tmp_path / "arxiv" / "vector" / "2606" / "29148" / "v1.sqlite",
    }
    for asset_type, expected_path in modern_asset_paths.items():
        assert Path(object_path(tmp_path, "2606.29148", 1, asset_type)) == expected_path
    assert Path(object_path(tmp_path, "hep-th/9901001", 1, "pdf")) == (
        tmp_path / "arxiv" / "pdf" / "hep-th" / "9901001" / "v1.pdf"
    )


@pytest.mark.parametrize(
    ("helper_name", "args", "message"),
    [
        ("asset_key", ("not an arxiv id", 1, "pdf"), "arxiv"),
        ("asset_key", ("2606.29148", 0, "pdf"), "version"),
        ("asset_key", ("2606.29148", 1, "thumbnail"), "asset_type"),
        ("job_key", ("send_email", "2606.29148", 1), "job_type"),
        ("job_key", ("download_pdf", "not an arxiv id", 1), "arxiv"),
        ("job_key", ("download_pdf", "2606.29148", 0), "version"),
    ],
)
def test_paperstream_helpers_reject_invalid_asset_and_job_inputs(helper_name, args, message):
    """Invalid asset/job identifiers fail before creating durable state."""
    import researchfs.paperstream as paperstream

    with pytest.raises(ValueError, match=message):
        getattr(paperstream, helper_name)(*args)


@pytest.mark.parametrize(
    ("helper_name", "arxiv_id", "version", "asset_type", "message"),
    [
        ("object_path", "not an arxiv id", 1, "pdf", "arxiv"),
        ("object_path", "2606.29148", 0, "pdf", "version"),
        ("object_path", "2606.29148", 1, "thumbnail", "asset_type"),
        ("arxiv_storage_path", "not an arxiv id", 1, "pdf", "arxiv"),
        ("arxiv_storage_path", "2606.29148", 0, "pdf", "version"),
        ("arxiv_storage_path", "2606.29148", 1, "thumbnail", "asset_type"),
    ],
)
def test_paperstream_object_path_helpers_reject_invalid_inputs_before_path_construction(
    tmp_path, helper_name, arxiv_id, version, asset_type, message
):
    """Invalid path helper inputs fail validation instead of producing object keys."""
    import researchfs.paperstream as paperstream

    with pytest.raises(ValueError, match=message):
        getattr(paperstream, helper_name)(tmp_path, arxiv_id, version, asset_type)


def test_paperstream_asset_upsert_is_idempotent_and_updates_state():
    """Asset upserts keep a single exact asset row while refreshing mutable fields."""
    from researchfs.paperstream import asset_key, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)

    key = upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29148",
        version=1,
        asset_type="metadata",
        status="seen",
        storage_path="/object-store/arxiv/metadata/2606/29148/v1.json",
        checksum="sha256:first",
        byte_size=100,
        source="oai",
    )
    second_key = upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29148",
        version=1,
        asset_type="metadata",
        status="imported",
        storage_path="/object-store/arxiv/metadata/2606/29148/v1.json",
        checksum="sha256:second",
        byte_size=120,
        source="oai",
    )

    assert key == second_key == asset_key("2606.29148", 1, "metadata")
    rows = conn.execute("SELECT * FROM arxiv_assets").fetchall()
    assert len(rows) == 1
    assert rows[0]["asset_key"] == key
    assert rows[0]["status"] == "imported"
    assert rows[0]["checksum"] == "sha256:second"
    assert rows[0]["byte_size"] == 120



def _paperstream_running_job(
    conn,
    arxiv_id,
    job_type="download_pdf",
    asset_type="pdf",
    *,
    version=1,
    priority=100,
    attempts=0,
    max_attempts=5,
    locked_at="2026-07-03 00:00:00",
    locked_by="worker-before-crash",
    run_after="2026-07-03 00:00:00",
    payload=None,
):
    from researchfs.paperstream import enqueue_arxiv_job

    _insert_paperstream_paper(conn, arxiv_id, version=version)
    job_key = enqueue_arxiv_job(
        conn,
        job_type=job_type,
        arxiv_id=arxiv_id,
        version=version,
        asset_type=asset_type,
        priority=priority,
        payload=payload if payload is not None else {},
        max_attempts=max_attempts,
        run_after=run_after,
    )
    conn.execute(
        """
        UPDATE arxiv_jobs
        SET status = 'running',
            locked_by = ?,
            locked_at = ?,
            attempts = ?,
            max_attempts = ?,
            updated_at = ?
        WHERE job_key = ?
        """,
        (locked_by, locked_at, attempts, max_attempts, locked_at, job_key),
    )
    return job_key


def test_paperstream_requeue_stale_arxiv_jobs_requeues_only_stale_running_job():
    """Explicit stale-lock maintenance requeues only stale running jobs and records audit events."""
    from researchfs.paperstream import requeue_stale_arxiv_jobs

    conn = _paperstream_conn()
    stale_key = _paperstream_running_job(
        conn,
        "2606.29148",
        attempts=2,
        locked_at="2026-07-03 22:00:00",
        payload={"local_path": "/private/paper.pdf"},
    )
    fresh_key = _paperstream_running_job(
        conn,
        "2606.29149",
        attempts=1,
        locked_at="2026-07-03 23:30:00",
        payload={"local_path": "/private/fresh.pdf"},
    )

    result = requeue_stale_arxiv_jobs(
        conn,
        stale_after_seconds=3600,
        now="2026-07-04 00:00:00",
        limit=10,
    )

    assert result["requeued"] == 1
    assert [job["job_key"] for job in result["jobs"]] == [stale_key]
    assert "payload_json" not in result["jobs"][0]
    assert "payload" not in result["jobs"][0]
    encoded_result = json.dumps(result, sort_keys=True)
    assert "/private" not in encoded_result
    _assert_paperstream_public_payload_has_no_local_asset_fields(result)
    stale = conn.execute(
        "SELECT status, attempts, locked_by, locked_at FROM arxiv_jobs WHERE job_key = ?",
        (stale_key,),
    ).fetchone()
    fresh = conn.execute(
        "SELECT status, attempts, locked_by, locked_at FROM arxiv_jobs WHERE job_key = ?",
        (fresh_key,),
    ).fetchone()
    assert dict(stale) == {
        "status": "queued",
        "attempts": 2,
        "locked_by": None,
        "locked_at": None,
    }
    assert dict(fresh) == {
        "status": "running",
        "attempts": 1,
        "locked_by": "worker-before-crash",
        "locked_at": "2026-07-03 23:30:00",
    }
    event = conn.execute(
        """
        SELECT event_type, job_key
        FROM arxiv_events
        WHERE event_type = 'job_requeued'
        """
    ).fetchone()
    assert dict(event) == {"event_type": "job_requeued", "job_key": stale_key}


def test_paperstream_requeue_stale_arxiv_jobs_skips_exhausted_attempts():
    """Stale running jobs at max_attempts remain running for deliberate failure handling."""
    from researchfs.paperstream import requeue_stale_arxiv_jobs

    conn = _paperstream_conn()
    exhausted_key = _paperstream_running_job(
        conn,
        "2606.29148",
        attempts=3,
        max_attempts=3,
        locked_at="2026-07-03 20:00:00",
    )

    result = requeue_stale_arxiv_jobs(
        conn,
        stale_after_seconds=3600,
        now="2026-07-04 00:00:00",
        limit=10,
    )

    assert result["requeued"] == 0
    assert result["jobs"] == []
    row = conn.execute(
        "SELECT status, attempts, max_attempts, locked_by, locked_at FROM arxiv_jobs WHERE job_key = ?",
        (exhausted_key,),
    ).fetchone()
    assert dict(row) == {
        "status": "running",
        "attempts": 3,
        "max_attempts": 3,
        "locked_by": "worker-before-crash",
        "locked_at": "2026-07-03 20:00:00",
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_requeued'"
    ).fetchone()[0] == 0


def test_paperstream_requeue_stale_arxiv_jobs_honors_job_types_and_limit_deterministically():
    """Stale-lock requeueing is bounded, type-filtered, and ordered like queue claims."""
    from researchfs.paperstream import requeue_stale_arxiv_jobs

    conn = _paperstream_conn()
    download_key = _paperstream_running_job(
        conn,
        "2606.29148",
        "download_pdf",
        "pdf",
        priority=0,
        locked_at="2026-07-03 20:00:00",
    )
    first_parse_key = _paperstream_running_job(
        conn,
        "2606.29149",
        "parse_pdf",
        "parsed_text",
        priority=1,
        locked_at="2026-07-03 20:00:00",
    )
    second_parse_key = _paperstream_running_job(
        conn,
        "2606.29150",
        "parse_pdf",
        "parsed_text",
        priority=2,
        locked_at="2026-07-03 20:00:00",
    )

    result = requeue_stale_arxiv_jobs(
        conn,
        stale_after_seconds=3600,
        now="2026-07-04 00:00:00",
        job_types=("parse_pdf",),
        limit=1,
    )

    assert result["requeued"] == 1
    assert [job["job_key"] for job in result["jobs"]] == [first_parse_key]
    rows = {
        row["job_key"]: row["status"]
        for row in conn.execute("SELECT job_key, status FROM arxiv_jobs")
    }
    assert rows == {
        download_key: "running",
        first_parse_key: "queued",
        second_parse_key: "running",
    }


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stale_after_seconds": 0}, "stale_after_seconds"),
        ({"stale_after_seconds": True}, "stale_after_seconds"),
        ({"stale_after_seconds": 60, "limit": 0}, "limit"),
        ({"stale_after_seconds": 60, "limit": True}, "limit"),
        ({"stale_after_seconds": 60, "job_types": ("not_a_job",)}, "job_types"),
        ({"stale_after_seconds": 60, "job_types": ()}, "job_types"),
    ],
)
def test_paperstream_requeue_stale_arxiv_jobs_rejects_invalid_args_before_sqlite_work(
    kwargs, message
):
    """Invalid stale-lock maintenance filters fail before touching PaperStream tables."""
    from researchfs.paperstream import requeue_stale_arxiv_jobs

    conn = _memory_conn()
    with pytest.raises((TypeError, ValueError), match=message):
        requeue_stale_arxiv_jobs(conn, **kwargs)


def test_paperstream_claim_next_arxiv_job_waits_for_explicit_stale_requeue():
    """Claiming never steals stale running locks; explicit requeue makes the job claimable."""
    from researchfs.paperstream import claim_next_arxiv_job, requeue_stale_arxiv_jobs

    conn = _paperstream_conn()
    job_key = _paperstream_running_job(
        conn,
        "2606.29148",
        locked_at="2026-07-03 20:00:00",
    )

    assert (
        claim_next_arxiv_job(conn, worker_id="worker-steal", now="2026-07-04 00:00:00")
        is None
    )

    requeue_stale_arxiv_jobs(
        conn,
        stale_after_seconds=3600,
        now="2026-07-04 00:00:00",
        limit=10,
    )
    claimed = claim_next_arxiv_job(
        conn,
        worker_id="worker-after-requeue",
        now="2026-07-04 00:00:01",
    )

    assert claimed is not None
    assert claimed["job_key"] == job_key
    assert claimed["locked_by"] == "worker-after-requeue"


def test_paperstream_cli_requeue_stale_jobs_prints_json_and_updates_db_state(tmp_path):
    """The stale-lock maintenance CLI prints path-safe JSON and persists the requeue."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    stale_key = _paperstream_running_job(
        conn,
        "2606.29148",
        attempts=2,
        locked_at="2000-01-01 00:00:00",
        payload={"local_path": "/private/cli.pdf"},
    )
    fresh_key = _paperstream_running_job(
        conn,
        "2606.29149",
        locked_at="9999-01-01 00:00:00",
        payload={"local_path": "/private/fresh-cli.pdf"},
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "requeue-stale-jobs",
            "--stale-after-seconds",
            "3600",
            "--job-types",
            "download_pdf",
            "--limit",
            "10",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["requeued"] == 1
    assert [job["job_key"] for job in payload["jobs"]] == [stale_key]
    assert "payload_json" not in payload["jobs"][0]
    assert "payload" not in payload["jobs"][0]
    encoded_payload = json.dumps(payload, sort_keys=True)
    assert "/private" not in encoded_payload
    _assert_paperstream_public_payload_has_no_local_asset_fields(payload)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        states = {
            row["job_key"]: dict(row)
            for row in conn.execute(
                """
                SELECT job_key, status, attempts, locked_by, locked_at
                FROM arxiv_jobs
                ORDER BY job_key
                """
            )
        }
        assert states[stale_key] == {
            "job_key": stale_key,
            "status": "queued",
            "attempts": 2,
            "locked_by": None,
            "locked_at": None,
        }
        assert states[fresh_key]["status"] == "running"
        assert conn.execute(
            "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_requeued' AND job_key = ?",
            (stale_key,),
        ).fetchone()[0] == 1
    finally:
        conn.close()



def _paperstream_job_row(conn, job_key):
    return conn.execute(
        """
        SELECT job_key, job_type, status, attempts, max_attempts, run_after,
               locked_by, locked_at, error_message
        FROM arxiv_jobs
        WHERE job_key = ?
        """,
        (job_key,),
    ).fetchone()


def test_paperstream_retry_failed_arxiv_jobs_retries_only_failed_jobs_and_is_claimable():
    """Manual failed-job retry resets only terminal failures into claimable queued work."""
    from researchfs.paperstream import (
        claim_next_arxiv_job,
        enqueue_arxiv_job,
        retry_failed_arxiv_jobs,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    _insert_paperstream_paper(conn, "2606.29150", version=1)
    _insert_paperstream_paper(conn, "2606.29151", version=1)
    failed_key = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29148",
        1,
        "pdf",
        priority=0,
        max_attempts=5,
        run_after="2099-01-01 00:00:00",
        payload={"local_path": "/private/terminal-failed.pdf", "storage_path": "/tmp/leak.pdf"},
    )
    succeeded_key = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29149",
        1,
        "pdf",
        priority=1,
        max_attempts=5,
        run_after="2026-07-03 00:00:00",
    )
    queued_key = enqueue_arxiv_job(
        conn,
        "download_source",
        "2606.29150",
        1,
        "source",
        priority=1,
        max_attempts=5,
        run_after="2026-07-03 00:00:00",
    )
    running_key = _paperstream_running_job(
        conn,
        "2606.29151",
        "download_pdf",
        "pdf",
        priority=1,
        attempts=1,
        max_attempts=5,
        locked_at="2026-07-03 23:00:00",
    )
    conn.execute(
        """
        UPDATE arxiv_jobs
        SET status = 'failed',
            attempts = 4,
            locked_by = 'worker-terminal',
            locked_at = '2026-07-03 22:00:00',
            error_message = 'permanent outage'
        WHERE job_key = ?
        """,
        (failed_key,),
    )
    conn.execute(
        """
        UPDATE arxiv_jobs
        SET status = 'succeeded',
            attempts = 2,
            locked_by = NULL,
            locked_at = NULL,
            error_message = NULL
        WHERE job_key = ?
        """,
        (succeeded_key,),
    )

    result = retry_failed_arxiv_jobs(
        conn,
        now="2026-07-04 00:00:00",
        limit=10,
    )

    assert result["ok"] is True
    assert result["retried"] == 1
    assert [job["job_key"] for job in result["jobs"]] == [failed_key]
    assert result["jobs"][0]["previous_attempts"] == 4
    assert result["jobs"][0]["max_attempts"] == 5
    assert result["jobs"][0]["queued_at"] == "2026-07-04 00:00:00"
    assert "payload_json" not in result["jobs"][0]
    assert "payload" not in result["jobs"][0]
    encoded_result = json.dumps(result, sort_keys=True)
    assert "/private" not in encoded_result
    _assert_paperstream_public_payload_has_no_local_asset_fields(result)
    assert dict(_paperstream_job_row(conn, failed_key)) == {
        "job_key": failed_key,
        "job_type": "download_pdf",
        "status": "queued",
        "attempts": 0,
        "max_attempts": 5,
        "run_after": "2026-07-04 00:00:00",
        "locked_by": None,
        "locked_at": None,
        "error_message": None,
    }
    assert _paperstream_job_row(conn, succeeded_key)["status"] == "succeeded"
    assert _paperstream_job_row(conn, queued_key)["status"] == "queued"
    assert _paperstream_job_row(conn, running_key)["status"] == "running"
    event = conn.execute(
        """
        SELECT event_type, job_key
        FROM arxiv_events
        WHERE event_type = 'job_retry_queued'
        """
    ).fetchone()
    assert dict(event) == {"event_type": "job_retry_queued", "job_key": failed_key}

    claimed = claim_next_arxiv_job(
        conn,
        worker_id="worker-after-manual-retry",
        job_types=("download_pdf",),
        now="2026-07-04 00:00:01",
    )

    assert claimed is not None
    assert claimed["job_key"] == failed_key
    assert claimed["attempts"] == 0
    assert claimed["locked_by"] == "worker-after-manual-retry"


def test_paperstream_retry_failed_arxiv_jobs_honors_job_types_and_limit_deterministically():
    """Manual failed-job retry is bounded, type-filtered, and ordered like queued claims."""
    from researchfs.paperstream import enqueue_arxiv_job, retry_failed_arxiv_jobs

    conn = _paperstream_conn()
    for arxiv_id in ("2606.29148", "2606.29149", "2606.29150"):
        _insert_paperstream_paper(conn, arxiv_id, version=1)
    download_key = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29148",
        1,
        "pdf",
        priority=0,
        run_after="2026-07-03 00:00:00",
    )
    first_parse_key = enqueue_arxiv_job(
        conn,
        "parse_pdf",
        "2606.29149",
        1,
        "parsed_text",
        priority=1,
        run_after="2026-07-03 00:00:00",
    )
    second_parse_key = enqueue_arxiv_job(
        conn,
        "parse_pdf",
        "2606.29150",
        1,
        "parsed_text",
        priority=2,
        run_after="2026-07-03 00:00:00",
    )
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 2 WHERE job_key = ?", (download_key,))
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 3 WHERE job_key = ?", (first_parse_key,))
    conn.execute("UPDATE arxiv_jobs SET status = 'failed', attempts = 4 WHERE job_key = ?", (second_parse_key,))

    result = retry_failed_arxiv_jobs(
        conn,
        now="2026-07-04 00:00:00",
        job_types=("parse_pdf",),
        limit=1,
    )

    assert result["retried"] == 1
    assert [job["job_key"] for job in result["jobs"]] == [first_parse_key]
    rows = {
        row["job_key"]: dict(row)
        for row in conn.execute("SELECT job_key, status, attempts FROM arxiv_jobs")
    }
    assert rows == {
        download_key: {"job_key": download_key, "status": "failed", "attempts": 2},
        first_parse_key: {"job_key": first_parse_key, "status": "queued", "attempts": 0},
        second_parse_key: {"job_key": second_parse_key, "status": "failed", "attempts": 4},
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_retry_queued'"
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"limit": 0}, "limit"),
        ({"limit": True}, "limit"),
        ({"limit": "100"}, "limit"),
        ({"job_types": ("not_a_job",)}, "job_types"),
        ({"job_types": ()}, "job_types"),
    ],
)
def test_paperstream_retry_failed_arxiv_jobs_rejects_invalid_args_before_sqlite_work(
    kwargs, message
):
    """Invalid manual-retry filters fail before touching PaperStream tables."""
    from researchfs.paperstream import retry_failed_arxiv_jobs

    conn = _memory_conn()
    with pytest.raises((TypeError, ValueError), match=message):
        retry_failed_arxiv_jobs(conn, **kwargs)


def test_paperstream_retry_failed_arxiv_jobs_rejects_limits_above_cap_before_sqlite_work():
    """Manual retry enforces the queue-maintenance cap before SQLite work."""
    import researchfs.paperstream as paperstream

    class ExecuteForbidden:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("retry_failed_arxiv_jobs should validate before sqlite execution")

    with pytest.raises(ValueError, match="limit"):
        paperstream.retry_failed_arxiv_jobs(
            ExecuteForbidden(),
            limit=paperstream._PAPERSTREAM_REQUEUE_MAX_LIMIT + 1,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["retry-failed-jobs", "--job-types", "not_a_job"],
        ["retry-failed-jobs", "--limit", "0"],
    ],
)
def test_paperstream_cli_retry_failed_jobs_invalid_args_fail_before_broad_data(
    tmp_path, argv
):
    """Invalid manual-retry CLI filters fail without printing broad PaperStream data."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    rejected = subprocess.run(
        [sys.executable, "-m", "researchfs.paperstream", "--db-path", str(db_path), *argv],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode != 0
    assert "2606.29148" not in rejected.stdout


def test_paperstream_cli_retry_failed_jobs_prints_json_and_persists_state(tmp_path):
    """The manual failed-job retry CLI prints path-safe JSON and persists queued state."""
    from researchfs import db
    from researchfs.paperstream import enqueue_arxiv_job

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    failed_key = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29148",
        1,
        "pdf",
        max_attempts=4,
        payload={"local_path": "/private/cli-retry.pdf", "storage_path": "/tmp/cli-retry.pdf"},
    )
    succeeded_key = enqueue_arxiv_job(
        conn,
        "download_pdf",
        "2606.29149",
        1,
        "pdf",
        max_attempts=4,
    )
    conn.execute(
        """
        UPDATE arxiv_jobs
        SET status = 'failed',
            attempts = 4,
            locked_by = 'worker-terminal',
            locked_at = '2026-07-03 22:00:00',
            error_message = 'permanent outage'
        WHERE job_key = ?
        """,
        (failed_key,),
    )
    conn.execute(
        "UPDATE arxiv_jobs SET status = 'succeeded', attempts = 1 WHERE job_key = ?",
        (succeeded_key,),
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "retry-failed-jobs",
            "--job-types",
            "download_pdf",
            "--limit",
            "10",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["retried"] == 1
    assert [job["job_key"] for job in payload["jobs"]] == [failed_key]
    assert payload["jobs"][0]["previous_attempts"] == 4
    assert payload["jobs"][0]["max_attempts"] == 4
    assert "queued_at" in payload["jobs"][0]
    assert "payload_json" not in payload["jobs"][0]
    assert "payload" not in payload["jobs"][0]
    encoded_payload = json.dumps(payload, sort_keys=True)
    assert "/private" not in encoded_payload
    _assert_paperstream_public_payload_has_no_local_asset_fields(payload)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        retried = conn.execute(
            """
            SELECT status, attempts, max_attempts, locked_by, locked_at, error_message
            FROM arxiv_jobs
            WHERE job_key = ?
            """,
            (failed_key,),
        ).fetchone()
        assert dict(retried) == {
            "status": "queued",
            "attempts": 0,
            "max_attempts": 4,
            "locked_by": None,
            "locked_at": None,
            "error_message": None,
        }
        assert conn.execute(
            "SELECT status FROM arxiv_jobs WHERE job_key = ?",
            (succeeded_key,),
        ).fetchone()[0] == "succeeded"
        assert conn.execute(
            "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'job_retry_queued' AND job_key = ?",
            (failed_key,),
        ).fetchone()[0] == 1
    finally:
        conn.close()

def test_paperstream_job_enqueue_suppresses_duplicates_and_claim_locks_one_job():
    """The durable queue suppresses duplicate keys and claims one eligible job at a time."""
    from researchfs.paperstream import claim_next_arxiv_job, enqueue_arxiv_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)

    first_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=5,
        payload={"reason": "unit-test"},
        max_attempts=3,
        run_after="2026-07-03T00:00:00Z",
    )
    duplicate_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=1,
        payload={"reason": "duplicate"},
        max_attempts=3,
        run_after="2026-07-03T00:00:00Z",
    )
    enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29149",
        version=1,
        asset_type="pdf",
        priority=1,
        payload={},
        max_attempts=3,
        run_after="2026-07-04T00:00:00Z",
    )

    assert duplicate_key == first_key
    assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs").fetchone()[0] == 2

    claimed = claim_next_arxiv_job(
        conn, worker_id="worker-a", now="2026-07-03T00:00:01Z"
    )
    assert claimed is not None
    assert claimed["job_key"] == first_key
    assert claimed["status"] == "running"
    assert claimed["locked_by"] == "worker-a"

    assert (
        claim_next_arxiv_job(conn, worker_id="worker-b", now="2026-07-03T00:00:01Z")
        is None
    )
    locked = conn.execute(
        "SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?", (first_key,)
    ).fetchone()
    assert dict(locked) == {"status": "running", "locked_by": "worker-a"}


def test_paperstream_claim_next_arxiv_job_filters_job_types_without_claiming_earlier_nonmatch():
    """A job_types filter skips an earlier eligible nonmatch and claims the matching job."""
    from researchfs.paperstream import claim_next_arxiv_job, enqueue_arxiv_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    nonmatching_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=1,
        payload={},
        run_after="2026-07-03T00:00:00Z",
    )
    matching_key = enqueue_arxiv_job(
        conn,
        job_type="parse_pdf",
        arxiv_id="2606.29149",
        version=1,
        asset_type="parsed_text",
        priority=10,
        payload={},
        run_after="2026-07-03T00:00:00Z",
    )

    claimed = claim_next_arxiv_job(
        conn,
        worker_id="worker-filter",
        job_types=["parse_pdf"],
        now="2026-07-03T00:00:01Z",
    )

    assert claimed is not None
    assert claimed["job_key"] == matching_key
    assert claimed["job_type"] == "parse_pdf"
    assert claimed["locked_by"] == "worker-filter"
    assert dict(
        conn.execute(
            "SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?",
            (nonmatching_key,),
        ).fetchone()
    ) == {"status": "queued", "locked_by": None}


def test_paperstream_mark_arxiv_job_succeeded_appends_safe_audit_event():
    """Successful job completion records one redacted-free audit event after the row update."""
    from researchfs.paperstream import (
        claim_next_arxiv_job,
        enqueue_arxiv_job,
        mark_arxiv_job_failed,
        mark_arxiv_job_succeeded,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    job_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=5,
        payload={"source_path": "/tmp/private/source.pdf"},
        max_attempts=2,
        run_after="2026-07-03T00:00:00Z",
    )

    claim_next_arxiv_job(conn, worker_id="worker-a", now="2026-07-03 00:00:01")
    succeeded = mark_arxiv_job_succeeded(conn, job_key, now="2026-07-03 00:00:02")

    assert succeeded is not None
    assert succeeded["status"] == "succeeded"
    events = conn.execute(
        """
        SELECT event_type, arxiv_id, version, asset_type, job_key, payload_json
        FROM arxiv_events
        ORDER BY event_id
        """
    ).fetchall()
    assert len(events) == 1
    event = events[0]
    assert dict(event) | {"payload_json": event["payload_json"]} == {
        "event_type": "job_succeeded",
        "arxiv_id": "2606.29148",
        "version": 1,
        "asset_type": "pdf",
        "job_key": job_key,
        "payload_json": event["payload_json"],
    }
    payload = json.loads(event["payload_json"])
    assert payload == {
        "attempts": 0,
        "error_message": None,
        "job_type": "download_pdf",
        "max_attempts": 2,
        "run_after": "2026-07-03T00:00:00Z",
        "status": "succeeded",
        "succeeded_at": "2026-07-03 00:00:02",
    }
    assert "/tmp/private" not in event["payload_json"]

    repeated = mark_arxiv_job_succeeded(conn, job_key, now="2026-07-03 00:00:03")
    assert repeated is None
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 1
    contradicted = mark_arxiv_job_failed(
        conn,
        job_key,
        error_message="contradictory failure /tmp/private/after-success.pdf",
        now="2026-07-03 00:00:04",
    )
    assert contradicted is None
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 1


def test_paperstream_job_failure_requeues_with_backoff_then_terminal_fails():
    """Failed jobs increment attempts and stop retrying once max attempts is reached."""
    from researchfs.paperstream import (
        claim_next_arxiv_job,
        enqueue_arxiv_job,
        mark_arxiv_job_failed,
        mark_arxiv_job_succeeded,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    job_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=5,
        payload={},
        max_attempts=2,
        run_after="2026-07-03T00:00:00Z",
    )

    claim_next_arxiv_job(conn, worker_id="worker-a", now="2026-07-03 00:00:01")
    mark_arxiv_job_failed(
        conn,
        job_key,
        error_message="temporary outage /tmp/private/retry.pdf",
        now="2026-07-03 00:00:02",
        backoff_seconds=60,
    )
    retry = conn.execute(
        "SELECT status, attempts, run_after, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (job_key,),
    ).fetchone()
    assert retry["status"] == "queued"
    assert retry["attempts"] == 1
    assert retry["locked_by"] is None
    assert retry["run_after"] > "2026-07-03 00:00:02"
    assert retry["error_message"] == "temporary outage /tmp/private/retry.pdf"

    claim_next_arxiv_job(conn, worker_id="worker-b", now="2026-07-03 00:02:03")
    mark_arxiv_job_failed(
        conn,
        job_key,
        error_message="permanent outage C:\\private\\terminal.pdf",
        now="2026-07-03 00:02:04",
        backoff_seconds=60,
    )
    terminal = conn.execute(
        "SELECT status, attempts, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (job_key,),
    ).fetchone()
    assert dict(terminal) == {
        "status": "failed",
        "attempts": 2,
        "locked_by": None,
        "error_message": "permanent outage C:\\private\\terminal.pdf",
    }
    events = conn.execute(
        """
        SELECT event_type, arxiv_id, version, asset_type, job_key, payload_json
        FROM arxiv_events
        ORDER BY event_id
        """
    ).fetchall()
    assert [event["event_type"] for event in events] == ["job_failed", "job_failed"]
    first_payload = json.loads(events[0]["payload_json"])
    second_payload = json.loads(events[1]["payload_json"])
    assert dict(events[0]) | {"payload_json": events[0]["payload_json"]} == {
        "event_type": "job_failed",
        "arxiv_id": "2606.29148",
        "version": 1,
        "asset_type": "pdf",
        "job_key": job_key,
        "payload_json": events[0]["payload_json"],
    }
    assert first_payload == {
        "attempts": 1,
        "error_message": "[redacted]",
        "failed_at": "2026-07-03 00:00:02",
        "job_type": "download_pdf",
        "max_attempts": 2,
        "run_after": "2026-07-03 00:01:02",
        "status": "queued",
    }
    assert second_payload == {
        "attempts": 2,
        "error_message": "[redacted]",
        "failed_at": "2026-07-03 00:02:04",
        "job_type": "download_pdf",
        "max_attempts": 2,
        "run_after": "2026-07-03 00:02:04",
        "status": "failed",
    }
    payload_text = "\n".join(event["payload_json"] for event in events)
    assert "/tmp/private" not in payload_text
    assert "\\private\\" not in payload_text

    repeated_terminal = mark_arxiv_job_failed(
        conn,
        job_key,
        error_message="duplicate failure /tmp/private/duplicate.pdf",
        now="2026-07-03 00:02:05",
        backoff_seconds=60,
    )
    assert repeated_terminal is None
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 2
    contradicted_terminal = mark_arxiv_job_succeeded(conn, job_key, now="2026-07-03 00:02:06")
    assert contradicted_terminal is None
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 2


def test_paperstream_missing_job_terminal_helpers_return_none_without_events():
    """Missing job terminal transitions are no-ops and do not append audit rows."""
    from researchfs.paperstream import mark_arxiv_job_failed, mark_arxiv_job_succeeded

    conn = _paperstream_conn()

    assert mark_arxiv_job_succeeded(conn, "missing:2606.29148:v1", now="2026-07-03 00:00:01") is None
    assert (
        mark_arxiv_job_failed(
            conn,
            "missing:2606.29148:v1",
            error_message="network outage",
            now="2026-07-03 00:00:02",
        )
        is None
    )
    assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0


def test_paperstream_enqueue_s3_bin_import_requires_existing_bin_and_is_idempotent():
    """S3 bin import jobs are bin-scoped, durable, and require known manifest rows."""
    from researchfs.paperstream import enqueue_arxiv_s3_bin_import_job

    conn = _paperstream_conn()
    missing_bin_key = "pdf:pdf/arXiv_pdf_2606_missing.tar"
    with pytest.raises(ValueError, match="arxiv_s3_bins"):
        enqueue_arxiv_s3_bin_import_job(conn, missing_bin_key, source="s3://arxiv/pdf/missing.tar")

    first_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_001.tar",
        size_bytes=128,
        checksum="sha256:first",
    )
    second_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="source",
        filename="src/arXiv_src_2606_001.tar",
        size_bytes=256,
        checksum="sha256:second",
    )

    first_source = "s3://arxiv/pdf/arXiv_pdf_2606_001.tar"
    first_job_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        first_bin_key,
        source=first_source,
        priority=7,
        max_attempts=3,
        run_after="2026-07-04T00:00:00Z",
    )
    duplicate_job_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        first_bin_key,
        source="duplicate",
        priority=1,
        max_attempts=1,
        run_after="2026-07-05T00:00:00Z",
    )
    second_job_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        second_bin_key,
        source="unit-test",
        priority=9,
        run_after="2026-07-04T00:00:00Z",
    )

    assert duplicate_job_key == first_job_key
    assert second_job_key != first_job_key
    rows = conn.execute(
        """
        SELECT job_key, job_type, asset_type, status, priority, max_attempts, run_after, payload_json
        FROM arxiv_jobs
        WHERE job_type = 'import_s3_bin'
        ORDER BY job_key
        """
    ).fetchall()
    assert len(rows) == 2
    first_row = next(row for row in rows if row["job_key"] == first_job_key)
    assert dict(first_row) | {"payload_json": None} == {
        "job_key": first_job_key,
        "job_type": "import_s3_bin",
        "asset_type": "pdf",
        "status": "queued",
        "priority": 7,
        "max_attempts": 3,
        "run_after": "2026-07-04T00:00:00Z",
        "payload_json": None,
    }
    first_payload = json.loads(first_row["payload_json"])
    assert first_payload["bin_key"] == first_bin_key
    assert first_payload["source"] == first_source
    assert json.loads(
        next(row for row in rows if row["job_key"] == second_job_key)["payload_json"]
    )["bin_key"] == second_bin_key


def test_paperstream_s3_bin_import_worker_writes_bin_and_enqueues_extract(tmp_path):
    """A claimed S3 bin import stores injected bytes and advances bin/job state."""
    import hashlib

    from researchfs.paperstream import (
        enqueue_arxiv_s3_bin_import_job,
        s3_bin_storage_path,
        work_next_arxiv_s3_bin_import_job,
    )

    conn = _paperstream_conn()
    filename = "pdf/arXiv_pdf_2606_010.tar"
    payload = b"paperstream s3 bin fixture\n"
    checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename=filename,
        size_bytes=len(payload),
        checksum=checksum,
    )
    source = f"s3://arxiv/{filename}"
    import_job_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        bin_key,
        source=source,
        run_after="2026-07-04T00:00:00Z",
    )
    before_event_count = _paperstream_row_count(conn, "arxiv_events")
    fetch_calls = []

    def fetch_fn(source_arg):
        fetch_calls.append(source_arg)
        return payload

    result = work_next_arxiv_s3_bin_import_job(
        conn,
        worker_id="s3-import-worker",
        object_root=tmp_path,
        fetch_fn=fetch_fn,
        now="2026-07-04T00:00:01Z",
    )

    expected_path = Path(s3_bin_storage_path(tmp_path, "pdf", filename))
    assert result["worked"] is True
    assert result["ok"] is True
    assert fetch_calls == [source]
    assert expected_path.read_bytes() == payload
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [expected_path]

    bin_row = conn.execute(
        "SELECT status, storage_path, checksum, size_bytes FROM arxiv_s3_bins WHERE bin_key = ?",
        (bin_key,),
    ).fetchone()
    assert dict(bin_row) == {
        "status": "downloaded",
        "storage_path": str(expected_path),
        "checksum": checksum,
        "size_bytes": len(payload),
    }
    import_job = conn.execute(
        "SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?",
        (import_job_key,),
    ).fetchone()
    assert dict(import_job) == {"status": "succeeded", "locked_by": None}
    extract_jobs = conn.execute(
        "SELECT job_type, asset_type, status, payload_json FROM arxiv_jobs WHERE job_type = 'extract_s3_bin'"
    ).fetchall()
    assert len(extract_jobs) == 1
    assert dict(extract_jobs[0]) | {"payload_json": None} == {
        "job_type": "extract_s3_bin",
        "asset_type": "pdf",
        "status": "queued",
        "payload_json": None,
    }
    assert json.loads(extract_jobs[0]["payload_json"])["bin_key"] == bin_key
    assert _paperstream_row_count(conn, "arxiv_events") > before_event_count
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM arxiv_events WHERE job_key = ?",
            (import_job_key,),
        ).fetchone()[0]
        >= 1
    )


def test_paperstream_s3_bin_import_worker_skips_existing_downloaded_bin(tmp_path):
    """A stale S3 bin import job does not refetch or overwrite durable downloaded bytes."""
    import hashlib

    from researchfs.paperstream import (
        enqueue_arxiv_s3_bin_import_job,
        s3_bin_storage_path,
        work_next_arxiv_s3_bin_import_job,
    )

    conn = _paperstream_conn()
    filename = "src/arXiv_src_2606_011.tar"
    payload = b"already downloaded s3 bin\n"
    checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="source",
        filename=filename,
        size_bytes=len(payload),
        checksum=checksum,
    )
    existing_path = Path(s3_bin_storage_path(tmp_path, "source", filename))
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(payload)
    conn.execute(
        """
        UPDATE arxiv_s3_bins
        SET status = 'downloaded', storage_path = ?, downloaded_at = ?
        WHERE bin_key = ?
        """,
        (str(existing_path), "2026-07-03T00:00:00Z", bin_key),
    )
    import_job_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        bin_key,
        source=f"s3://arxiv/{filename}",
        run_after="2026-07-04T00:00:00Z",
    )

    def fail_fetch(_source):
        raise AssertionError("S3 bin import worker should not refetch downloaded bins")

    result = work_next_arxiv_s3_bin_import_job(
        conn,
        worker_id="s3-import-skip-worker",
        object_root=tmp_path,
        fetch_fn=fail_fetch,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result["skipped"] is True
    assert existing_path.read_bytes() == payload
    bin_row = conn.execute(
        "SELECT status, storage_path, checksum, size_bytes FROM arxiv_s3_bins WHERE bin_key = ?",
        (bin_key,),
    ).fetchone()
    assert dict(bin_row) == {
        "status": "downloaded",
        "storage_path": str(existing_path),
        "checksum": checksum,
        "size_bytes": len(payload),
    }
    job = conn.execute(
        "SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?",
        (import_job_key,),
    ).fetchone()
    assert dict(job) == {"status": "succeeded", "locked_by": None}
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [existing_path]


@pytest.mark.parametrize(
    "manifest_fields,payload,error_fragment",
    [
        (
            {
                "size_bytes": len(b"paperstream bad size bin\n") + 1,
                "checksum": "sha256:"
                + __import__("hashlib").sha256(b"paperstream bad size bin\n").hexdigest(),
            },
            b"paperstream bad size bin\n",
            "size",
        ),
        (
            {
                "size_bytes": len(b"paperstream bad checksum bin\n"),
                "checksum": "sha256:" + ("0" * 64),
            },
            b"paperstream bad checksum bin\n",
            "checksum",
        ),
    ],
)
def test_paperstream_s3_bin_import_worker_fails_mismatched_bin_without_extract(
    tmp_path, manifest_fields, payload, error_fragment
):
    """Size/checksum mismatches fail the bin/job and leave no completed object."""
    from researchfs.paperstream import (
        enqueue_arxiv_s3_bin_import_job,
        s3_bin_storage_path,
        work_next_arxiv_s3_bin_import_job,
    )

    conn = _paperstream_conn()
    filename = f"pdf/arXiv_pdf_2606_{error_fragment}.tar"
    bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename=filename,
        **manifest_fields,
    )
    import_job_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        bin_key,
        source=f"s3://arxiv/{filename}",
        max_attempts=1,
        run_after="2026-07-04T00:00:00Z",
    )

    def fetch_fn(_source):
        return payload

    result = work_next_arxiv_s3_bin_import_job(
        conn,
        worker_id=f"s3-import-{error_fragment}-worker",
        object_root=tmp_path,
        fetch_fn=fetch_fn,
        now="2026-07-04T00:00:01Z",
    )

    expected_path = Path(s3_bin_storage_path(tmp_path, "pdf", filename))
    assert result["worked"] is True
    assert result["ok"] is False
    assert error_fragment in result["error"].lower()
    assert not expected_path.exists()
    bin_row = conn.execute(
        "SELECT status, storage_path FROM arxiv_s3_bins WHERE bin_key = ?",
        (bin_key,),
    ).fetchone()
    assert dict(bin_row) == {"status": "failed", "storage_path": None}
    job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (import_job_key,),
    ).fetchone()
    assert job["status"] == "failed"
    assert job["locked_by"] is None
    assert error_fragment in job["error_message"].lower()
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'extract_s3_bin'"
    ).fetchone()[0] == 0
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM arxiv_events WHERE job_key = ?",
            (import_job_key,),
        ).fetchone()[0]
        >= 1
    )



def test_paperstream_download_pdf_job_writes_asset_and_enqueues_parse(tmp_path):
    """A claimed PDF download stores injected bytes and advances asset/job state."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        work_next_arxiv_download_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    download_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=2,
        asset_type="pdf",
        priority=5,
        payload={"reason": "unit-test"},
        run_after="2026-07-04T00:00:00Z",
    )
    payload = b"%PDF-1.7\nfixture pdf bytes\n"
    requested_urls = []

    def fetch_fn(url):
        requested_urls.append(url)
        return payload

    result = work_next_arxiv_download_job(
        conn,
        worker_id="download-worker",
        object_root=tmp_path,
        fetch_fn=fetch_fn,
        now="2026-07-04T00:00:01Z",
    )

    expected_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "pdf"))
    expected_checksum = "sha256:" + hashlib.sha256(payload).hexdigest()
    assert result["worked"] is True
    assert requested_urls == ["https://arxiv.org/pdf/2606.29148.pdf"]
    assert expected_path.read_bytes() == payload
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [expected_path]

    asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v2:pdf",),
    ).fetchone()
    assert dict(asset) == {
        "status": "downloaded",
        "storage_path": str(expected_path),
        "checksum": expected_checksum,
        "byte_size": len(payload),
        "source": "arxiv",
        "error_message": None,
    }
    jobs = conn.execute(
        "SELECT job_key, job_type, asset_type, status FROM arxiv_jobs ORDER BY job_type"
    ).fetchall()
    assert [dict(row) for row in jobs] == [
        {
            "job_key": download_key,
            "job_type": "download_pdf",
            "asset_type": "pdf",
            "status": "succeeded",
        },
        {
            "job_key": "parse_pdf:2606.29148:v2",
            "job_type": "parse_pdf",
            "asset_type": "pdf",
            "status": "queued",
        },
    ]
    event = conn.execute(
        "SELECT event_type, arxiv_id, version, asset_type, job_key FROM arxiv_events "
        "WHERE event_type = 'asset_downloaded'"
    ).fetchone()
    assert dict(event) == {
        "event_type": "asset_downloaded",
        "arxiv_id": "2606.29148",
        "version": 2,
        "asset_type": "pdf",
        "job_key": download_key,
    }


def test_paperstream_download_source_job_enqueues_parse_source(tmp_path):
    """A successful source download queues source parsing rather than PDF parsing."""
    from researchfs.paperstream import enqueue_arxiv_job, work_next_arxiv_download_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    enqueue_arxiv_job(
        conn,
        job_type="download_source",
        arxiv_id="2606.29148",
        version=1,
        asset_type="source",
        run_after="2026-07-04T00:00:00Z",
    )
    requested_urls = []

    def fetch_fn(url):
        requested_urls.append(url)
        return b"source tarball bytes"


    result = work_next_arxiv_download_job(
        conn,
        worker_id="source-worker",
        object_root=tmp_path,
        fetch_fn=fetch_fn,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert requested_urls == ["https://arxiv.org/src/2606.29148"]
    parse_jobs = conn.execute(
        "SELECT job_key, job_type, asset_type, status FROM arxiv_jobs "
        "WHERE job_type LIKE 'parse_%'"
    ).fetchall()
    assert [dict(row) for row in parse_jobs] == [
        {
            "job_key": "parse_source:2606.29148:v1",
            "job_type": "parse_source",
            "asset_type": "source",
            "status": "queued",
        }
    ]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'parse_pdf'"
    ).fetchone()[0] == 0



def test_paperstream_download_job_skips_preexisting_downloaded_asset(tmp_path):
    """A stale download job does not refetch or overwrite an available asset."""
    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_download_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    existing_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "pdf"))
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"already downloaded")
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="downloaded",
        storage_path=existing_path,
        checksum="sha256:old",
        byte_size=len(b"already downloaded"),
        source="arxiv",
    )
    download_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )

    def fail_fetch(_url):
        raise AssertionError("download worker should not refetch downloaded asset")

    result = work_next_arxiv_download_job(
        conn,
        worker_id="skip-worker",
        object_root=tmp_path,
        fetch_fn=fail_fetch,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result["skipped"] is True
    assert existing_path.read_bytes() == b"already downloaded"
    asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:pdf",),
    ).fetchone()
    assert dict(asset) == {
        "status": "downloaded",
        "storage_path": str(existing_path),
        "checksum": "sha256:old",
        "byte_size": len(b"already downloaded"),
    }
    job = conn.execute("SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?", (download_key,)).fetchone()
    assert dict(job) == {"status": "succeeded", "locked_by": None}
    assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'parse_pdf'").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_download_skipped'"
    ).fetchone()[0] == 1


def test_paperstream_download_default_fetch_sleeps_before_fetch(monkeypatch, tmp_path):
    """Live default arXiv download fetches are rate-limited before network I/O."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    calls = []

    def sleep_fn(seconds):
        calls.append(f"sleep:{seconds}")

    def fetch_url_bytes(url):
        calls.append(f"fetch:{url}")
        return b"%PDF rate limited fixture\n"

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fetch_url_bytes)

    result = paperstream.work_next_arxiv_download_job(
        conn,
        worker_id="rate-limit-default-fetch-worker",
        object_root=tmp_path,
        fetch_fn=None,
        sleep_fn=sleep_fn,
        rate_limit_seconds=1.25,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert calls == ["sleep:1.25", "fetch:https://arxiv.org/pdf/2606.29148.pdf"]


def test_paperstream_download_skip_existing_default_fetch_does_not_sleep_or_fetch(
    monkeypatch, tmp_path
):
    """Already-downloaded default-fetch jobs skip without rate-limiter or network I/O."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    existing_path = Path(paperstream.arxiv_storage_path(tmp_path, "2606.29148", 1, "pdf"))
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"already downloaded")
    paperstream.upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="downloaded",
        storage_path=existing_path,
        checksum="sha256:old",
        byte_size=len(b"already downloaded"),
        source="arxiv",
    )
    download_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    calls = []

    def sleep_fn(seconds):
        calls.append(f"sleep:{seconds}")

    def fetch_url_bytes(url):
        calls.append(f"fetch:{url}")
        raise AssertionError("download worker should not refetch downloaded asset")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fetch_url_bytes)

    result = paperstream.work_next_arxiv_download_job(
        conn,
        worker_id="rate-limit-skip-worker",
        object_root=tmp_path,
        fetch_fn=None,
        sleep_fn=sleep_fn,
        rate_limit_seconds=2.0,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result["skipped"] is True
    assert calls == []
    job = conn.execute(
        "SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?",
        (download_key,),
    ).fetchone()
    assert dict(job) == {"status": "succeeded", "locked_by": None}


def test_paperstream_download_injected_fetch_does_not_sleep(tmp_path):
    """Injected fetch functions remain local no-network hooks and are not rate-limited."""
    from researchfs.paperstream import enqueue_arxiv_job, work_next_arxiv_download_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    requested_urls = []

    def fetch_fn(url):
        requested_urls.append(url)
        return b"%PDF injected fixture\n"

    def fail_sleep(_seconds):
        raise AssertionError("injected fetch_fn should not be rate-limited")

    result = work_next_arxiv_download_job(
        conn,
        worker_id="rate-limit-injected-fetch-worker",
        object_root=tmp_path,
        fetch_fn=fetch_fn,
        sleep_fn=fail_sleep,
        rate_limit_seconds=9.0,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert requested_urls == ["https://arxiv.org/pdf/2606.29148.pdf"]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"rate_limit_seconds": True}, ValueError),
        ({"rate_limit_seconds": -1}, ValueError),
        ({"rate_limit_seconds": float("nan")}, ValueError),
        ({"rate_limit_seconds": 0.25, "sleep_fn": object()}, TypeError),
    ],
)
def test_paperstream_download_invalid_rate_limit_inputs_do_not_claim_job(
    monkeypatch, tmp_path, kwargs, error_type
):
    """Invalid live rate-limit helpers fail before a queued download job is claimed."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )

    def fail_fetch(_url):
        raise AssertionError("invalid rate-limit inputs must fail before fetching")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fail_fetch)

    with pytest.raises(error_type):
        paperstream.work_next_arxiv_download_job(
            conn,
            worker_id="invalid-rate-limit-worker",
            object_root=tmp_path,
            fetch_fn=None,
            now="2026-07-04T00:00:01Z",
            **kwargs,
        )

    job = conn.execute(
        "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
        (download_key,),
    ).fetchone()
    assert dict(job) == {
        "status": "queued",
        "locked_by": None,
        "locked_at": None,
        "attempts": 0,
    }
    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
    assert list(tmp_path.rglob("*")) == []


def test_paperstream_cli_work_rejects_nan_rate_limit_before_claim(tmp_path):
    """CLI non-finite download rate limits fail before claiming queued work."""
    from researchfs import db
    from researchfs.paperstream import enqueue_arxiv_job

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "download_pdf",
            "--daily-download-cap",
            "1",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
            "--rate-limit-seconds",
            "nan",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert "rate_limit_seconds" in rejected.stderr
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute(
            "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
            (download_key,),
        ).fetchone()
        assert dict(job) == {
            "status": "queued",
            "locked_by": None,
            "locked_at": None,
            "attempts": 0,
        }
        assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
    finally:
        conn.close()
    assert not (tmp_path / "objects").exists()

def test_paperstream_download_worker_reports_no_queued_download_job(tmp_path):
    """With no eligible download job, the worker reports no work and mutates nothing."""
    from researchfs.paperstream import enqueue_arxiv_job, work_next_arxiv_download_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    enqueue_arxiv_job(
        conn,
        job_type="parse_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="parsed_text",
        run_after="2026-07-04T00:00:00Z",
    )

    def fail_fetch(_url):
        raise AssertionError("download worker should not fetch without a download job")

    result = work_next_arxiv_download_job(
        conn,
        worker_id="idle-worker",
        object_root=tmp_path,
        fetch_fn=fail_fetch,
        now="2026-07-04T00:00:01Z",
    )

    assert result == {"worked": False}
    assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
    idle_job = conn.execute(
        "SELECT status, locked_by FROM arxiv_jobs WHERE job_type = 'parse_pdf'"
    ).fetchone()
    assert dict(idle_job) == {"status": "queued", "locked_by": None}
    assert list(tmp_path.rglob("*")) == []


def test_paperstream_parse_pdf_job_writes_parsed_text_and_enqueues_chunk(tmp_path):
    """A claimed PDF parse reads the downloaded asset and queues chunking."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_parse_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    pdf_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "pdf"))
    pdf_path.parent.mkdir(parents=True)
    pdf_bytes = b"%PDF-1.7\nlocal parse fixture\n"
    pdf_path.write_bytes(pdf_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "pdf",
        status="downloaded",
        storage_path=pdf_path,
        checksum="sha256:" + hashlib.sha256(pdf_bytes).hexdigest(),
        byte_size=len(pdf_bytes),
        source="arxiv",
    )
    parse_key = enqueue_arxiv_job(
        conn,
        job_type="parse_pdf",
        arxiv_id="2606.29148",
        version=2,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    parser_calls = []

    def pdf_parser_fn(path):
        parser_calls.append(Path(path).read_bytes())
        return "Extracted PDF text from fixture."

    result = work_next_arxiv_parse_job(
        conn,
        worker_id="parse-pdf-worker",
        object_root=tmp_path,
        pdf_parser_fn=pdf_parser_fn,
        now="2026-07-04T00:00:01Z",
        job_types=("parse_pdf",),
    )

    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "parsed_text"))
    parsed_bytes = parsed_path.read_bytes()
    parsed_payload = json.loads(parsed_bytes)
    assert result["worked"] is True
    assert result["ok"] is True
    assert parser_calls == [pdf_bytes]
    assert parsed_payload["arxiv_id"] == "2606.29148"
    assert parsed_payload["version"] == 2
    assert parsed_payload["source_asset_type"] == "pdf"
    assert parsed_payload["text"] == "Extracted PDF text from fixture."

    parsed_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v2:parsed_text",),
    ).fetchone()
    assert dict(parsed_asset) == {
        "status": "parsed",
        "storage_path": str(parsed_path),
        "checksum": "sha256:" + hashlib.sha256(parsed_bytes).hexdigest(),
        "byte_size": len(parsed_bytes),
        "source": "parse_pdf",
        "error_message": None,
    }
    parse_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (parse_key,),
    ).fetchone()
    assert dict(parse_job) == {"status": "succeeded", "locked_by": None, "error_message": None}
    chunk_jobs = conn.execute(
        "SELECT job_type, arxiv_id, version, status FROM arxiv_jobs WHERE job_type = 'chunk_text'"
    ).fetchall()
    assert [dict(row) for row in chunk_jobs] == [
        {
            "job_type": "chunk_text",
            "arxiv_id": "2606.29148",
            "version": 2,
            "status": "queued",
        }
    ]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_parsed'"
    ).fetchone()[0] == 1



def _run_paperstream_source_parse_fixture(tmp_path, source_bytes, *, source_parser_fn=None):
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_parse_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    source_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "source"))
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "source",
        status="downloaded",
        storage_path=source_path,
        checksum="sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        byte_size=len(source_bytes),
        source="arxiv",
    )
    parse_key = enqueue_arxiv_job(
        conn,
        job_type="parse_source",
        arxiv_id="2606.29148",
        version=1,
        asset_type="source",
        max_attempts=1,
        run_after="2026-07-04T00:00:00Z",
    )
    result = work_next_arxiv_parse_job(
        conn,
        worker_id="parse-source-worker",
        object_root=tmp_path,
        source_parser_fn=source_parser_fn,
        now="2026-07-04T00:00:01Z",
        job_types=("parse_source",),
    )
    return conn, source_path, parse_key, result


def test_paperstream_parse_source_job_extracts_tarball_text_and_enqueues_chunk(tmp_path):
    """Source parsing safely extracts local tar.gz TeX bytes before chunking."""
    import hashlib
    import tarfile

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_parse_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    source_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "source"))
    source_path.parent.mkdir(parents=True)
    source_bytes_io = io.BytesIO()
    tex_bytes = b"\\title{Local Tarball Source}\\begin{document}Deterministic source text.\\end{document}"
    with tarfile.open(fileobj=source_bytes_io, mode="w:gz") as archive:
        member = tarfile.TarInfo("paper/main.tex")
        member.size = len(tex_bytes)
        archive.addfile(member, io.BytesIO(tex_bytes))
    source_bytes = source_bytes_io.getvalue()
    source_path.write_bytes(source_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "source",
        status="downloaded",
        storage_path=source_path,
        checksum="sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        byte_size=len(source_bytes),
        source="arxiv",
    )
    enqueue_arxiv_job(
        conn,
        job_type="parse_source",
        arxiv_id="2606.29148",
        version=1,
        asset_type="source",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_parse_job(
        conn,
        worker_id="parse-source-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
        job_types=("parse_source",),
    )

    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "parsed_text"))
    parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    assert result["worked"] is True
    assert result["ok"] is True
    assert parsed_payload["source_asset_type"] == "source"
    assert "Deterministic source text." in parsed_payload["text"]
    parsed_asset = conn.execute(
        "SELECT status, storage_path, source FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:parsed_text",),
    ).fetchone()
    assert dict(parsed_asset) == {
        "status": "parsed",
        "storage_path": str(parsed_path),
        "source": "parse_source",
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'chunk_text' AND status = 'queued'"
    ).fetchone()[0] == 1


def test_paperstream_parse_source_job_extracts_plain_tar_tex_and_bbl_in_order(tmp_path):
    """Plain source tar files include safe TeX and BBL text in deterministic path order."""
    from researchfs.paperstream import arxiv_storage_path

    source_path = _write_paperstream_tar(
        arxiv_storage_path(tmp_path, "2606.29148", 1, "source"),
        {
            "z-last.tex": b"Last TeX text.",
            ".hidden.tex": b"Hidden text must be ignored.",
            "a-main.tex": b"First TeX text.",
            "notes/paper.aux": b"Generated aux text must be ignored.",
            "notes/references.bbl": b"Bibliography text.",
            "__MACOSX/._a-main.tex": b"Generated AppleDouble text must be ignored.",
        },
    )

    conn, _source_path, _parse_key, result = _run_paperstream_source_parse_fixture(
        tmp_path, Path(source_path).read_bytes()
    )

    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "parsed_text"))
    parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    assert result["worked"] is True
    assert result["ok"] is True
    assert parsed_payload["source_asset_type"] == "source"
    assert parsed_payload["text"].find("First TeX text.") < parsed_payload["text"].find(
        "Bibliography text."
    ) < parsed_payload["text"].find("Last TeX text.")
    assert "Hidden text must be ignored." not in parsed_payload["text"]
    assert "Generated aux text must be ignored." not in parsed_payload["text"]
    assert "Generated AppleDouble text must be ignored." not in parsed_payload["text"]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'chunk_text' AND status = 'queued'"
    ).fetchone()[0] == 1


def test_paperstream_parse_source_job_normalizes_gzip_and_raw_tex_bytes(tmp_path):
    """Single-file gzip and raw source bytes normalize CRLF, NULs, and blank runs."""
    import gzip

    gzip_conn, _gzip_source_path, _gzip_parse_key, gzip_result = _run_paperstream_source_parse_fixture(
        tmp_path / "gzip",
        gzip.compress(b"\\section{Intro}\r\nAlpha\x00\r\n\r\n\r\nBeta\r\n"),
    )
    raw_conn, _raw_source_path, _raw_parse_key, raw_result = _run_paperstream_source_parse_fixture(
        tmp_path / "raw",
        b"\\begin{document}\r\nRaw\x00 text\r\n\r\n\r\nMore text\r\n\\end{document}",
    )

    from researchfs.paperstream import arxiv_storage_path

    gzip_payload = json.loads(
        Path(arxiv_storage_path(tmp_path / "gzip", "2606.29148", 1, "parsed_text")).read_text(
            encoding="utf-8"
        )
    )
    raw_payload = json.loads(
        Path(arxiv_storage_path(tmp_path / "raw", "2606.29148", 1, "parsed_text")).read_text(
            encoding="utf-8"
        )
    )
    assert gzip_result["worked"] is True
    assert gzip_result["ok"] is True
    assert raw_result["worked"] is True
    assert raw_result["ok"] is True
    assert gzip_payload["text"] == "\\section{Intro}\nAlpha\n\nBeta"
    assert raw_payload["text"] == "\\begin{document}\nRaw text\n\nMore text\n\\end{document}"
    assert "\x00" not in gzip_payload["text"]
    assert "\r" not in gzip_payload["text"]
    assert "\x00" not in raw_payload["text"]
    assert "\r" not in raw_payload["text"]
    assert gzip_conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'chunk_text' AND status = 'queued'"
    ).fetchone()[0] == 1
    assert raw_conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'chunk_text' AND status = 'queued'"
    ).fetchone()[0] == 1


def test_paperstream_parse_job_missing_source_asset_fails_without_file(tmp_path):
    """Missing parse input records durable failure state without creating text JSON."""
    from researchfs.paperstream import arxiv_storage_path, enqueue_arxiv_job, work_next_arxiv_parse_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    parse_key = enqueue_arxiv_job(
        conn,
        job_type="parse_source",
        arxiv_id="2606.29148",
        version=1,
        asset_type="source",
        max_attempts=1,
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_parse_job(
        conn,
        worker_id="parse-missing-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
        job_types=("parse_source",),
    )

    assert result["worked"] is True
    assert result["ok"] is False
    assert not Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "parsed_text")).exists()
    parse_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (parse_key,),
    ).fetchone()
    assert parse_job["status"] == "failed"
    assert parse_job["locked_by"] is None
    assert parse_job["error_message"]
    parsed_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:parsed_text",),
    ).fetchone()
    assert parsed_asset["status"] == "failed"
    assert parsed_asset["storage_path"] is None
    assert parsed_asset["checksum"] is None
    assert parsed_asset["byte_size"] is None
    assert parsed_asset["source"] == "parse_source"
    assert parsed_asset["error_message"]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_parse_failed'"
    ).fetchone()[0] == 1
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_paperstream_parse_failure_sanitizes_local_paths_in_public_errors(tmp_path):
    """Parser exceptions that mention local object paths are sanitized everywhere public."""
    from researchfs.paperstream import arxiv_storage_path, get_arxiv_paper_state

    source_bytes = b"\\begin{document}broken fixture\\end{document}"
    expected_source_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "source"))

    def failing_source_parser(_data):
        raise RuntimeError(f"cannot parse local source file {expected_source_path}")

    conn, source_path, parse_key, result = _run_paperstream_source_parse_fixture(
        tmp_path,
        source_bytes,
        source_parser_fn=failing_source_parser,
    )
    assert source_path == expected_source_path
    assert result["worked"] is True
    assert result["ok"] is False

    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "parsed_text"))
    assert not parsed_path.exists()

    parse_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (parse_key,),
    ).fetchone()
    parsed_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:parsed_text",),
    ).fetchone()
    event_payload = json.loads(
        conn.execute(
            "SELECT payload_json FROM arxiv_events WHERE event_type = 'asset_parse_failed'"
        ).fetchone()["payload_json"]
    )
    state = get_arxiv_paper_state(conn, "2606.29148")

    assert result["worked"] is True
    assert result["ok"] is False
    assert parse_job["status"] == "failed"
    assert parse_job["locked_by"] is None
    assert parsed_asset["status"] == "failed"
    assert parsed_asset["storage_path"] is None
    assert parsed_asset["checksum"] is None
    assert parsed_asset["byte_size"] is None
    assert parsed_asset["source"] == "parse_source"
    assert state["assets"]["parsed_text"]["status"] == "failed"

    local_path = str(source_path)
    local_root = str(tmp_path)
    error_messages = [
        result["error"],
        result["asset"]["error_message"],
        result["job"]["error_message"],
        parse_job["error_message"],
        parsed_asset["error_message"],
        event_payload["error"],
        state["assets"]["parsed_text"]["error_message"],
    ]
    assert all(error_messages)
    for message in error_messages:
        assert local_path not in message
        assert local_root not in message
    for public_payload in (result, event_payload, state):
        encoded = json.dumps(public_payload, sort_keys=True)
        assert local_path not in encoded
        assert local_root not in encoded
        assert "storage_path" not in encoded


def test_paperstream_parse_job_skips_preexisting_parsed_text_without_overwrite(tmp_path):
    """A stale parse job leaves an existing parsed_text object untouched."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_parse_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    pdf_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "pdf"))
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF available but should not be parsed")
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "pdf",
        status="downloaded",
        storage_path=pdf_path,
        checksum="sha256:pdf",
        byte_size=pdf_path.stat().st_size,
        source="arxiv",
    )
    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "parsed_text"))
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = b'{"text":"already parsed"}\n'
    parsed_path.write_bytes(existing_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "parsed_text",
        status="parsed",
        storage_path=parsed_path,
        checksum="sha256:" + hashlib.sha256(existing_bytes).hexdigest(),
        byte_size=len(existing_bytes),
        source="previous_parse",
    )
    parse_key = enqueue_arxiv_job(
        conn,
        job_type="parse_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )

    def fail_parser(_path):
        raise AssertionError("parser should not run for an existing parsed_text asset")

    result = work_next_arxiv_parse_job(
        conn,
        worker_id="parse-skip-worker",
        object_root=tmp_path,
        pdf_parser_fn=fail_parser,
        now="2026-07-04T00:00:01Z",
        job_types=("parse_pdf",),
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result["skipped"] is True
    assert parsed_path.read_bytes() == existing_bytes
    parsed_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:parsed_text",),
    ).fetchone()
    assert dict(parsed_asset) == {
        "status": "parsed",
        "storage_path": str(parsed_path),
        "checksum": "sha256:" + hashlib.sha256(existing_bytes).hexdigest(),
        "byte_size": len(existing_bytes),
        "source": "previous_parse",
    }
    parse_job = conn.execute("SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?", (parse_key,)).fetchone()
    assert dict(parse_job) == {"status": "succeeded", "locked_by": None}
    assert conn.execute("SELECT COUNT(*) FROM arxiv_jobs WHERE job_type = 'chunk_text'").fetchone()[0] == 0


def test_paperstream_chunk_text_job_writes_chunks_and_enqueues_downstream_jobs(tmp_path):
    """A chunk_text job reads parsed_text JSON, stores chunks, and queues downstream work."""
    import hashlib

    from researchfs.extractor import chunk_text
    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_chunk_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "parsed_text"))
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_payload = {
        "text": "abcdefghi",
        "arxiv_id": "2606.29148",
        "version": 2,
        "source_asset_type": "pdf",
        "created_at": "2026-07-04T00:00:00Z",
    }
    parsed_bytes = json.dumps(parsed_payload, sort_keys=True).encode("utf-8")
    parsed_path.write_bytes(parsed_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "parsed_text",
        status="parsed",
        storage_path=parsed_path,
        checksum="sha256:" + hashlib.sha256(parsed_bytes).hexdigest(),
        byte_size=len(parsed_bytes),
        source="parse_pdf",
    )
    chunk_key = enqueue_arxiv_job(
        conn,
        job_type="chunk_text",
        arxiv_id="2606.29148",
        version=2,
        asset_type="parsed_text",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_chunk_job(
        conn,
        worker_id="chunk-worker",
        object_root=tmp_path,
        chunk_size=4,
        overlap=1,
        now="2026-07-04T00:00:01Z",
    )

    chunks_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "chunks"))
    chunks_bytes = chunks_path.read_bytes()
    chunks_payload = json.loads(chunks_bytes)
    assert result["worked"] is True
    assert result["ok"] is True
    assert chunks_payload["arxiv_id"] == "2606.29148"
    assert chunks_payload["version"] == 2
    assert chunks_payload["source_asset_type"] == "pdf"
    assert chunks_payload["chunk_size"] == 4
    assert chunks_payload["overlap"] == 1
    assert chunks_payload["chunks"] == chunk_text("abcdefghi", chunk_size=4, overlap=1)

    chunks_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v2:chunks",),
    ).fetchone()
    assert dict(chunks_asset) == {
        "status": "parsed",
        "storage_path": str(chunks_path),
        "checksum": "sha256:" + hashlib.sha256(chunks_bytes).hexdigest(),
        "byte_size": len(chunks_bytes),
        "source": "chunk_text",
        "error_message": None,
    }
    chunk_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (chunk_key,),
    ).fetchone()
    assert dict(chunk_job) == {"status": "succeeded", "locked_by": None, "error_message": None}
    downstream_jobs = conn.execute(
        "SELECT job_key, job_type, asset_type, status FROM arxiv_jobs "
        "WHERE job_type IN ('embed_chunks', 'index_bm25') ORDER BY job_type"
    ).fetchall()
    assert [dict(row) for row in downstream_jobs] == [
        {
            "job_key": "embed_chunks:2606.29148:v2",
            "job_type": "embed_chunks",
            "asset_type": "chunks",
            "status": "queued",
        },
        {
            "job_key": "index_bm25:2606.29148:v2",
            "job_type": "index_bm25",
            "asset_type": "chunks",
            "status": "queued",
        },
    ]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_chunked'"
    ).fetchone()[0] == 1


def test_paperstream_chunk_text_job_missing_parsed_text_fails_without_file(tmp_path):
    """Missing parsed_text input records chunks failure without writing an object."""
    from researchfs.paperstream import arxiv_storage_path, enqueue_arxiv_job, work_next_arxiv_chunk_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    chunk_key = enqueue_arxiv_job(
        conn,
        job_type="chunk_text",
        arxiv_id="2606.29148",
        version=1,
        asset_type="parsed_text",
        max_attempts=1,
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_chunk_job(
        conn,
        worker_id="chunk-missing-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is False
    assert not Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "chunks")).exists()
    chunk_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (chunk_key,),
    ).fetchone()
    assert chunk_job["status"] == "failed"
    assert chunk_job["locked_by"] is None
    assert chunk_job["error_message"]
    chunks_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:chunks",),
    ).fetchone()
    assert chunks_asset["status"] == "failed"
    assert chunks_asset["storage_path"] is None
    assert chunks_asset["checksum"] is None
    assert chunks_asset["byte_size"] is None
    assert chunks_asset["source"] == "chunk_text"
    assert chunks_asset["error_message"]
    event_payload = json.loads(
        conn.execute(
            "SELECT payload_json FROM arxiv_events WHERE event_type = 'asset_chunk_failed'"
        ).fetchone()["payload_json"]
    )
    worker_errors = [
        result["error"],
        result["asset"]["error_message"],
        result["job"]["error_message"],
        chunk_job["error_message"],
        chunks_asset["error_message"],
        event_payload["error"],
    ]
    assert all("parsed_text asset" in message for message in worker_errors)
    assert all("parse job failed" not in message for message in worker_errors)
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_paperstream_chunk_text_job_skips_preexisting_chunks_without_overwrite(tmp_path):
    """A stale chunk_text job leaves existing chunks untouched and queues no downstream work."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_chunk_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "parsed_text"))
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_bytes = b'{"text":"parsed text that should not be chunked"}\n'
    parsed_path.write_bytes(parsed_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "parsed_text",
        status="parsed",
        storage_path=parsed_path,
        checksum="sha256:" + hashlib.sha256(parsed_bytes).hexdigest(),
        byte_size=len(parsed_bytes),
        source="parse_pdf",
    )
    chunks_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "chunks"))
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = b'{"chunks":["already chunked"],"arxiv_id":"2606.29148","version":1}\n'
    chunks_path.write_bytes(existing_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "chunks",
        status="parsed",
        storage_path=chunks_path,
        checksum="sha256:" + hashlib.sha256(existing_bytes).hexdigest(),
        byte_size=len(existing_bytes),
        source="previous_chunk",
    )
    chunk_key = enqueue_arxiv_job(
        conn,
        job_type="chunk_text",
        arxiv_id="2606.29148",
        version=1,
        asset_type="parsed_text",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_chunk_job(
        conn,
        worker_id="chunk-skip-worker",
        object_root=tmp_path,
        chunk_size=4,
        overlap=1,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result["skipped"] is True
    assert chunks_path.read_bytes() == existing_bytes
    chunks_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:chunks",),
    ).fetchone()
    assert dict(chunks_asset) == {
        "status": "parsed",
        "storage_path": str(chunks_path),
        "checksum": "sha256:" + hashlib.sha256(existing_bytes).hexdigest(),
        "byte_size": len(existing_bytes),
        "source": "previous_chunk",
    }
    chunk_job = conn.execute("SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?", (chunk_key,)).fetchone()
    assert dict(chunk_job) == {"status": "succeeded", "locked_by": None}
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type IN ('embed_chunks', 'index_bm25')"
    ).fetchone()[0] == 0


def _paperstream_fts_match_count(index_path, term):
    conn = sqlite3.connect(index_path)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND lower(sql) LIKE '%fts5%' ORDER BY name"
        ).fetchall()
        assert tables
        for (table_name,) in tables:
            quoted_table = '"' + table_name.replace('"', '""') + '"'
            try:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_table} MATCH ?",
                    (term,),
                ).fetchone()[0]
            except sqlite3.OperationalError:
                continue
            if count:
                return count
        return 0
    finally:
        conn.close()


def test_paperstream_bm25_job_writes_queryable_fts_asset_and_state(tmp_path):
    """A BM25 job builds a local SQLite FTS artifact and records only public state fields."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        get_arxiv_paper_state,
        upsert_arxiv_asset,
        work_next_arxiv_bm25_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    chunks_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "chunks"))
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_payload = {
        "arxiv_id": "2606.29148",
        "version": 2,
        "source_asset_type": "pdf",
        "chunks": [
            "ordinary retrieval text",
            "The unique quasarionomega marker belongs in the BM25 index.",
        ],
    }
    chunks_bytes = json.dumps(chunks_payload, sort_keys=True).encode("utf-8")
    chunks_path.write_bytes(chunks_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "chunks",
        status="parsed",
        storage_path=chunks_path,
        checksum="sha256:" + hashlib.sha256(chunks_bytes).hexdigest(),
        byte_size=len(chunks_bytes),
        source="chunk_text",
    )
    bm25_job_key = enqueue_arxiv_job(
        conn,
        job_type="index_bm25",
        arxiv_id="2606.29148",
        version=2,
        asset_type="chunks",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_bm25_job(
        conn,
        worker_id="bm25-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
    )

    bm25_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "bm25_index"))
    bm25_bytes = bm25_path.read_bytes()
    expected_checksum = "sha256:" + hashlib.sha256(bm25_bytes).hexdigest()
    assert result["worked"] is True
    assert result["ok"] is True
    assert _paperstream_fts_match_count(bm25_path, "quasarionomega") == 1
    bm25_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v2:bm25_index",),
    ).fetchone()
    assert dict(bm25_asset) == {
        "status": "indexed",
        "storage_path": str(bm25_path),
        "checksum": expected_checksum,
        "byte_size": len(bm25_bytes),
        "source": "index_bm25",
        "error_message": None,
    }
    bm25_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (bm25_job_key,),
    ).fetchone()
    assert dict(bm25_job) == {"status": "succeeded", "locked_by": None, "error_message": None}
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_bm25_indexed'"
    ).fetchone()[0] == 1
    public_state = get_arxiv_paper_state(conn, "2606.29148")
    assert public_state["assets"]["bm25_index"]["status"] == "indexed"
    assert public_state["assets"]["bm25_index"]["byte_size"] == len(bm25_bytes)
    _assert_paperstream_public_payload_has_no_local_asset_fields(public_state)


def test_paperstream_bm25_job_missing_chunks_asset_fails_without_artifact(tmp_path):
    """Missing BM25 input records durable failure state without creating an index file."""
    from researchfs.paperstream import arxiv_storage_path, enqueue_arxiv_job, work_next_arxiv_bm25_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    bm25_job_key = enqueue_arxiv_job(
        conn,
        job_type="index_bm25",
        arxiv_id="2606.29148",
        version=1,
        asset_type="chunks",
        max_attempts=1,
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_bm25_job(
        conn,
        worker_id="bm25-missing-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is False
    assert not Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "bm25_index")).exists()
    bm25_job = conn.execute(
        "SELECT status, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (bm25_job_key,),
    ).fetchone()
    assert bm25_job["status"] == "failed"
    assert bm25_job["locked_by"] is None
    assert bm25_job["error_message"]
    bm25_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:bm25_index",),
    ).fetchone()
    assert bm25_asset["status"] == "failed"
    assert bm25_asset["storage_path"] is None
    assert bm25_asset["checksum"] is None
    assert bm25_asset["byte_size"] is None
    assert bm25_asset["source"] == "index_bm25"
    assert bm25_asset["error_message"]
    event_payload = json.loads(
        conn.execute(
            "SELECT payload_json FROM arxiv_events WHERE event_type = 'asset_bm25_failed'"
        ).fetchone()["payload_json"]
    )
    worker_errors = [
        result["error"],
        result["asset"]["error_message"],
        result["job"]["error_message"],
        bm25_job["error_message"],
        bm25_asset["error_message"],
        event_payload["error"],
    ]
    assert all("chunks asset" in message for message in worker_errors)
    assert all("parse job failed" not in message for message in worker_errors)
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_paperstream_bm25_job_skips_preexisting_index_without_overwrite(tmp_path):
    """A stale BM25 job leaves an indexed artifact untouched and marks the job succeeded."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_bm25_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    bm25_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "bm25_index"))
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = b"existing index bytes should not be overwritten\n"
    bm25_path.write_bytes(existing_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "bm25_index",
        status="indexed",
        storage_path=bm25_path,
        checksum="sha256:" + hashlib.sha256(existing_bytes).hexdigest(),
        byte_size=len(existing_bytes),
        source="previous_index_bm25",
    )
    bm25_job_key = enqueue_arxiv_job(
        conn,
        job_type="index_bm25",
        arxiv_id="2606.29148",
        version=1,
        asset_type="chunks",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_bm25_job(
        conn,
        worker_id="bm25-skip-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result["skipped"] is True
    assert bm25_path.read_bytes() == existing_bytes
    bm25_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:bm25_index",),
    ).fetchone()
    assert dict(bm25_asset) == {
        "status": "indexed",
        "storage_path": str(bm25_path),
        "checksum": "sha256:" + hashlib.sha256(existing_bytes).hexdigest(),
        "byte_size": len(existing_bytes),
        "source": "previous_index_bm25",
    }
    bm25_job = conn.execute("SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?", (bm25_job_key,)).fetchone()
    assert dict(bm25_job) == {"status": "succeeded", "locked_by": None}
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_bm25_skipped'"
    ).fetchone()[0] == 1


def test_paperstream_bm25_job_rebuilds_missing_preexisting_index(tmp_path):
    """An indexed DB row with a missing artifact does not permanently skip rebuilds."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_bm25_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    chunks_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "chunks"))
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_payload = {
        "arxiv_id": "2606.29148",
        "version": 1,
        "source_asset_type": "pdf",
        "chunks": ["The rebuilt berylliumfalcon term must be searchable."],
    }
    chunks_bytes = json.dumps(chunks_payload, sort_keys=True).encode("utf-8")
    chunks_path.write_bytes(chunks_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "chunks",
        status="parsed",
        storage_path=chunks_path,
        checksum="sha256:" + hashlib.sha256(chunks_bytes).hexdigest(),
        byte_size=len(chunks_bytes),
        source="chunk_text",
    )
    bm25_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "bm25_index"))
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "bm25_index",
        status="indexed",
        storage_path=bm25_path,
        checksum="sha256:missing",
        byte_size=123,
        source="previous_index_bm25",
    )
    enqueue_arxiv_job(
        conn,
        job_type="index_bm25",
        arxiv_id="2606.29148",
        version=1,
        asset_type="chunks",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_bm25_job(
        conn,
        worker_id="bm25-rebuild-worker",
        object_root=tmp_path,
        now="2026-07-04T00:00:01Z",
    )

    assert result["worked"] is True
    assert result["ok"] is True
    assert result.get("skipped") is not True
    assert bm25_path.is_file()
    assert _paperstream_fts_match_count(bm25_path, "berylliumfalcon") == 1
    bm25_asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source FROM arxiv_assets WHERE asset_key = ?",
        ("2606.29148:v1:bm25_index",),
    ).fetchone()
    bm25_bytes = bm25_path.read_bytes()
    assert dict(bm25_asset) == {
        "status": "indexed",
        "storage_path": str(bm25_path),
        "checksum": "sha256:" + hashlib.sha256(bm25_bytes).hexdigest(),
        "byte_size": len(bm25_bytes),
        "source": "index_bm25",
    }
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_bm25_indexed'"
    ).fetchone()[0] == 1


def _update_paperstream_search_metadata(
    conn,
    arxiv_id,
    *,
    version=1,
    title,
    abstract,
    authors,
    category,
    categories=None,
):
    categories = categories or [category]
    authors_json = json.dumps(authors)
    categories_json = json.dumps(categories)
    conn.execute(
        """
        UPDATE arxiv_papers
        SET latest_version = ?,
            title = ?,
            abstract = ?,
            authors_json = ?,
            primary_category = ?,
            categories_json = ?
        WHERE arxiv_id = ?
        """,
        (version, title, abstract, authors_json, category, categories_json, arxiv_id),
    )
    conn.execute(
        """
        UPDATE arxiv_versions
        SET title = ?,
            abstract = ?,
            authors_json = ?,
            categories_json = ?
        WHERE arxiv_id = ? AND version = ?
        """,
        (title, abstract, authors_json, categories_json, arxiv_id, version),
    )


def _create_paperstream_bm25_fixture(
    conn,
    object_root,
    *,
    arxiv_id="2606.29148",
    version=1,
    chunks=None,
):
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        upsert_arxiv_asset,
        work_next_arxiv_bm25_job,
    )

    chunks = chunks or ["deterministic PaperStream search chunk"]
    chunks_path = Path(arxiv_storage_path(object_root, arxiv_id, version, "chunks"))
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_payload = {
        "arxiv_id": arxiv_id,
        "version": version,
        "source_asset_type": "pdf",
        "chunks": chunks,
    }
    chunks_bytes = json.dumps(chunks_payload, sort_keys=True).encode("utf-8")
    chunks_path.write_bytes(chunks_bytes)
    upsert_arxiv_asset(
        conn,
        arxiv_id,
        version,
        "chunks",
        status="parsed",
        storage_path=chunks_path,
        checksum="sha256:" + hashlib.sha256(chunks_bytes).hexdigest(),
        byte_size=len(chunks_bytes),
        source="chunk_text",
    )
    enqueue_arxiv_job(
        conn,
        job_type="index_bm25",
        arxiv_id=arxiv_id,
        version=version,
        asset_type="chunks",
        run_after="2026-07-04T00:00:00Z",
    )
    result = work_next_arxiv_bm25_job(
        conn,
        worker_id="search-bm25-worker",
        object_root=object_root,
        now="2026-07-04T00:00:01Z",
    )
    assert result["worked"] is True
    assert result["ok"] is True
    return Path(arxiv_storage_path(object_root, arxiv_id, version, "bm25_index"))

def _write_paperstream_chunks_asset(
    conn,
    object_root,
    *,
    arxiv_id="2606.29148",
    version=1,
    chunks=None,
    status="parsed",
    source="chunk_text",
):
    import hashlib

    from researchfs.paperstream import arxiv_storage_path, upsert_arxiv_asset

    chunks = chunks or ["deterministic PaperStream materialized chunk"]
    chunks_path = Path(arxiv_storage_path(object_root, arxiv_id, version, "chunks"))
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_payload = {
        "arxiv_id": arxiv_id,
        "version": version,
        "asset_type": "chunks",
        "source_asset_type": "pdf",
        "chunk_size": 128,
        "overlap": 0,
        "chunks": chunks,
    }
    chunks_bytes = json.dumps(chunks_payload, sort_keys=True).encode("utf-8")
    chunks_path.write_bytes(chunks_bytes)
    upsert_arxiv_asset(
        conn,
        arxiv_id,
        version,
        "chunks",
        status=status,
        storage_path=chunks_path,
        checksum="sha256:" + hashlib.sha256(chunks_bytes).hexdigest(),
        byte_size=len(chunks_bytes),
        source=source,
    )
    return chunks_path


def _paperstream_materialized_chunks(conn, arxiv_id, version):
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT chunk_index, text
            FROM arxiv_chunks
            WHERE arxiv_id = ? AND version = ?
            ORDER BY chunk_index
            """,
            (arxiv_id, version),
        ).fetchall()
    ]


def _assert_paperstream_payload_has_no_storage_path_leak(payload, *forbidden_fragments):
    encoded = json.dumps(payload, sort_keys=True)
    assert "storage_path" not in encoded
    for fragment in forbidden_fragments:
        assert str(fragment) not in encoded



def test_paperstream_paper_chunks_returns_latest_materialized_public_payload(tmp_path):
    """Chunk retrieval reads latest materialized rows, not local chunk artifacts."""
    from researchfs.paperstream import get_arxiv_paper_chunks, materialize_paperstream_chunks

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.IR")
    _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["old version chunk must not be returned"],
    )
    latest_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=2,
        chunks=[
            "latest alpha chunk with enough text to truncate",
            "latest beta chunk",
        ],
    )
    materialized = materialize_paperstream_chunks(conn, limit=10)
    latest_chunks_path.unlink()

    payload = get_arxiv_paper_chunks(
        conn,
        "2606.29148",
        limit=1,
        offset=0,
        max_chars=18,
    )

    assert materialized["ok"] is True
    assert payload["arxiv_id"] == "2606.29148"
    assert payload["version"] == 2
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["links"] == {
        "arxiv_abs": "https://arxiv.org/abs/2606.29148",
        "arxiv_pdf": "https://arxiv.org/pdf/2606.29148.pdf",
        "arxiv_source": "https://arxiv.org/e-print/2606.29148",
    }
    _assert_paperstream_payload_has_arxiv_attribution(payload)
    assert payload["chunks"] == [
        {
            "chunk_index": 0,
            "text": "latest alpha chunk",
            "source_asset_type": "pdf",
            "chunks_asset_key": "2606.29148:v2:chunks",
            "chunks_checksum": payload["chunks"][0]["chunks_checksum"],
        }
    ]
    assert payload["chunks"][0]["chunks_checksum"].startswith("sha256:")
    _assert_paperstream_payload_has_no_storage_path_leak(payload, latest_chunks_path, tmp_path)


def test_paperstream_paper_chunks_version_offset_and_zero_max_chars_are_deterministic(tmp_path):
    """Explicit versions, pagination offsets, and zero-width text bounds are stable."""
    from researchfs.paperstream import get_arxiv_paper_chunks, materialize_paperstream_chunks

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.IR")
    old_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["v1 zero chars text", "v1 offset text"],
    )
    _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=2,
        chunks=["v2 should not be selected"],
    )
    materialize_paperstream_chunks(conn, limit=10)
    old_chunks_path.write_text("artifact should not be reopened", encoding="utf-8")

    payload = get_arxiv_paper_chunks(
        conn,
        "2606.29148",
        version=1,
        limit=5,
        offset=1,
        max_chars=0,
    )

    assert payload["arxiv_id"] == "2606.29148"
    assert payload["version"] == 1
    assert payload["total"] == 2
    assert payload["limit"] == 5
    assert payload["offset"] == 1
    assert payload["chunks"] == [
        {
            "chunk_index": 1,
            "text": "",
            "source_asset_type": "pdf",
            "chunks_asset_key": "2606.29148:v1:chunks",
            "chunks_checksum": payload["chunks"][0]["chunks_checksum"],
        }
    ]
    assert payload["chunks"][0]["chunks_checksum"].startswith("sha256:")
    assert get_arxiv_paper_chunks(conn, "2606.29148", version=3) is None
    assert get_arxiv_paper_chunks(conn, "2606.00000") is None
    _assert_paperstream_payload_has_no_storage_path_leak(payload, old_chunks_path, tmp_path)


class _PaperstreamNoSqliteAccess:
    def execute(self, *_args, **_kwargs):
        raise AssertionError("SQLite should not be accessed for invalid chunk arguments")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"arxiv_id": ""}, "arxiv_id"),
        ({"arxiv_id": "2606.29148v2"}, "arxiv_id"),
        ({"arxiv_id": "not an id"}, "arxiv_id"),
        ({"version": True}, "version"),
        ({"version": 0}, "version"),
        ({"version": "v1"}, "version"),
        ({"limit": True}, "limit"),
        ({"limit": 0}, "limit"),
        ({"limit": 501}, "limit"),
        ({"offset": True}, "offset"),
        ({"offset": -1}, "offset"),
        ({"max_chars": True}, "max_chars"),
        ({"max_chars": -1}, "max_chars"),
        ({"max_chars": 20001}, "max_chars"),
    ],
)
def test_paperstream_paper_chunks_invalid_args_fail_before_sqlite(kwargs, message):
    """Invalid chunk retrieval filters are rejected before broad SQLite reads."""
    from researchfs.paperstream import get_arxiv_paper_chunks

    arxiv_id = kwargs.pop("arxiv_id", "2606.29148")
    with pytest.raises((TypeError, ValueError), match=message):
        get_arxiv_paper_chunks(_PaperstreamNoSqliteAccess(), arxiv_id, **kwargs)


def test_paperstream_chunk_state_counts_after_materialization_and_bm25_rebuild(tmp_path):
    """Paper state reports latest chunk and BM25 readiness without local paths."""
    from researchfs.paperstream import (
        get_arxiv_paper_state,
        materialize_paperstream_chunks,
        rebuild_paperstream_bm25_indexes,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.IR")
    old_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["old materialized chunk"],
    )
    latest_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=2,
        chunks=["first latest chunk", "second latest chunk"],
    )
    materialized = materialize_paperstream_chunks(conn, limit=10)
    rebuilt = rebuild_paperstream_bm25_indexes(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=2,
        limit=10,
    )

    state = get_arxiv_paper_state(conn, "2606.29148")

    assert materialized["ok"] is True
    assert rebuilt["ok"] is True
    assert state["latest_version"] == 2
    assert state["chunks"] == {
        "total": 3,
        "materialized_versions": [1, 2],
        "latest_version_count": 2,
        "by_version": {"1": 1, "2": 2},
    }
    assert state["indexes"] == {
        "bm25_indexed": True,
        "materialized_chunks": True,
    }
    _assert_paperstream_public_payload_has_no_local_asset_fields(state)
    _assert_paperstream_payload_has_no_storage_path_leak(
        state,
        old_chunks_path,
        latest_chunks_path,
        tmp_path,
    )


def test_paperstream_cli_paper_chunks_prints_json_and_honors_bounds(tmp_path):
    """The paper-chunks CLI prints bounded JSON from materialized rows."""
    from researchfs import db
    from researchfs.paperstream import materialize_paperstream_chunks

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["alpha cli chunk", "bravo cli chunk"],
    )
    materialized = materialize_paperstream_chunks(conn, limit=10)
    chunks_path.unlink()
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "paper-chunks",
            "2606.29148",
            "--limit",
            "1",
            "--offset",
            "1",
            "--max-chars",
            "5",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert materialized["ok"] is True
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["arxiv_id"] == "2606.29148"
    assert payload["version"] == 1
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 1
    assert payload["chunks"] == [
        {
            "chunk_index": 1,
            "text": "bravo",
            "source_asset_type": "pdf",
            "chunks_asset_key": "2606.29148:v1:chunks",
            "chunks_checksum": payload["chunks"][0]["chunks_checksum"],
        }
    ]
    assert payload["chunks"][0]["chunks_checksum"].startswith("sha256:")
    _assert_paperstream_payload_has_no_storage_path_leak(payload, chunks_path, tmp_path)

def test_paperstream_materialize_chunks_asset_populates_rows_and_searches_public(tmp_path):
    """Existing chunks JSON assets can be materialized and searched without leaking paths."""
    from researchfs.paperstream import materialize_paperstream_chunks, search_paperstream

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=[
            "ordinary materialization text",
            "zephyrmango appears only in the materialized PaperStream chunk",
        ],
    )
    assert _paperstream_row_count(conn, "arxiv_chunks") == 0

    result = materialize_paperstream_chunks(conn, limit=10)

    assert result["ok"] is True
    assert result["processed"] == 1
    assert result["materialized"] == 1
    assert result["errors"] == []
    assert _paperstream_materialized_chunks(conn, "2606.29148", 1) == [
        {"chunk_index": 0, "text": "ordinary materialization text"},
        {"chunk_index": 1, "text": "zephyrmango appears only in the materialized PaperStream chunk"},
    ]
    search_result = search_paperstream(conn, "zephyrmango", limit=3, max_chunk_chars=48)
    assert search_result["ok"] is True
    _assert_paperstream_payload_has_arxiv_attribution(search_result)
    assert search_result["errors"] == []
    assert search_result["chunks"]
    assert search_result["chunks"][0]["arxiv_id"] == "2606.29148"
    assert search_result["chunks"][0]["version"] == 1
    assert search_result["chunks"][0]["chunk_index"] == 1
    assert search_result["chunks"][0]["text"].startswith("zephyrmango")
    assert len(search_result["chunks"][0]["text"]) <= 48
    _assert_paperstream_payload_has_no_storage_path_leak(result, chunks_path, tmp_path)
    _assert_paperstream_search_payload_is_public(search_result, chunks_path, tmp_path)


def test_paperstream_chunk_text_job_writes_json_and_materializes_rows_idempotently(tmp_path):
    """chunk_text writes chunks JSON and arxiv_chunks rows; later materialization is idempotent."""
    import hashlib

    from researchfs.paperstream import (
        arxiv_storage_path,
        enqueue_arxiv_job,
        materialize_paperstream_chunks,
        upsert_arxiv_asset,
        work_next_arxiv_chunk_job,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2)
    parsed_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "parsed_text"))
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_payload = {
        "text": "alphariver betariver gammariver deltariver",
        "arxiv_id": "2606.29148",
        "version": 2,
        "source_asset_type": "pdf",
    }
    parsed_bytes = json.dumps(parsed_payload, sort_keys=True).encode("utf-8")
    parsed_path.write_bytes(parsed_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        2,
        "parsed_text",
        status="parsed",
        storage_path=parsed_path,
        checksum="sha256:" + hashlib.sha256(parsed_bytes).hexdigest(),
        byte_size=len(parsed_bytes),
        source="parse_pdf",
    )
    enqueue_arxiv_job(
        conn,
        job_type="chunk_text",
        arxiv_id="2606.29148",
        version=2,
        asset_type="parsed_text",
        run_after="2026-07-04T00:00:00Z",
    )

    result = work_next_arxiv_chunk_job(
        conn,
        worker_id="chunk-materialize-worker",
        object_root=tmp_path,
        chunk_size=12,
        overlap=0,
        now="2026-07-04T00:00:01Z",
    )

    chunks_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 2, "chunks"))
    chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    expected_rows = [
        {"chunk_index": index, "text": text}
        for index, text in enumerate(chunks_payload["chunks"])
    ]
    assert result["worked"] is True
    assert result["ok"] is True
    assert chunks_payload["arxiv_id"] == "2606.29148"
    assert chunks_payload["version"] == 2
    assert _paperstream_materialized_chunks(conn, "2606.29148", 2) == expected_rows

    before_count = _paperstream_row_count(conn, "arxiv_chunks")
    repeat = materialize_paperstream_chunks(
        conn,
        limit=10,
        arxiv_id="2606.29148",
        version=2,
    )
    forced = materialize_paperstream_chunks(
        conn,
        limit=10,
        arxiv_id="2606.29148",
        version=2,
        force=True,
    )

    assert repeat["ok"] is True
    assert forced["ok"] is True
    assert repeat["errors"] == []
    assert forced["errors"] == []
    assert _paperstream_row_count(conn, "arxiv_chunks") == before_count
    assert _paperstream_materialized_chunks(conn, "2606.29148", 2) == expected_rows
    _assert_paperstream_payload_has_no_storage_path_leak(repeat, chunks_path, tmp_path)
    _assert_paperstream_payload_has_no_storage_path_leak(forced, chunks_path, tmp_path)


def test_paperstream_rebuild_bm25_indexes_creates_skips_and_forces_public_summary(tmp_path):
    """Bounded BM25 rebuilds create missing indexes, skip valid ones, and force replacement."""
    from researchfs.paperstream import (
        arxiv_storage_path,
        rebuild_paperstream_bm25_indexes,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29149", version=2, category="cs.IR")
    first_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["zirconotter should be indexed by the rebuild helper"],
    )
    existing_bm25_path = _create_paperstream_bm25_fixture(
        conn,
        tmp_path,
        arxiv_id="2606.29149",
        version=2,
        chunks=["oldskipterm lives in the already valid BM25 artifact"],
    )
    existing_before = existing_bm25_path.read_bytes()

    result = rebuild_paperstream_bm25_indexes(conn, tmp_path, limit=10)

    first_bm25_path = Path(arxiv_storage_path(tmp_path, "2606.29148", 1, "bm25_index"))
    assert result["ok"] is True
    assert result["processed"] == 2
    assert result["indexed"] == 1
    assert result["skipped"] == 1
    assert result["errors"] == []
    assert first_bm25_path.is_file()
    assert _paperstream_fts_match_count(first_bm25_path, "zirconotter") == 1
    assert existing_bm25_path.read_bytes() == existing_before
    assert _paperstream_fts_match_count(existing_bm25_path, "oldskipterm") == 1
    assert _paperstream_fts_match_count(existing_bm25_path, "newforcelemur") == 0
    _assert_paperstream_payload_has_no_storage_path_leak(
        result,
        first_chunks_path,
        first_bm25_path,
        existing_bm25_path,
        tmp_path,
    )

    second_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29149",
        version=2,
        chunks=["newforcelemur should appear only after forced rebuild"],
        source="updated_chunk_text",
    )

    forced = rebuild_paperstream_bm25_indexes(
        conn,
        tmp_path,
        limit=10,
        arxiv_id="2606.29149",
        version=2,
        force=True,
    )

    assert forced["ok"] is True
    assert forced["processed"] == 1
    assert forced["indexed"] == 1
    assert forced["skipped"] == 0
    assert forced["errors"] == []
    assert _paperstream_fts_match_count(existing_bm25_path, "newforcelemur") == 1
    assert _paperstream_fts_match_count(existing_bm25_path, "oldskipterm") == 0
    _assert_paperstream_payload_has_no_storage_path_leak(
        forced,
        second_chunks_path,
        existing_bm25_path,
        tmp_path,
    )


def test_paperstream_materialize_invalid_chunks_reports_error_without_mutating_good_state(tmp_path):
    """Invalid chunks JSON is reported without deleting good rows or touching other assets."""
    import hashlib

    from researchfs.paperstream import materialize_paperstream_chunks, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29149", version=1, category="cs.IR")
    good_chunks_path = _write_paperstream_chunks_asset(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["goodrowterm remains materialized"],
    )
    good_result = materialize_paperstream_chunks(conn, limit=10)
    assert good_result["ok"] is True
    good_rows = _paperstream_materialized_chunks(conn, "2606.29148", 1)
    invalid_chunks_path = Path(tmp_path / "private-invalid-chunks.json")
    invalid_chunks_path.write_text("{not valid json", encoding="utf-8")
    bad_bytes = invalid_chunks_path.read_bytes()
    upsert_arxiv_asset(
        conn,
        "2606.29149",
        1,
        "chunks",
        status="parsed",
        storage_path=invalid_chunks_path,
        checksum="sha256:" + hashlib.sha256(bad_bytes).hexdigest(),
        byte_size=len(bad_bytes),
        source="chunk_text",
    )
    unrelated_path = tmp_path / "private-unrelated.pdf"
    unrelated_bytes = b"%PDF unrelated asset state must not change\n"
    unrelated_path.write_bytes(unrelated_bytes)
    upsert_arxiv_asset(
        conn,
        "2606.29149",
        1,
        "pdf",
        status="downloaded",
        storage_path=unrelated_path,
        checksum="sha256:" + hashlib.sha256(unrelated_bytes).hexdigest(),
        byte_size=len(unrelated_bytes),
        source="download_pdf",
    )
    before_bad_asset = dict(
        conn.execute(
            "SELECT status, storage_path, checksum, byte_size, source, error_message "
            "FROM arxiv_assets WHERE asset_key = ?",
            ("2606.29149:v1:chunks",),
        ).fetchone()
    )
    before_unrelated = dict(
        conn.execute(
            "SELECT status, storage_path, checksum, byte_size, source, error_message "
            "FROM arxiv_assets WHERE asset_key = ?",
            ("2606.29149:v1:pdf",),
        ).fetchone()
    )

    result = materialize_paperstream_chunks(conn, limit=10)

    assert result["errors"]
    assert "2606.29149:v1:chunks" in json.dumps(result["errors"], sort_keys=True)
    assert "unreadable_chunks_asset" in json.dumps(result["errors"], sort_keys=True)
    assert _paperstream_materialized_chunks(conn, "2606.29148", 1) == good_rows
    assert _paperstream_materialized_chunks(conn, "2606.29149", 1) == []
    assert dict(
        conn.execute(
            "SELECT status, storage_path, checksum, byte_size, source, error_message "
            "FROM arxiv_assets WHERE asset_key = ?",
            ("2606.29149:v1:chunks",),
        ).fetchone()
    ) == before_bad_asset
    assert dict(
        conn.execute(
            "SELECT status, storage_path, checksum, byte_size, source, error_message "
            "FROM arxiv_assets WHERE asset_key = ?",
            ("2606.29149:v1:pdf",),
        ).fetchone()
    ) == before_unrelated
    _assert_paperstream_payload_has_no_storage_path_leak(
        result,
        good_chunks_path,
        invalid_chunks_path,
        unrelated_path,
        tmp_path,
    )


def test_paperstream_cli_rebuild_indexes_prints_json_and_honors_limit_filters(tmp_path):
    """The rebuild-indexes CLI is bounded, filterable, JSON-safe, and path-silent."""
    from researchfs import db
    from researchfs.paperstream import arxiv_storage_path

    db_path = tmp_path / "paperstream.db"
    object_root = tmp_path / "objects"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _insert_paperstream_paper(conn, "2606.29149", version=2, category="cs.IR")
    _write_paperstream_chunks_asset(
        conn,
        object_root,
        arxiv_id="2606.29148",
        version=1,
        chunks=["clirebuildone marker for the first bounded CLI rebuild"],
    )
    _write_paperstream_chunks_asset(
        conn,
        object_root,
        arxiv_id="2606.29149",
        version=2,
        chunks=["clirebuildtwo marker for the filtered CLI rebuild"],
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "rebuild-indexes",
            "--object-root",
            str(object_root),
            "--limit",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    first_payload = json.loads(first.stdout)
    assert first_payload["ok"] is True
    assert first_payload["limit"] == 1
    assert first_payload["processed"] == 1
    assert first_payload["indexed"] == 1
    _assert_paperstream_payload_has_no_storage_path_leak(first_payload, db_path, object_root, tmp_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        indexed = [
            (row["arxiv_id"], row["version"])
            for row in conn.execute(
                "SELECT arxiv_id, version FROM arxiv_assets "
                "WHERE asset_type = 'bm25_index' AND status = 'indexed' "
                "ORDER BY arxiv_id, version"
            ).fetchall()
        ]
        assert len(indexed) == 1
    finally:
        conn.close()
    missing = ("2606.29149", 2) if indexed[0] == ("2606.29148", 1) else ("2606.29148", 1)

    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "rebuild-indexes",
            "--object-root",
            str(object_root),
            "--limit",
            "5",
            "--arxiv-id",
            missing[0],
            "--version",
            str(missing[1]),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["ok"] is True
    assert second_payload["limit"] == 5
    assert second_payload["processed"] == 1
    assert second_payload["indexed"] == 1
    _assert_paperstream_payload_has_no_storage_path_leak(second_payload, db_path, object_root, tmp_path)

    first_bm25_path = Path(arxiv_storage_path(object_root, "2606.29148", 1, "bm25_index"))
    second_bm25_path = Path(arxiv_storage_path(object_root, "2606.29149", 2, "bm25_index"))
    assert _paperstream_fts_match_count(first_bm25_path, "clirebuildone") == 1
    assert _paperstream_fts_match_count(second_bm25_path, "clirebuildtwo") == 1


def _assert_paperstream_search_payload_is_public(payload, *forbidden_fragments):
    _assert_paperstream_public_payload_has_no_local_asset_fields(payload)
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in forbidden_fragments:
        assert str(fragment) not in encoded


def test_paperstream_search_metadata_only_matches_title_abstract_category_and_author():
    """Metadata-only search finds papers across public metadata without leaking asset paths."""
    from researchfs.paperstream import search_paperstream, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=3, category="astro-ph.IM")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        version=3,
        title="Neuralquokka Calibration for Local Archives",
        abstract="The abstractlemur protocol keeps PaperStream search deterministic.",
        authors=["Katherine Johnson", "Alan Turing"],
        category="astro-ph.IM",
        categories=["astro-ph.IM", "cs.IR"],
    )
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        3,
        "pdf",
        status="downloaded",
        storage_path="/private/paperstream/search/metadata.pdf",
        checksum="sha256:metadata-private-path",
        byte_size=128,
        source="download_pdf",
    )
    _insert_paperstream_paper(conn, "2606.29149", version=1, category="cs.AI")

    cases = {
        "title": ("neuralquokka", {"title"}),
        "abstract": ("abstractlemur", {"abstract"}),
        "category": ("astro-ph.IM", {"categories", "primary_category"}),
        "authors": ("Katherine Johnson", {"authors"}),
    }
    for _case_name, (query, expected_sources) in cases.items():
        result = search_paperstream(conn, query, limit=3, include_chunks=False)

        assert result["ok"] is True
        assert result["query"] == query
        assert result["limit"] == 3
        assert result["chunks"] == []
        assert result["errors"] == []
        _assert_paperstream_payload_has_arxiv_attribution(result)
        matches = [paper for paper in result["papers"] if paper["arxiv_id"] == "2606.29148"]
        assert matches
        paper = matches[0]
        assert paper["latest_version"] == 3
        assert paper["title"] == "Neuralquokka Calibration for Local Archives"
        assert paper["abstract"] == "The abstractlemur protocol keeps PaperStream search deterministic."
        assert paper["primary_category"] == "astro-ph.IM"
        assert paper["categories"] == ["astro-ph.IM", "cs.IR"]
        assert paper["metadata_score"] > 0
        assert expected_sources.intersection(paper["match_sources"])
        _assert_paperstream_search_payload_is_public(result, "/private/paperstream/search")


def test_paperstream_search_bm25_chunks_are_bounded_and_public(tmp_path):
    """BM25 search returns matching chunks with paper metadata and bounded public text."""
    from researchfs.paperstream import search_paperstream

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=2, category="cs.IR")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        version=2,
        title="PaperStream Chunk Retrieval Fixture",
        abstract="Metadata is intentionally quiet for this chunk search.",
        authors=["Grace Hopper"],
        category="cs.IR",
        categories=["cs.IR"],
    )
    bm25_path = _create_paperstream_bm25_fixture(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=2,
        chunks=[
            "ordinary unrelated chunk",
            "quasarionomega starts this chunk and the remaining text must be truncated away",
        ],
    )

    result = search_paperstream(conn, "quasarionomega", limit=5, max_chunk_chars=32)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["chunks"]
    chunk = result["chunks"][0]
    assert chunk["arxiv_id"] == "2606.29148"
    assert chunk["version"] == 2
    assert chunk["chunk_index"] == 1
    assert chunk["title"] == "PaperStream Chunk Retrieval Fixture"
    assert chunk["primary_category"] == "cs.IR"
    assert chunk["categories"] == ["cs.IR"]
    assert chunk["text"].startswith("quasarionomega")
    assert len(chunk["text"]) <= 32
    assert "truncated away" not in chunk["text"]
    _assert_paperstream_search_payload_is_public(result, bm25_path, tmp_path)


@pytest.mark.parametrize("artifact_state", ["missing", "corrupt"])
def test_paperstream_search_missing_or_corrupt_bm25_keeps_metadata_results_and_errors(
    tmp_path, artifact_state
):
    """Unreadable BM25 artifacts do not block metadata hits or mutate stored asset state."""
    from researchfs.paperstream import search_paperstream, upsert_arxiv_asset

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        title="Gracefulmetadata Search Fallback",
        abstract="Metadata survives a broken local text index.",
        authors=["Barbara Liskov"],
        category="cs.IR",
        categories=["cs.IR"],
    )
    index_path = tmp_path / f"{artifact_state}-bm25.sqlite"
    if artifact_state == "corrupt":
        index_path.write_bytes(b"not a sqlite database")
    upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "bm25_index",
        status="indexed",
        storage_path=index_path,
        checksum="sha256:stale-index",
        byte_size=19,
        source="index_bm25",
    )
    before_asset = dict(
        conn.execute(
            "SELECT status, storage_path, checksum, byte_size, source FROM arxiv_assets WHERE asset_key = ?",
            ("2606.29148:v1:bm25_index",),
        ).fetchone()
    )

    result = search_paperstream(conn, "gracefulmetadata", limit=5, include_chunks=True)

    assert [paper["arxiv_id"] for paper in result["papers"]] == ["2606.29148"]
    assert result["errors"]
    assert "bm25_index" in json.dumps(result["errors"], sort_keys=True)
    after_asset = dict(
        conn.execute(
            "SELECT status, storage_path, checksum, byte_size, source FROM arxiv_assets WHERE asset_key = ?",
            ("2606.29148:v1:bm25_index",),
        ).fetchone()
    )
    assert after_asset == before_asset
    _assert_paperstream_search_payload_is_public(result, index_path, tmp_path)


def test_paperstream_search_bm25_oserror_does_not_leak_storage_path(
    tmp_path, monkeypatch
):
    """BM25 OSError details are sanitized while preserving metadata search results."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        title="Pathleak Sentinel Search Fallback",
        abstract="Metadata survives a local BM25 open failure.",
        authors=["Mary Jackson"],
        category="cs.IR",
        categories=["cs.IR"],
    )
    index_path = tmp_path / "private-bm25-index.sqlite"
    paperstream.upsert_arxiv_asset(
        conn,
        "2606.29148",
        1,
        "bm25_index",
        status="indexed",
        storage_path=index_path,
        checksum="sha256:path-leak-sentinel",
        byte_size=64,
        source="index_bm25",
    )

    def fail_open_bm25(storage_path):
        raise OSError(f"cannot open {storage_path}")

    monkeypatch.setattr(paperstream, "_paperstream_open_bm25_index", fail_open_bm25)

    result = paperstream.search_paperstream(
        conn, "pathleak sentinel", limit=5, include_chunks=True
    )

    assert result["ok"] is True
    assert [paper["arxiv_id"] for paper in result["papers"]] == ["2606.29148"]
    assert result["errors"] == [
        {
            "asset_key": "2606.29148:v1:bm25_index",
            "arxiv_id": "2606.29148",
            "version": 1,
            "error": "unreadable_bm25_index",
            "message": "BM25 index file is unreadable",
        }
    ]
    _assert_paperstream_search_payload_is_public(result, index_path, tmp_path)


@pytest.mark.parametrize(
    ("query", "limit", "max_chunk_chars", "message"),
    [
        ("", 1, 20, "query"),
        (" ,.!? ", 1, 20, "query"),
        ("valid", 0, 20, "limit"),
        ("valid", True, 20, "limit"),
        ("valid", 1, -1, "max_chunk_chars"),
        ("valid", 1, True, "max_chunk_chars"),
    ],
)
def test_paperstream_search_invalid_inputs_fail_before_sqlite(
    query, limit, max_chunk_chars, message
):
    """Invalid search inputs fail clearly before SQLite query or FTS execution."""
    from researchfs.paperstream import search_paperstream

    class ExecuteForbidden:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("search_paperstream should validate before sqlite execution")

    with pytest.raises((TypeError, ValueError), match=message):
        search_paperstream(
            ExecuteForbidden(),
            query,
            limit=limit,
            max_chunk_chars=max_chunk_chars,
        )


def test_paperstream_search_rejects_bounds_above_caps_before_sqlite():
    """Search result bounds must be capped before any SQLite access."""
    from researchfs.paperstream import (
        _PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS,
        _PAPERSTREAM_SEARCH_MAX_LIMIT,
        search_paperstream,
    )

    class ExecuteForbidden:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("search_paperstream should validate before sqlite execution")

    with pytest.raises(ValueError, match="limit"):
        search_paperstream(
            ExecuteForbidden(),
            "valid",
            limit=_PAPERSTREAM_SEARCH_MAX_LIMIT + 1,
        )

    with pytest.raises(ValueError, match="max_chunk_chars"):
        search_paperstream(
            ExecuteForbidden(),
            "valid",
            max_chunk_chars=_PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS + 1,
        )


def test_paperstream_search_accepts_bounds_at_caps_on_empty_db():
    """Search caps are inclusive while still returning a bounded public payload."""
    from researchfs.paperstream import (
        _PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS,
        _PAPERSTREAM_SEARCH_MAX_LIMIT,
        search_paperstream,
    )

    conn = _paperstream_conn()

    result = search_paperstream(
        conn,
        "nomatches",
        limit=_PAPERSTREAM_SEARCH_MAX_LIMIT,
        max_chunk_chars=_PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS,
    )

    assert result == {
        "ok": True,
        "attribution": _PAPERSTREAM_ARXIV_ATTRIBUTION,
        "query": "nomatches",
        "limit": _PAPERSTREAM_SEARCH_MAX_LIMIT,
        "papers": [],
        "chunks": [],
        "errors": [],
    }


@pytest.mark.parametrize(
    ("helper_name", "message"),
    [
        ("materialize_paperstream_chunks", "materialize_paperstream_chunks"),
        ("rebuild_paperstream_bm25_indexes", "rebuild_paperstream_bm25_indexes"),
    ],
)
def test_paperstream_index_maintenance_rejects_limit_above_cap_before_sqlite(
    tmp_path, helper_name, message
):
    """Index maintenance limits are capped before scans or materialization work."""
    import researchfs.paperstream as paperstream

    class ExecuteForbidden:
        def execute(self, *_args, **_kwargs):
            raise AssertionError(f"{message} should validate before sqlite execution")

    helper = getattr(paperstream, helper_name)
    too_large = paperstream._PAPERSTREAM_REBUILD_MAX_LIMIT + 1

    with pytest.raises(ValueError, match="limit"):
        if helper_name == "rebuild_paperstream_bm25_indexes":
            helper(ExecuteForbidden(), tmp_path, limit=too_large)
        else:
            helper(ExecuteForbidden(), limit=too_large)


def test_paperstream_index_maintenance_accepts_limit_at_cap_on_empty_db(tmp_path):
    """Index maintenance caps are inclusive and no-op cleanly on empty PaperStream DBs."""
    import researchfs.paperstream as paperstream

    conn = _paperstream_conn()
    limit = paperstream._PAPERSTREAM_REBUILD_MAX_LIMIT

    materialized = paperstream.materialize_paperstream_chunks(conn, limit=limit)
    rebuilt = paperstream.rebuild_paperstream_bm25_indexes(conn, tmp_path, limit=limit)

    assert materialized == {
        "ok": True,
        "limit": limit,
        "processed": 0,
        "materialized": 0,
        "skipped": 0,
        "errors": [],
        "results": [],
    }
    assert rebuilt == {
        "ok": True,
        "limit": limit,
        "processed": 0,
        "materialized": 0,
        "indexed": 0,
        "skipped": 0,
        "errors": [],
        "materialization": materialized,
        "results": [],
    }


def test_paperstream_search_cli_prints_json_and_no_chunks_suppresses_hits(tmp_path):
    """The search CLI prints bounded JSON and --no-chunks suppresses BM25 chunk hits."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1, category="cs.IR")
    _update_paperstream_search_metadata(
        conn,
        "2606.29148",
        title="Clidragonfruit Retrieval from PaperStream",
        abstract="The CLI fixture also has metadata for the same term.",
        authors=["Edsger Dijkstra"],
        category="cs.IR",
        categories=["cs.IR"],
    )
    _create_paperstream_bm25_fixture(
        conn,
        tmp_path,
        arxiv_id="2606.29148",
        version=1,
        chunks=["clidragonfruit chunk text should appear only when chunks are enabled"],
    )
    conn.commit()
    conn.close()
    env = {**os.environ, "RESEARCHFS_DB_PATH": str(db_path)}

    with_chunks = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "search",
            "--query",
            "clidragonfruit",
            "--limit",
            "2",
            "--max-chunk-chars",
            "40",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    no_chunks = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "search",
            "--query",
            "clidragonfruit",
            "--limit",
            "2",
            "--no-chunks",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    with_chunks_payload = json.loads(with_chunks.stdout)
    no_chunks_payload = json.loads(no_chunks.stdout)
    assert with_chunks_payload["ok"] is True
    assert with_chunks_payload["query"] == "clidragonfruit"
    assert with_chunks_payload["limit"] == 2
    assert [paper["arxiv_id"] for paper in with_chunks_payload["papers"]] == ["2606.29148"]
    assert with_chunks_payload["chunks"]
    assert with_chunks_payload["chunks"][0]["arxiv_id"] == "2606.29148"
    assert len(with_chunks_payload["chunks"][0]["text"]) <= 40
    assert no_chunks_payload["ok"] is True
    assert [paper["arxiv_id"] for paper in no_chunks_payload["papers"]] == ["2606.29148"]
    assert no_chunks_payload["chunks"] == []
    _assert_paperstream_search_payload_is_public(with_chunks_payload, db_path, tmp_path)
    _assert_paperstream_search_payload_is_public(no_chunks_payload, db_path, tmp_path)

@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [(0, 0, "chunk_size"), (4, -1, "overlap"), (4, 4, "overlap")],
)
def test_paperstream_chunk_text_helper_invalid_params_do_not_claim_job(
    tmp_path, chunk_size, overlap, message
):
    """Invalid helper chunk parameters fail before claiming queued chunk_text work."""
    from researchfs.paperstream import enqueue_arxiv_job, work_next_arxiv_chunk_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    chunk_key = enqueue_arxiv_job(
        conn,
        job_type="chunk_text",
        arxiv_id="2606.29148",
        version=1,
        asset_type="parsed_text",
        run_after="2026-07-04T00:00:00Z",
    )

    with pytest.raises(ValueError, match=message):
        work_next_arxiv_chunk_job(
            conn,
            worker_id="invalid-chunk-worker",
            object_root=tmp_path,
            chunk_size=chunk_size,
            overlap=overlap,
            now="2026-07-04T00:00:01Z",
        )

    chunk_job = conn.execute(
        "SELECT status, locked_by, attempts FROM arxiv_jobs WHERE job_key = ?",
        (chunk_key,),
    ).fetchone()
    assert dict(chunk_job) == {"status": "queued", "locked_by": None, "attempts": 0}


def _paperstream_file_db(tmp_path):
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    return db_path, conn


def _paperstream_today_timestamp():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")


def _run_paperstream_work_cli(db_path, object_root, *args):
    import researchfs.paperstream as paperstream

    stdout = io.StringIO()
    code = paperstream.main(
        [
            "--db-path",
            str(db_path),
            "work",
            "--max-jobs",
            "1",
            "--object-root",
            str(object_root),
            "--rate-limit-seconds",
            "0",
            *args,
        ],
        stdout=stdout,
    )
    assert code == 0
    return json.loads(stdout.getvalue())


def _assert_paperstream_cap_stop_payload(payload, *, count, cap):
    assert payload["jobs_attempted"] == 0
    assert payload["stop_reason"] == "daily_download_cap"
    assert payload["daily_download_count"] == count
    assert payload["daily_download_cap"] == cap

def test_paperstream_cli_work_default_skips_download_jobs_before_claim_or_write(
    monkeypatch, tmp_path
):
    """Bare work defaults to local jobs and leaves queued live downloads untouched."""
    import researchfs.paperstream as paperstream

    db_path, conn = _paperstream_file_db(tmp_path)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    pdf_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=1,
        run_after="2026-07-04T00:00:00Z",
    )
    source_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_source",
        arxiv_id="2606.29149",
        version=1,
        asset_type="source",
        priority=2,
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()
    object_root = tmp_path / "objects"

    def fail_fetch(_url):
        raise AssertionError("bare paperstream work must not fetch downloads")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fail_fetch)

    payload = _run_paperstream_work_cli(db_path, object_root)

    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = {
            row["job_key"]: dict(row)
            for row in conn.execute(
                """
                SELECT job_key, status, locked_by, locked_at, attempts
                FROM arxiv_jobs
                ORDER BY job_key
                """
            )
        }
        assert rows == {
            pdf_key: {
                "job_key": pdf_key,
                "status": "queued",
                "locked_by": None,
                "locked_at": None,
                "attempts": 0,
            },
            source_key: {
                "job_key": source_key,
                "status": "queued",
                "locked_by": None,
                "locked_at": None,
                "attempts": 0,
            },
        }
        assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0
    finally:
        conn.close()
    assert not object_root.exists()


def test_paperstream_cli_work_default_skips_s3_import_and_extract_jobs(tmp_path):
    """Bare work defaults to local jobs and leaves queued S3 jobs untouched."""
    from researchfs.paperstream import enqueue_arxiv_s3_bin_import_job

    db_path, conn = _paperstream_file_db(tmp_path)
    import_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="pdf",
        filename="pdf/arXiv_pdf_2606_010.tar",
        size_bytes=128,
    )
    extract_bin_key = _insert_paperstream_s3_bin(
        conn,
        asset_type="source",
        filename="src/arXiv_src_2606_010.tar",
        size_bytes=256,
    )
    import_key = enqueue_arxiv_s3_bin_import_job(
        conn,
        import_bin_key,
        source="s3://arxiv/pdf/arXiv_pdf_2606_010.tar",
        priority=1,
        run_after="2026-07-04T00:00:00Z",
    )
    extract_key = _enqueue_paperstream_extract_s3_bin_job(
        conn,
        extract_bin_key,
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()
    object_root = tmp_path / "objects"

    payload = _run_paperstream_work_cli(db_path, object_root)

    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = {
            row["job_key"]: dict(row)
            for row in conn.execute(
                """
                SELECT job_key, status, locked_by, locked_at, attempts
                FROM arxiv_jobs
                ORDER BY job_key
                """
            )
        }
        assert rows == {
            extract_key: {
                "job_key": extract_key,
                "status": "queued",
                "locked_by": None,
                "locked_at": None,
                "attempts": 0,
            },
            import_key: {
                "job_key": import_key,
                "status": "queued",
                "locked_by": None,
                "locked_at": None,
                "attempts": 0,
            },
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM arxiv_events WHERE event_type IN ('job_succeeded', 'job_failed')"
        ).fetchone()[0] == 0
    finally:
        conn.close()
    assert not object_root.exists()


def test_paperstream_cli_work_download_filter_requires_daily_cap_before_claim(tmp_path):
    """Explicit download filters require a daily cap before any queued job is claimed."""
    import researchfs.paperstream as paperstream

    db_path, conn = _paperstream_file_db(tmp_path)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()
    object_root = tmp_path / "objects"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "download_pdf",
            "--max-jobs",
            "1",
            "--object-root",
            str(object_root),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "daily_download_cap" in result.stderr
    assert "required" in result.stderr
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute(
            "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
            (download_key,),
        ).fetchone()
        assert dict(job) == {
            "status": "queued",
            "locked_by": None,
            "locked_at": None,
            "attempts": 0,
        }
        assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0
    finally:
        conn.close()
    assert not object_root.exists()


def test_paperstream_cli_work_daily_download_cap_zero_stops_before_claim_fetch_or_write(
    monkeypatch, tmp_path
):
    """A zero daily cap stops queued downloads before claim, fetch, or object writes."""
    import researchfs.paperstream as paperstream

    db_path, conn = _paperstream_file_db(tmp_path)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()
    object_root = tmp_path / "objects"

    def fail_fetch(_url):
        raise AssertionError("daily cap must stop before fetching")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fail_fetch)

    payload = _run_paperstream_work_cli(
        db_path,
        object_root,
        "--job-types",
        "download_pdf",
        "--daily-download-cap",
        "0",
    )

    _assert_paperstream_cap_stop_payload(payload, count=0, cap=0)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute(
            "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
            (download_key,),
        ).fetchone()
        assert dict(job) == {
            "status": "queued",
            "locked_by": None,
            "locked_at": None,
            "attempts": 0,
        }
        assert conn.execute("SELECT COUNT(*) FROM arxiv_assets").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM arxiv_events").fetchone()[0] == 0
    finally:
        conn.close()
    assert not object_root.exists()


def test_paperstream_cli_work_daily_download_cap_counts_durable_attempt_history(
    monkeypatch, tmp_path
):
    """Today’s durable attempt event blocks a later worker invocation before claim/fetch."""
    import researchfs.paperstream as paperstream

    db_path, conn = _paperstream_file_db(tmp_path)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.execute(
        """
        INSERT INTO arxiv_events (
            event_type, arxiv_id, version, asset_type, job_key, payload_json, created_at
        )
        VALUES ('asset_download_attempted', '2606.00001', 1, 'pdf', 'download_pdf:2606.00001:v1', '{}', ?)
        """,
        (_paperstream_today_timestamp(),),
    )
    conn.commit()
    conn.close()

    def fail_fetch(_url):
        raise AssertionError("durable daily cap history must stop before fetching")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fail_fetch)

    payload = _run_paperstream_work_cli(
        db_path,
        tmp_path / "objects",
        "--job-types",
        "download_pdf",
        "--daily-download-cap",
        "1",
    )

    _assert_paperstream_cap_stop_payload(payload, count=1, cap=1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute(
            "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
            (download_key,),
        ).fetchone()
        assert dict(job) == {
            "status": "queued",
            "locked_by": None,
            "locked_at": None,
            "attempts": 0,
        }
    finally:
        conn.close()


def test_paperstream_cli_work_failed_download_attempt_consumes_daily_cap(
    monkeypatch, tmp_path
):
    """A failed fetch records an attempt event that blocks a second cap-1 invocation."""
    import researchfs.paperstream as paperstream

    db_path, conn = _paperstream_file_db(tmp_path)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    first_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        max_attempts=1,
        priority=1,
        run_after="2026-07-04T00:00:00Z",
    )
    second_key = paperstream.enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29149",
        version=1,
        asset_type="pdf",
        priority=2,
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()
    fetch_calls = []

    def failing_fetch(url):
        fetch_calls.append(url)
        raise RuntimeError("fixture download outage")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", failing_fetch)

    first_payload = _run_paperstream_work_cli(
        db_path,
        tmp_path / "objects",
        "--job-types",
        "download_pdf",
        "--daily-download-cap",
        "1",
    )

    assert first_payload["jobs_attempted"] == 1
    assert first_payload["jobs_failed"] == 1
    assert fetch_calls == ["https://arxiv.org/pdf/2606.29148.pdf"]

    def fail_second_fetch(_url):
        raise AssertionError("failed attempt must consume the cap before a second fetch")

    monkeypatch.setattr(paperstream, "_fetch_url_bytes", fail_second_fetch)

    second_payload = _run_paperstream_work_cli(
        db_path,
        tmp_path / "objects",
        "--job-types",
        "download_pdf",
        "--daily-download-cap",
        "1",
    )

    _assert_paperstream_cap_stop_payload(second_payload, count=1, cap=1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        first_job = conn.execute(
            "SELECT status, attempts FROM arxiv_jobs WHERE job_key = ?",
            (first_key,),
        ).fetchone()
        second_job = conn.execute(
            "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
            (second_key,),
        ).fetchone()
        assert dict(first_job) == {"status": "failed", "attempts": 1}
        assert dict(second_job) == {
            "status": "queued",
            "locked_by": None,
            "locked_at": None,
            "attempts": 0,
        }
        assert conn.execute(
            "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_download_attempted'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_paperstream_cli_work_non_download_filter_ignores_reached_download_cap(tmp_path):
    """Download cap history does not block non-download job filters."""
    db_path, conn = _paperstream_file_db(tmp_path)
    conn.execute(
        """
        INSERT INTO arxiv_events (
            event_type, arxiv_id, version, asset_type, job_key, payload_json, created_at
        )
        VALUES ('asset_download_attempted', '2606.00001', 1, 'pdf', 'download_pdf:2606.00001:v1', '{}', ?)
        """,
        (_paperstream_today_timestamp(),),
    )
    conn.commit()
    conn.close()

    payload = _run_paperstream_work_cli(
        db_path,
        tmp_path / "objects",
        "--job-types",
        "parse_pdf",
        "--daily-download-cap",
        "1",
    )

    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert "stop_reason" not in payload
    assert payload["results"] == []


def test_paperstream_cli_work_negative_daily_download_cap_fails_clearly(tmp_path):
    """Negative daily caps are rejected before queued download jobs are claimed."""
    from researchfs import db
    from researchfs.paperstream import enqueue_arxiv_job

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--daily-download-cap",
            "-1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--daily-download-cap" in result.stderr
    assert "non-negative" in result.stderr
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute(
            "SELECT status, locked_by, locked_at, attempts FROM arxiv_jobs WHERE job_key = ?",
            (download_key,),
        ).fetchone()
        assert dict(job) == {
            "status": "queued",
            "locked_by": None,
            "locked_at": None,
            "attempts": 0,
        }
    finally:
        conn.close()
    assert not (tmp_path / "objects").exists()

def test_paperstream_cli_work_parse_pdf_reports_no_jobs(tmp_path):
    """The worker CLI accepts parse job filters and reports zero work safely."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "parse_pdf",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    assert not (tmp_path / "objects").exists()


def test_paperstream_cli_work_extract_s3_bin_reports_no_jobs(tmp_path):
    """The worker CLI accepts extract_s3_bin filters and reports zero work safely."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "extract_s3_bin",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    assert not (tmp_path / "objects").exists()


def test_paperstream_cli_work_chunk_text_reports_no_jobs(tmp_path):
    """The worker CLI accepts chunk_text filters and reports zero work safely."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "chunk_text",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    assert not (tmp_path / "objects").exists()

def test_paperstream_cli_work_index_bm25_reports_no_jobs(tmp_path):
    """The worker CLI accepts index_bm25 filters and reports zero work safely."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "index_bm25",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    assert not (tmp_path / "objects").exists()

def test_paperstream_cli_work_import_s3_bin_reports_no_jobs(tmp_path):
    """The worker CLI accepts import_s3_bin filters and reports zero work safely."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "import_s3_bin",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    assert not (tmp_path / "objects").exists()



def test_paperstream_cli_work_invalid_job_type_fails_clearly(tmp_path):
    """The worker CLI rejects unknown job types before doing queue work."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "not_a_job",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--job-types" in result.stderr
    assert "index_bm25" in result.stderr
    assert not (tmp_path / "objects").exists()


def test_paperstream_cli_work_invalid_chunk_params_do_not_claim_job(tmp_path):
    """Invalid worker CLI chunk options fail before mutating queued chunk_text jobs."""
    from researchfs import db
    from researchfs.paperstream import enqueue_arxiv_job

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    chunk_key = enqueue_arxiv_job(
        conn,
        job_type="chunk_text",
        arxiv_id="2606.29148",
        version=1,
        asset_type="parsed_text",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "chunk_text",
            "--chunk-size",
            "4",
            "--overlap",
            "4",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "overlap" in result.stderr
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute(
            "SELECT status, locked_by, attempts FROM arxiv_jobs WHERE job_key = ?",
            (chunk_key,),
        ).fetchone()
        assert dict(job) == {"status": "queued", "locked_by": None, "attempts": 0}
    finally:
        conn.close()


def test_paperstream_cli_work_invalid_rate_limit_does_not_claim_job(tmp_path):
    """Invalid worker CLI options fail before mutating queued download jobs."""
    from researchfs import db
    from researchfs.paperstream import enqueue_arxiv_job

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        run_after="2026-07-04T00:00:00Z",
    )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--job-types",
            "download_pdf",
            "--daily-download-cap",
            "1",
            "--rate-limit-seconds",
            "-1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "rate_limit_seconds" in result.stderr
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        job = conn.execute("SELECT status, locked_by FROM arxiv_jobs WHERE job_key = ?", (download_key,)).fetchone()
        assert dict(job) == {"status": "queued", "locked_by": None}
    finally:
        conn.close()


def test_paperstream_cli_work_reports_no_download_jobs(tmp_path):
    """The opt-in worker CLI is safe to run when no download jobs are queued."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    db.initialize_db(conn)
    conn.commit()
    conn.close()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchfs.paperstream",
            "--db-path",
            str(db_path),
            "work",
            "--max-jobs",
            "1",
            "--object-root",
            str(tmp_path / "objects"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["jobs_attempted"] == 0
    assert payload["results"] == []
    assert not (tmp_path / "objects").exists()


def test_paperstream_download_fetch_failure_marks_job_and_asset_failed(tmp_path):
    """Fetch failures are recorded without parse jobs or leftover object files."""
    from researchfs.paperstream import enqueue_arxiv_job, work_next_arxiv_download_job

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    download_key = enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        max_attempts=1,
        run_after="2026-07-04 00:00:00",
    )

    def fetch_fn(url):
        assert url == "https://arxiv.org/pdf/2606.29148.pdf"
        raise RuntimeError("fixture outage")

    result = work_next_arxiv_download_job(
        conn,
        worker_id="failure-worker",
        object_root=tmp_path,
        fetch_fn=fetch_fn,
        now="2026-07-04 00:00:01",
    )

    assert result["worked"] is True
    job = conn.execute(
        "SELECT status, attempts, locked_by, error_message FROM arxiv_jobs WHERE job_key = ?",
        (download_key,),
    ).fetchone()
    assert job["status"] == "failed"
    assert job["attempts"] == 1
    assert job["locked_by"] is None
    assert "fixture outage" in job["error_message"]
    asset = conn.execute(
        "SELECT status, storage_path, checksum, byte_size, source, error_message "
        "FROM arxiv_assets WHERE asset_key = '2606.29148:v1:pdf'"
    ).fetchone()
    assert asset["status"] == "failed"
    assert asset["storage_path"] is None
    assert asset["checksum"] is None
    assert asset["byte_size"] is None
    assert asset["source"] == "arxiv"
    assert "fixture outage" in asset["error_message"]
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'asset_failed' "
        "AND job_key = ? AND asset_type = 'pdf'",
        (download_key,),
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM arxiv_jobs WHERE job_type LIKE 'parse_%'"
    ).fetchone()[0] == 0
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []

def test_paperstream_bloom_rebuild_save_and_load_preserves_membership():
    """Bloom filters rebuild from exact DB keys and persist enough state to reload."""
    from researchfs.paperstream import (
        BloomFilter,
        bloom_db_contains,
        load_bloom_filter,
        rebuild_bloom_filter,
        save_bloom_filter,
        upsert_arxiv_asset,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    key_one = upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29148",
        version=1,
        asset_type="metadata",
        status="seen",
    )
    key_two = upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29149",
        version=1,
        asset_type="metadata",
        status="seen",
    )

    manual = BloomFilter(capacity=10, false_positive_rate=0.01)
    manual.add(key_one)
    assert key_one in manual
    assert manual.item_count == 1

    source_keys = [
        row["asset_key"]
        for row in conn.execute("SELECT asset_key FROM arxiv_assets ORDER BY asset_key")
    ]
    rebuilt = rebuild_bloom_filter(
        conn,
        filter_name="metadata_assets",
        keys=source_keys,
        source_table="arxiv_assets",
        source_column="asset_key",
        false_positive_rate=0.01,
    )
    assert key_one in rebuilt
    assert key_two in rebuilt
    assert rebuilt.item_count == 2

    save_bloom_filter(
        conn,
        filter_name="metadata_assets",
        bloom_filter=rebuilt,
        source_table="arxiv_assets",
        source_column="asset_key",
    )
    loaded = load_bloom_filter(conn, "metadata_assets")
    assert loaded is not None
    assert key_one in loaded
    assert key_two in loaded
    assert "2606.29150:v1:metadata" not in loaded
    assert loaded.item_count == 2
    assert loaded.false_positive_rate == pytest.approx(0.01)

    all_bits = BloomFilter.from_bytes(
        b"\xff",
        bit_count=8,
        hash_count=1,
        item_count=1,
        false_positive_rate=0.01,
    )
    missing_key = "2606.29150:v1:metadata"
    assert missing_key in all_bits
    assert bloom_db_contains(conn, all_bits, "arxiv_assets", "asset_key", key_one) is True
    assert bloom_db_contains(conn, all_bits, "arxiv_assets", "asset_key", missing_key) is False
    assert bloom_db_contains(conn, rebuilt, "arxiv_assets", "asset_key", missing_key) is False
    with pytest.raises(ValueError, match="SQLite identifier"):
        bloom_db_contains(
            conn, all_bits, "arxiv_assets", "asset_key; DROP TABLE arxiv_assets", key_one
        )

def test_paperstream_status_sync_failed_only_does_not_report_success():
    """Failed-only sync state exposes latest run counters without a false success."""
    from researchfs.paperstream import get_paperstream_status

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        status="failed",
        committed_from="2026-07-01T00:00:00Z",
        last_response_date="2026-07-04T12:00:00Z",
        pending_response_date="2026-07-04T12:00:00Z",
        last_resumption_token="failed-cursor-token",
        records_seen_last_run=37,
        records_updated_last_run=11,
        jobs_queued_last_run=5,
        error_message="fixture failure",
        started_at="2026-07-04T12:00:00Z",
        finished_at="2026-07-04T12:00:05Z",
        updated_at="2026-07-04T12:00:06Z",
    )

    sync = get_paperstream_status(conn)["sync"]

    assert sync["last_successful_oai_sync"] is None
    assert sync["last_response_date"] == "2026-07-04T12:00:00Z"
    assert sync["last_committed_cursor"] == "2026-07-01T00:00:00Z"
    assert sync["pending_cursor"] == "2026-07-04T12:00:00Z"
    assert sync["records_seen_last_run"] == 37
    assert sync["records_updated_last_run"] == 11
    assert sync["jobs_queued_last_run"] == 5



def test_paperstream_status_exposes_inserted_counters_top_level_and_nested_sync():
    """Status payload carries last-run inserted counters at both public sync surfaces."""
    from researchfs.paperstream import get_paperstream_status

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        status="succeeded",
        committed_from="2026-07-03T12:00:00Z",
        last_response_date="2026-07-03T12:00:00Z",
        records_seen_last_run=15,
        records_updated_last_run=6,
        records_inserted_last_run=4,
        versions_inserted_last_run=9,
        jobs_queued_last_run=2,
        finished_at="2026-07-03T12:00:01Z",
        updated_at="2026-07-03T12:00:01Z",
    )

    status = get_paperstream_status(conn)

    assert status["records_inserted_last_run"] == 4
    assert status["versions_inserted_last_run"] == 9
    assert status["sync"]["records_inserted_last_run"] == 4
    assert status["sync"]["versions_inserted_last_run"] == 9


def test_paperstream_status_sync_success_fields_ignore_newer_failed_row():
    """Successful sync timestamp, response date, and cursor ignore newer failed rows."""
    from researchfs.paperstream import get_paperstream_status

    conn = _paperstream_conn()
    _insert_paperstream_sync_state(
        conn,
        sync_name="successful",
        status="succeeded",
        committed_from="2026-07-03T12:00:00Z",
        last_response_date="2026-07-03T12:00:00Z",
        records_seen_last_run=12,
        records_updated_last_run=7,
        jobs_queued_last_run=2,
        started_at="2026-07-03T12:00:00Z",
        finished_at="2026-07-03T12:00:01Z",
        updated_at="2026-07-03T12:00:01Z",
    )
    _insert_paperstream_sync_state(
        conn,
        sync_name="newer_failed",
        status="failed",
        committed_from="2026-07-04T12:00:00Z",
        last_response_date="2026-07-04T12:00:00Z",
        pending_response_date="2026-07-04T12:00:00Z",
        last_resumption_token="newer-failed-token",
        records_seen_last_run=99,
        records_updated_last_run=88,
        jobs_queued_last_run=77,
        error_message="newer fixture failure",
        started_at="2026-07-04T12:00:00Z",
        finished_at="2026-07-04T12:00:05Z",
        updated_at="2026-07-04T12:00:06Z",
    )

    sync = get_paperstream_status(conn)["sync"]

    assert sync["last_successful_oai_sync"] == "2026-07-03T12:00:01Z"
    assert sync["last_response_date"] == "2026-07-03T12:00:00Z"
    assert sync["last_committed_cursor"] == "2026-07-03T12:00:00Z"


def test_paperstream_cli_status_sync_failed_only_json_has_no_success(tmp_path):
    """CLI status JSON mirrors API behavior for failed-only sync state."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_sync_state(
        conn,
        status="failed",
        committed_from="2026-07-01T00:00:00Z",
        last_response_date="2026-07-04T12:00:00Z",
        pending_response_date="2026-07-04T12:00:00Z",
        last_resumption_token="failed-cli-token",
        records_seen_last_run=21,
        records_updated_last_run=8,
        jobs_queued_last_run=3,
        error_message="fixture CLI failure",
        started_at="2026-07-04T12:00:00Z",
        finished_at="2026-07-04T12:00:05Z",
        updated_at="2026-07-04T12:00:06Z",
    )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [sys.executable, "-m", "researchfs.paperstream", "--db-path", str(db_path), "status"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    sync = json.loads(result.stdout)["sync"]
    assert sync["last_successful_oai_sync"] is None
    assert sync["last_response_date"] == "2026-07-04T12:00:00Z"
    assert sync["last_committed_cursor"] == "2026-07-01T00:00:00Z"
    assert sync["pending_cursor"] == "2026-07-04T12:00:00Z"
    assert sync["records_seen_last_run"] == 21
    assert sync["records_updated_last_run"] == 8
    assert sync["jobs_queued_last_run"] == 3


def test_paperstream_cli_status_json_includes_inserted_counters(tmp_path):
    """CLI status JSON includes last-run inserted counters at top level and under sync."""
    from researchfs import db

    db_path = tmp_path / "paperstream.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.initialize_db(conn)
    _insert_paperstream_sync_state(
        conn,
        status="succeeded",
        committed_from="2026-07-03T12:00:00Z",
        last_response_date="2026-07-03T12:00:00Z",
        records_seen_last_run=13,
        records_updated_last_run=5,
        records_inserted_last_run=3,
        versions_inserted_last_run=8,
        jobs_queued_last_run=1,
        finished_at="2026-07-03T12:00:01Z",
        updated_at="2026-07-03T12:00:01Z",
    )
    conn.commit()
    conn.close()

    result = subprocess.run(
        [sys.executable, "-m", "researchfs.paperstream", "--db-path", str(db_path), "status"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["records_inserted_last_run"] == 3
    assert payload["versions_inserted_last_run"] == 8
    assert payload["sync"]["records_inserted_last_run"] == 3
    assert payload["sync"]["versions_inserted_last_run"] == 8



def test_paperstream_status_metrics_report_sync_job_asset_and_bloom_counts():
    """Status metrics summarize sync state, asset rows, jobs, and persisted Bloom filters."""
    from researchfs.paperstream import (
        enqueue_arxiv_job,
        get_paperstream_status,
        rebuild_bloom_filter,
        save_bloom_filter,
        upsert_arxiv_asset,
    )

    conn = _paperstream_conn()
    _insert_paperstream_paper(conn, "2606.29148", version=1)
    _insert_paperstream_paper(conn, "2606.29149", version=1)
    upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29148",
        version=1,
        asset_type="metadata",
        status="seen",
    )
    upsert_arxiv_asset(
        conn,
        arxiv_id="2606.29149",
        version=1,
        asset_type="pdf",
        status="downloaded",
        storage_path="/object-store/arxiv/pdf/2606/29149/v1.pdf",
    )
    enqueue_arxiv_job(
        conn,
        job_type="download_pdf",
        arxiv_id="2606.29148",
        version=1,
        asset_type="pdf",
        priority=10,
        payload={},
        run_after="2026-07-03T00:00:00Z",
    )
    enqueue_arxiv_job(
        conn,
        job_type="parse_pdf",
        arxiv_id="2606.29149",
        version=1,
        asset_type="parsed_text",
        priority=1,
        payload={},
        run_after="2026-07-04T00:00:00Z",
    )
    conn.execute(
        """
        INSERT INTO arxiv_sync_state (
            sync_name,
            committed_from,
            last_response_date,
            pending_response_date,
            status,
            records_seen_last_run,
            records_updated_last_run,
            jobs_queued_last_run,
            finished_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "default",
            "2026-07-01",
            "2026-07-03T00:00:00Z",
            "2026-07-03T00:00:00Z",
            "succeeded",
            12,
            7,
            2,
            "2026-07-03T00:00:01Z",
        ),
    )
    source_keys = [
        row["asset_key"]
        for row in conn.execute("SELECT asset_key FROM arxiv_assets ORDER BY asset_key")
    ]
    bloom = rebuild_bloom_filter(
        conn,
        filter_name="asset_keys",
        keys=source_keys,
        source_table="arxiv_assets",
        source_column="asset_key",
        false_positive_rate=0.01,
    )
    save_bloom_filter(
        conn,
        filter_name="asset_keys",
        bloom_filter=bloom,
        source_table="arxiv_assets",
        source_column="asset_key",
    )

    status = get_paperstream_status(conn)

    assert status["papers"]["total"] == 2
    assert status["assets"]["total"] == 2
    assert status["assets"]["by_type"] == {"metadata": 1, "pdf": 1}
    assert status["assets"]["by_status"] == {"downloaded": 1, "seen": 1}
    assert status["jobs"]["total"] == 2
    assert status["jobs"]["by_status"] == {"queued": 2}
    assert status["jobs"]["oldest_queued_job"] == "2026-07-03T00:00:00Z"
    sync = status["sync"]
    assert sync["last_successful_oai_sync"] == "2026-07-03T00:00:01Z"
    assert sync["last_committed_cursor"] == "2026-07-01"
    assert sync["pending_cursor"] == "2026-07-03T00:00:00Z"
    assert sync["records_seen_last_run"] == 12
    assert sync["records_updated_last_run"] == 7
    assert sync["jobs_queued_last_run"] == 2

    bloom_status = status["bloom_filters"]["asset_keys"]
    assert bloom_status["item_count"] == 2
    assert bloom_status["false_positive_rate"] == pytest.approx(0.01)
    assert bloom_status["source_table"] == "arxiv_assets"
    assert bloom_status["source_column"] == "asset_key"
