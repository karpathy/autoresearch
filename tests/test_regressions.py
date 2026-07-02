import ast
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


def _attr_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_is_fill_one(call):
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "fill_"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 1.0
    )


def test_train_init_weights_explicitly_initializes_all_rmsnorm_scales_without_importing_training():
    """RMSNorm parameters created on the meta device are reset to neutral scale before training."""
    tree = ast.parse(Path("train.py").read_text())
    gpt_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPT"
    )
    init_weights = next(
        node
        for node in gpt_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_weights"
    )

    filled_paths = {
        _attr_path(call.func.value)
        for call in ast.walk(init_weights)
        if _call_is_fill_one(call)
    }

    assert {
        "block.attn.q_norm.scale",
        "block.attn.k_norm.scale",
        "block.norm1.scale",
        "block.norm2.scale",
        "self.ln_f.scale",
    } <= filled_paths


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


def test_missing_token_usage_renders_unknown_not_zero_savings():
    """Runs without token_usage rows are reported as unknown, not as zero-token savings."""
    from evals.nanochat_bpb import metrics

    conn = _init_memory_db()
    baseline_id = _insert_run(conn, "baseline")
    loopgraph_id = _insert_run(conn, "loopgraph")

    comparison = metrics.compare_runs(baseline_id, loopgraph_id, conn=conn)
    assert comparison["baseline"]["has_token_usage"] is False
    assert comparison["researchfs"]["has_token_usage"] is False
    assert comparison["token_reduction_ratio"] is None

    rendered = metrics.format_comparison(comparison)
    assert "| Prompt tokens | N/A | N/A |" in rendered
    assert "| Completion tokens | N/A | N/A |" in rendered
    assert "| Total tokens | N/A | N/A |" in rendered
    assert "fewer tokens" not in rendered
    assert "more tokens" not in rendered
    assert "same number of tokens" not in rendered


def test_codex_dirty_guard_allows_preexisting_non_train_diff_but_rejects_new_outside_edits(tmp_path, monkeypatch):
    """Codex guard covers tracked/untracked changes and scoped ignored artifacts."""
    import run_agent
    import hashlib

    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Regression Tester",
            "GIT_AUTHOR_EMAIL": "regression@example.invalid",
            "GIT_COMMITTER_NAME": "Regression Tester",
            "GIT_COMMITTER_EMAIL": "regression@example.invalid",
        }
    )
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, env=env)
    (tmp_path / "train.py").write_text("print('base')\n")
    (tmp_path / "notes.txt").write_text("base notes\n")
    (tmp_path / ".gitignore").write_text("results.tsv\nresults/\n")
    subprocess.run(["git", "add", "train.py", "notes.txt", ".gitignore"], cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, env=env)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("pre-existing user note\n")
    preexisting_untracked = tmp_path / "preexisting-untracked.txt"
    preexisting_untracked.write_text("pre-existing untracked user note\n")
    ignored_results = tmp_path / "results.tsv"
    ignored_results_content = "step\tval_bpb\n1\t2.5\n"
    ignored_results.write_text(ignored_results_content)
    ignored_dir = tmp_path / "results"
    ignored_dir.mkdir()
    ignored_dir_result = ignored_dir / "seed.tsv"
    ignored_dir_result_content = "seed\tval_bpb\n1\t2.6\n"
    ignored_dir_result.write_text(ignored_dir_result_content)
    before_snapshot = run_agent._snapshot_tracked_non_train_diff(env)
    untracked_path, untracked_size, untracked_hash = before_snapshot["untracked"].splitlines()[0].split("\t")
    assert untracked_path == "preexisting-untracked.txt"
    assert int(untracked_size) == len("pre-existing untracked user note\n")
    assert len(untracked_hash) == 64
    ignored_entries = {
        path: (size, digest)
        for path, size, digest in (line.split("\t") for line in before_snapshot["ignored"].splitlines())
    }
    assert ignored_entries["results.tsv"] == (
        str(len(ignored_results_content)),
        hashlib.sha256(ignored_results_content.encode()).hexdigest(),
    )
    assert ignored_entries["results/seed.tsv"] == (
        str(len(ignored_dir_result_content)),
        hashlib.sha256(ignored_dir_result_content.encode()).hexdigest(),
    )

    (tmp_path / "train.py").write_text("print('codex experiment')\n")
    after_train_only_snapshot = run_agent._ensure_codex_only_changed_train_py(before_snapshot, env)
    assert after_train_only_snapshot == before_snapshot

    ignored_dir_result.write_text("seed\tval_bpb\n1\t9.9\n")
    with pytest.raises(RuntimeError, match=r"outside train\.py .*ignored"):
        run_agent._ensure_codex_only_changed_train_py(before_snapshot, env)
    ignored_dir_result.write_text(ignored_dir_result_content)

    preexisting_untracked.write_text("codex changed existing untracked note\n")
    with pytest.raises(RuntimeError, match=r"outside train\.py .*untracked"):
        run_agent._ensure_codex_only_changed_train_py(before_snapshot, env)
    preexisting_untracked.write_text("pre-existing untracked user note\n")

    (tmp_path / "notes.txt").write_text("new codex-side outside edit\n")
    with pytest.raises(RuntimeError, match="notes.txt|non-train|outside"):
        run_agent._ensure_codex_only_changed_train_py(before_snapshot, env)

    (tmp_path / "notes.txt").write_text("pre-existing user note\n")
    (tmp_path / "codex-output.txt").write_text("new codex-side untracked output\n")
    with pytest.raises(RuntimeError, match=r"outside train\.py .*untracked"):
        run_agent._ensure_codex_only_changed_train_py(before_snapshot, env)


def test_asep_branch_selection_deletes_single_candidate_but_refuses_multiple_new_candidates():
    """ASEP cleanup selects one current autoresearch branch and refuses ambiguous new branches."""
    import asep_loopgraph_benchmark as asep

    before = {"autoresearch/exp_existing"}
    after_one = before | {"autoresearch/exp_new"}
    assert (
        asep._temporary_branch_from_this_run(
            "autoresearch/exp_new", "main", before, branches_after=after_one
        )
        == "autoresearch/exp_new"
    )

    after_two = before | {"autoresearch/exp_new", "autoresearch/exp_other"}
    assert (
        asep._temporary_branch_from_this_run(
            "autoresearch/exp_new", "main", before, branches_after=after_two
        )
        is None
    )


def test_asep_cleanup_checks_out_original_branch_before_deleting_single_generated_branch(monkeypatch):
    """Cleanup returns to the original branch before deleting the generated autoresearch branch."""
    import asep_loopgraph_benchmark as asep

    calls = []
    monkeypatch.setattr(asep, "current_git_branch", lambda: "autoresearch/exp_new")

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd == ["git", "branch", "--list", "autoresearch/exp_*"]:
            return SimpleNamespace(
                returncode=0,
                stdout="  autoresearch/exp_existing\n* autoresearch/exp_new\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(asep.subprocess, "run", fake_run)

    asep.cleanup_git_state("main", {"autoresearch/exp_existing"})

    checkout_call = ["git", "checkout", "main"]
    delete_call = ["git", "branch", "-D", "autoresearch/exp_new"]
    assert checkout_call in calls
    assert delete_call in calls
    assert calls.index(checkout_call) < calls.index(delete_call)


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


def test_empty_search_replace_replacement_deletes_matched_text():
    """An empty REPLACE block is a deletion, not a malformed hunk or no-op."""
    from agents.editor import apply_hunks

    content = "alpha\nremove me\nomega"
    response = "<<<<<<< SEARCH\nremove me\n=======\n>>>>>>>"

    assert apply_hunks(content, response) == "alpha\n\nomega"


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


def _search_result(search_fn, query, *, conn, max_chars):
    result = search_fn(query, conn=conn, max_chars=max_chars)
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


def test_get_paper_memory_missing_positive_paper_id_still_raises_lookup_error():
    """Positive but nonexistent paper IDs keep the documented not-found behavior."""
    from researchfs.retrieve import get_paper_memory

    conn = _init_memory_db()

    with pytest.raises(LookupError, match="No paper found"):
        get_paper_memory(999_999, conn=conn)


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