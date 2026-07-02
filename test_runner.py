from run_agent import (
    extract_code_block,
    extract_description,
    parse_log,
    assemble_system_prompt,
    assemble_user_prompt,
    assemble_codex_prompt,
    extract_codex_description
)

def test_extract_code_block_python():
    response = """
Here is my change to train.py:
```python
def main():
    print("hello world")
```
Hope this helps!
"""
    code = extract_code_block(response)
    assert code == 'def main():\n    print("hello world")'

def test_extract_code_block_py():
    response = """
```py
def foo():
    pass
```
"""
    code = extract_code_block(response)
    assert code == "def foo():\n    pass"

def test_extract_code_block_generic():
    response = """
Some text
```
def bar():
    return 42
```
"""
    code = extract_code_block(response)
    assert code == "def bar():\n    return 42"

def test_extract_code_block_requires_python_markdown_block():
    response = "def baz():\n    return 1"
    try:
        extract_code_block(response)
    except ValueError as exc:
        assert "Python markdown code block" in str(exc)
    else:
        raise AssertionError("extract_code_block should reject raw prose/code responses")

def test_extract_description_found():
    response = """
I optimized attention heads.
Description: Optimized GQA and reduced KV heads.
Some other text.
"""
    desc = extract_description(response, "default_desc")
    assert desc == "Optimized GQA and reduced KV heads."

def test_extract_description_case_insensitive():
    response = """
description: RMSNorm scaling update
"""
    desc = extract_description(response, "default_desc")
    assert desc == "RMSNorm scaling update"

def test_extract_description_not_found():
    response = "No description field in here"
    desc = extract_description(response, "my_default")
    assert desc == "my_default"

def test_parse_log_success():
    log_text = """
Step 1: training
Step 2: evaluating
---
val_bpb:          0.145600
training_seconds: 55.4
total_seconds:    62.1
peak_vram_mb:     2048.0
mfu_percent:      15.2
"""
    val_bpb, peak_vram_mb, status, error_msg = parse_log(log_text, exit_code=0)
    assert status == "success"
    assert val_bpb == 0.1456
    assert peak_vram_mb == 2048.0
    assert error_msg == ""

def test_parse_log_crash():
    log_text = """
Some initial logs
Traceback (most recent call last):
  File "train.py", line 55, in <module>
    AssertionError: Emb LR must be > 0
"""
    val_bpb, peak_vram_mb, status, error_msg = parse_log(log_text, exit_code=1)
    assert status == "crash"
    assert "AssertionError: Emb LR must be > 0" in error_msg
    assert val_bpb is None
    assert peak_vram_mb is None

def test_parse_log_silent_fail():
    log_text = "Some random print without val_bpb or crash"
    val_bpb, peak_vram_mb, status, error_msg = parse_log(log_text, exit_code=0)
    assert status == "crash"
    assert "No val_bpb found in log" in error_msg

def test_assemble_system_prompt_baseline():
    prompt = assemble_system_prompt("baseline")
    assert "Method Pack" not in prompt
    assert "Experiment Brief" not in prompt
    assert "val_bpb" in prompt

def test_assemble_system_prompt_loopgraph():
    method_pack = "Custom Method Pack Reference Content"
    experiment_brief = "Custom Experiment Brief Rules Content"
    prompt = assemble_system_prompt("loopgraph", method_pack, experiment_brief)
    assert "Method Pack" in prompt
    assert "Experiment Brief" in prompt
    assert "Custom Method Pack Reference Content" in prompt
    assert "Custom Experiment Brief Rules Content" in prompt

def test_assemble_user_prompt():
    code = "import torch\n..."
    results = "commit\tval_bpb\tstatus\n..."
    prompt = assemble_user_prompt(code, results)
    assert code in prompt
    assert results in prompt

def test_assemble_codex_prompt():
    prompt = assemble_codex_prompt("loopgraph", "commit\tval_bpb\tstatus\n...", "Custom Method Pack Reference Content", "Custom Experiment Brief Rules Content")
    assert "Custom Method Pack Reference Content" in prompt
    assert "Custom Experiment Brief Rules Content" in prompt
    assert "DESCRIPTION:" in prompt
    
    prompt_search = assemble_codex_prompt("loopgraph", "commit\tval_bpb\tstatus\n...", "Custom Method Pack Reference Content", "Custom Experiment Brief Rules Content", search_enabled=True)
    assert "web/arXiv search" in prompt_search

def test_extract_codex_description():
    stdout = "Blah blah.\nDESCRIPTION: My cool optimization\nMore output."
    desc = extract_codex_description(stdout, "default")
    assert desc == "My cool optimization"

import json
import sqlite3
from researchfs.db import initialize_db, migrate_db
from researchfs.evaluation import evaluate_extraction
from researchfs.ingest import (
    seed_initial_data,
    ingest_paper,
    _normalize_extracted_method,
    load_local_file_metadata,
    _apply_source_type_override,
    _source_type_label,
)
from researchfs import retrieve as rfs
from researchfs.extractor import METHOD_CATEGORY_TOKENS, PROMPT_VERSION, assemble_extractor_prompt, chunk_text, extract_json_block, extract_methods_from_chunks
from researchfs.fetcher import clean_latex, fetch_github_readme, _parse_github_repo_path
from researchfs.api import ResearchFSLabAPI, _make_handler
from evals.nanochat_bpb import metrics
from agents.editor import apply_hunks, check_syntax, HunkError


def _in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    seed_initial_data(conn)
    return conn

def _eval_test_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    return conn

def _write_minimal_pdf(path, text):
    """Write a tiny valid text PDF fixture for local ingestion tests."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode("ascii")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\n"
            b"endobj\n"
        ),
        (
            b"4 0 obj\n<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(data))
        data.extend(obj)
    xref_start = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(bytes(data))


def _insert_eval_paper(conn, method_specs, confidence_score=0.8, confidence_reason="stored confidence"):
    cur = conn.cursor()
    cur.execute("INSERT INTO papers (title) VALUES (?)", ("Eval Paper",))
    paper_id = cur.lastrowid
    chunk_ids = []
    for index in range(3):
        cur.execute(
            "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
            (paper_id, f"chunk-{index}", f"chunk {index} evidence text"),
        )
        chunk_ids.append(cur.lastrowid)
    for name, source_indices, evidence_indices in method_specs:
        cur.execute("INSERT INTO methods (name) VALUES (?)", (name,))
        method_id = cur.lastrowid
        for source_index in source_indices:
            evidence_text = (
                f"evidence for {name} in chunk {source_index}"
                if source_index in evidence_indices
                else None
            )
            cur.execute(
                """
                INSERT INTO method_sources (
                    method_id, chunk_id, evidence_text, evidence_start, evidence_end, evidence_strategy
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    method_id,
                    chunk_ids[source_index],
                    evidence_text,
                    0 if evidence_text else None,
                    len(evidence_text) if evidence_text else None,
                    "test" if evidence_text else None,
                ),
            )
    cur.execute(
        """
        INSERT INTO paper_extraction_status (
            paper_id, status, chunk_count, method_count, source_char_count,
            compressed_char_count, confidence_score, confidence_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            "extracted",
            len(chunk_ids),
            len(method_specs),
            100,
            50,
            confidence_score,
            confidence_reason,
        ),
    )
    conn.commit()
    return paper_id


def test_evaluate_extraction_perfect_match_reports_metrics_and_confidence():
    conn = _eval_test_conn()
    paper_id = _insert_eval_paper(
        conn,
        [
            ("SwiGLU MLP", [0], {0}),
            ("Grouped Query Attention (GQA)", [1], {1}),
        ],
    )

    result = evaluate_extraction(
        conn,
        paper_id,
        [
            {"name": "swiglu mlp", "source_chunk_indices": [0]},
            {"name": "Grouped-Query Attention", "source_chunk_indices": [1]},
        ],
    )

    assert result["paper_id"] == paper_id
    assert result["expected_count"] == 2
    assert result["extracted_count"] == 2
    assert result["matched_count"] == 2
    assert result["missing_methods"] == []
    assert result["extra_methods"] == []
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["source_link_coverage"] == 1.0
    assert result["evidence_coverage"] == 1.0
    assert result["confidence_score"] == 0.8
    assert result["confidence_reason"] == "stored confidence"
    assert result["passed"] is True


def test_evaluate_extraction_missing_method_and_extra_method():
    conn = _eval_test_conn()
    paper_id = _insert_eval_paper(
        conn,
        [
            ("Muon Optimizer", [0], {0}),
            ("Extra Rotary Cache", [1], {1}),
        ],
    )

    result = evaluate_extraction(
        conn,
        paper_id,
        ["Muon optimizer", "Sliding Window Attention"],
    )

    assert result["matched_count"] == 1
    assert result["missing_methods"] == ["Sliding Window Attention"]
    assert result["extra_methods"] == ["Extra Rotary Cache"]
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["passed"] is False


def test_evaluate_extraction_source_and_evidence_coverage_are_separate():
    conn = _eval_test_conn()
    paper_id = _insert_eval_paper(
        conn,
        [("LayerScale Residual", [0, 1], {0})],
    )

    result = evaluate_extraction(
        conn,
        paper_id,
        [{"name": "layerscale residual", "source_chunk_indices": [0, 1]}],
        min_evidence_coverage=0.75,
    )

    assert result["matched_count"] == 1
    assert result["source_link_coverage"] == 1.0
    assert result["evidence_coverage"] == 0.5
    assert result["passed"] is False


def test_evaluate_extraction_missing_evidence_text_fails_coverage():
    conn = _eval_test_conn()
    paper_id = _insert_eval_paper(
        conn,
        [("Quote Matched Method", [0], {0})],
    )

    result = evaluate_extraction(
        conn,
        paper_id,
        [
            {
                "name": "quote matched method",
                "source_chunk_indices": [0],
                "evidence_contains": ["absent citation quote"],
            }
        ],
        min_evidence_coverage=1.0,
    )

    assert result["matched_count"] == 1
    assert result["source_link_coverage"] == 1.0
    assert result["evidence_coverage"] == 0.5
    assert result["matched_methods"][0]["source_expectation_met"] is False
    assert result["thresholds"]["min_evidence_coverage"]["passed"] is False
    assert result["passed"] is False


def test_evaluate_extraction_requires_expected_methods():
    conn = _eval_test_conn()
    paper_id = _insert_eval_paper(conn, [("Any Method", [0], {0})])

    try:
        evaluate_extraction(conn, paper_id, [])
        assert False, "Expected empty expected_methods to raise ValueError"
    except ValueError as exc:
        assert "expected_methods" in str(exc)


def test_evaluate_extraction_is_pure_and_echoes_notes():
    conn = _eval_test_conn()
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'extraction_evals'"
    ).fetchone()[0] == 0

    paper_id = _insert_eval_paper(conn, [("Evidence Method", [0], {0})])
    result = evaluate_extraction(conn, paper_id, ["Evidence Method"], notes="lab eval")

    assert result["matched_count"] == 1
    assert result["passed"] is True
    assert result["notes"] == "lab eval"
    assert "persisted" not in result
    assert "eval_id" not in result
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'extraction_evals'"
    ).fetchone()[0] == 0


def test_initialize_db_enables_foreign_keys_and_cascades():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    cur = conn.cursor()
    cur.execute("INSERT INTO papers (title) VALUES (?)", ("paper",))
    paper_id = cur.lastrowid
    cur.execute("INSERT INTO chunks (paper_id, content) VALUES (?, ?)", (paper_id, "chunk"))
    chunk_id = cur.lastrowid
    cur.execute("INSERT INTO methods (name) VALUES (?)", ("method",))
    method_id = cur.lastrowid
    cur.execute(
        "INSERT INTO method_sources (method_id, chunk_id) VALUES (?, ?)",
        (method_id, chunk_id),
    )
    conn.commit()

    cur.execute("DELETE FROM methods WHERE method_id = ?", (method_id,))
    conn.commit()
    remaining = cur.execute("SELECT COUNT(*) FROM method_sources").fetchone()[0]
    assert remaining == 0


def test_initialize_db_creates_graph_tables_and_cascades_links():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)

    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "code_symbols" in tables
    assert "code_symbols_fts" in tables
    assert "method_code_targets" in tables
    assert "experiment_methods" in tables
    assert "lessons" in tables

    fts_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'code_symbols_fts'"
    ).fetchone()["sql"].lower()
    normalized_fts_sql = fts_sql.replace('"', "'").replace(" ", "")
    assert (
        "content='code_symbols'" in normalized_fts_sql
        or "content=code_symbols" in normalized_fts_sql
    )
    assert (
        "content_rowid='symbol_id'" in normalized_fts_sql
        or "content_rowid=symbol_id" in normalized_fts_sql
    )

    cur = conn.cursor()
    cur.execute("INSERT INTO methods (name) VALUES (?)", ("Cascade Graph Method",))
    method_id = cur.lastrowid
    cur.execute(
        "INSERT INTO experiments (commit_hash, status) VALUES (?, ?)",
        ("abc123", "keep"),
    )
    experiment_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO method_code_targets
        (method_id, file_path, symbol_name, symbol_kind, rationale)
        VALUES (?, ?, ?, ?, ?)
        """,
        (method_id, "train.py", "MLP", "class", "method maps to MLP"),
    )
    cur.execute(
        """
        INSERT INTO experiment_methods
        (experiment_id, method_id, role, note)
        VALUES (?, ?, ?, ?)
        """,
        (experiment_id, method_id, "retrieved", "seed note"),
    )
    conn.commit()

    cur.execute("DELETE FROM methods WHERE method_id = ?", (method_id,))
    conn.commit()
    assert cur.execute("SELECT COUNT(*) FROM method_code_targets").fetchone()[0] == 0
    assert cur.execute("SELECT COUNT(*) FROM experiment_methods").fetchone()[0] == 0

    cur.execute("INSERT INTO methods (name) VALUES (?)", ("Experiment Cascade Method",))
    method_id = cur.lastrowid
    cur.execute(
        "INSERT INTO experiments (commit_hash, status) VALUES (?, ?)",
        ("def456", "discard"),
    )
    experiment_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO experiment_methods
        (experiment_id, method_id, role, note)
        VALUES (?, ?, ?, ?)
        """,
        (experiment_id, method_id, "retrieved", None),
    )
    conn.commit()

    cur.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
    conn.commit()
    assert cur.execute("SELECT COUNT(*) FROM experiment_methods").fetchone()[0] == 0


def test_method_feedback_schema_fresh_and_migrated():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)

    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert "method_feedback" in tables
    columns = {
        row["name"]: row
        for row in conn.execute("PRAGMA table_info(method_feedback)")
    }
    assert columns["event_id"]["notnull"] == 0
    assert columns["method_id"]["notnull"] == 1
    assert columns["rating"]["notnull"] == 1

    table_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'method_feedback'"
    ).fetchone()["sql"]
    assert "rating >= -1.0" in table_sql
    assert "rating <= 1.0" in table_sql
    assert "ON DELETE SET NULL" in table_sql

    conn.execute("DROP TABLE method_feedback")
    migrate_db(conn)
    migrate_db(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'method_feedback'"
    ).fetchone()[0] == 1


def test_paper_extraction_status_metadata_schema_fresh_and_migrated():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)

    columns = {
        row["name"]: row
        for row in conn.execute("PRAGMA table_info(paper_extraction_status)")
    }
    assert columns["confidence_score"]["type"] == "REAL"
    assert columns["confidence_score"]["notnull"] == 0
    assert columns["confidence_reason"]["type"] == "TEXT"
    assert columns["confidence_reason"]["notnull"] == 0
    assert columns["extractor_model"]["type"] == "TEXT"
    assert columns["extractor_model"]["notnull"] == 0
    assert columns["extractor_prompt_version"]["type"] == "TEXT"
    assert columns["extractor_prompt_version"]["notnull"] == 0

    legacy = sqlite3.connect(":memory:")
    legacy.row_factory = sqlite3.Row
    legacy.executescript(
        """
        CREATE TABLE paper_extraction_status (
            paper_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('skipped', 'extracted', 'failed')),
            source_type TEXT,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            method_count INTEGER NOT NULL DEFAULT 0,
            source_char_count INTEGER NOT NULL DEFAULT 0,
            compressed_char_count INTEGER NOT NULL DEFAULT 0,
            compression_ratio REAL,
            error_text TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    migrate_db(legacy)
    migrate_db(legacy)
    migrated_columns = {
        row["name"]: row
        for row in legacy.execute("PRAGMA table_info(paper_extraction_status)")
    }
    assert migrated_columns["confidence_score"]["type"] == "REAL"
    assert migrated_columns["confidence_reason"]["type"] == "TEXT"
    assert migrated_columns["extractor_model"]["type"] == "TEXT"
    assert migrated_columns["extractor_prompt_version"]["type"] == "TEXT"



def test_refresh_code_map_indexes_temp_python_symbols_and_searches():
    import tempfile
    from pathlib import Path
    from researchfs import codegraph

    conn = _in_memory_db()
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "model_snippet.py"
        source_path.write_text(
            '''"""Temporary module for code graph tests."""

class TinyBlock:
    """Tiny block class docstring."""

    def forward(self, x):
        return x + 1


def top_attention(x):
    """Applies attention transform."""
    return TinyBlock().forward(x)
''',
            encoding="utf-8",
        )

        indexed = codegraph.refresh_code_map(paths=(str(source_path),), conn=conn)
        indexed_pairs = {
            (item["symbol_name"], item["symbol_kind"])
            for item in indexed
        }
        assert ("TinyBlock", "class") in indexed_pairs
        assert ("top_attention", "function") in indexed_pairs

        stored_pairs = {
            (row["symbol_name"], row["symbol_kind"])
            for row in conn.execute("SELECT symbol_name, symbol_kind FROM code_symbols")
        }
        assert ("TinyBlock", "class") in stored_pairs
        assert ("top_attention", "function") in stored_pairs

        class_hits = codegraph.search_code_symbols("TinyBlock", conn=conn)
        function_hits = codegraph.search_code_symbols("top_attention", conn=conn)
        assert "TinyBlock" in class_hits
        assert "Tiny block class docstring" in class_hits
        assert "top_attention" in function_hits
        assert "Applies attention transform" in function_hits


def test_link_methods_to_code_links_seeded_methods_to_temp_train_symbols():
    import tempfile
    from pathlib import Path
    from researchfs import codegraph

    conn = _in_memory_db()
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.py"
        train_path.write_text(
            '''class CausalSelfAttention:
    """Attention implementation target."""
    pass


class MLP:
    """Feed-forward MLP target."""
    pass


class MuonAdamW:
    """Muon optimizer target."""
    pass
''',
            encoding="utf-8",
        )
        codegraph.refresh_code_map(paths=(str(train_path),), conn=conn)
        linked_count = codegraph.link_methods_to_code(conn=conn)

    assert linked_count >= 3
    rows = conn.execute(
        """
        SELECT m.name, t.file_path, t.symbol_name, t.symbol_kind
        FROM method_code_targets t
        JOIN methods m ON m.method_id = t.method_id
        """
    ).fetchall()

    def has_link(name_part, symbol_name):
        return any(
            name_part in row["name"]
            and row["symbol_name"] == symbol_name
            and row["symbol_kind"] == "class"
            and row["file_path"].endswith("train.py")
            for row in rows
        )

    assert has_link("MLP", "MLP")
    assert has_link("Attention", "CausalSelfAttention")
    assert has_link("Muon", "MuonAdamW")


def test_chunk_text_overlap_and_boundaries():
    text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # Consecutive chunks must overlap (last chunk may be shorter / non-overlapping tail)
    for a, b in zip(chunks[:-1], chunks[1:]):
        assert any(fragment in b for fragment in a[-20:]), "Chunks should overlap"

    # Short text should return a single non-empty chunk
    short = "Tiny."
    assert chunk_text(short, chunk_size=1000, overlap=100) == ["Tiny."]


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("long enough text", chunk_size=10, overlap=10)
        assert False, "Expected invalid overlap to raise ValueError"
    except ValueError as exc:
        assert "overlap" in str(exc)


def test_extract_json_block_variants():
    block = '```json\n{"methods": [{"name": "x"}]}\n```'
    assert extract_json_block(block) == '{"methods": [{"name": "x"}]}'

    raw = '{"methods": [{"name": "y"}]}'
    assert extract_json_block(raw) == raw

    fallback = "plain text"
    assert extract_json_block(fallback) == fallback

def test_normalize_extracted_method_filters_invalid_source_indices():
    method = _normalize_extracted_method(
        {
            "name": "Index Filter Method",
            "description": "Filters bad source indices.",
            "implementation_hint": "Use only integer chunk indices.",
            "risks": "none",
            "category": "attention",
            "source_chunk_indices": [0, True, "1", 2.0, 3, False, -1],
        }
    )
    assert method is not None
    assert method["source_chunk_indices"] == [0, 3]
    assert method["category"] == "attention"

    method_with_mixed_case_category = _normalize_extracted_method(
        {"name": "Category Case", "category": " Optimizer ", "source_chunk_indices": [0]}
    )
    assert method_with_mixed_case_category is not None
    assert method_with_mixed_case_category["category"] == "optimizer"

    method_with_unknown_category = _normalize_extracted_method(
        {"name": "Category Fallback", "category": "scheduler", "source_chunk_indices": [0]}
    )
    assert method_with_unknown_category is not None
    assert method_with_unknown_category["category"] == "method"

    method_with_scalar_indices = _normalize_extracted_method(
        {"name": "Scalar Indices", "source_chunk_indices": "0,1"}
    )
    assert method_with_scalar_indices is not None
    assert method_with_scalar_indices["source_chunk_indices"] == []


def test_assemble_extractor_prompt_json_example_has_no_comments():
    prompt = assemble_extractor_prompt(
        {"title": "Prompt Safety", "authors": "Tester", "abstract": "JSON examples must parse."},
        ["Chunk zero describes a method.", "Chunk one supports it."],
    )
    assert f"Prompt contract version: {PROMPT_VERSION}" in prompt
    assert "using the local CHUNK labels shown in this prompt" in prompt
    assert "Do not infer methods from title, authors, abstract, or outside knowledge" in prompt
    assert PROMPT_VERSION == "researchfs-extractor-windowed-methods-v3"
    assert "research artifact" in prompt
    assert "paper, lab note, experiment report, or repository README" in prompt
    assert "implementation lessons" in prompt
    expected_categories = [
        "attention",
        "mlp",
        "optimizer",
        "normalization",
        "precision",
        "regularization",
        "embeddings",
        "method",
        "architecture",
        "evaluation",
        "dataset",
        "instrumentation",
        "protocol",
        "implementation",
        "risk_mitigation",
    ]
    expected_category_contract = (
        '      "category": "one of: ' + ", ".join(METHOD_CATEGORY_TOKENS) + '",'
    )
    assert list(METHOD_CATEGORY_TOKENS) == expected_categories
    assert expected_category_contract in prompt.splitlines()
    assert "deep learning paper" not in prompt
    assert "showing how to write it in PyTorch" not in prompt
    json_example = extract_json_block(prompt)
    assert "//" not in json_example
    assert json.loads(json_example)["methods"][0]["source_chunk_indices"] == [0, 1]


def test_extract_methods_from_chunks_windows_all_chunks_and_remaps_indices():
    chunks = [f"chunk {i} evidence" for i in range(35)]
    calls = []

    def stub_llm(prompt):
        calls.append(prompt)
        assert "=== CHUNK 30 ===" not in prompt
        methods = []
        if "chunk 29 evidence" in prompt:
            methods.append(
                {
                    "name": "Window Union Method",
                    "description": "First support in the end of the first window.",
                    "implementation_hint": "Use the late first-window evidence.",
                    "risks": "none",
                    "category": "attention",
                    "source_chunk_indices": [29],
                }
            )
        if "chunk 30 evidence" in prompt:
            methods.append(
                {
                    "name": "Window Union Method",
                    "description": "Duplicate support in the start of the second window.",
                    "implementation_hint": "Use the late second-window evidence.",
                    "risks": "none",
                    "category": "attention",
                    "source_chunk_indices": [0],
                }
            )
        if "chunk 31 evidence" in prompt:
            methods.append(
                {
                    "name": "Late Window Method",
                    "description": "Support appears only after the first extractor window.",
                    "implementation_hint": "Read the late local chunk.",
                    "risks": "none",
                    "category": "optimizer",
                    "source_chunk_indices": [1],
                }
            )
        return "```json\n" + json.dumps({"methods": methods}) + "\n```"

    result = extract_methods_from_chunks(
        {"title": "Windowed Extraction", "authors": "Tester", "abstract": ""},
        chunks,
        stub_llm,
    )

    assert len(calls) == 2
    methods = {method["name"]: method for method in result["methods"]}
    assert methods["Window Union Method"]["source_chunk_indices"] == [29, 30]
    assert methods["Late Window Method"]["source_chunk_indices"] == [31]


def test_extract_methods_from_chunks_preserves_invalid_methods_list_failure():
    def stub_llm(prompt):
        if "chunk 30 evidence" in prompt:
            return "```json\n" + json.dumps({"methods": "not a list"}) + "\n```"
        return "```json\n" + json.dumps({"methods": []}) + "\n```"

    try:
        extract_methods_from_chunks(
            {"title": "Invalid Window", "authors": "Tester", "abstract": ""},
            [f"chunk {i} evidence" for i in range(31)],
            stub_llm,
        )
    except ValueError as exc:
        assert "methods' must be a list" in str(exc)
    else:
        raise AssertionError("Expected invalid extractor methods field to fail")


def test_extract_json_block_parses_extractor_style_response():
    response = """```json
{
  "methods": [
    {
      "name": "Grouped Query Attention",
      "description": "Shares key/value heads across query groups to reduce KV cache memory.",
      "implementation_hint": "Configure n_kv_heads lower than n_heads in attention.",
      "risks": "May reduce quality if the number of KV heads is too small.",
      "category": "attention",
      "source_chunk_indices": [0, 2]
    }
  ]
}
```"""
    parsed = json.loads(extract_json_block(response))
    assert parsed["methods"][0]["name"] == "Grouped Query Attention"
    assert parsed["methods"][0]["source_chunk_indices"] == [0, 2]


def test_clean_latex():
    latex = "Hello % comment here\nworld!\n\n% another comment\n"
    cleaned = clean_latex(latex)
    assert "comment" not in cleaned
    assert "another" not in cleaned
    assert "Hello " in cleaned
    assert "world!" in cleaned
    # Blank lines should be removed
    assert "\n\n" not in cleaned


def test_get_method_pack_contains_methods():
    conn = _in_memory_db()
    pack = rfs.get_method_pack(conn)
    assert "SwiGLU MLP" in pack
    assert "Grouped-Query Attention (GQA)" in pack
    assert "Muon LR Tuning" in pack
    assert "Sliding Window Attention Patterns" in pack
    assert "Hashed n-gram Value Embeddings" in pack
    assert "Token Shifting" in pack
    assert "FP8 / bf16 Training" in pack
    assert "https://arxiv.org/abs/2305.13245" in pack

def test_researchfs_package_root_exports_public_api():
    import researchfs
    from researchfs import bundle, codegraph, ingest, retrieve

    expected_exports = {
        "ResearchFSClient": researchfs.ResearchFSClient,
        "ResearchFSLabClient": researchfs.ResearchFSLabClient,
        "ResearchFSLabClientError": researchfs.ResearchFSLabClientError,
        "export_paper_memory_bundle": bundle.export_paper_memory_bundle,
        "import_paper_memory_bundle": bundle.import_paper_memory_bundle,
        "ensure_seed": ingest.ensure_seed,
        "finish_agent_run": retrieve.finish_agent_run,
        "get_code_map": codegraph.get_code_map,
        "get_connection": researchfs.get_connection,
        "get_experiment_brief": retrieve.get_experiment_brief,
        "get_graph_context": retrieve.get_graph_context,
        "get_method_history": retrieve.get_method_history,
        "get_method_pack": retrieve.get_method_pack,
        "initialize_db": researchfs.initialize_db,
        "link_experiment_methods": retrieve.link_experiment_methods,
        "link_methods_to_code": codegraph.link_methods_to_code,
        "log_experiment": retrieve.log_experiment,
        "log_lesson": retrieve.log_lesson,
        "log_retrieval_event": retrieve.log_retrieval_event,
        "log_token_usage": retrieve.log_token_usage,
        "recommend_next_experiments": retrieve.recommend_next_experiments,
        "refresh_code_map": codegraph.refresh_code_map,
        "search_code_symbols": codegraph.search_code_symbols,
        "search_methods": retrieve.search_methods,
        "seed_initial_data": ingest.seed_initial_data,
        "start_agent_run": retrieve.start_agent_run,
        "update_experiment": retrieve.update_experiment,
    }

    for name, expected_obj in expected_exports.items():
        assert name in researchfs.__all__
        assert getattr(researchfs, name) is expected_obj





def test_search_methods():
    conn = _in_memory_db()
    
    # Test specific keyword
    pack = rfs.search_methods("SwiGLU", conn)
    assert "SwiGLU MLP" in pack
    assert "Grouped-Query Attention" not in pack
    
    # Test category search
    pack_attn = rfs.search_methods("attention", conn)
    assert "Grouped-Query Attention" in pack_attn
    assert "Token Shifting" in pack_attn
    assert "SwiGLU MLP" not in pack_attn
    
    # Test case-insensitivity
    pack_case = rfs.search_methods("muon", conn)
    assert "Muon LR Tuning" in pack_case
    
    # Test empty query fallback
    pack_empty = rfs.search_methods("", conn)
    assert "SwiGLU MLP" in pack_empty
    assert "Grouped-Query Attention" in pack_empty


def test_search_methods_no_match_does_not_append_general_methods():
    conn = _in_memory_db()

    pack, ids = rfs.search_methods_and_ids("definitelynomatchsentinel", conn)

    assert ids == []
    assert "No methods found matching query" in pack
    assert "# General Optimization & Hyperparameter Tuning Methods" not in pack
    assert "## " not in pack



def test_lesson_retrieval_search_methods_direct_and_experiment_links():
    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        (
            "Lesson Linked Retrieval",
            "A method whose searchable evidence lives only in lab lessons.",
            "Use the implementation described by the retrieved lesson.",
            "Lesson evidence can go stale if not linked.",
            "memory",
        ),
    )
    direct_method_id = cur.lastrowid
    direct_lesson_id = rfs.log_lesson(
        "Direct lesson retrieval gate",
        "Direct lessononlyretrievaltoken should retrieve the linked method without paper text.",
        outcome="keep",
        method_id=direct_method_id,
        conn=conn,
    )

    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        (
            "Experiment Linked Retrieval",
            "A method linked through an experiment lesson rather than its own text.",
            "Follow the experiment method association.",
            "Missing experiment links hide useful lessons.",
            "memory",
        ),
    )
    experiment_method_id = cur.lastrowid
    experiment_id = rfs.log_experiment(
        "lesson-exp-link",
        val_bpb=1.9,
        memory_gb=0.5,
        status="keep",
        description="Experiment connected to lesson retrieval.",
        conn=conn,
    )
    rfs.link_experiment_methods(experiment_id, [experiment_method_id], conn=conn)
    experiment_lesson_id = rfs.log_lesson(
        "Experiment lesson retrieval gate",
        "Experiment lessonexperimentretrievaltoken should retrieve the experiment-linked method.",
        outcome="keep",
        experiment_id=experiment_id,
        conn=conn,
    )

    direct_pack, direct_ids = rfs.search_methods_and_ids(
        "lessononlyretrievaltoken", conn, max_chars=1200
    )
    assert direct_method_id in direct_ids
    assert "Lesson Linked Retrieval" in direct_pack
    assert "Direct lesson retrieval gate" in direct_pack
    assert "Direct lessononlyretrievaltoken" in direct_pack

    scored_pack, scored_ids, scores = rfs.search_methods_with_scores(
        "lessononlyretrievaltoken", conn, max_chars=1200
    )
    assert scored_pack == direct_pack
    assert direct_method_id in scored_ids
    assert any(score["method_id"] == direct_method_id for score in scores)
    direct_score = next(score for score in scores if score["method_id"] == direct_method_id)
    assert direct_score["lesson_match_count"] > 0
    assert direct_score["lesson_match_score"] > 0
    assert str(direct_lesson_id) in scored_pack or "Direct lesson retrieval gate" in scored_pack

    experiment_pack, experiment_ids = rfs.search_methods_and_ids(
        "lessonexperimentretrievaltoken", conn, max_chars=1200
    )
    assert experiment_method_id in experiment_ids
    assert "Experiment Linked Retrieval" in experiment_pack
    assert "Experiment lesson retrieval gate" in experiment_pack
    assert "lessonexperimentretrievaltoken" in experiment_pack
    assert str(experiment_lesson_id) in experiment_pack or "Experiment lesson retrieval gate" in experiment_pack

def test_fts_retrieves_method_by_chunk_term():
    conn = _in_memory_db()
    cur = conn.cursor()

    # Seed a paper + chunk whose content contains terms absent from the method name/description.
    cur.execute(
        "INSERT INTO papers (title, authors, abstract, url, publish_date) VALUES (?, ?, ?, ?, ?)",
        ("Muon Optimizer Paper", "Test Author", "Abstract", "https://example.com/muon", "2026-06-27"),
    )
    paper_id = cur.lastrowid
    cur.execute(
        "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
        (
            paper_id,
            "Introduction",
            "The Newton-Schulz iterations produce orthogonal weight matrices for stable optimization.",
        ),
    )
    chunk_id = cur.lastrowid

    # Method name/description deliberately do NOT mention Newton-Schulz/orthogonal.
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Muon", "Matrix optimizer via polar decomposition", "Use with large batches", "Instability", "optimizer"),
    )
    method_id = cur.lastrowid
    cur.execute(
        "INSERT INTO method_sources (method_id, chunk_id) VALUES (?, ?)",
        (method_id, chunk_id),
    )

    pack, ids = rfs.search_methods_and_ids("Newton-Schulz", conn)
    assert method_id in ids, "Expected Muon to be retrieved via chunk provenance for 'Newton-Schulz'"
    assert "Muon" in pack

    pack2, ids2 = rfs.search_methods_and_ids("orthogonal", conn)
    assert method_id in ids2, "Expected Muon to be retrieved via chunk provenance for 'orthogonal'"


def test_fts_ranking_prefers_successful_method():
    conn = _in_memory_db()
    cur = conn.cursor()

    # Two methods match the same query term in their descriptions.
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Alpha Method", "A rankingtest optimizer for experiments", "Hint", "Risk", "test"),
    )
    alpha_id = cur.lastrowid
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Beta Method", "Another rankingtest optimizer for experiments", "Hint", "Risk", "test"),
    )
    beta_id = cur.lastrowid

    def make_run_and_experiment(method_id, final_status):
        run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
        exp_id = rfs.log_experiment("abc123", 1.0, 0.5, "pending", "test", run_id, conn)
        rfs.log_retrieval_event(
            "search", "rankingtest", "content", agent_run_id=run_id, experiment_id=exp_id, conn=conn, method_ids=[method_id]
        )
        rfs.update_experiment(exp_id, status=final_status, conn=conn)
        return exp_id

    make_run_and_experiment(alpha_id, "keep")
    make_run_and_experiment(beta_id, "discard")

    pack, ids = rfs.search_methods_and_ids("rankingtest", conn)
    assert alpha_id in ids
    assert beta_id in ids
    assert ids.index(alpha_id) < ids.index(beta_id), "Expected the 100% successful method to rank first"


def test_score_methods_prefers_better_bpb_delta():
    conn = _in_memory_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Delta Good", "A deltatest optimizer for experiments", "Hint", "Risk", "test"),
    )
    good_id = cur.lastrowid
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Delta Bad", "Another deltatest optimizer for experiments", "Hint", "Risk", "test"),
    )
    bad_id = cur.lastrowid

    def make_keep_run(method_id, val_bpb):
        run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
        rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", run_id, conn)
        exp_id = rfs.log_experiment("abc123", val_bpb, 0.5, "pending", "test", run_id, conn)
        rfs.log_retrieval_event(
            "search", "deltatest", "content", agent_run_id=run_id, experiment_id=exp_id, conn=conn, method_ids=[method_id]
        )
        rfs.update_experiment(exp_id, status="keep", conn=conn)
        return run_id

    make_keep_run(good_id, 1.8)  # delta -0.2
    make_keep_run(bad_id, 1.9)   # delta -0.1

    cur.execute(
        "SELECT method_id, name, description, implementation_hint, risks, category FROM methods WHERE method_id IN (?, ?)",
        (good_id, bad_id),
    )
    methods = [dict(row) for row in cur.fetchall()]
    scored, ids = rfs._score_methods(
        methods, conn, {good_id: 1.0, bad_id: 1.0}
    )
    assert ids.index(good_id) < ids.index(bad_id), "Expected better BPB delta to rank first"


def test_score_methods_attaches_json_safe_score_details():
    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Score Detail", "A scoredetail optimizer for explainability", "Hint", "Risk", "test"),
    )
    method_id = cur.lastrowid
    rfs.log_method_feedback(method_id, 1.0, label="useful", conn=conn)

    agent_run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    experiment_id = rfs.log_experiment("abc123", 2.1, 0.5, "pending", "test", agent_run_id, conn)
    rfs.log_retrieval_event(
        "agent_search",
        "scoredetail",
        "content",
        agent_run_id=agent_run_id,
        experiment_id=experiment_id,
        conn=conn,
        method_ids=[method_id],
    )
    rfs.update_experiment(experiment_id, status="discard", conn=conn)

    cur.execute(
        "SELECT method_id, name, description, implementation_hint, risks, category FROM methods WHERE method_id = ?",
        (method_id,),
    )
    scored, ids = rfs._score_methods(
        [dict(cur.fetchone())], conn, {method_id: -1.0}, agent_run_id=agent_run_id
    )

    assert ids == [method_id]
    details = scored[0]["score_details"]
    for key in (
        "final_score",
        "base_score",
        "rank_score",
        "delta_score",
        "fatigue_multiplier",
        "feedback_multiplier",
        "stats",
    ):
        assert key in details
    assert details["method_id"] == method_id
    assert details["method_rank"] == -1.0
    assert details["normalized_rank"] == 0.0
    assert details["fatigue_multiplier"] == 0.5
    assert details["feedback_multiplier"] == rfs.MAX_FEEDBACK_MULTIPLIER
    assert details["stats"]["discards"] == 1
    assert details["stats"]["feedback_count"] == 1
    json.dumps(details)


def test_feedback_multiplier_is_bounded():
    assert rfs._feedback_multiplier(10, 9.0) == rfs.MAX_FEEDBACK_MULTIPLIER
    assert rfs._feedback_multiplier(10, -9.0) == rfs.MIN_FEEDBACK_MULTIPLIER
    assert rfs._feedback_multiplier(0, 9.0) == 1.0


def test_search_methods_with_scores_matches_method_id_order():
    conn = _in_memory_db()

    pack, ids = rfs.search_methods_and_ids("SwiGLU", conn)
    scored_pack, scored_ids, scores = rfs.search_methods_with_scores("SwiGLU", conn)

    assert scored_pack == pack
    assert scored_ids == ids
    assert [score["method_id"] for score in scores] == scored_ids
    assert all("final_score" in score for score in scores)


def test_log_method_feedback_validates_rating_and_summarizes():
    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Feedback Summary", "A feedbacksummary optimizer", "Hint", "Risk", "test"),
    )
    method_id = cur.lastrowid
    event_id = rfs.log_retrieval_event(
        "agent_search",
        "feedbacksummary",
        "content",
        conn=conn,
        method_ids=[method_id],
    )

    feedback_id = rfs.log_method_feedback(
        method_id,
        0.75,
        event_id=event_id,
        label="useful",
        note="helped select an experiment",
        source="pytest",
        conn=conn,
    )
    assert feedback_id > 0

    row = conn.execute(
        """
        SELECT event_id, method_id, rating, label, note, source
        FROM method_feedback
        WHERE feedback_id = ?
        """,
        (feedback_id,),
    ).fetchone()
    assert row["event_id"] == event_id
    assert row["method_id"] == method_id
    assert row["rating"] == 0.75
    assert row["label"] == "useful"
    assert row["source"] == "pytest"

    summary = rfs.get_method_feedback_summary(method_id, conn)
    assert summary["feedback_count"] == 1
    assert summary["feedback_avg_rating"] == 0.75
    assert summary["positive_feedback_count"] == 1
    stats = rfs._get_empirical_stats(method_id, conn, {})
    assert stats["feedback_count"] == 1
    assert stats["feedback_avg_rating"] == 0.75

    for invalid_rating in (-1.01, 1.01, "good", True):
        try:
            rfs.log_method_feedback(method_id, invalid_rating, conn=conn)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid feedback rating {invalid_rating!r} to fail")

    assert rfs.get_method_feedback_summary(method_id, conn)["feedback_count"] == 1


def test_positive_feedback_boosts_comparable_same_query_method():
    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Feedback Neutral Alpha", "feedbackpos comparable optimizer", "Hint", "Risk", "test"),
    )
    neutral_id = cur.lastrowid
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Feedback Boosted Zeta", "feedbackpos comparable optimizer", "Hint", "Risk", "test"),
    )
    boosted_id = cur.lastrowid
    rfs.log_method_feedback(boosted_id, 1.0, label="useful", conn=conn)

    pack, ids = rfs.search_methods_and_ids("feedbackpos", conn)

    assert neutral_id in ids
    assert boosted_id in ids
    assert ids.index(boosted_id) < ids.index(neutral_id)
    assert "Lab Feedback:".lower() in pack.lower()


def test_negative_feedback_penalizes_comparable_same_query_method():
    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Feedback Penalized Alpha", "feedbackneg comparable optimizer", "Hint", "Risk", "test"),
    )
    penalized_id = cur.lastrowid
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Feedback Neutral Zeta", "feedbackneg comparable optimizer", "Hint", "Risk", "test"),
    )
    neutral_id = cur.lastrowid
    rfs.log_method_feedback(penalized_id, -1.0, label="not_useful", conn=conn)

    _pack, ids = rfs.search_methods_and_ids("feedbackneg", conn)

    assert neutral_id in ids
    assert penalized_id in ids
    assert ids.index(neutral_id) < ids.index(penalized_id)


def test_fatigue_penalizes_discarded_method():
    conn = _in_memory_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Fatigue Tainted", "A fatiguetest optimizer for experiments", "Hint", "Risk", "test"),
    )
    tainted_id = cur.lastrowid
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Fatigue Clean", "Another fatiguetest optimizer for experiments", "Hint", "Risk", "test"),
    )
    clean_id = cur.lastrowid

    # Align empirical success rates: tainted has one previous keep (delta zero)
    # plus a current discard; clean has no runs. Both end at 0.5 success rate.
    prev_run = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", prev_run, conn)
    exp_prev = rfs.log_experiment("abc123", 2.0, 0.5, "pending", "test", prev_run, conn)
    rfs.log_retrieval_event(
        "agent_search", "fatiguetest", "content", agent_run_id=prev_run, experiment_id=exp_prev, conn=conn, method_ids=[tainted_id]
    )
    rfs.update_experiment(exp_prev, status="keep", conn=conn)

    # Current run: tainted method is retrieved then discarded.
    current_run = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", current_run, conn)
    exp_cur = rfs.log_experiment("abc456", 2.1, 0.5, "pending", "test", current_run, conn)
    rfs.log_retrieval_event(
        "agent_search", "fatiguetest", "content", agent_run_id=current_run, experiment_id=exp_cur, conn=conn, method_ids=[tainted_id]
    )
    rfs.update_experiment(exp_cur, status="discard", conn=conn)

    cur.execute(
        "SELECT method_id, name, description, implementation_hint, risks, category FROM methods WHERE method_id IN (?, ?)",
        (tainted_id, clean_id),
    )
    methods = [dict(row) for row in cur.fetchall()]
    scored, ids = rfs._score_methods(
        methods, conn, {tainted_id: 1.0, clean_id: 1.0}, agent_run_id=current_run
    )
    assert ids.index(clean_id) < ids.index(tainted_id), "Expected clean method to rank above discarded-fatigue method"


def test_fatigue_boosts_kept_method():
    conn = _in_memory_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Fatigue Boosted", "A fatiguetest optimizer for experiments", "Hint", "Risk", "test"),
    )
    boosted_id = cur.lastrowid
    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Fatigue Neutral", "Another fatiguetest optimizer for experiments", "Hint", "Risk", "test"),
    )
    neutral_id = cur.lastrowid

    # Align empirical success rates: neutral has one previous keep and one previous
    # discard (delta zero), giving 0.5. Boosted has a previous discard and a current
    # keep, also 0.5, plus the kept-method fatigue boost.
    prev_run_neutral = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", prev_run_neutral, conn)
    exp_n_keep = rfs.log_experiment("abc123", 2.0, 0.5, "pending", "test", prev_run_neutral, conn)
    rfs.log_retrieval_event(
        "agent_search", "fatiguetest", "content", agent_run_id=prev_run_neutral, experiment_id=exp_n_keep, conn=conn, method_ids=[neutral_id]
    )
    rfs.update_experiment(exp_n_keep, status="keep", conn=conn)
    exp_n_discard = rfs.log_experiment("abc124", 2.1, 0.5, "pending", "test", prev_run_neutral, conn)
    rfs.log_retrieval_event(
        "agent_search", "fatiguetest", "content", agent_run_id=prev_run_neutral, experiment_id=exp_n_discard, conn=conn, method_ids=[neutral_id]
    )
    rfs.update_experiment(exp_n_discard, status="discard", conn=conn)

    prev_run_boosted = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", prev_run_boosted, conn)
    exp_b_discard = rfs.log_experiment("abc123", 2.1, 0.5, "pending", "test", prev_run_boosted, conn)
    rfs.log_retrieval_event(
        "agent_search", "fatiguetest", "content", agent_run_id=prev_run_boosted, experiment_id=exp_b_discard, conn=conn, method_ids=[boosted_id]
    )
    rfs.update_experiment(exp_b_discard, status="discard", conn=conn)

    # Current run: boosted method is retrieved and kept.
    current_run = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", current_run, conn)
    exp_cur = rfs.log_experiment("abc456", 2.0, 0.5, "pending", "test", current_run, conn)
    rfs.log_retrieval_event(
        "agent_search", "fatiguetest", "content", agent_run_id=current_run, experiment_id=exp_cur, conn=conn, method_ids=[boosted_id]
    )
    rfs.update_experiment(exp_cur, status="keep", conn=conn)

    cur.execute(
        "SELECT method_id, name, description, implementation_hint, risks, category FROM methods WHERE method_id IN (?, ?)",
        (boosted_id, neutral_id),
    )
    methods = [dict(row) for row in cur.fetchall()]
    scored, ids = rfs._score_methods(
        methods, conn, {boosted_id: 1.0, neutral_id: 1.0}, agent_run_id=current_run
    )
    assert ids.index(boosted_id) < ids.index(neutral_id), "Expected kept-fatigue method to rank above neutral method"

def test_fatigue_ignores_non_search_events_with_method_ids():
    conn = _in_memory_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO methods (name, description, implementation_hint, risks, category) VALUES (?, ?, ?, ?, ?)",
        ("Fatigue Non Search", "A fatiguefilter optimizer for experiments", "Hint", "Risk", "test"),
    )
    method_id = cur.lastrowid

    current_run = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    exp_non_search = rfs.log_experiment("abc123", 2.1, 0.5, "pending", "test", current_run, conn)
    rfs.log_retrieval_event(
        "method_pack",
        "all",
        "content",
        agent_run_id=current_run,
        experiment_id=exp_non_search,
        conn=conn,
        method_ids=[method_id],
    )
    rfs.update_experiment(exp_non_search, status="discard", conn=conn)

    assert method_id not in rfs._get_method_fatigue_scores(current_run, conn)

    exp_search = rfs.log_experiment("abc456", 2.0, 0.5, "pending", "test", current_run, conn)
    rfs.log_retrieval_event(
        "agent_search_reuse",
        "fatiguefilter",
        "content",
        agent_run_id=current_run,
        experiment_id=exp_search,
        conn=conn,
        method_ids=[method_id],
    )
    rfs.update_experiment(exp_search, status="keep", conn=conn)

    assert rfs._get_method_fatigue_scores(current_run, conn)[method_id] == 1.1



def test_ingest_paper_without_llm():
    conn = _in_memory_db()
    meta = {
        "title": "Test Paper Without LLM",
        "authors": "Test Author",
        "abstract": "A paper about testing.",
        "url": "https://example.com/test",
        "publish_date": "2026-06-26",
        "content": "\n\n".join([f"This is paragraph {i} of the test paper." for i in range(5)]),
    }
    result = ingest_paper(meta, conn=conn)

    # Verify paper was inserted
    assert result["paper_id"] is not None
    cur = conn.cursor()
    cur.execute("SELECT * FROM papers WHERE paper_id = ?", (result["paper_id"],))
    row = cur.fetchone()
    assert row["title"] == meta["title"]

    # Verify chunks were inserted
    assert len(result["chunk_ids"]) > 0
    cur.execute("SELECT COUNT(*) FROM chunks WHERE paper_id = ?", (result["paper_id"],))
    assert cur.fetchone()[0] == len(result["chunk_ids"])
    status = result["extraction_status"]
    assert status["status"] == "skipped"
    assert status["paper_id"] == result["paper_id"]
    assert status["chunk_count"] == len(result["chunk_ids"])
    assert status["method_count"] == 0
    assert status["source_char_count"] == len(meta["content"])
    assert status["compressed_char_count"] == 0

    assert status["confidence_score"] == 0.0
    assert status["confidence_reason"] == "skipped"

    cur.execute(
        "SELECT status, chunk_count, method_count, confidence_score, confidence_reason FROM paper_extraction_status WHERE paper_id = ?",
        (result["paper_id"],),
    )
    status_row = cur.fetchone()
    assert status_row["status"] == "skipped"
    assert status_row["chunk_count"] == len(result["chunk_ids"])
    assert status_row["method_count"] == 0
    assert status_row["confidence_score"] == 0.0
    assert status_row["confidence_reason"] == "skipped"


    # search_methods should still return the seeded methods without crashing
    pack = rfs.search_methods("", conn)
    assert "SwiGLU MLP" in pack


def test_ingest_paper_with_stub_llm():
    conn = _in_memory_db()
    stub_response = '{"methods":[{"name":"Stub Test Method","description":"A stub keyword test method","implementation_hint":"do nothing","risks":"none","category":"attention","source_chunk_indices":[0]}]}'

    def stub_llm(_prompt):
        return f"```json\n{stub_response}\n```"

    meta = {
        "title": "Test Paper With Stub LLM",
        "authors": "Test Author",
        "abstract": "A paper about stub LLMs.",
        "url": "https://example.com/stub",
        "publish_date": "2026-06-26",
        "content": "This is the content. The stub keyword appears in the evidence window for this method.",
    }
    result = ingest_paper(meta, llm_fn=stub_llm, conn=conn)

    assert result["paper_id"] is not None
    assert len(result["chunk_ids"]) > 0
    assert len(result["method_ids"]) == 1

    status = result["extraction_status"]
    assert status["status"] == "extracted"
    assert status["paper_id"] == result["paper_id"]
    assert status["chunk_count"] == len(result["chunk_ids"])
    assert status["method_count"] == 1
    assert status["compressed_char_count"] > 0
    assert status["compression_ratio"] > 0

    assert status["extractor_prompt_version"] == PROMPT_VERSION
    cur = conn.cursor()
    assert 0.0 < status["confidence_score"] <= 1.0
    assert "methods=1" in status["confidence_reason"]
    assert "valid_source_links=1/1" in status["confidence_reason"]
    assert "method_source_coverage=1/1" in status["confidence_reason"]

    cur = conn.cursor()
    cur.execute(
        "SELECT status, method_count, error_text, confidence_score, confidence_reason, extractor_prompt_version FROM paper_extraction_status WHERE paper_id = ?",
        (result["paper_id"],),
    )
    status_row = cur.fetchone()
    assert status_row["status"] == "extracted"
    assert status_row["method_count"] == 1
    assert status_row["error_text"] is None
    assert status_row["confidence_score"] == status["confidence_score"]
    assert status_row["confidence_reason"] == status["confidence_reason"]
    assert status_row["extractor_prompt_version"] == PROMPT_VERSION
    source_row = cur.execute(
        """
        SELECT ms.evidence_text, ms.evidence_start, ms.evidence_end, ms.evidence_strategy, c.content
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE ms.method_id = ? AND ms.chunk_id = ?
        """,
        (result["method_ids"][0], result["chunk_ids"][0]),
    ).fetchone()
    assert "stub keyword" in source_row["evidence_text"]
    assert source_row["evidence_strategy"] == "matched_term:stub"
    assert source_row["evidence_start"] >= 0
    assert source_row["evidence_end"] > source_row["evidence_start"]
    assert source_row["evidence_text"] == source_row["content"][
        source_row["evidence_start"]:source_row["evidence_end"]
    ]


    pack = rfs.search_methods("stub keyword", conn)
    assert "Stub Test Method" in pack



def test_ingest_paper_links_extracted_late_window_method_to_late_chunk():
    conn = _in_memory_db()
    late_token = "lateprovenancetoken"
    sections = []
    for i in range(38):
        token = late_token if i == 34 else f"fillertoken{i}"
        sections.append(f"Section {i}. {token} " + ("x" * 2100))

    def stub_llm(prompt):
        methods = []
        for part in prompt.split("=== CHUNK ")[1:]:
            idx_text, chunk_text_part = part.split(" ===", 1)
            if late_token in chunk_text_part:
                methods.append(
                    {
                        "name": "Zyxqv Provenance Sentinel",
                        "description": "A unique method whose evidence appears only in a late extractor window.",
                        "implementation_hint": "Apply the lateprovenancetoken transform.",
                        "risks": "none",
                        "category": "optimizer",
                        "source_chunk_indices": [int(idx_text)],
                    }
                )
                break
        return "```json\n" + json.dumps({"methods": methods}) + "\n```"

    result = ingest_paper(
        {
            "title": "Late Window Provenance Paper",
            "authors": "Window Tester",
            "abstract": "A paper with evidence beyond the first extractor window.",
            "url": "https://example.com/late-window-provenance",
            "publish_date": "2026-07-01",
            "content": "\n\n".join(sections),
        },
        llm_fn=stub_llm,
        conn=conn,
    )

    assert len(result["chunk_ids"]) > 30
    source_row = conn.execute(
        """
        SELECT c.chunk_id, c.content
        FROM method_sources ms
        JOIN methods m ON m.method_id = ms.method_id
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE m.name = ? AND c.paper_id = ?
        """,
        ("Zyxqv Provenance Sentinel", result["paper_id"]),
    ).fetchone()
    assert source_row is not None
    linked_index = result["chunk_ids"].index(source_row["chunk_id"])
    assert linked_index > 30
    assert late_token in source_row["content"]
    assert source_row["chunk_id"] != result["chunk_ids"][1]


def test_ingest_paper_duplicate_reextract_links_late_window_method_to_late_chunk():
    conn = _in_memory_db()
    late_token = "reextractlateprovenancetoken"
    sections = []
    for i in range(38):
        token = late_token if i == 34 else f"reextractfillertoken{i}"
        sections.append(f"Section {i}. {token} " + ("y" * 2100))
    meta = {
        "title": "Late Window Duplicate Reextract Paper",
        "authors": "Window Tester",
        "abstract": "A duplicate paper with evidence beyond the first extractor window.",
        "url": "https://example.com/late-window-reextract",
        "publish_date": "2026-07-01",
        "content": "\n\n".join(sections),
    }
    first = ingest_paper(meta, conn=conn)
    assert first["extraction_status"]["status"] == "skipped"
    assert len(first["chunk_ids"]) > 30

    def stub_llm(prompt):
        methods = []
        for part in prompt.split("=== CHUNK ")[1:]:
            idx_text, chunk_text_part = part.split(" ===", 1)
            if late_token in chunk_text_part:
                methods.append(
                    {
                        "name": "Zyxqv Reextract Sentinel",
                        "description": "A unique re-extracted method whose evidence appears only in a late window.",
                        "implementation_hint": "Apply the reextractlateprovenancetoken transform.",
                        "risks": "none",
                        "category": "optimizer",
                        "source_chunk_indices": [int(idx_text)],
                    }
                )
                break
        return "```json\n" + json.dumps({"methods": methods}) + "\n```"

    second = ingest_paper({"title": meta["title"]}, llm_fn=stub_llm, conn=conn)
    assert second["duplicate"] is True
    assert second["paper_id"] == first["paper_id"]
    assert second["chunk_ids"] == first["chunk_ids"]

    source_row = conn.execute(
        """
        SELECT c.chunk_id, c.content
        FROM method_sources ms
        JOIN methods m ON m.method_id = ms.method_id
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE m.name = ? AND c.paper_id = ?
        """,
        ("Zyxqv Reextract Sentinel", first["paper_id"]),
    ).fetchone()
    assert source_row is not None
    linked_index = first["chunk_ids"].index(source_row["chunk_id"])
    assert linked_index > 30
    assert late_token in source_row["content"]
    assert source_row["chunk_id"] != first["chunk_ids"][1]

def test_ingest_duplicate_skipped_paper_reextracts_with_llm_and_remains_idempotent():
    conn = _in_memory_db()
    meta = {
        "title": "Duplicate Reextract Paper",
        "authors": "Retry Author",
        "abstract": "A paper about adding compressed memory after a skipped ingest.",
        "url": "https://example.com/reextract",
        "publish_date": "2026-06-27",
        "content": "The Zorblax Quindle technique is described with enough detail for provenance.",
    }
    first = ingest_paper(meta, conn=conn)
    assert first["method_ids"] == []
    assert first["extraction_status"]["status"] == "skipped"

    class StubExtractor:
        model = "stub-extractor-v1"
        prompt_version = "duplicate-reextract-v2"

        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            assert "Duplicate Reextract Paper" in prompt
            assert "Zorblax Quindle" in prompt
            payload = {
                "methods": [
                    {
                        "name": "Zorblax Quindle",
                        "description": "A provenance-backed compressed technique.",
                        "implementation_hint": "Apply the Zorblax Quindle transform before evaluation.",
                        "risks": "Can overfit tiny validation sets.",
                        "category": "optimizer",
                        "source_chunk_indices": [0],
                    }
                ]
            }
            return "```json\n" + json.dumps(payload) + "\n```"

    llm = StubExtractor()
    second = ingest_paper({"title": meta["title"]}, llm_fn=llm, conn=conn)

    assert llm.calls == 1
    assert second["duplicate"] is True
    assert second["paper_id"] == first["paper_id"]
    assert second["chunk_ids"] == first["chunk_ids"]
    assert len(second["method_ids"]) == 1

    status = second["extraction_status"]
    assert status["status"] == "extracted"
    assert status["paper_id"] == first["paper_id"]
    assert status["method_count"] == 1
    assert status["extractor_model"] == "stub-extractor-v1"
    assert status["extractor_prompt_version"] == "duplicate-reextract-v2"
    assert status["error_text"] is None
    assert status["confidence_score"] > 0.0
    assert "methods=1" in status["confidence_reason"]
    assert "valid_source_links=1/1" in status["confidence_reason"]

    source_row = conn.execute(
        """
        SELECT m.name, ms.evidence_text, ms.evidence_start, ms.evidence_end,
               ms.evidence_strategy, c.content
        FROM method_sources ms
        JOIN methods m ON m.method_id = ms.method_id
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        """,
        (first["paper_id"],),
    ).fetchone()
    assert source_row["name"] == "Zorblax Quindle"
    assert "Zorblax Quindle" in source_row["evidence_text"]
    assert source_row["evidence_strategy"] == "matched_term:zorblax"
    assert source_row["evidence_start"] >= 0
    assert source_row["evidence_end"] > source_row["evidence_start"]
    assert source_row["evidence_text"] == source_row["content"][
        source_row["evidence_start"]:source_row["evidence_end"]
    ]

    status_row = conn.execute(
        """
        SELECT status, method_count, extractor_model, extractor_prompt_version
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (first["paper_id"],),
    ).fetchone()
    assert status_row["status"] == "extracted"
    assert status_row["method_count"] == 1
    assert status_row["extractor_model"] == "stub-extractor-v1"
    assert status_row["extractor_prompt_version"] == "duplicate-reextract-v2"

    method_count = conn.execute(
        "SELECT COUNT(*) FROM methods WHERE name = ?",
        ("Zorblax Quindle",),
    ).fetchone()[0]
    source_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        """,
        (first["paper_id"],),
    ).fetchone()[0]

    third = ingest_paper({"title": meta["title"]}, llm_fn=llm, conn=conn)
    assert third["duplicate"] is True
    assert third["method_ids"] == []
    assert llm.calls == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM methods WHERE name = ?",
        ("Zorblax Quindle",),
    ).fetchone()[0] == method_count
    assert conn.execute(
        """
        SELECT COUNT(*)
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        """,
        (first["paper_id"],),
    ).fetchone()[0] == source_count



def test_ingest_duplicate_explicit_source_type_relabels_status_only():
    conn = _in_memory_db()

    class StubExtractor:
        model = "source-label-extractor"
        prompt_version = "source-label-prompt-v1"

        def __call__(self, _prompt):
            payload = {
                "methods": [
                    {
                        "name": "Explicit Relabel Method",
                        "description": "Uses explicitrelabeltoken provenance.",
                        "implementation_hint": "Keep compressed memory while relabeling.",
                        "risks": "Incorrect relabeling can hide source provenance.",
                        "category": "ingestion",
                        "source_chunk_indices": [0],
                    }
                ]
            }
            return "```json\n" + json.dumps(payload) + "\n```"

    initial_meta = {
        "title": "Duplicate Explicit Source Relabel Paper",
        "authors": "Source Label Lab",
        "abstract": "Explicit source relabel duplicate paper.",
        "url": "https://example.com/source-relabel",
        "publish_date": "2026-07-01",
        "content": "The explicitrelabeltoken method records durable compressed memory.",
    }
    _apply_source_type_override(initial_meta, "paper")
    first = ingest_paper(initial_meta, llm_fn=StubExtractor(), conn=conn)
    before_status = dict(
        conn.execute(
            "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
            (first["paper_id"],),
        ).fetchone()
    )
    assert before_status["source_type"] == "paper"

    duplicate_meta = {"title": initial_meta["title"], "source_type": " lab_note "}
    second = ingest_paper(duplicate_meta, conn=conn)

    assert second["duplicate"] is True
    assert second["paper_id"] == first["paper_id"]
    assert second["chunk_ids"] == []
    assert second["method_ids"] == []
    assert second["extraction_status"]["source_type"] == "lab_note"

    after_status = dict(
        conn.execute(
            "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
            (first["paper_id"],),
        ).fetchone()
    )
    assert after_status["source_type"] == "lab_note"
    for key in (
        "status",
        "chunk_count",
        "method_count",
        "source_char_count",
        "compressed_char_count",
        "compression_ratio",
        "confidence_score",
        "confidence_reason",
        "extractor_model",
        "extractor_prompt_version",
        "error_text",
    ):
        assert after_status[key] == before_status[key]


def test_ingest_duplicate_default_local_file_source_type_does_not_relabel():
    import tempfile
    from pathlib import Path

    conn = _in_memory_db()
    initial_meta = {
        "title": "Duplicate Local Default Source Paper",
        "authors": "Local Source Lab",
        "abstract": "A local note that already has a curated source label.",
        "url": "https://example.com/local-default-source",
        "publish_date": "2026-07-01",
        "content": "The localdefaultsourcetoken note keeps its curated lab note label.",
    }
    _apply_source_type_override(initial_meta, "lab_note")
    first = ingest_paper(initial_meta, conn=conn)
    assert first["extraction_status"]["source_type"] == "lab_note"

    with tempfile.TemporaryDirectory() as tmpdir:
        note_path = Path(tmpdir) / "duplicate-local-default-source.md"
        note_path.write_text(
            "# Duplicate Local Default Source\n\n"
            "The localdefaultsourcetoken file default must not overwrite lab_note.",
            encoding="utf-8",
        )
        duplicate_meta = load_local_file_metadata(
            note_path,
            {"title": initial_meta["title"]},
        )
        assert duplicate_meta["source_type"] == "file"

        second = ingest_paper(duplicate_meta, conn=conn)

    assert second["duplicate"] is True
    assert second["paper_id"] == first["paper_id"]
    assert second["extraction_status"]["source_type"] == "lab_note"
    status = conn.execute(
        "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
        (first["paper_id"],),
    ).fetchone()
    assert status["source_type"] == "lab_note"

def test_ingest_duplicate_skipped_paper_failed_reextract_updates_status():
    conn = _in_memory_db()
    meta = {
        "title": "Duplicate Failed Reextract Paper",
        "authors": "Retry Author",
        "abstract": "A skipped paper that later hits an extractor failure.",
        "url": "https://example.com/reextract-failed",
        "publish_date": "2026-06-27",
        "content": "Stored content must remain available even when the retry fails.",
    }
    first = ingest_paper(meta, conn=conn)

    def failing_llm(prompt):
        assert "Stored content" in prompt
        raise RuntimeError("retry extractor unavailable")

    second = ingest_paper({"title": meta["title"]}, llm_fn=failing_llm, conn=conn)

    assert second["duplicate"] is True
    assert second["paper_id"] == first["paper_id"]
    assert second["chunk_ids"] == first["chunk_ids"]
    assert second["method_ids"] == []
    assert second["extraction_status"]["status"] == "failed"
    assert "retry extractor unavailable" in second["extraction_status"]["error_text"]

    row = conn.execute(
        """
        SELECT status, method_count, error_text, confidence_score, confidence_reason
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (first["paper_id"],),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["method_count"] == 0
    assert "retry extractor unavailable" in row["error_text"]
    assert row["confidence_score"] == 0.0
    assert "retry extractor unavailable" in row["confidence_reason"]
    assert conn.execute(
        """
        SELECT COUNT(*)
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        """,
        (first["paper_id"],),
    ).fetchone()[0] == 0


def test_ingest_paper_records_failed_extraction_status_and_keeps_paper_chunks():
    conn = _in_memory_db()

    def failing_llm(_prompt):
        raise RuntimeError("local extractor unavailable")

    meta = {
        "title": "Test Paper With Failed LLM",
        "authors": "Test Author",
        "abstract": "A paper about failing extractors.",
        "url": "https://example.com/failed",
        "publish_date": "2026-06-26",
        "content": "This paper still needs durable chunks even when extraction fails.",
    }
    result = ingest_paper(meta, llm_fn=failing_llm, conn=conn)

    assert result["paper_id"] is not None
    assert len(result["chunk_ids"]) > 0
    assert result["method_ids"] == []
    status = result["extraction_status"]
    assert status["status"] == "failed"
    assert status["paper_id"] == result["paper_id"]
    assert status["chunk_count"] == len(result["chunk_ids"])
    assert status["method_count"] == 0
    assert status["source_char_count"] == len(meta["content"])
    assert status["compressed_char_count"] == 0
    assert "local extractor unavailable" in status["error_text"]

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers WHERE paper_id = ?", (result["paper_id"],))
    assert cur.fetchone()[0] == 1
    cur.execute("SELECT COUNT(*) FROM chunks WHERE paper_id = ?", (result["paper_id"],))
    assert cur.fetchone()[0] == len(result["chunk_ids"])
    assert status["confidence_score"] == 0.0
    assert status["confidence_reason"].startswith("failed: ")
    assert "local extractor unavailable" in status["confidence_reason"]
    cur.execute(
        "SELECT status, error_text, confidence_score, confidence_reason FROM paper_extraction_status WHERE paper_id = ?",
        (result["paper_id"],),
    )
    row = cur.fetchone()
    assert row["status"] == "failed"
    assert "local extractor unavailable" in row["error_text"]
    assert row["confidence_score"] == 0.0
    assert "local extractor unavailable" in row["confidence_reason"]


def test_ingest_paper_status_migration_handles_existing_database():
    conn = _in_memory_db()
    conn.execute("DROP TABLE paper_extraction_status")
    migrate_db(conn)

    meta = {
        "title": "Migrated Status Paper",
        "authors": "Migration Author",
        "abstract": "Migration abstract.",
        "url": "https://example.com/migrated",
        "publish_date": "2026-06-26",
        "content": "Migration content becomes one durable chunk.",
    }
    result = ingest_paper(meta, conn=conn)

    assert result["extraction_status"]["status"] == "skipped"
    assert result["extraction_status"]["confidence_score"] == 0.0
    assert result["extraction_status"]["confidence_reason"] == "skipped"
    row = conn.execute(
        "SELECT status, chunk_count, confidence_score, confidence_reason FROM paper_extraction_status WHERE paper_id = ?",
        (result["paper_id"],),
    ).fetchone()
    assert row["status"] == "skipped"
    assert row["chunk_count"] == len(result["chunk_ids"])
    assert row["confidence_score"] == 0.0
    assert row["confidence_reason"] == "skipped"


def test_ingest_duplicate_backfills_missing_confidence_status():
    conn = _in_memory_db()
    row = conn.execute(
        """
        SELECT paper_id, title, abstract, url, publish_date
        FROM papers
        WHERE title = 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints'
        """
    ).fetchone()
    conn.execute(
        """
        UPDATE paper_extraction_status
        SET confidence_score = NULL, confidence_reason = NULL
        WHERE paper_id = ?
        """,
        (row["paper_id"],),
    )

    result = ingest_paper(
        {
            "title": row["title"],
            "abstract": row["abstract"],
            "url": row["url"],
            "publish_date": row["publish_date"],
            "content": row["abstract"],
        },
        conn=conn,
    )

    assert result["duplicate"] is True
    assert result["extraction_status"]["confidence_score"] is not None
    assert "valid_source_links=" in result["extraction_status"]["confidence_reason"]
    stored = conn.execute(
        """
        SELECT confidence_score, confidence_reason
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (row["paper_id"],),
    ).fetchone()
    assert stored["confidence_score"] == result["extraction_status"]["confidence_score"]
    assert stored["confidence_reason"] == result["extraction_status"]["confidence_reason"]



def test_method_sources_evidence_migration_handles_existing_table():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
            content TEXT NOT NULL
        );
        CREATE TABLE methods (
            method_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE method_sources (
            method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
            chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            PRIMARY KEY (method_id, chunk_id)
        );
        INSERT INTO papers (title) VALUES ('Legacy Paper');
        INSERT INTO chunks (paper_id, content) VALUES (1, 'legacy evidence chunk');
        INSERT INTO methods (name) VALUES ('Legacy Method');
        INSERT INTO method_sources (method_id, chunk_id) VALUES (1, 1);
        """
    )

    migrate_db(conn)
    migrate_db(conn)

    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(method_sources)")
    }
    assert {"evidence_text", "evidence_start", "evidence_end", "evidence_strategy"} <= columns
    row = conn.execute(
        """
        SELECT evidence_text, evidence_start, evidence_end, evidence_strategy
        FROM method_sources
        WHERE method_id = 1 AND chunk_id = 1
        """
    ).fetchone()
    assert row["evidence_text"] == "legacy evidence chunk"
    assert row["evidence_start"] == 0
    assert row["evidence_end"] == len("legacy evidence chunk")
    assert row["evidence_strategy"] == "migration_leading_excerpt"


def test_seed_initial_data_populates_method_source_evidence():
    conn = _in_memory_db()
    rows = conn.execute(
        """
        SELECT ms.evidence_text, ms.evidence_start, ms.evidence_end, ms.evidence_strategy, c.content
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        """
    ).fetchall()

    assert rows
    for row in rows:
        assert row["evidence_text"]
        assert row["evidence_start"] >= 0
        assert row["evidence_end"] > row["evidence_start"]
        assert row["evidence_text"] == row["content"][
            row["evidence_start"]:row["evidence_end"]
        ]
        assert row["evidence_strategy"].startswith(("matched_term:", "leading_excerpt"))


def test_seed_initial_data_populates_extraction_confidence_statuses():
    conn = _in_memory_db()
    paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    status_rows = conn.execute(
        """
        SELECT p.title, pes.status, pes.method_count, pes.confidence_score, pes.confidence_reason
        FROM papers p
        JOIN paper_extraction_status pes ON pes.paper_id = p.paper_id
        ORDER BY p.paper_id
        """
    ).fetchall()

    assert len(status_rows) == paper_count
    assert status_rows
    for row in status_rows:
        assert row["confidence_score"] is not None
        assert row["confidence_reason"]
        assert 0.0 <= row["confidence_score"] <= 1.0
    extracted_rows = [row for row in status_rows if row["status"] == "extracted"]
    assert extracted_rows
    assert any(row["method_count"] > 0 for row in extracted_rows)
    assert all("valid_source_links=" in row["confidence_reason"] for row in extracted_rows)


def test_ingest_paper_dry_run_includes_confidence_status():
    result = ingest_paper(
        {
            "title": "Dry Run Confidence Paper",
            "authors": "Dry Runner",
            "abstract": "Dry run abstract.",
            "url": "https://example.com/dry-confidence",
            "publish_date": "2026-06-30",
            "content": "Dry run content is chunked but not persisted.",
        },
        dry_run=True,
    )

    status = result["planned"]["extraction_status"]
    assert result["extraction_status"] == status
    assert status["status"] == "skipped"
    assert status["confidence_score"] == 0.0
    assert status["confidence_reason"] == "skipped"

def test_load_local_markdown_file_ingests_and_retrieves_method():
    import tempfile
    from pathlib import Path

    conn = _in_memory_db()

    with tempfile.TemporaryDirectory() as tmpdir:
        note_path = Path(tmpdir) / "local-gate-notes.md"
        note_path.write_text(
            "# Local Gate Notes\n\n"
            "The localfiletoken schedule gates residual updates after warmup for tiny transformers.",
            encoding="utf-8",
        )
        meta = load_local_file_metadata(note_path, {"authors": "Lab Notebook"})

        assert meta["title"] == "Local Gate Notes"
        assert meta["authors"] == "Lab Notebook"
        assert meta["source_type"] == "file"
        assert meta["url"].startswith("file://")
        assert "localfiletoken schedule" in meta["content"]

        def stub_llm(_prompt):
            return (
                "```json\n"
                '{"methods":[{"name":"Local File Gate","description":"A localfiletoken residual gate",'
                '"implementation_hint":"Gate residual updates after warmup","risks":"Can underfit",'
                '"category":"schedule","source_chunk_indices":[0]}]}'
                "\n```"
            )

        result = ingest_paper(meta, llm_fn=stub_llm, conn=conn)

        assert result["paper_id"] is not None
        assert result["method_ids"]
        assert result["extraction_status"]["source_type"] == "file"

        paper = conn.execute(
            "SELECT title, authors, url FROM papers WHERE paper_id = ?",
            (result["paper_id"],),
        ).fetchone()
        assert paper["title"] == "Local Gate Notes"
        assert paper["authors"] == "Lab Notebook"
        assert paper["url"] == meta["url"]

        chunk = conn.execute(
            "SELECT content FROM chunks WHERE paper_id = ?",
            (result["paper_id"],),
        ).fetchone()
        assert "localfiletoken schedule" in chunk["content"]

        pack = rfs.search_methods("localfiletoken", conn=conn)
        assert "Local File Gate" in pack
        assert "Local Gate Notes" in pack
        assert meta["url"] in pack



def test_load_local_file_metadata_preserves_source_type_override():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        note_path = Path(tmpdir) / "source-type-lab-note.md"
        note_path.write_text(
            "# Source Type Lab Note\n\n"
            "The sourcetypetoken note records an experiment report imported from a lab run.",
            encoding="utf-8",
        )

        meta = load_local_file_metadata(note_path, {"source_type": "lab_note"})

        assert meta["title"] == "Source Type Lab Note"
        assert meta["source_type"] == "lab_note"
        assert meta["url"] == note_path.resolve().as_uri()
        assert "sourcetypetoken note" in meta["content"]


def test_cli_source_type_override_applies_to_loaded_local_metadata():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        note_path = Path(tmpdir) / "cli-source-type-lab-note.md"
        note_path.write_text(
            "# CLI Source Type Lab Note\n\n"
            "The clisourcetypetoken note records explicit source class metadata.",
            encoding="utf-8",
        )

        meta = load_local_file_metadata(note_path)
        assert meta["source_type"] == "file"

        _apply_source_type_override(meta, " experiment_report ")

        assert meta["title"] == "Cli Source Type Lab Note"
        assert meta["source_type"] == "experiment_report"
        assert "clisourcetypetoken note" in meta["content"]

        conn = _in_memory_db()
        result = ingest_paper(meta, conn=conn)
        assert result["extraction_status"]["source_type"] == "experiment_report"


def test_cli_source_type_override_applies_to_fetched_metadata_shapes():
    arxiv_meta = {
        "title": "Fetched Arxiv Paper",
        "url": "https://arxiv.org/abs/2305.13245",
        "content": "arxiv content",
    }
    repo_meta = {
        "title": "Repository README",
        "url": "https://github.com/example/repo",
        "source_type": "repo_readme",
        "content": "repo content",
    }

    assert _apply_source_type_override(arxiv_meta, " paper ") is arxiv_meta
    assert _apply_source_type_override(repo_meta, " lab_artifact ") is repo_meta

    assert arxiv_meta["source_type"] == "paper"
    assert repo_meta["source_type"] == "lab_artifact"


def test_cli_source_type_label_rejects_empty_values():
    import argparse

    assert _source_type_label(" lab_note ") == "lab_note"
    try:
        _source_type_label(" \t ")
    except argparse.ArgumentTypeError as exc:
        assert "--source-type must not be empty" in str(exc)
    else:
        raise AssertionError("Expected blank source type label to be rejected")

def test_load_local_pdf_file_extracts_text_and_metadata():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "compressed-paper-memory.pdf"
        _write_minimal_pdf(
            pdf_path,
            "The pdffixturetoken schedule is extracted from an actual PDF content stream.",
        )

        meta = load_local_file_metadata(
            pdf_path,
            {
                "authors": "PDF Lab",
                "abstract": "PDF ingestion smoke fixture.",
                "publish_date": "2026-07-01",
            },
        )

        assert meta["title"] == "Compressed Paper Memory"
        assert meta["authors"] == "PDF Lab"
        assert meta["abstract"] == "PDF ingestion smoke fixture."
        assert meta["publish_date"] == "2026-07-01"
        assert meta["source_type"] == "file"
        assert meta["url"] == pdf_path.resolve().as_uri()
        assert "pdffixturetoken schedule" in meta["content"]


def test_load_local_malformed_pdf_reports_extraction_error():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "malformed-paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nnot a valid cross-reference table\n")

        try:
            load_local_file_metadata(pdf_path)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected malformed PDF ingestion to report extraction failure")

        assert "Could not extract text from PDF" in message


def test_ingest_deduplication_exact_name():
    conn = _in_memory_db()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM methods")
    initial_methods_count = cur.fetchone()[0]

    cur.execute("SELECT method_id FROM methods WHERE LOWER(name) = LOWER('SwiGLU MLP')")
    existing_method_id = cur.fetchone()[0]

    stub_response = '{"methods":[{"name":"swiglu mlp","description":"A case-insensitive duplicate SwiGLU","implementation_hint":"use silu","risks":"none","category":"mlp","source_chunk_indices":[0]}]}'

    def stub_llm(_prompt):
        return f"```json\n{stub_response}\n```"

    meta = {
        "title": "Exact Name Duplicate Paper",
        "authors": "Duplicate Author",
        "abstract": "Testing exact name duplicate resolution.",
        "url": "https://example.com/exact_name_dup",
        "publish_date": "2026-06-27",
        "content": "This is content about swiglu mlp.",
    }

    result = ingest_paper(meta, llm_fn=stub_llm, conn=conn)

    assert result["paper_id"] is not None
    assert len(result["chunk_ids"]) > 0
    assert result["method_ids"] == [existing_method_id]

    # Verify no new method was inserted
    cur.execute("SELECT COUNT(*) FROM methods")
    new_methods_count = cur.fetchone()[0]
    assert new_methods_count == initial_methods_count

    # Verify method_sources receives the new chunk ID mapping for the resolved method
    new_chunk_id = result["chunk_ids"][0]
    cur.execute(
        """
        SELECT evidence_text, evidence_start, evidence_end, evidence_strategy
        FROM method_sources
        WHERE method_id = ? AND chunk_id = ?
        """,
        (existing_method_id, new_chunk_id),
    )
    duplicate_source = cur.fetchone()
    assert duplicate_source is not None, "Expected method_sources to link the existing method to the new chunk"
    assert "swiglu mlp" in duplicate_source["evidence_text"]
    assert duplicate_source["evidence_start"] == 0
    assert duplicate_source["evidence_end"] == len(meta["content"])
    assert duplicate_source["evidence_strategy"] == "matched_term:swiglu"

    # Verify search_methods lists the new paper source
    pack = rfs.search_methods("swiglu", conn=conn)
    assert "Exact Name Duplicate Paper" in pack
    assert "https://example.com/exact_name_dup" in pack


def test_ingest_deduplication_llm():
    conn = _in_memory_db()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM methods")
    initial_methods_count = cur.fetchone()[0]

    cur.execute("SELECT method_id FROM methods WHERE LOWER(name) = LOWER('SwiGLU MLP')")
    existing_method_id = cur.fetchone()[0]

    # 1. Test case: similar name but not exact, LLM returns YES -> should deduplicate / merge
    stub_response = '{"methods":[{"name":"SwiGLU Activation Function","description":"SwiGLU activation.","implementation_hint":"use silu","risks":"none","category":"mlp","source_chunk_indices":[0]}]}'

    def stub_llm_yes(prompt):
        if "exact same optimization method" in prompt:
            if "SwiGLU MLP" in prompt:
                return "YES"
            return "NO"
        return f"```json\n{stub_response}\n```"

    meta_yes = {
        "title": "LLM Merge SwiGLU Paper",
        "authors": "Merge Author",
        "abstract": "Testing LLM merge behavior.",
        "url": "https://example.com/llm_merge",
        "publish_date": "2026-06-27",
        "content": "This is content describing SwiGLU Activation Function.",
    }

    result_yes = ingest_paper(meta_yes, llm_fn=stub_llm_yes, conn=conn)

    assert result_yes["paper_id"] is not None
    assert len(result_yes["chunk_ids"]) > 0
    assert result_yes["method_ids"] == [existing_method_id]

    # Verify no new method was inserted
    cur.execute("SELECT COUNT(*) FROM methods")
    assert cur.fetchone()[0] == initial_methods_count

    # Verify method_sources receives the new chunk ID mapping
    new_chunk_id_yes = result_yes["chunk_ids"][0]
    cur.execute("SELECT 1 FROM method_sources WHERE method_id = ? AND chunk_id = ?", (existing_method_id, new_chunk_id_yes))
    assert cur.fetchone() is not None

    # Verify search_methods lists the new paper source
    pack_yes = rfs.search_methods("swiglu", conn=conn)
    assert "LLM Merge SwiGLU Paper" in pack_yes
    assert "https://example.com/llm_merge" in pack_yes

    # 2. Test case: similar name but not exact, LLM returns NO -> should NOT deduplicate (should insert new method)
    stub_response_no = '{"methods":[{"name":"SwiGLU Attention Layer","description":"A completely different SwiGLU based attention.","implementation_hint":"none","risks":"none","category":"attention","source_chunk_indices":[0]}]}'

    def stub_llm_no(prompt):
        if "exact same optimization method" in prompt:
            return "NO"
        return f"```json\n{stub_response_no}\n```"

    meta_no = {
        "title": "LLM No-Merge SwiGLU Paper",
        "authors": "No-Merge Author",
        "abstract": "Testing LLM no-merge behavior.",
        "url": "https://example.com/llm_nomerge",
        "publish_date": "2026-06-27",
        "content": "This is content describing SwiGLU Attention Layer.",
    }

    result_no = ingest_paper(meta_no, llm_fn=stub_llm_no, conn=conn)

    assert result_no["paper_id"] is not None
    assert len(result_no["chunk_ids"]) > 0
    new_method_id = result_no["method_ids"][0]
    assert new_method_id != existing_method_id

    # Verify a new method WAS inserted
    cur.execute("SELECT COUNT(*) FROM methods")
    assert cur.fetchone()[0] == initial_methods_count + 1

    # Verify method_sources receives the new chunk ID mapping for the NEW method
    new_chunk_id_no = result_no["chunk_ids"][0]
    cur.execute("SELECT 1 FROM method_sources WHERE method_id = ? AND chunk_id = ?", (new_method_id, new_chunk_id_no))
    assert cur.fetchone() is not None

    # Verify search_methods lists the new paper source under the NEW method
    pack_no = rfs.search_methods("Attention Layer", conn=conn)
    assert "SwiGLU Attention Layer" in pack_no
    assert "LLM No-Merge SwiGLU Paper" in pack_no
    assert "https://example.com/llm_nomerge" in pack_no

def test_get_experiment_brief():
    conn = _in_memory_db()
    brief = rfs.get_experiment_brief("nanochat_autoresearch", conn)
    assert "minimize validation bits per byte" in brief.lower()


def test_log_and_update_experiment():
    conn = _in_memory_db()
    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    exp_id = rfs.log_experiment("abc123", None, None, "pending", "test", run_id, conn)
    rfs.update_experiment(exp_id, 2.123, 0.5, "keep", "updated", conn)
    cur = conn.cursor()
    cur.execute("SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,))
    row = cur.fetchone()
    assert row["status"] == "keep"
    assert abs(row["val_bpb"] - 2.123) < 1e-6


def test_log_retrieval_and_token_usage():
    conn = _in_memory_db()
    run_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    exp_id = rfs.log_experiment("def456", 2.0, 0.4, "keep", "test", run_id, conn)
    rfs.log_retrieval_event("method_pack", "all", "content", run_id, exp_id, conn)
    rfs.log_token_usage(100, 50, 0.001, exp_id, conn)
    summary = rfs.get_agent_run_summary(run_id, conn)
    assert summary["retrieval_count"] == 1
    assert summary["tokens"]["prompt_tokens"] == 100
    assert summary["tokens"]["completion_tokens"] == 50


def test_load_run_calculates_best_and_mean_bpb():
    conn = _in_memory_db()
    run_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", run_id, conn)
    rfs.log_experiment("abc", 2.450, 0.5, "keep", "better", run_id, conn)
    rfs.log_experiment("def", 2.480, 0.5, "discard", "worse", run_id, conn)
    rfs.log_experiment("ghi", None, None, "crash", "crash", run_id, conn)
    summary = metrics.load_run(run_id, conn)
    assert abs(summary["baseline_bpb"] - 2.500) < 1e-6
    assert abs(summary["best_bpb"] - 2.450) < 1e-6
    assert abs(summary["improvement"] - 0.050) < 1e-6
    assert summary["total_experiments"] == 4
    assert summary["kept"] == 2
    assert summary["crashed"] == 1
    assert summary["discarded"] == 1


def test_load_run_preserves_logged_zero_cost_for_priced_model():
    conn = _in_memory_db()
    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    exp_id = rfs.log_experiment("abc", 2.450, 0.5, "keep", "zero cost", run_id, conn)
    rfs.log_token_usage(1_000_000, 200_000, 0.0, exp_id, conn)

    summary = metrics.load_run(run_id, conn)

    assert summary["prompt_tokens"] == 1_000_000
    assert summary["completion_tokens"] == 200_000
    assert summary["cost"] == 0.0

def test_compare_runs_rejects_wrong_modes():
    conn = _in_memory_db()
    baseline_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    loopgraph_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", baseline_id, conn)
    rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", loopgraph_id, conn)
    comparison = metrics.compare_runs(baseline_id, loopgraph_id, conn)
    assert comparison["baseline"]["mode"] == "baseline"
    assert comparison["researchfs"]["mode"] == "loopgraph"


def test_format_comparison_includes_key_metrics():
    conn = _in_memory_db()
    baseline_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    loopgraph_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    baseline_exp_id = rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", baseline_id, conn)
    rfs.log_token_usage(10_000_000, 2_000_000, None, baseline_exp_id, conn)
    exp_id = rfs.log_experiment("abc", 2.450, 0.5, "keep", "better", loopgraph_id, conn)
    rfs.log_token_usage(1_000_000, 200_000, None, exp_id, conn)
    rfs.log_retrieval_event("method_pack", "all", "content", loopgraph_id, None, conn)
    rfs.log_retrieval_event("experiment_brief", "nanochat", "content", loopgraph_id, None, conn)
    comparison = metrics.compare_runs(baseline_id, loopgraph_id, conn)
    text = metrics.format_comparison(comparison)
    assert "Agent Run Comparison" in text
    assert "2.500000" in text
    assert "2.450000" in text
    assert "fewer tokens" in text


def test_benchmark_comparison_and_reward_hack_screening():
    conn = _in_memory_db()
    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 1.000, 0.5, "keep", "baseline", run_id, conn)
    rfs.log_experiment("abc", 0.880, 0.5, "keep", "suspicious", run_id, conn)
    rfs.log_experiment("missing", None, 0.5, "keep", "missing metric", run_id, conn)

    comparison = metrics.compare_run_to_benchmarks(run_id, conn=conn)
    assert comparison["agent_run_id"] == run_id
    assert comparison["strongest_benchmark"] == "recursive_best"
    assert comparison["gap_to_sota"] < 0

    text = metrics.format_benchmark_comparison(comparison)
    assert "Benchmark Comparison" in text
    assert "recursive_best" in text
    assert "0.880000" in text

    run = metrics.load_run(run_id, conn)
    flags = metrics.detect_reward_hacks(run)
    assert any("multi-seed validation" in flag["reason"] for flag in flags)
    assert any("no val_bpb" in flag["reason"] for flag in flags)
    report = metrics.format_reward_hack_report(run, flags)
    assert "suspicious pattern" in report


def test_aggregate_seeded_runs_tracks_valid_run_ids():
    conn = _in_memory_db()
    run_ids = []
    for idx, best_bpb in enumerate([0.950, 0.940, None]):
        run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
        run_ids.append(run_id)
        rfs.log_experiment("baseline", 1.000, 0.5, "keep", "baseline", run_id, conn)
        if best_bpb is not None:
            rfs.log_experiment(f"seed{idx}", best_bpb, 0.5, "keep", "seed", run_id, conn)
        else:
            rfs.log_experiment(f"seed{idx}", None, 0.5, "crash", "seed", run_id, conn)

    aggregate = metrics.aggregate_seeded_runs(run_ids, conn=conn)
    assert aggregate["n_seeds"] == 3
    assert aggregate["valid_seeds"] == 3
    assert aggregate["valid_run_ids"] == run_ids
    assert abs(aggregate["mean_bpb"] - 0.9633333333333333) < 1e-9

    text = metrics.format_seeded_aggregate(aggregate)
    assert "Multi-Seed Aggregate Results" in text
    for run_id in run_ids:
        assert str(run_id) in text


def test_aggregate_seeded_runs_skips_runs_without_valid_bpb():
    conn = _in_memory_db()
    valid_ids = []
    for best_bpb in [0.950, 0.940]:
        run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
        valid_ids.append(run_id)
        rfs.log_experiment("baseline", 1.000, 0.5, "keep", "baseline", run_id, conn)
        rfs.log_experiment("seed", best_bpb, 0.5, "keep", "seed", run_id, conn)

    invalid_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("crashed", None, None, "crash", "no metric", invalid_id, conn)

    aggregate = metrics.aggregate_seeded_runs([valid_ids[0], invalid_id, valid_ids[1]], conn=conn)
    assert aggregate["n_seeds"] == 3
    assert aggregate["valid_seeds"] == 2
    assert aggregate["valid_run_ids"] == valid_ids
    assert aggregate["invalid_run_ids"] == [invalid_id]

    text = metrics.format_seeded_aggregate(aggregate)
    assert f"Skipped run IDs without valid best BPB: {invalid_id}" in text
    assert f"| {invalid_id} |" not in text


def test_aggregate_seeded_runs_rejects_mixed_modes_or_models():
    conn = _in_memory_db()
    baseline_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    loopgraph_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    for run_id in [baseline_id, loopgraph_id]:
        rfs.log_experiment("baseline", 1.000, 0.5, "keep", "baseline", run_id, conn)

    try:
        metrics.aggregate_seeded_runs([baseline_id, loopgraph_id], conn=conn)
    except ValueError as exc:
        assert "same mode and model" in str(exc)
    else:
        assert False, "Expected mixed mode aggregation to fail"


def test_cost_estimate_uses_model_rates():
    cost = metrics.estimate_cost(1_000_000, 200_000, "gpt-4o-mini")
    assert cost > 0
    assert cost == (1_000_000 * 0.00015 + 200_000 * 0.0006) / 1000


def test_build_comparison_report_matches_default_two_run_sections():
    import compare_runs

    conn = _in_memory_db()
    baseline_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    loopgraph_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    baseline_exp_id = rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", baseline_id, conn)
    rfs.log_token_usage(10_000, 2_000, None, baseline_exp_id, conn)
    loopgraph_exp_id = rfs.log_experiment("improved", 2.450, 0.5, "keep", "better", loopgraph_id, conn)
    rfs.log_token_usage(1_000, 200, None, loopgraph_exp_id, conn)

    comparison = metrics.compare_runs(baseline_id, loopgraph_id, conn)
    expected = "\n\n".join(
        [
            metrics.format_comparison(comparison),
            metrics.format_reward_hack_report(
                comparison["baseline"],
                metrics.detect_reward_hacks(comparison["baseline"]),
            ),
            metrics.format_reward_hack_report(
                comparison["researchfs"],
                metrics.detect_reward_hacks(comparison["researchfs"]),
            ),
        ]
    )

    assert compare_runs.build_comparison_report(baseline_id, loopgraph_id, conn=conn) == expected


def test_build_comparison_report_can_include_benchmark_sections():
    import compare_runs

    conn = _in_memory_db()
    baseline_id = rfs.start_agent_run("baseline", "gpt-4o-mini", conn)
    loopgraph_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", baseline_id, conn)
    rfs.log_experiment("improved", 2.450, 0.5, "keep", "better", loopgraph_id, conn)

    report = compare_runs.build_comparison_report(
        baseline_id,
        loopgraph_id,
        include_benchmarks=True,
        conn=conn,
    )

    assert "## Benchmark comparison: Baseline" in report
    assert "## Benchmark comparison: ResearchFS" in report
    assert "Reward-hack screening" in report


def test_compare_runs_main_uses_report_builder_for_two_run_comparison():
    import io
    import sys
    import unittest.mock
    import compare_runs

    out = io.StringIO()
    with unittest.mock.patch.object(sys, "argv", ["compare_runs.py", "101", "202"]), \
         unittest.mock.patch("compare_runs.build_comparison_report", return_value="REPORT") as build_report, \
         unittest.mock.patch("sys.stdout", out):
        compare_runs.main()

    build_report.assert_called_once_with(101, 202, include_benchmarks=False)
    assert out.getvalue() == "REPORT\n"


def test_run_comparison_uses_report_builder_without_cli_global_mutation():
    import io
    import sys
    import unittest.mock
    import run_comparison

    original_argv = ["run_comparison.py", "--model", "test-model"]
    out = io.StringIO()
    helper_observations = []

    def fake_run_agent(mode, *args):
        return {"baseline": 101, "loopgraph": 202}[mode]

    def fake_build_report(baseline_run_id, loopgraph_run_id):
        helper_observations.append(
            (
                baseline_run_id,
                loopgraph_run_id,
                sys.argv is original_argv,
                sys.stdout is out,
            )
        )
        return "DIRECT REPORT"

    with unittest.mock.patch.object(sys, "argv", original_argv), \
         unittest.mock.patch("sys.stdout", out), \
         unittest.mock.patch("run_comparison.initialize_db"), \
         unittest.mock.patch("run_comparison.ensure_seed"), \
         unittest.mock.patch("run_comparison._current_branch", return_value="main"), \
         unittest.mock.patch("run_comparison._checkout"), \
         unittest.mock.patch("run_comparison._run_agent", side_effect=fake_run_agent), \
         unittest.mock.patch("run_comparison.build_comparison_report", side_effect=fake_build_report):
        run_comparison.main()
        assert sys.argv is original_argv
        assert sys.stdout is out

    assert helper_observations == [(101, 202, True, True)]
    assert "DIRECT REPORT\n" in out.getvalue()


def test_prepare_data_checks_prepare_cache_dir_before_running():
    import types
    import unittest.mock
    import run_comparison

    fake_prepare = types.ModuleType("prepare")
    fake_prepare.DATA_DIR = "/tmp/autoresearch-cache/data"
    checked_paths = []

    def fake_isdir(path):
        checked_paths.append(path)
        return path == fake_prepare.DATA_DIR

    def fail_run(*args, **kwargs):
        assert False, "prepare.py should not run when prepare.DATA_DIR exists"

    with unittest.mock.patch.dict("sys.modules", {"prepare": fake_prepare}), \
         unittest.mock.patch("run_comparison.os.path.isdir", fake_isdir), \
         unittest.mock.patch("run_comparison.subprocess.run", fail_run):
        run_comparison._prepare_data({"TEST": "1"})

    assert checked_paths == [fake_prepare.DATA_DIR]


def test_prepare_data_runs_prepare_when_cache_dir_missing():
    import types
    import unittest.mock
    import run_comparison

    fake_prepare = types.ModuleType("prepare")
    fake_prepare.DATA_DIR = "/tmp/autoresearch-cache/data"
    checked_paths = []
    calls = []

    def fake_isdir(path):
        checked_paths.append(path)
        return False

    def record_run(args, **kwargs):
        calls.append((args, kwargs))
        class Result:
            returncode = 0
        return Result()

    env = {"TEST": "1"}
    with unittest.mock.patch.dict("sys.modules", {"prepare": fake_prepare}), \
         unittest.mock.patch("run_comparison.os.path.isdir", fake_isdir), \
         unittest.mock.patch("run_comparison.subprocess.run", record_run):
        run_comparison._prepare_data(env)

    assert checked_paths == [fake_prepare.DATA_DIR]
    assert calls == [(["uv", "run", "python", "prepare.py"], {"env": env})]


def test_asep_cleanup_deletes_only_current_generated_branch():
    import unittest.mock
    import asep_loopgraph_benchmark as asep

    calls = []

    def record_run(args, **kwargs):
        calls.append(args)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    branches_before = {"autoresearch/exp_existing"}
    with unittest.mock.patch("asep_loopgraph_benchmark.current_git_branch", return_value="autoresearch/exp_generated"), \
         unittest.mock.patch("asep_loopgraph_benchmark.autoresearch_branches", side_effect=AssertionError("cleanup should not scan when still on this run's temp branch")), \
         unittest.mock.patch("asep_loopgraph_benchmark.subprocess.run", record_run):
        asep.cleanup_git_state("main", branches_before)

    assert calls == [
        ["git", "checkout", "main"],
        ["git", "branch", "-D", "autoresearch/exp_generated"],
    ]


def test_asep_cleanup_deletes_single_new_generated_branch_when_already_original():
    import unittest.mock
    import asep_loopgraph_benchmark as asep

    calls = []

    def record_run(args, **kwargs):
        calls.append(args)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    branches_before = {"autoresearch/exp_existing"}
    branches_after = {"autoresearch/exp_existing", "autoresearch/exp_generated"}
    with unittest.mock.patch("asep_loopgraph_benchmark.current_git_branch", return_value="main"), \
         unittest.mock.patch("asep_loopgraph_benchmark.autoresearch_branches", return_value=branches_after), \
         unittest.mock.patch("asep_loopgraph_benchmark.subprocess.run", record_run):
        asep.cleanup_git_state("main", branches_before)

    assert calls == [["git", "branch", "-D", "autoresearch/exp_generated"]]


def test_asep_cleanup_does_not_delete_ambiguous_new_generated_branches():
    import unittest.mock
    import asep_loopgraph_benchmark as asep

    calls = []

    def record_run(args, **kwargs):
        calls.append(args)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    branches_before = {"autoresearch/exp_existing"}
    branches_after = {
        "autoresearch/exp_existing",
        "autoresearch/exp_generated_one",
        "autoresearch/exp_generated_two",
    }
    with unittest.mock.patch("asep_loopgraph_benchmark.current_git_branch", return_value="main"), \
         unittest.mock.patch("asep_loopgraph_benchmark.autoresearch_branches", return_value=branches_after), \
         unittest.mock.patch("asep_loopgraph_benchmark.subprocess.run", record_run):
        asep.cleanup_git_state("main", branches_before)

    assert calls == []


def test_asep_main_emits_numeric_penalty_for_missing_bpb():
    import io
    import json
    import types
    import unittest.mock
    import asep_loopgraph_benchmark as asep

    args = types.SimpleNamespace(
        steps_budget=1,
        time_budget=1,
        device="cpu",
        api_provider="openai",
        model="model",
        mode="loopgraph",
        max_experiments=1,
        max_method_chars=None,
        metric="val_bpb",
    )
    out = io.StringIO()
    with unittest.mock.patch("asep_loopgraph_benchmark.parse_args", return_value=args), \
         unittest.mock.patch("asep_loopgraph_benchmark.set_fast_train_env"), \
         unittest.mock.patch("asep_loopgraph_benchmark.current_git_branch", return_value="main"), \
         unittest.mock.patch("asep_loopgraph_benchmark.autoresearch_branches", return_value=set()), \
         unittest.mock.patch("asep_loopgraph_benchmark.build_llm_fn", return_value=lambda prompt: ""), \
         unittest.mock.patch("asep_loopgraph_benchmark.run_agent_mode", return_value=123), \
         unittest.mock.patch("asep_loopgraph_benchmark.load_run", return_value={"best_bpb": None, "total_tokens": 0, "cost": 0.0}), \
         unittest.mock.patch("asep_loopgraph_benchmark.cleanup_git_state"), \
         unittest.mock.patch("sys.stdout", out):
        assert asep.main() == 0

    payload = json.loads(out.getvalue())
    assert payload["val_bpb"] == 1e9


def test_apply_hunks_success():
    content = "line 1\nline 2\nline 3"
    response = "some text\n<<<<<<< SEARCH\nline 2\n=======\nline 2 modified\n>>>>>>>"
    res = apply_hunks(content, response)
    assert res == "line 1\nline 2 modified\nline 3"


def test_apply_hunks_preserves_indentation():
    content = "def fn():\n    x = 1\n    return x\n"
    response = "<<<<<<< SEARCH\n    x = 1\n=======\n    x = 2\n>>>>>>>"
    res = apply_hunks(content, response)
    assert res == "def fn():\n    x = 2\n    return x\n"


def test_apply_hunks_not_found():
    content = "line 1\nline 2\nline 3"
    response = "<<<<<<< SEARCH\nmissing line\n=======\nreplacement\n>>>>>>>"
    try:
        apply_hunks(content, response)
        assert False, "Should raise HunkError"
    except HunkError:
        pass


def test_check_syntax_valid_and_invalid_temp_files():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        valid_file = tmp_path / "valid.py"
        invalid_file = tmp_path / "invalid.py"

        valid_file.write_text("def ok():\n    return 42\n", encoding="utf-8")
        invalid_file.write_text("def broken(:\n    pass\n", encoding="utf-8")

        valid_ok, valid_error = check_syntax(valid_file)
        invalid_ok, invalid_error = check_syntax(invalid_file)

    assert valid_ok is True
    assert valid_error == ""
    assert invalid_ok is False
    assert "SyntaxError" in invalid_error


def test_empirical_feedback_and_stats():
    conn = _in_memory_db()

    # Get method IDs
    cur = conn.cursor()
    cur.execute("SELECT method_id, name FROM methods")
    methods_dict = {row["name"]: row["method_id"] for row in cur.fetchall()}
    swiglu_id = methods_dict["SwiGLU MLP"]
    gqa_id = methods_dict["Grouped-Query Attention (GQA)"]

    # 1. Agent run 1
    run_id1 = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)

    # Baseline for run 1
    baseline_id1 = rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", run_id1, conn)

    # Exp 1: SwiGLU keep (val_bpb = 2.450 -> delta = -0.050)
    exp1_id = rfs.log_experiment("commit1", 2.450, 0.5, "keep", "better SwiGLU", run_id1, conn)
    rfs.log_retrieval_event("agent_search", "SwiGLU", "retrieved content 1", run_id1, exp1_id, conn, method_ids=[swiglu_id])

    # Exp 2: SwiGLU discard (val_bpb = 2.520)
    exp2_id = rfs.log_experiment("commit2", 2.520, 0.5, "discard", "worse SwiGLU", run_id1, conn)
    rfs.log_retrieval_event("agent_search", "SwiGLU", "retrieved content 2", run_id1, exp2_id, conn, method_ids=[swiglu_id])

    # Exp 3: SwiGLU crash (val_bpb = None)
    exp3_id = rfs.log_experiment("commit3", None, 0.0, "crash", "crashed SwiGLU", run_id1, conn)
    rfs.log_retrieval_event("agent_search", "SwiGLU", "retrieved content 3", run_id1, exp3_id, conn, method_ids=[swiglu_id])

    # Exp 4: GQA keep (val_bpb = 2.400 -> delta = -0.100)
    exp4_id = rfs.log_experiment("commit4", 2.400, 0.5, "keep", "better GQA", run_id1, conn)
    rfs.log_retrieval_event("agent_search", "GQA", "retrieved content 4", run_id1, exp4_id, conn, method_ids=[gqa_id])

    # Exp 5: GQA keep (val_bpb = 2.380 -> delta = -0.120)
    exp5_id = rfs.log_experiment("commit5", 2.380, 0.5, "keep", "even better GQA", run_id1, conn)
    rfs.log_retrieval_event("agent_search", "GQA", "retrieved content 5", run_id1, exp5_id, conn, method_ids=[gqa_id])

    # Verify retrieved_methods table population
    cur.execute("SELECT COUNT(*) FROM retrieved_methods")
    assert cur.fetchone()[0] == 5

    # Get method pack for search query "SwiGLU"
    pack, ids = rfs.search_methods_and_ids("SwiGLU", conn)
    assert swiglu_id in ids

    # Check stats for SwiGLU
    # 3 runs, 33% success (1 keeps, 1 discards, 1 crashes), avg -0.050000 BPB.
    expected_swiglu_stats = "3 runs, 33% success (1 keeps, 1 discards, 1 crashes), avg -0.050000 BPB"
    assert expected_swiglu_stats in pack

    # Get method pack for GQA
    pack_gqa, ids_gqa = rfs.search_methods_and_ids("Grouped-Query", conn)
    assert gqa_id in ids_gqa
    expected_gqa_stats = "2 runs, 100% success (2 keeps, 0 discards, 0 crashes), avg -0.110000 BPB"
    assert expected_gqa_stats in pack_gqa

    # Verify sorting in the full method pack
    # GQA success rate: 1.0 (keeps=2)
    # Unrun methods success rate: 0.5 (keeps=0)
    # SwiGLU success rate: 0.33 (keeps=1)
    # So GQA should be first, followed by unrun methods (e.g. Muon), followed by SwiGLU MLP.
    full_pack = rfs.get_method_pack(conn)

    # Let's verify sorting order via positions of the headers
    pos_gqa = full_pack.find("Grouped-Query Attention (GQA)")
    pos_swiglu = full_pack.find("SwiGLU MLP")
    pos_muon = full_pack.find("Muon LR Tuning")

    assert pos_gqa < pos_muon
    assert pos_muon < pos_swiglu

def test_log_retrieval_event_with_methods():
    conn = _in_memory_db()
    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    exp_id = rfs.log_experiment("def456", 2.0, 0.4, "keep", "test", run_id, conn)

    event_id = rfs.log_retrieval_event(
        event_type="agent_search",
        query="attention",
        retrieved_content="content",
        agent_run_id=run_id,
        experiment_id=exp_id,
        conn=conn,
        method_ids=[1, 2],
    )

    cur = conn.cursor()
    cur.execute("SELECT * FROM retrieved_methods WHERE event_id = ? ORDER BY method_id", (event_id,))
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0]["method_id"] == 1
    assert rows[1]["method_id"] == 2


def test_log_retrieval_event_with_experiment_methods():
    conn = _in_memory_db()
    method_ids = [
        row["method_id"]
        for row in conn.execute(
            "SELECT method_id FROM methods ORDER BY method_id LIMIT 2"
        )
    ]
    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    experiment_id = rfs.log_experiment(
        "graph123",
        2.0,
        0.4,
        "keep",
        "graph retrieval test",
        run_id,
        conn,
    )

    rfs.log_retrieval_event(
        event_type="agent_search",
        query="graph attention",
        retrieved_content="content",
        agent_run_id=run_id,
        experiment_id=experiment_id,
        conn=conn,
        method_ids=[method_ids[0], method_ids[0], method_ids[1]],
    )

    rows = conn.execute(
        """
        SELECT method_id, role, note
        FROM experiment_methods
        WHERE experiment_id = ?
        ORDER BY method_id
        """,
        (experiment_id,),
    ).fetchall()
    assert [row["method_id"] for row in rows] == method_ids
    assert [row["role"] for row in rows] == ["retrieved", "retrieved"]
    assert [row["note"] for row in rows] == [None, None]


def test_recommendations_and_graph_context_include_code_targets_and_lessons():
    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO methods (name, description, implementation_hint, risks, category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Graph Context MLP",
            "Graph context method for MLP routing.",
            "Modify the MLP class in train.py.",
            "Avoid widening the hidden layer.",
            "architecture",
        ),
    )
    method_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO code_symbols
        (file_path, symbol_name, symbol_kind, start_line, end_line, docstring, code)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "train.py",
            "MLP",
            "class",
            134,
            151,
            "Feed-forward block.",
            "class MLP:\n    pass\n",
        ),
    )
    cur.execute(
        """
        INSERT INTO method_code_targets
        (method_id, file_path, symbol_name, symbol_kind, rationale)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            method_id,
            "train.py",
            "MLP",
            "class",
            "MLP owns the feed-forward path for this method.",
        ),
    )
    conn.commit()
    rfs.log_lesson(
        "mlp-width-regression",
        "Prior lesson: widening MLP increased BPB; prefer a narrower gate.",
        outcome="discard",
        method_id=method_id,
        conn=conn,
    )

    recommendations = rfs.recommend_next_experiments(
        "Graph Context MLP",
        conn=conn,
        max_items=3,
    )
    graph_context = rfs.get_graph_context(
        "Graph Context MLP",
        conn=conn,
        max_chars=4000,
    )

    for text in (recommendations, graph_context):
        assert "Graph Context MLP" in text
        assert "train.py" in text
        assert "MLP" in text
        assert "Prior lesson: widening MLP increased BPB" in text



def test_recommendations_fallback_to_catalog_on_over_specific_query():
    conn = _in_memory_db()

    recommendations = rfs.recommend_next_experiments(
        "SwiGLU optimizer combined query with no exact shared method",
        conn=conn,
        max_items=3,
    )

    assert "No direct method matches" in recommendations
    assert "Recommended Next Experiments" in recommendations
    assert "## 1." in recommendations

def test_method_stats_calculation():
    conn = _in_memory_db()

    cur = conn.cursor()
    cur.execute("SELECT method_id FROM methods WHERE name = ?", ("Grouped-Query Attention (GQA)",))
    method_id = cur.fetchone()["method_id"]

    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)

    # Experiment 1 (baseline): val_bpb = 2.0
    exp1_id = rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", run_id, conn)

    # Experiment 2 (keep): val_bpb = 1.95, linked to Method A
    exp2_id = rfs.log_experiment("commit2", 1.95, 0.5, "keep", "keep exp", run_id, conn)
    rfs.log_retrieval_event("agent_search", "query", "retrieved", run_id, exp2_id, conn, method_ids=[method_id])

    # Experiment 3 (discard): val_bpb = 2.05, linked to Method A
    exp3_id = rfs.log_experiment("commit3", 2.05, 0.5, "discard", "discard exp", run_id, conn)
    rfs.log_retrieval_event("agent_search", "query", "retrieved", run_id, exp3_id, conn, method_ids=[method_id])

    # Experiment 4 (crash): val_bpb = None, linked to Method A
    exp4_id = rfs.log_experiment("commit4", None, None, "crash", "crash exp", run_id, conn)
    rfs.log_retrieval_event("agent_search", "query", "retrieved", run_id, exp4_id, conn, method_ids=[method_id])

    pack = rfs.search_methods("Grouped-Query Attention", conn)
    expected_stats = "3 runs, 33% success (1 keeps, 1 discards, 1 crashes), avg -0.050000 BPB"
    assert expected_stats in pack


def test_method_pack_stats_formatting():
    conn = _in_memory_db()

    cur = conn.cursor()
    cur.execute("SELECT method_id, name FROM methods")
    methods_dict = {row["name"]: row["method_id"] for row in cur.fetchall()}
    swiglu_id = methods_dict["SwiGLU MLP"]
    gqa_id = methods_dict["Grouped-Query Attention (GQA)"]
    muon_id = methods_dict["Muon LR Tuning"]

    run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", run_id, conn)

    exp_gqa = rfs.log_experiment("gqa_commit", 1.9, 0.5, "keep", "GQA keep", run_id, conn)
    rfs.log_retrieval_event("agent_search", "GQA", "retrieved", run_id, exp_gqa, conn, method_ids=[gqa_id])

    exp_swiglu = rfs.log_experiment("swiglu_commit", 2.1, 0.5, "discard", "SwiGLU discard", run_id, conn)
    rfs.log_retrieval_event("agent_search", "SwiGLU", "retrieved", run_id, exp_swiglu, conn, method_ids=[swiglu_id])

    pack = rfs.get_method_pack(conn)

    assert "1 runs, 100% success (1 keeps, 0 discards, 0 crashes), avg -0.100000 BPB" in pack
    assert "1 runs, 0% success (0 keeps, 1 discards, 0 crashes), avg +0.000000 BPB" in pack
    assert "No runs yet" in pack

    pos_gqa = pack.find("Grouped-Query Attention (GQA)")
    pos_muon = pack.find("Muon LR Tuning")
    pos_swiglu = pack.find("SwiGLU MLP")

    assert pos_gqa < pos_muon
    assert pos_muon < pos_swiglu

    pack_search_gqa = rfs.search_methods("Grouped-Query Attention", conn)
    assert "1 runs, 100% success (1 keeps, 0 discards, 0 crashes), avg -0.100000 BPB" in pack_search_gqa

    pack_search_muon = rfs.search_methods("Muon", conn)
    assert "No runs yet" in pack_search_muon


def test_run_agent_mode_reuses_query():
    import os
    import sqlite3
    import tempfile
    import unittest.mock
    from agents import harness_agent
    from researchfs.db import initialize_db
    from researchfs.ingest import ensure_seed
    from researchfs import retrieve as rfs

    # Use a named shared in-memory database so functions that close their
    # connection do not destroy the database.
    db_uri = "file:reuse_test_mem?mode=memory&cache=shared"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    def patched_get_connection():
        c = sqlite3.connect(db_uri, uri=True)
        c.row_factory = sqlite3.Row
        return c

    with tempfile.TemporaryDirectory() as tmpdir, \
         unittest.mock.patch("researchfs.db.get_connection", patched_get_connection), \
         unittest.mock.patch("researchfs.retrieve.get_connection", patched_get_connection):
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            initialize_db()
            ensure_seed()

            # Create a simple train.py for the agent to edit.
            with open("train.py", "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            # Set up a prior agent run with a baseline and a kept, improved experiment
            # that has an associated agent_search retrieval event.
            prior_run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini")
            rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", prior_run_id)
            prior_exp_id = rfs.log_experiment("abc123", 2.400, 0.5, "keep", "better", prior_run_id)
            rfs.log_retrieval_event(
                "agent_search", "prior query", "content", prior_run_id, prior_exp_id, method_ids=[1]
            )

            search_calls = []

            def mock_search_methods_and_ids(query, conn=None, max_chars=None, agent_run_id=None):
                search_calls.append(query)
                return f"pack for {query}", [1]

            train_calls = [0]

            def mock_run_train(time_budget, steps_budget, env, log_path):
                if train_calls[0] == 0:
                    train_calls[0] += 1
                    return (2.500, 1024.0, "success", "")
                return (2.350, 1024.0, "success", "")

            def mock_subprocess_run(args, **kwargs):
                class Result:
                    returncode = 0
                    stdout = "abc123"
                    stderr = ""
                res = Result()
                if len(args) >= 5 and args[:3] == ["git", "diff", "--quiet"] and args[-1] == "train.py":
                    res.returncode = 0
                return res

            def mock_llm(prompt):
                return "Description: test\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>>"

            with unittest.mock.patch("researchfs.retrieve.search_methods_and_ids", mock_search_methods_and_ids), \
                 unittest.mock.patch("agents.harness_agent._run_train", mock_run_train), \
                 unittest.mock.patch("subprocess.run", mock_subprocess_run), \
                 unittest.mock.patch("agents.harness_agent._current_branch", return_value="main"), \
                 unittest.mock.patch("agents.harness_agent._checkout"), \
                 unittest.mock.patch("agents.harness_agent.rfs.start_agent_run", return_value=prior_run_id):
                harness_agent.run_agent_mode(
                    mode="loopgraph",
                    model="gpt-4o-mini",
                    max_experiments=1,
                    time_budget=60,
                    llm_fn=mock_llm,
                    steps_budget=100,
                )

            assert "prior query" in search_calls, f"Expected 'prior query' in search calls, got {search_calls}"
        finally:
            os.chdir(old_cwd)


def test_run_agent_mode_reuses_query_saves_tokens_across_experiments():
    import os
    import sqlite3
    import tempfile
    import unittest.mock
    from agents import harness_agent
    from researchfs.db import initialize_db
    from researchfs.ingest import ensure_seed
    from researchfs import retrieve as rfs

    db_uri = "file:reuse_test_mem2?mode=memory&cache=shared"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    def patched_get_connection():
        c = sqlite3.connect(db_uri, uri=True)
        c.row_factory = sqlite3.Row
        return c

    with tempfile.TemporaryDirectory() as tmpdir, \
         unittest.mock.patch("researchfs.db.get_connection", patched_get_connection), \
         unittest.mock.patch("researchfs.retrieve.get_connection", patched_get_connection):
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            initialize_db()
            ensure_seed()

            with open("train.py", "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            prior_run_id = rfs.start_agent_run("loopgraph", "gpt-4o-mini")
            rfs.log_experiment("baseline", 2.500, 0.5, "keep", "baseline", prior_run_id)

            search_calls = []
            retrieval_llm_calls = []

            def mock_search_methods_and_ids(query, conn=None, max_chars=None, agent_run_id=None):
                search_calls.append(query)
                return f"pack for {query}", [1]

            train_calls = [0]

            def mock_run_train(time_budget, steps_budget, env, log_path):
                val = [2.500, 2.400, 2.300][train_calls[0]]
                train_calls[0] += 1
                return (val, 1024.0, "success", "")

            def mock_subprocess_run(args, **kwargs):
                class Result:
                    returncode = 0
                    stdout = "abc123"
                    stderr = ""
                res = Result()
                if len(args) >= 5 and args[:3] == ["git", "diff", "--quiet"] and args[-1] == "train.py":
                    res.returncode = 0
                return res

            def mock_llm(prompt):
                # Distinguish retrieval-query prompts from code-proposal prompts.
                if "search query" in prompt.lower():
                    retrieval_llm_calls.append("query")
                    return "Muon LR"
                return "Description: test\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>>"

            with unittest.mock.patch("researchfs.retrieve.search_methods_and_ids", mock_search_methods_and_ids), \
                 unittest.mock.patch("agents.harness_agent._run_train", mock_run_train), \
                 unittest.mock.patch("subprocess.run", mock_subprocess_run), \
                 unittest.mock.patch("agents.harness_agent._current_branch", return_value="main"), \
                 unittest.mock.patch("agents.harness_agent._checkout"), \
                 unittest.mock.patch("agents.harness_agent.rfs.start_agent_run", return_value=prior_run_id):
                harness_agent.run_agent_mode(
                    mode="loopgraph",
                    model="gpt-4o-mini",
                    max_experiments=2,
                    time_budget=60,
                    llm_fn=mock_llm,
                    steps_budget=100,
                )

            # Experiment 1 performs an agent_search; experiment 2 reuses the query.
            assert len(search_calls) == 1, f"Expected 1 search call, got {search_calls}"
            assert len(retrieval_llm_calls) == 1, (
                f"Expected 1 retrieval LLM call, got {len(retrieval_llm_calls)}"
            )
        finally:
            os.chdir(old_cwd)


def test_run_agent_mode_cleans_up_after_baseline_failure():
    import os
    import tempfile
    import unittest.mock
    from agents import harness_agent

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("train.py", "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            def mock_subprocess_run(args, **kwargs):
                class Result:
                    returncode = 0
                    stdout = "abc123"
                    stderr = ""
                return Result()

            finish_mock = unittest.mock.Mock()
            checkout_mock = unittest.mock.Mock()

            with unittest.mock.patch("agents.harness_agent.initialize_db"), \
                 unittest.mock.patch("agents.harness_agent.ensure_seed"), \
                 unittest.mock.patch("agents.harness_agent.subprocess.run", mock_subprocess_run), \
                 unittest.mock.patch("agents.harness_agent._current_branch", return_value="main"), \
                 unittest.mock.patch("agents.harness_agent._run_train", return_value=(None, None, "crash", "boom")), \
                 unittest.mock.patch("agents.harness_agent.rfs.start_agent_run", return_value=4321), \
                 unittest.mock.patch("agents.harness_agent.rfs.finish_agent_run", finish_mock), \
                 unittest.mock.patch("agents.harness_agent._checkout", checkout_mock):
                try:
                    harness_agent.run_agent_mode(
                        mode="baseline",
                        model="gpt-4o-mini",
                        max_experiments=1,
                        time_budget=60,
                        llm_fn=lambda prompt: "unused",
                        steps_budget=100,
                    )
                except RuntimeError as exc:
                    assert "Baseline run failed: boom" in str(exc)
                else:
                    assert False, "Expected baseline failure to propagate"

            finish_mock.assert_called_once_with(4321)
            checkout_mock.assert_called_once()
            assert checkout_mock.call_args[0][0] == "main"
        finally:
            os.chdir(old_cwd)


def test_run_agent_mode_cleans_up_after_mid_loop_exception():
    import os
    import tempfile
    import unittest.mock
    from agents import harness_agent

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with open("train.py", "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            def mock_subprocess_run(args, **kwargs):
                class Result:
                    returncode = 0
                    stdout = "abc123"
                    stderr = ""
                return Result()

            def failing_llm(prompt):
                raise ValueError("agent exploded")

            client_mock = unittest.mock.Mock()
            finish_mock = unittest.mock.Mock()
            checkout_mock = unittest.mock.Mock()

            with unittest.mock.patch("agents.harness_agent.initialize_db"), \
                 unittest.mock.patch("agents.harness_agent.ensure_seed"), \
                 unittest.mock.patch("agents.harness_agent.subprocess.run", mock_subprocess_run), \
                 unittest.mock.patch("agents.harness_agent._current_branch", return_value="main"), \
                 unittest.mock.patch("agents.harness_agent._run_train", return_value=(2.500, 1024.0, "success", "")), \
                 unittest.mock.patch("agents.harness_agent.rfs.start_agent_run", return_value=5432), \
                 unittest.mock.patch("agents.harness_agent.rfs.finish_agent_run", finish_mock), \
                 unittest.mock.patch("agents.harness_agent.rfs.log_experiment", return_value=6543), \
                 unittest.mock.patch("agents.harness_agent.ResearchFSClient", return_value=client_mock), \
                 unittest.mock.patch("agents.harness_agent._checkout", checkout_mock):
                try:
                    harness_agent.run_agent_mode(
                        mode="baseline",
                        model="gpt-4o-mini",
                        max_experiments=1,
                        time_budget=60,
                        llm_fn=failing_llm,
                        steps_budget=100,
                    )
                except ValueError as exc:
                    assert "agent exploded" in str(exc)
                else:
                    assert False, "Expected mid-loop exception to propagate"

            finish_mock.assert_called_once_with(5432)
            checkout_mock.assert_called_once()
            assert checkout_mock.call_args[0][0] == "main"
            client_mock.close.assert_called_once_with()
        finally:
            os.chdir(old_cwd)


def test_search_methods_max_chars():
    conn = _in_memory_db()

    # Empty query returns the full catalog; use budgets that truncate at method boundaries.
    small_pack = rfs.search_methods("", conn, max_chars=600)
    large_pack = rfs.search_methods("", conn, max_chars=2000)

    small_method_count = small_pack.count("## ")
    large_method_count = large_pack.count("## ")

    assert small_method_count > 0, "Small budget should still include at least one whole method"
    assert small_method_count < large_method_count, "Larger budget should include more methods"
    assert "... and " in small_pack and "more methods omitted" in small_pack, (
        "Truncated pack should report the omitted method count"
    )

    # search_methods_and_ids should honor the same budget and return matching ids.
    pack_and_ids, ids = rfs.search_methods_and_ids("", conn, max_chars=600)
    assert pack_and_ids == small_pack
    assert len(ids) == small_method_count



def test_get_query_history_str():
    from researchfs.client import ResearchFSClient, ResearchFSSession
    conn = _in_memory_db()
    cur = conn.cursor()

    # 1. Create an agent run
    cur.execute("INSERT INTO agent_runs (mode, model) VALUES ('loopgraph', 'test-model')")
    agent_run_id = cur.lastrowid

    # 2. Insert some experiments
    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'hash1', 2.153019, 'keep', 'experiment_1')
        """,
        (agent_run_id,)
    )
    exp_id1 = cur.lastrowid

    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'hash2', 2.164321, 'discard', 'experiment_2')
        """,
        (agent_run_id,)
    )
    exp_id2 = cur.lastrowid

    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'hash3', NULL, 'crash', 'experiment_3')
        """,
        (agent_run_id,)
    )
    exp_id3 = cur.lastrowid

    # 3. Insert matching retrieval events
    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search', 'SwiGLU', 'content1')
        """,
        (agent_run_id, exp_id1)
    )

    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search_reuse', 'Muon', 'content2')
        """,
        (agent_run_id, exp_id2)
    )

    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search', 'Attention', 'content3')
        """,
        (agent_run_id, exp_id3)
    )

    # Also log a retrieval event of a different type to ensure it is filtered out
    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'method_pack', 'all', 'content4')
        """,
        (agent_run_id, exp_id1)
    )

    # 4. Use ResearchFSSession directly with the in-memory connection
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False
    session = ResearchFSSession(client, agent_run_id)

    history_str = session.get_query_history()

    # 5. Assert expectations
    expected_lines = [
        "Query: 'SwiGLU' -> Outcome: Kept (improved BPB to 2.153019)",
        "Query: 'Muon' -> Outcome: Discarded (BPB: 2.164321)",
        "Query: 'Attention' -> Outcome: Crashed"
    ]
    for line in expected_lines:
        assert line in history_str

    # Ensure method_pack is NOT in history
    assert "Query: 'all'" not in history_str


def test_researchfs_client_session():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()

    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    # Start a session manually using a pre-created run id
    cur = conn.cursor()
    cur.execute("INSERT INTO agent_runs (mode, model) VALUES ('loopgraph', 'test-model')")
    agent_run_id = cur.lastrowid

    session = ResearchFSSession(client, agent_run_id, baseline_metric=2.18)
    assert isinstance(session, ResearchFSSession)

    # Test query reuse: nothing yet should return None
    assert session.get_reusable_query() is None

    # Test suggest_query with a stub LLM that returns a JSON decision
    def fake_llm(prompt: str) -> str:
        assert "validation bits per byte" in prompt
        return '{"search": true, "query": "SwiGLU", "reason": "Try SwiGLU"}'

    decision = session.suggest_query("results", "brief", fake_llm)
    assert decision["search"] is True
    assert decision["query"] == "SwiGLU"
    assert decision["reused"] is False

    # Test context retrieval
    pack, ids = session.retrieve_context("SwiGLU")
    assert "SwiGLU MLP" in pack
    assert ids is not None

    # Second retrieval should hit the cache
    pack2, ids2 = session.retrieve_context("SwiGLU")
    assert pack2 == pack
    assert ids2 == ids

    # Test log_outcome and log_retrieval
    exp_id = session.log_outcome(val_metric=2.15, status="keep", description="SwiGLU experiment")
    assert exp_id is not None
    ret_id = session.log_retrieval("agent_search", "SwiGLU", pack[:2000], experiment_id=exp_id)
    assert ret_id is not None

    # Now the same query should be reusable since it improved over baseline
    reusable = session.get_reusable_query()
    assert reusable == "SwiGLU"


def test_researchfs_client_start_session_uses_configured_db_path():
    import tempfile
    import unittest.mock
    from researchfs.client import ResearchFSClient, ResearchFSSession

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        client = ResearchFSClient(db_path=tmp.name)
        try:
            with unittest.mock.patch(
                "researchfs.retrieve.get_connection",
                side_effect=AssertionError("default ResearchFS DB should not be opened"),
            ):
                session = client.start_session(mode="loopgraph", model="test-model", baseline_metric=2.18)

            assert isinstance(session, ResearchFSSession)

            conn = client.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT mode, model FROM agent_runs WHERE agent_run_id = ?", (session.agent_run_id,))
            row = cur.fetchone()
            assert row is not None
            assert row["mode"] == "loopgraph"
            assert row["model"] == "test-model"
        finally:
            client.close()


def test_researchfs_client_initializes_cached_connection_once_and_closes():
    import unittest.mock
    from researchfs.client import ResearchFSClient

    class FakeConn:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_conn = FakeConn()
    client = ResearchFSClient(db_path="custom.db")
    with unittest.mock.patch("researchfs.client.db_module.get_connection", return_value=fake_conn) as get_conn, \
         unittest.mock.patch("researchfs.client.db_module.initialize_db") as init_db:
        assert client.get_connection() is fake_conn
        assert client.get_connection() is fake_conn
        get_conn.assert_called_once_with("custom.db")
        init_db.assert_called_once_with(fake_conn)

    client.close()
    assert fake_conn.closed is True
    assert client._conn is None
    assert client._conn_initialized is False



def test_lab_client_ingest_list_get_retrieve_feedback_evaluate():
    from researchfs import ResearchFSLabClient

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    client = ResearchFSLabClient(conn=conn)

    ingest_body = client.ingest(
        title="Lab Client Residual Gate Memory",
        authors="SDK Lab",
        abstract="A paper about labclienttoken residual gates.",
        url="https://example.com/lab-client-residual-gate",
        publish_date="2026-06-30",
        content="The labclienttoken residual gate stabilizes compact transformer runs.",
        methods=[
            {
                "name": "Lab Client Residual Gate",
                "description": "Use labclienttoken gates to stabilize compact runs.",
                "implementation_hint": "Scale residual branches with a learned gate.",
                "risks": "A closed gate can slow early optimization.",
                "category": "normalization",
                "source_chunk_indices": [0],
            }
        ],
    )
    assert ingest_body["ok"] is True
    paper_id = ingest_body["paper_id"]
    method_id = ingest_body["method_ids"][0]

    papers_body = client.list_papers()
    assert papers_body["ok"] is True
    assert [paper["paper_id"] for paper in papers_body["papers"]] == [paper_id]

    paper_body = client.get_paper(paper_id)
    assert paper_body["paper"]["title"] == "Lab Client Residual Gate Memory"
    assert paper_body["methods"][0]["method_id"] == method_id
    assert paper_body["chunks"][0]["method_ids"] == [method_id]

    method_body = client.get_method(method_id)
    assert method_body["method"]["name"] == "Lab Client Residual Gate"
    assert method_body["provenance_chunks"][0]["paper_id"] == paper_id

    retrieve_body = client.retrieve(
        "labclienttoken residual",
        max_chars=1200,
        log=True,
        include_scores=True,
        event_type="lab_client_test_search",
    )
    assert retrieve_body["ok"] is True
    assert method_id in retrieve_body["method_ids"]
    assert retrieve_body["event_id"] is not None
    assert retrieve_body["scores"]
    event = conn.execute(
        "SELECT event_type, query, retrieved_content FROM retrieval_events WHERE event_id = ?",
        (retrieve_body["event_id"],),
    ).fetchone()
    assert event["event_type"] == "lab_client_test_search"
    assert event["query"] == "labclienttoken residual"
    assert event["retrieved_content"] == retrieve_body["context"]

    feedback_body = client.feedback(
        method_id,
        0.75,
        event_id=retrieve_body["event_id"],
        label="useful",
        note="SDK wrapper feedback path",
        source="test",
    )
    assert feedback_body["ok"] is True
    feedback_row = conn.execute(
        "SELECT rating, label, note, source FROM method_feedback WHERE feedback_id = ?",
        (feedback_body["feedback_id"],),
    ).fetchone()
    assert feedback_row["rating"] == 0.75
    assert feedback_row["label"] == "useful"
    assert feedback_row["note"] == "SDK wrapper feedback path"
    assert feedback_row["source"] == "test"

    evaluation_body = client.evaluate_extraction(
        paper_id,
        [{"name": "Lab Client Residual Gate", "source_chunk_indices": [0]}],
        min_recall=1.0,
        min_evidence_coverage=0.0,
        notes="SDK eval",
    )
    evaluation = evaluation_body["evaluation"]
    assert evaluation["matched_count"] == 1
    assert evaluation["missing_methods"] == []
    assert evaluation["passed"] is True
    assert evaluation["notes"] == "SDK eval"

    stats_body = client.stats()
    assert stats_body["counts"]["papers"] == 1
    assert stats_body["counts"]["methods"] == 1
    assert stats_body["feedback"]["count"] == 1



def test_lab_client_ingest_accepts_full_payload_mapping():
    from researchfs import ResearchFSLabClient

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    client = ResearchFSLabClient(conn=conn)

    body = client.ingest(
        {
            "title": "Copy Paste SDK Memory",
            "authors": "SDK Lab",
            "abstract": "A paper about sdkpayloadtoken direct payload ingestion.",
            "url": "https://example.com/sdk-payload",
            "publish_date": "2026-06-30",
            "content": "The sdkpayloadtoken method should be retrievable after direct payload ingestion.",
            "methods": [
                {
                    "name": "SDK Payload Method",
                    "description": "Use sdkpayloadtoken direct payload ingestion.",
                    "implementation_hint": "Keep top-level methods when wrapping SDK calls.",
                    "risks": "Nested methods would be ignored by extraction selection.",
                    "category": "sdk",
                    "source_chunk_indices": [0],
                }
            ],
        }
    )

    assert body["ok"] is True
    assert len(body["method_ids"]) == 1
    retrieved = client.retrieve("sdkpayloadtoken", max_chars=800)
    assert body["method_ids"][0] in retrieved["method_ids"]

def test_lab_client_ingest_file_accepts_pdf_and_reports_bad_pdf_and_missing_route():
    import tempfile
    from pathlib import Path

    from researchfs import ResearchFSLabClient, ResearchFSLabClientError

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    client = ResearchFSLabClient(conn=conn)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "sdk-pdf-memory.pdf"
        _write_minimal_pdf(
            pdf_path,
            "The sdkpdftoken method should be retrievable after SDK PDF ingestion.",
        )
        body = client.ingest_file(
            pdf_path,
            authors="SDK PDF Lab",
            methods=[
                {
                    "name": "SDK PDF Memory Route",
                    "description": "Load sdkpdftoken PDFs through the SDK.",
                    "implementation_hint": "Call ResearchFSLabClient.ingest_file with a PDF path.",
                    "risks": "Unreadable PDF files must fail explicitly.",
                    "category": "sdk",
                    "source_chunk_indices": [0],
                }
            ],
        )

        assert body["ok"] is True
        assert len(body["method_ids"]) == 1
        paper = conn.execute(
            "SELECT title, authors, url FROM papers WHERE paper_id = ?",
            (body["paper_id"],),
        ).fetchone()
        assert paper["title"] == "Sdk Pdf Memory"
        assert paper["authors"] == "SDK PDF Lab"
        assert paper["url"] == pdf_path.resolve().as_uri()

        retrieved = client.retrieve("sdkpdftoken", max_chars=800)
        assert body["method_ids"][0] in retrieved["method_ids"]

        malformed_path = Path(tmpdir) / "malformed-sdk-paper.pdf"
        malformed_path.write_bytes(b"%PDF-1.7\nbroken local PDF\n")
        try:
            client.ingest_file(malformed_path)
            assert False, "Expected malformed PDF to raise ResearchFSLabClientError"
        except ResearchFSLabClientError as exc:
            assert exc.status == 400
            assert exc.code == "extraction_failed"
            assert "Could not extract text from PDF" in exc.message
            assert exc.body["ok"] is False

    try:
        client.dispatch("GET", "/missing")
        assert False, "Expected missing route to raise ResearchFSLabClientError"
    except ResearchFSLabClientError as exc:
        assert exc.status == 404
        assert exc.code == "not_found"
        assert "No route for GET /missing" in exc.message


def test_lab_client_ingest_repo_uses_configured_fetcher():
    from researchfs import ResearchFSLabClient

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    fetched_repos = []

    def stub_fetcher(repo):
        fetched_repos.append(repo)
        return {
            "title": "GitHub: example/sdk-repo",
            "authors": "example",
            "abstract": "SDK repo ingestion exposes sdkrepotoken README memory.",
            "url": "https://github.com/example/sdk-repo",
            "publish_date": "n/a",
            "content": "The sdkrepotoken method is documented in the repository README.",
        }

    client = ResearchFSLabClient(conn=conn, repo_fetcher=stub_fetcher)
    body = client.ingest_repo(
        "example/sdk-repo",
        methods=[
            {
                "name": "SDK Repo README Route",
                "description": "Load sdkrepotoken README content through the SDK.",
                "implementation_hint": "Call ResearchFSLabClient.ingest_repo with owner/repo.",
                "risks": "Repository fetch failures must surface through the local API.",
                "category": "sdk",
                "source_chunk_indices": [0],
            }
        ],
    )

    assert body["ok"] is True
    assert fetched_repos == ["example/sdk-repo"]
    assert body["repo"] == "example/sdk-repo"
    assert body["extraction_status"]["source_type"] == "repo_readme"
    retrieved = client.retrieve("sdkrepotoken", max_chars=800)
    assert body["method_ids"][0] in retrieved["method_ids"]

def test_researchfs_session_log_retrieval_records_unique_method_ids():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    cur = conn.cursor()
    cur.execute("SELECT method_id FROM methods ORDER BY method_id LIMIT 2")
    method_ids = [row["method_id"] for row in cur.fetchall()]
    assert len(method_ids) == 2

    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False
    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    session = ResearchFSSession(client, run_id, baseline_metric=2.18)

    event_id = session.log_retrieval(
        "agent_search",
        "SwiGLU",
        "content",
        method_ids=[method_ids[0], method_ids[1], method_ids[0]],
    )

    cur.execute("SELECT method_id FROM retrieved_methods WHERE event_id = ? ORDER BY method_id", (event_id,))
    rows = cur.fetchall()
    assert [row["method_id"] for row in rows] == method_ids


def test_search_methods_hybrid_general_appends():
    conn = _in_memory_db()

    # 1. Test search with non-empty query.
    # When query is "SwiGLU", the main search results should contain SwiGLU MLP.
    # The general optimization methods (like Muon, Dropout, Warmup, Clipping) should be appended.
    pack, ids = rfs.search_methods_and_ids("SwiGLU", conn)

    # Check that SwiGLU MLP is present
    assert "SwiGLU MLP" in pack

    # Check that General Methods header is present
    assert "# General Optimization & Hyperparameter Tuning Methods" in pack

    # Check that Muon LR Tuning (optimizer) is appended
    assert "Muon LR Tuning" in pack

    # Check that Dropout Regularization (regularization) is appended
    assert "Dropout Regularization" in pack

    # Check that general method IDs are merged in `ids`
    cur = conn.cursor()
    cur.execute("SELECT method_id FROM methods WHERE name = 'SwiGLU MLP'")
    swiglu_id = cur.fetchone()["method_id"]
    cur.execute("SELECT method_id FROM methods WHERE name = 'Muon LR Tuning'")
    muon_id = cur.fetchone()["method_id"]
    cur.execute("SELECT method_id FROM methods WHERE name = 'Dropout Regularization'")
    dropout_id = cur.fetchone()["method_id"]

    assert swiglu_id in ids
    assert muon_id in ids
    assert dropout_id in ids

    # 2. Test search with non-empty query that matches a general method directly.
    # When query is "Muon", the main search results will already contain "Muon LR Tuning".
    # General methods should still be appended, but "Muon LR Tuning" should NOT be duplicated in the pack
    # or the IDs.
    pack2, ids2 = rfs.search_methods_and_ids("Muon", conn)

    # "Muon LR Tuning" should appear only once.
    assert pack2.count("Muon LR Tuning") == 1
    assert ids2.count(muon_id) == 1

    # Other general methods (like Dropout Regularization) should still be appended.
    assert "Dropout Regularization" in pack2
    assert dropout_id in ids2

    # 3. Test budget constraint
    pack_budget, ids_budget = rfs.search_methods_and_ids("SwiGLU", conn, max_chars=len(pack) - 100)
    assert len(ids_budget) < len(ids)

def test_researchfs_query_reuse_scorer():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    cur = conn.cursor()
    cur.execute("INSERT INTO agent_runs (mode, model) VALUES ('loopgraph', 'test-model')")
    run_id = cur.lastrowid

    session = ResearchFSSession(client, run_id, baseline_metric=2.18)

    # No history -> nothing reusable.
    assert session.get_reusable_query() is None
    assert session._query_outcomes() == []

    # A kept, improved experiment makes its query reusable.
    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'h1', 2.15, 'keep', 'swiglu')
        """,
        (run_id,),
    )
    exp1 = cur.lastrowid
    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search', 'SwiGLU', 'content')
        """,
        (run_id, exp1),
    )

    assert session.get_reusable_query() == "SwiGLU"
    outcomes = {o["query"]: o for o in session._query_outcomes()}
    assert outcomes["SwiGLU"]["kept_count"] == 1
    assert outcomes["SwiGLU"]["total_count"] == 1
    assert abs(outcomes["SwiGLU"]["best_delta"] - 0.03) < 1e-6
    assert abs(outcomes["SwiGLU"]["avg_delta"] - 0.03) < 1e-6
    assert outcomes["SwiGLU"]["consecutive_uses"] == 1

    # Consecutive reuse of the same query blocks further reuse.
    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'h2', 2.14, 'keep', 'swiglu again')
        """,
        (run_id,),
    )
    exp2 = cur.lastrowid
    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search_reuse', 'SwiGLU', 'content')
        """,
        (run_id, exp2),
    )
    assert session.get_reusable_query() is None

    # A newer, different query is penalized for being the most recent experiment,
    # so the older, still-promising SwiGLU query wins again once the streak is broken.
    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'h3', 2.17, 'keep', 'muon')
        """,
        (run_id,),
    )
    exp3 = cur.lastrowid
    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search', 'Muon', 'content')
        """,
        (run_id, exp3),
    )
    reusable = session.get_reusable_query()
    assert reusable == "SwiGLU"

    # A discarded query is never reused.
    cur.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, status, description)
        VALUES (?, 'h4', 2.20, 'discard', 'attention')
        """,
        (run_id,),
    )
    exp4 = cur.lastrowid
    cur.execute(
        """
        INSERT INTO retrieval_events (agent_run_id, experiment_id, event_type, query, retrieved_content)
        VALUES (?, ?, 'agent_search', 'Attention', 'content')
        """,
        (run_id, exp4),
    )
    outcomes = {o["query"]: o for o in session._query_outcomes()}
    assert outcomes["Attention"]["kept_count"] == 0
    assert session.get_reusable_query() == "SwiGLU"


def test_query_scorer_prefers_best_query():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    baseline = 2.50
    session = ResearchFSSession(client, run_id, baseline_metric=baseline)

    # Muon LR is the better query (delta 0.10). Insert it first so that the
    # weaker SwiGLU query (delta 0.05) is the most recent and gets penalised.
    exp1 = rfs.log_experiment(
        "h1", 2.40, None, "keep", "muon lr",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search", "Muon LR", "content",
        agent_run_id=run_id, experiment_id=exp1, conn=conn,
    )
    exp2 = rfs.log_experiment(
        "h2", 2.45, None, "keep", "swiglu",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search", "SwiGLU", "content",
        agent_run_id=run_id, experiment_id=exp2, conn=conn,
    )

    assert session._pick_reusable_query() == "Muon LR"


def test_query_scorer_penalizes_recent_use():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    baseline = 2.50
    session = ResearchFSSession(client, run_id, baseline_metric=baseline)

    # SwiGLU has a smaller improvement but is not the most recent experiment.
    exp1 = rfs.log_experiment(
        "h1", 2.42, None, "keep", "swiglu",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search", "SwiGLU", "content",
        agent_run_id=run_id, experiment_id=exp1, conn=conn,
    )

    # Muon LR is the most recent and has a larger raw improvement, but its
    # score is halved by the recency penalty.
    exp2 = rfs.log_experiment(
        "h2", 2.40, None, "keep", "muon lr",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search", "Muon LR", "content",
        agent_run_id=run_id, experiment_id=exp2, conn=conn,
    )

    assert session._pick_reusable_query() == "SwiGLU"


def test_query_scorer_blocks_overused_query():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    baseline = 2.50
    session = ResearchFSSession(client, run_id, baseline_metric=baseline)

    exp1 = rfs.log_experiment(
        "h1", 2.40, None, "keep", "muon lr",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search", "Muon LR", "content",
        agent_run_id=run_id, experiment_id=exp1, conn=conn,
    )

    exp2 = rfs.log_experiment(
        "h2", 2.42, None, "keep", "muon lr again",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search_reuse", "Muon LR", "content",
        agent_run_id=run_id, experiment_id=exp2, conn=conn,
    )

    reusable = session._pick_reusable_query()
    assert reusable != "Muon LR"
    assert reusable is None


def test_suggest_query_reuses_then_explores():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    baseline = 2.50
    session = ResearchFSSession(client, run_id, baseline_metric=baseline)

    exp1 = rfs.log_experiment(
        "h1", 2.45, None, "keep", "swiglu",
        agent_run_id=run_id, conn=conn,
    )
    rfs.log_retrieval_event(
        "agent_search", "SwiGLU", "content",
        agent_run_id=run_id, experiment_id=exp1, conn=conn,
    )

    llm_calls: list[str] = []

    def stub_llm(prompt: str) -> str:
        llm_calls.append(prompt)
        return '{"search": true, "query": "NewArea", "reason": "Explore elsewhere"}'

    # After a single kept, improved experiment, the scorer should reuse.
    decision1 = session.suggest_query("results", "brief", stub_llm)
    assert decision1["search"] is True
    assert decision1["query"] == "SwiGLU"
    assert decision1["reused"] is True
    assert len(llm_calls) == 0

    # Simulate a second consecutive kept use of the same query.
    exp2 = session.log_outcome(val_metric=2.44, status="keep", description="swiglu again")
    session.log_retrieval("agent_search_reuse", "SwiGLU", "content", experiment_id=exp2)

    # Overuse rules should now block reuse and fall back to the LLM.
    decision2 = session.suggest_query("results", "brief", stub_llm)
    assert decision2["search"] is True
    assert decision2["query"] == "NewArea"
    assert decision2["reused"] is False
    assert len(llm_calls) == 1


def test_suggest_query_rejects_empty_json_query():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    session = ResearchFSSession(client, run_id, baseline_metric=2.50)

    def empty_llm(prompt: str) -> str:
        return '{"search": true, "query": "", "reason": "No ideas"}'

    decision = session.suggest_query("results", "brief", empty_llm)
    assert decision["search"] is False
    assert decision["query"] == ""
    assert decision["reused"] is False
    assert "empty" in decision["reason"].lower()


def test_frontier_hint_lists_recent_queries():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    session = ResearchFSSession(client, run_id, baseline_metric=2.50)

    # Log discarded experiments so the queries appear in history but are not reusable.
    for query, val_bpb in [("Attention", 2.55), ("MLP", 2.56), ("Optimizer", 2.57)]:
        exp_id = rfs.log_experiment(
            "h", val_bpb, None, "discard", query,
            agent_run_id=run_id, conn=conn,
        )
        rfs.log_retrieval_event(
            "agent_search", query, "content",
            agent_run_id=run_id, experiment_id=exp_id, conn=conn,
        )

    prompts: list[str] = []

    def capture_llm(prompt: str) -> str:
        prompts.append(prompt)
        return '{"search": true, "query": "NewArea", "reason": "Explore elsewhere"}'

    decision = session.suggest_query("results", "brief", capture_llm)
    assert decision["search"] is True
    assert decision["query"] == "NewArea"
    assert decision["reused"] is False
    assert len(prompts) == 1

    prompt = prompts[0]
    assert "You have recently explored:" in prompt
    for query in ["Attention", "MLP", "Optimizer"]:
        assert query in prompt
    assert "Prefer a different optimization area" in prompt
    # The existing anti-overuse paragraph should remain.
    assert "Avoid repeating queries that were recently used" in prompt


def test_custom_task_descriptions_in_prompt():
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    session = ResearchFSSession(
        client,
        run_id,
        baseline_metric=2.50,
        task_target="the training script `train_custom.py`",
        objective="maximize validation accuracy (val_acc)",
        task_name="image classification",
    )

    prompts: list[str] = []

    def capture_llm(prompt: str) -> str:
        prompts.append(prompt)
        return '{"search": true, "query": "Augmentation", "reason": "Custom"}'

    session.suggest_query("results", "brief", capture_llm)
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "train_custom.py" in prompt
    assert "maximize validation accuracy (val_acc)" in prompt
    assert "image classification" in prompt


def test_suggest_query_async_returns_same_decision():
    import asyncio
    from researchfs.client import ResearchFSClient, ResearchFSSession

    conn = _in_memory_db()
    client = ResearchFSClient(db_path=":memory:")
    client._conn = conn
    client._close_on_done = False

    run_id = rfs.start_agent_run("loopgraph", "test-model", conn=conn)
    session = ResearchFSSession(client, run_id, baseline_metric=2.50)

    async def async_llm(prompt: str) -> str:
        return '{"search": true, "query": "AsyncArea", "reason": "Async decision"}'

    def sync_llm(prompt: str) -> str:
        return '{"search": true, "query": "AsyncArea", "reason": "Async decision"}'

    sync_decision = session.suggest_query("results", "brief", sync_llm)
    async_decision = asyncio.run(
        session.suggest_query_async("results", "brief", async_llm)
    )
    assert async_decision == sync_decision
    assert async_decision["search"] is True
    assert async_decision["query"] == "AsyncArea"
    assert async_decision["reused"] is False



def _landing_page_sources():
    from pathlib import Path

    return {
        "index.html": Path("index.html").read_text(encoding="utf-8"),
        "public/index.html": Path("public/index.html").read_text(encoding="utf-8"),
    }

def _landing_page_script(html):
    import re

    return "\n".join(
        re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)
    )


def _landing_page_controls(html):
    import re

    return " ".join(
        re.findall(
            r"<(?:button|input|textarea|select|label)\b[^>]*(?:>.*?</(?:button|textarea|select|label)>)?",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ).lower()


def _landing_page_value(html, element_id):
    import html as html_module
    import re

    input_match = re.search(
        rf"<input\b[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*\bvalue=[\"']([^\"']*)[\"']",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if input_match:
        return html_module.unescape(input_match.group(1))

    textarea_match = re.search(
        rf"<textarea\b[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*>(.*?)</textarea>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if textarea_match:
        return html_module.unescape(textarea_match.group(1))

    select_match = re.search(
        rf"<select\b[^>]*\bid=[\"']{re.escape(element_id)}[\"'][^>]*>(.*?)</select>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if select_match:
        options = re.findall(
            r"<option\b([^>]*)>(.*?)</option>",
            select_match.group(1),
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert options, element_id
        selected = next(
            (
                attributes
                for attributes, _label in options
                if re.search(r"\bselected\b", attributes, flags=re.IGNORECASE)
            ),
            options[0][0],
        )
        value_match = re.search(
            r"\bvalue=[\"']([^\"']*)[\"']",
            selected,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert value_match, element_id
        return html_module.unescape(value_match.group(1))

    raise AssertionError(f"missing landing control value: {element_id}")


def _landing_page_workflow_payloads(html, paper_id=None, method_id=None, event_id=None):
    import json

    title = _landing_page_value(html, "paper-title").strip() or "Untitled local note"
    content = _landing_page_value(html, "paper-content").strip()
    method_name = _landing_page_value(html, "method-name").strip() or "Local note method"
    method_hint = (
        _landing_page_value(html, "method-hint").strip()
        or "Review this note before the next related experiment."
    )
    query = _landing_page_value(html, "retrieve-query").strip() or "MLP SwiGLU"
    feedback_label = _landing_page_value(html, "feedback-label").strip() or "browser-loop"

    payloads = {
        "ingest": {
            "paper": {"title": title, "abstract": content[:400], "content": content},
            "methods": [
                {
                    "name": method_name,
                    "description": method_hint,
                    "implementation_hint": method_hint,
                    "source_chunk_indices": [0],
                }
            ],
        },
        "retrieve": {"query": query, "log": True, "include_scores": True},
    }
    if method_id is not None:
        payloads["feedback"] = {
            "method_id": method_id,
            "rating": float(_landing_page_value(html, "feedback-rating")),
            "label": feedback_label,
            "source": "landing-page",
        }
        if event_id is not None:
            payloads["feedback"]["event_id"] = event_id
    if paper_id is not None:
        payloads["evaluate"] = {
            "paper_id": paper_id,
            "expected_methods": json.loads(_landing_page_value(html, "expected-methods")),
            "min_recall": 1.0,
            "min_evidence_coverage": 0.0,
            "notes": "landing-page evaluation",
        }
    return payloads


def _script_window_for_route(script, route):
    route_index = script.find(route)
    assert route_index >= 0, route
    return script[max(0, route_index - 1200) : route_index + 1600]


def _script_windows_for_route(script, route):
    import re

    matches = list(re.finditer(re.escape(route), script))
    assert matches, route
    return [script[max(0, match.start() - 1200) : match.end() + 1600] for match in matches]


def _assert_post_route(script, route):
    import re

    shared_post_options = re.search(
        r"\bjsonOptions\b[\s\S]{0,250}\bmethod\s*:\s*['\"]POST['\"]",
        script,
    )
    for window in _script_windows_for_route(script, route):
        has_inline_post = "method" in window.lower() and "POST" in window
        has_shared_post_options = "jsonOptions" in window and shared_post_options
        if has_inline_post or has_shared_post_options:
            return window
    assert False, route


def _assert_literal_json_flag(window, name):
    import re

    assert re.search(rf"\b{name}\s*:\s*true\b", window), name


def _assert_no_direct_default_fetch(script, path):
    import re

    direct_default_fetch = re.compile(
        r"fetch\s*\(\s*(?:new\s+Request\s*\(\s*)?['\"`]https?://(?:127\.0\.0\.1|localhost):8765",
        flags=re.IGNORECASE,
    )
    assert not direct_default_fetch.search(script), path


def test_landing_pages_are_identical():
    pages = _landing_page_sources()

    assert pages["index.html"] == pages["public/index.html"]


def test_landing_pages_expose_local_full_loop_workflow_controls():
    pages = _landing_page_sources()

    for path, html in pages.items():
        controls = _landing_page_controls(html)
        assert controls.count("<button") >= 5, path
        for token in ("base", "url", "ingest", "retrieve", "feedback", "method"):
            assert token in controls, (path, token)
        assert "expected" in controls, (path, "expected")
        assert "evaluate" in controls or "evaluation" in controls, (path, "evaluate")
        ordered_steps = [
            "1. post /ingest",
            "2. post /retrieve",
            "3. inspect latest /method",
            "4. post /feedback",
            "5. post /evaluate/extraction",
        ]
        positions = [controls.find(step) for step in ordered_steps]
        assert all(position >= 0 for position in positions), (path, positions)
        assert positions == sorted(positions), (path, positions)


def test_landing_pages_scripts_call_existing_local_workflow_routes():
    pages = _landing_page_sources()

    for path, html in pages.items():
        script = _landing_page_script(html)
        lower_script = script.lower()

        assert "localstorage.getitem" in lower_script, path
        assert "localstorage.setitem" in lower_script, path
        assert "fetch" in lower_script, path
        _assert_no_direct_default_fetch(script, path)

        ingest_window = _assert_post_route(script, "/ingest")
        for key in ("title", "content", "methods"):
            assert key in ingest_window, (path, key)
        for response_key in ("paper_id", "method_ids"):
            assert response_key in script, (path, response_key)

        retrieve_window = _assert_post_route(script, "/retrieve")
        _assert_literal_json_flag(retrieve_window, "log")
        _assert_literal_json_flag(retrieve_window, "include_scores")
        for response_key in ("event_id", "method_ids"):
            assert response_key in script, (path, response_key)

        feedback_window = _assert_post_route(script, "/feedback")
        for key in ("method_id", "event_id"):
            assert key in feedback_window, (path, key)

        method_window = _script_window_for_route(script, "/method/")
        assert "method_id" in method_window or "methodId" in method_window, path

        evaluation_window = _assert_post_route(script, "/evaluate/extraction")
        for key in ("paper_id", "expected_methods"):
            assert key in evaluation_window, (path, key)


def test_landing_pages_default_payloads_complete_local_api_loop():
    import sqlite3

    from researchfs.api import ResearchFSLabAPI
    from researchfs.db import initialize_db

    pages = _landing_page_sources()
    html = pages["index.html"]
    assert html == pages["public/index.html"]

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    payloads = _landing_page_workflow_payloads(html)
    ingest_response = api.dispatch("POST", "/ingest", payloads["ingest"])
    assert ingest_response.status == 200
    assert ingest_response.body["paper_id"]
    assert ingest_response.body["method_ids"]

    retrieve_response = api.dispatch("POST", "/retrieve", payloads["retrieve"])
    assert retrieve_response.status == 200
    assert retrieve_response.body["event_id"] is not None
    assert retrieve_response.body["method_ids"]

    method_id = retrieve_response.body["method_ids"][0]
    detail_response = api.dispatch("GET", f"/method/{method_id}")
    assert detail_response.status == 200
    assert detail_response.body["provenance_chunks"][0]["evidence_text"]

    feedback_payload = _landing_page_workflow_payloads(
        html,
        method_id=method_id,
        event_id=retrieve_response.body["event_id"],
    )["feedback"]
    feedback_response = api.dispatch("POST", "/feedback", feedback_payload)
    assert feedback_response.status == 200
    assert feedback_response.body["feedback_id"] is not None

    evaluate_payload = _landing_page_workflow_payloads(
        html,
        paper_id=ingest_response.body["paper_id"],
    )["evaluate"]
    evaluate_response = api.dispatch("POST", "/evaluate/extraction", evaluate_payload)
    assert evaluate_response.status == 200
    assert evaluate_response.body["evaluation"]["passed"] is True

    stats_response = api.dispatch("GET", "/stats")
    assert stats_response.status == 200
    assert stats_response.body["counts"]["retrieval_events"] == 1
    assert stats_response.body["counts"]["method_feedback"] == 1
    assert stats_response.body["confidence"]["high_confidence_count"] == 1


def test_landing_pages_list_required_local_api_routes():
    pages = _landing_page_sources()

    for path, html in pages.items():
        for route in ("/health", "/stats", "/ingest", "/ingest/repo", "/retrieve", "/feedback", "/evaluate/extraction"):
            assert route in html, (path, route)
        assert "/method/&lt;id&gt;" in html or "/method/<id>" in html or "/method/" in html, path


def test_landing_pages_fetch_from_configured_base_url_not_hard_coded_default():
    pages = _landing_page_sources()

    for path, html in pages.items():
        _assert_no_direct_default_fetch(_landing_page_script(html), path)



def test_lab_api_index_and_health_dispatch_routes():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    index_response = api.dispatch("GET", "/")
    assert index_response.status == 200
    assert index_response.body["ok"] is True
    assert index_response.body["service"] == "ResearchFS lab API"
    assert index_response.body["local_only"] is True
    assert "Local-only" in index_response.body["caveat"]
    routes = {(route["method"], route["path"]) for route in index_response.body["routes"]}
    assert ("GET", "/") in routes
    assert ("GET", "/health") in routes
    assert ("POST", "/retrieve") in routes
    assert ("POST", "/ingest/file") in routes
    assert ("POST", "/ingest/repo") in routes
    assert ("GET", "/paper/{paper_id}/memory/export") in routes
    assert ("GET", "/memory/export") in routes
    assert ("POST", "/memory/import") in routes

    health_response = api.dispatch("GET", "/health")
    assert health_response.status == 200
    assert health_response.body == {
        "ok": True,
        "service": {"name": "ResearchFS lab API", "status": "ok"},
        "db": {"status": "ok"},
        "local_only": True,
    }


def test_lab_api_http_handler_adds_cors_and_options_preflight():
    import http.client
    import tempfile
    import threading
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        api = ResearchFSLabAPI(db_path=str(Path(tmpdir) / "researchfs_http.sqlite"))
        server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", "/", headers={"Origin": "http://localhost:5173"})
            get_response = conn.getresponse()
            get_body = json.loads(get_response.read().decode("utf-8"))
            assert get_response.status == 200
            assert get_response.getheader("Content-Type") == "application/json; charset=utf-8"
            assert get_response.getheader("Access-Control-Allow-Origin") == "*"
            assert get_response.getheader("Access-Control-Allow-Methods") == "GET, POST, OPTIONS"
            assert get_response.getheader("Access-Control-Allow-Headers") == "Content-Type"
            assert get_body["ok"] is True
            assert get_body["local_only"] is True

            conn.request(
                "OPTIONS",
                "/retrieve",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            options_response = conn.getresponse()
            assert options_response.status == 204
            assert options_response.read() == b""
            assert options_response.getheader("Access-Control-Allow-Origin") == "*"
            assert options_response.getheader("Access-Control-Allow-Methods") == "GET, POST, OPTIONS"
            assert options_response.getheader("Access-Control-Allow-Headers") == "Content-Type"
            assert options_response.getheader("Access-Control-Max-Age") == "600"
            assert options_response.getheader("Content-Length") == "0"
        finally:
            conn.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_lab_api_ingest_retrieve_detail_experiment_lesson_graph_context():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    method_payload = {
        "name": "Segmented Dropout Schedule",
        "description": "Use a segmented dropout schedule to regularize short lab runs.",
        "implementation_hint": "Increase dropout after the warmup segment only.",
        "risks": "Too much late dropout can underfit.",
        "category": "regularization",
        "source_chunk_indices": [0],
    }
    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Segmented Dropout for Tiny Transformers",
            "authors": "Lab Author",
            "abstract": "A paper about segmented dropout schedules.",
            "url": "https://example.com/segmented-dropout",
            "publish_date": "2026-06-30",
            "content": (
                "Segmented dropout schedules regularize compact transformer "
                "training by delaying stronger dropout until after warmup."
            ),
            "methods": [method_payload],
        },
    )
    assert ingest_response.status == 200
    assert ingest_response.body["ok"] is True
    assert ingest_response.body["paper_id"] is not None
    assert ingest_response.body["chunk_ids"]
    method_id = ingest_response.body["method_ids"][0]
    assert ingest_response.body["ingestion"]["compression_status"]
    assert ingest_response.body["ingestion"]["extraction_status"] in {
        "extracted",
        "completed",
        "succeeded",
    }
    assert "confidence_score" in ingest_response.body["extraction_status"]
    assert "confidence_reason" in ingest_response.body["extraction_status"]
    assert "confidence_score" in ingest_response.body["ingestion"]
    assert "confidence_reason" in ingest_response.body["ingestion"]

    retrieve_response = api.dispatch(
        "POST",
        "/retrieve",
        {"query": "segmented dropout", "max_chars": 700, "log": True},
    )
    assert retrieve_response.status == 200
    assert method_id in retrieve_response.body["method_ids"]
    assert "Segmented Dropout Schedule" in retrieve_response.body["context"]
    assert len(retrieve_response.body["context"]) <= 700

    detail_response = api.dispatch("GET", f"/method/{method_id}")
    assert detail_response.status == 200
    assert detail_response.body["method"]["name"] == "Segmented Dropout Schedule"
    assert detail_response.body["provenance_chunks"]
    provenance = detail_response.body["provenance_chunks"][0]
    assert provenance["title"] == "Segmented Dropout for Tiny Transformers"
    assert "Segmented dropout schedules" in provenance["content"]

    experiment_response = api.dispatch(
        "POST",
        "/experiment",
        {
            "commit_hash": "lab-api-commit",
            "status": "keep",
            "val_bpb": 2.345,
            "memory_gb": 1.25,
            "description": "Segmented dropout improved validation bpb.",
            "method_ids": [method_id],
        },
    )
    assert experiment_response.status == 200
    experiment_id = experiment_response.body["experiment_id"]

    lesson_response = api.dispatch(
        "POST",
        "/lesson",
        {
            "label": "Lesson-only API retrieval gate",
            "content": "lessononlyapiretrievaltoken proves lab lessons retrieve methods without full papers.",
            "outcome": "keep",
            "method_id": method_id,
            "experiment_id": experiment_id,
        },
    )
    assert lesson_response.status == 200
    assert lesson_response.body["lesson_id"] is not None

    lesson_retrieve_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "lessononlyapiretrievaltoken",
            "max_chars": 1800,
            "include_scores": True,
        },
    )
    assert lesson_retrieve_response.status == 200
    assert method_id in lesson_retrieve_response.body["method_ids"]
    assert "Lesson-only API retrieval gate" in lesson_retrieve_response.body["context"]
    assert "lessononlyapiretrievaltoken" in lesson_retrieve_response.body["context"]
    assert [score["method_id"] for score in lesson_retrieve_response.body["scores"]][0] == method_id

    graph_response = api.dispatch(
        "POST",
        "/retrieve",
        {"query": "lessononlyapiretrievaltoken", "max_chars": 1800, "mode": "graph"},
    )
    assert graph_response.status == 200
    assert method_id in graph_response.body["method_ids"]
    assert "Lesson-only API retrieval gate" in graph_response.body["context"]
    assert "lessononlyapiretrievaltoken" in graph_response.body["context"]

    stats_response = api.dispatch("GET", "/stats")
    assert stats_response.status == 200
    stats = stats_response.body
    assert stats["ok"] is True
    assert stats["counts"]["papers"] == 1
    assert stats["counts"]["chunks"] == len(ingest_response.body["chunk_ids"])
    assert stats["counts"]["methods"] == 1
    assert stats["counts"]["lessons"] == 1
    assert stats["counts"]["retrieval_events"] == 1
    assert stats["extraction_statuses"] == {"extracted": 1}
    assert stats["source_types"] == {"unknown": 1}
    assert stats["compression"]["paper_count"] == 1
    assert stats["compression"]["source_char_count"] == len(
        "Segmented dropout schedules regularize compact transformer "
        "training by delaying stronger dropout until after warmup."
    )
    assert stats["compression"]["compressed_char_count"] > 0
    assert stats["compression"]["average_compression_ratio"] > 0
    assert set(stats["confidence"]) == {
        "count",
        "average_score",
        "low_confidence_count",
        "high_confidence_count",
    }



def test_lab_api_ingest_preserves_source_type_payload_metadata():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    nested_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "paper": {
                "title": "Nested Source Type Payload",
                "authors": "Payload Lab",
                "content": "The nestedsourcetypetoken paper keeps an experiment_report label.",
                "source_type": "experiment_report",
            },
            "methods": [
                {
                    "name": "Nested Source Type Method",
                    "description": "Keep nestedsourcetypetoken labels.",
                    "implementation_hint": "Copy source_type from nested paper metadata.",
                    "risks": "Dropping labels hides provenance.",
                    "category": "ingestion",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert nested_response.status == 200
    assert nested_response.body["extraction_status"]["source_type"] == "experiment_report"
    nested_status = conn.execute(
        "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
        (nested_response.body["paper_id"],),
    ).fetchone()
    assert nested_status["source_type"] == "experiment_report"

    top_level_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Top Level Source Type Payload",
            "authors": "Payload Lab",
            "content": "The toplevelsourcetypetoken paper keeps an imported_file label.",
            "source_type": "imported_file",
            "methods": [
                {
                    "name": "Top Level Source Type Method",
                    "description": "Keep toplevelsourcetypetoken labels.",
                    "implementation_hint": "Copy source_type from top-level paper metadata.",
                    "risks": "Dropping labels hides provenance.",
                    "category": "ingestion",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert top_level_response.status == 200
    assert top_level_response.body["extraction_status"]["source_type"] == "imported_file"
    top_level_status = conn.execute(
        "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
        (top_level_response.body["paper_id"],),
    ).fetchone()
    assert top_level_status["source_type"] == "imported_file"


def test_lab_api_ingest_duplicate_source_type_payload_relabels_status_only():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    first_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "API Duplicate Source Relabel Payload",
            "authors": "API Source Lab",
            "content": "The apiduplicaterelabeltoken paper starts as a paper source.",
            "source_type": "paper",
            "methods": [
                {
                    "name": "API Duplicate Source Relabel",
                    "description": "Keep apiduplicaterelabeltoken compressed memory.",
                    "implementation_hint": "Relabel only the extraction status source type.",
                    "risks": "Duplicate relabeling must not reinsert methods.",
                    "category": "ingestion",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert first_response.status == 200
    paper_id = first_response.body["paper_id"]
    before_status = dict(
        conn.execute(
            "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
    )

    second_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "API Duplicate Source Relabel Payload",
            "content": "Duplicate payload can correct source metadata.",
            "source_type": "lab_note",
        },
    )

    assert second_response.status == 200
    assert second_response.body["duplicate"] is True
    assert second_response.body["paper_id"] == paper_id
    assert second_response.body["method_ids"] == []
    assert second_response.body["extraction_status"]["source_type"] == "lab_note"
    after_status = dict(
        conn.execute(
            "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
    )
    assert after_status["source_type"] == "lab_note"
    for key in (
        "status",
        "chunk_count",
        "method_count",
        "source_char_count",
        "compressed_char_count",
        "compression_ratio",
        "confidence_score",
        "confidence_reason",
        "extractor_model",
        "extractor_prompt_version",
        "error_text",
    ):
        assert after_status[key] == before_status[key]

def test_lab_api_evaluate_extraction_reports_metrics_and_source_coverage():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Extraction Eval API Memory",
            "authors": "Eval Lab",
            "abstract": "A paper about deterministic extraction evaluation.",
            "content": "The source evidence says exact eval methods should be measured.",
            "methods": [
                {
                    "name": "Exact Eval Method",
                    "description": "Measure expected methods against stored extraction rows.",
                    "implementation_hint": "Compare names and exact source evidence.",
                    "risks": "Missing source evidence hides bad memories.",
                    "category": "evaluation",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert ingest_response.status == 200
    paper_id = ingest_response.body["paper_id"]
    method_id = ingest_response.body["method_ids"][0]
    chunk_id = ingest_response.body["chunk_ids"][0]
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = ?, evidence_start = ?, evidence_end = ?, evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        ("exact eval methods should be measured", 25, 62, "exact", method_id, chunk_id),
    )
    conn.commit()

    response = api.dispatch(
        "POST",
        "/evaluate/extraction",
        {
            "paper_id": paper_id,
            "expected_methods": [
                {
                    "name": "Exact Eval Method",
                    "source_chunk_ids": [chunk_id],
                    "evidence_contains": ["exact eval methods"],
                }
            ],
            "min_recall": 1.0,
            "min_evidence_coverage": 1.0,
            "notes": "API smoke",
        },
    )

    assert response.status == 200
    body = response.body
    assert body["ok"] is True
    evaluation = body["evaluation"]
    assert evaluation["paper_id"] == paper_id
    assert evaluation["expected_count"] == 1
    assert evaluation["extracted_count"] == 1
    assert evaluation["matched_count"] == 1
    assert evaluation["precision"] == 1.0
    assert evaluation["recall"] == 1.0
    assert evaluation["f1"] == 1.0
    assert evaluation["source_link_coverage"] == 1.0
    assert evaluation["evidence_coverage"] == 1.0
    assert evaluation["passed"] is True
    assert "persisted" not in evaluation
    assert "eval_id" not in evaluation
    assert evaluation["matched_methods"][0]["method_id"] == method_id
    assert evaluation["matched_methods"][0]["source_expectation_met"] is True
    assert evaluation["thresholds"]["min_recall"]["passed"] is True
    assert evaluation["thresholds"]["min_evidence_coverage"]["passed"] is True

    negative_response = api.dispatch(
        "POST",
        "/evaluate/extraction",
        {
            "paper_id": paper_id,
            "expected_methods": [
                {
                    "name": "Exact Eval Method",
                    "source_chunk_ids": [chunk_id],
                    "evidence_contains": ["missing eval citation"],
                }
            ],
            "min_evidence_coverage": 1.0,
        },
    )
    assert negative_response.status == 200
    negative_evaluation = negative_response.body["evaluation"]
    assert negative_evaluation["evidence_coverage"] == 0.5
    assert negative_evaluation["matched_methods"][0]["source_expectation_met"] is False
    assert negative_evaluation["thresholds"]["min_evidence_coverage"]["passed"] is False
    assert negative_evaluation["passed"] is False


def test_lab_api_evaluate_extraction_validation_errors_are_consistent():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)
    paper_id = conn.execute("INSERT INTO papers (title) VALUES (?)", ("Eval Validation",)).lastrowid
    conn.commit()

    missing_expected = api.dispatch(
        "POST",
        "/evaluate/extraction",
        {"paper_id": paper_id},
    )
    assert missing_expected.status == 400
    assert missing_expected.body["error"]["code"] == "missing_field"

    invalid_threshold = api.dispatch(
        "POST",
        "/evaluate/extraction",
        {"paper_id": paper_id, "expected_methods": ["Any Method"], "min_recall": 1.5},
    )
    assert invalid_threshold.status == 400
    assert invalid_threshold.body["error"]["code"] == "invalid_field"

    missing_paper = api.dispatch(
        "POST",
        "/evaluate/extraction",
        {"paper_id": 999999, "expected_methods": ["Any Method"]},
    )
    assert missing_paper.status == 404
    assert missing_paper.body["error"]["code"] == "paper_not_found"


def test_lab_api_evaluate_extraction_uses_default_recall_threshold_and_remains_pure():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Pure Eval API Memory",
            "content": "Pure eval rows measure stored methods without persistence.",
            "methods": [
                {
                    "name": "Pure Eval Method",
                    "description": "Compare deterministic extraction metrics.",
                    "implementation_hint": "Report recall without storing evaluation rows.",
                    "risks": "Missing expected methods should fail default recall.",
                    "category": "evaluation",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert ingest_response.status == 200

    response = api.dispatch(
        "POST",
        "/evaluate/extraction",
        {
            "paper_id": ingest_response.body["paper_id"],
            "expected_methods": ["Pure Eval Method", "Missing Eval Method"],
            "notes": "api echo only",
        },
    )
    assert response.status == 200
    evaluation = response.body["evaluation"]
    assert evaluation["matched_count"] == 1
    assert evaluation["recall"] == 0.5
    assert evaluation["passed"] is False
    assert evaluation["notes"] == "api echo only"
    assert evaluation["thresholds"]["min_recall"] == {
        "threshold": 1.0,
        "observed": 0.5,
        "passed": False,
    }
    assert "persisted" not in evaluation
    assert "eval_id" not in evaluation
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'extraction_evals'"
    ).fetchone()[0] == 0



def test_lab_api_stats_aggregates_extraction_confidence():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    rows = [
        ("High Confidence Paper", "extracted", 0.9, "methods and provenance are well supported"),
        ("Low Confidence Paper", "failed", 0.2, "extraction failed before methods were available"),
        ("Unscored Legacy Paper", "skipped", None, None),
    ]
    for title, status, score, reason in rows:
        paper_id = conn.execute("INSERT INTO papers (title) VALUES (?)", (title,)).lastrowid
        conn.execute(
            """
            INSERT INTO paper_extraction_status (
                paper_id, status, source_type, chunk_count, method_count,
                source_char_count, compressed_char_count, compression_ratio,
                confidence_score, confidence_reason
            )
            VALUES (?, ?, 'unknown', 0, 0, 0, 0, NULL, ?, ?)
            """,
            (paper_id, status, score, reason),
        )
    conn.commit()

    response = api.dispatch("GET", "/stats")
    assert response.status == 200
    assert response.body["confidence"]["count"] == 2
    assert response.body["confidence"]["average_score"] == 0.55
    assert response.body["confidence"]["low_confidence_count"] == 1
    assert response.body["confidence"]["high_confidence_count"] == 1


def test_lab_api_lists_papers_with_extraction_status_and_detail_provenance():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Auditable Paper Memory",
            "authors": "Audit Lab",
            "abstract": "A paper about auditable paper memory.",
            "url": "https://example.com/auditable-paper-memory",
            "publish_date": "2026-06-30",
            "content": "Auditable paper memory links extracted methods to exact source chunks.",
            "methods": [
                {
                    "name": "Auditable Chunk Linking",
                    "description": "Link each extracted method to paper chunks for inspection.",
                    "implementation_hint": "Keep chunk provenance beside method metadata.",
                    "risks": "Missing provenance makes review difficult.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert ingest_response.status == 200
    paper_id = ingest_response.body["paper_id"]
    method_id = ingest_response.body["method_ids"][0]
    chunk_id = ingest_response.body["chunk_ids"][0]
    existing_evidence_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(method_sources)")
    }
    evidence_column_defs = {
        "evidence_text": "TEXT",
        "evidence_start": "INTEGER",
        "evidence_end": "INTEGER",
        "evidence_strategy": "TEXT",
    }
    for column, definition in evidence_column_defs.items():
        if column not in existing_evidence_columns:
            conn.execute(f"ALTER TABLE method_sources ADD COLUMN {column} {definition}")
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = ?,
            evidence_start = ?,
            evidence_end = ?,
            evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        (
            "extracted methods to exact source chunks",
            29,
            70,
            "exact",
            method_id,
            chunk_id,
        ),
    )
    conn.execute(
        """
        UPDATE paper_extraction_status
        SET confidence_score = ?, confidence_reason = ?
        WHERE paper_id = ?
        """,
        (0.92, "method has source chunk provenance and compression coverage", paper_id),
    )
    conn.commit()

    list_response = api.dispatch("GET", "/papers")
    assert list_response.status == 200
    papers = list_response.body["papers"]
    assert len(papers) == 1
    assert papers[0]["paper_id"] == paper_id
    assert papers[0]["title"] == "Auditable Paper Memory"
    assert papers[0]["extraction_status"]["status"] == "extracted"
    assert papers[0]["extraction_status"]["chunk_count"] == len(ingest_response.body["chunk_ids"])
    assert papers[0]["extraction_status"]["method_count"] == 1
    assert papers[0]["extraction_status"]["confidence_score"] == 0.92
    assert papers[0]["extraction_status"]["confidence_reason"] == (
        "method has source chunk provenance and compression coverage"
    )

    detail_response = api.dispatch("GET", f"/paper/{paper_id}")
    assert detail_response.status == 200
    detail = detail_response.body
    assert detail["paper"]["title"] == "Auditable Paper Memory"
    assert detail["extraction_status"]["status"] == "extracted"
    assert detail["extraction_status"]["confidence_score"] == 0.92
    assert detail["extraction_status"]["confidence_reason"] == (
        "method has source chunk provenance and compression coverage"
    )
    assert [method["method_id"] for method in detail["methods"]] == [method_id]
    assert detail["methods"][0]["name"] == "Auditable Chunk Linking"
    assert [chunk["chunk_id"] for chunk in detail["chunks"]] == [chunk_id]
    assert detail["chunks"][0]["method_ids"] == [method_id]
    assert detail["provenance"][0]["method_id"] == method_id
    assert detail["provenance"][0]["chunk_id"] == chunk_id
    assert "exact source chunks" in detail["provenance"][0]["content"]
    assert detail["provenance"][0]["evidence_text"] == "extracted methods to exact source chunks"
    assert detail["provenance"][0]["evidence_start"] == 29
    assert detail["provenance"][0]["evidence_end"] == 70
    assert detail["provenance"][0]["evidence_strategy"] == "exact"

    method_detail_response = api.dispatch("GET", f"/method/{method_id}")
    assert method_detail_response.status == 200
    method_provenance = method_detail_response.body["provenance_chunks"][0]
    assert method_provenance["evidence_text"] == "extracted methods to exact source chunks"
    assert method_provenance["evidence_start"] == 29
    assert method_provenance["evidence_end"] == 70
    assert method_provenance["evidence_strategy"] == "exact"


def _paper_memory_fixture():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Memory Card Contract Paper",
            "authors": "Memory Lab",
            "abstract": "A paper about deterministic paper memory cards.",
            "url": "https://example.com/memory-card-contract",
            "publish_date": "2026-07-01",
            "content": (
                "Memory card retrieval links exact evidence snippets, compact method metadata, "
                "lessons, experiments, and feedback without calling a summarizer."
            ),
            "methods": [
                {
                    "name": "Deterministic Memory Card Assembly",
                    "description": "Assemble compact paper memory from stored rows only.",
                    "implementation_hint": "Join paper, method, chunk, provenance, lesson, experiment, and feedback rows.",
                    "risks": "Invented summaries would break provenance.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert ingest_response.status == 200
    paper_id = ingest_response.body["paper_id"]
    method_id = ingest_response.body["method_ids"][0]
    chunk_id = ingest_response.body["chunk_ids"][0]

    existing_evidence_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(method_sources)")
    }
    evidence_column_defs = {
        "evidence_text": "TEXT",
        "evidence_start": "INTEGER",
        "evidence_end": "INTEGER",
        "evidence_strategy": "TEXT",
    }
    for column, definition in evidence_column_defs.items():
        if column not in existing_evidence_columns:
            conn.execute(f"ALTER TABLE method_sources ADD COLUMN {column} {definition}")
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = ?,
            evidence_start = ?,
            evidence_end = ?,
            evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        (
            "exact evidence snippets, compact method metadata, lessons, experiments, and feedback",
            33,
            117,
            "exact",
            method_id,
            chunk_id,
        ),
    )
    conn.execute(
        """
        UPDATE paper_extraction_status
        SET confidence_score = ?,
            confidence_reason = ?,
            extractor_model = ?,
            extractor_prompt_version = ?
        WHERE paper_id = ?
        """,
        (
            0.88,
            "stored method provenance covers the compact memory card",
            "paper-memory-test-extractor",
            "paper-memory-prompt-v2",
            paper_id,
        ),
    )
    experiment_id = rfs.log_experiment(
        "memory-card-commit",
        1.25,
        2.5,
        "keep",
        "Memory card experiment kept the deterministic assembly path.",
        conn=conn,
    )
    rfs.link_experiment_methods(
        experiment_id,
        [method_id],
        role="validated",
        note="Experiment validated the paper memory card route.",
        conn=conn,
    )
    lesson_id = rfs.log_lesson(
        "Prefer stored evidence",
        "Use stored provenance snippets rather than generating a fresh summary.",
        outcome="adopted",
        method_id=method_id,
        experiment_id=experiment_id,
        conn=conn,
    )
    feedback_response = api.dispatch(
        "POST",
        "/feedback",
        {
            "method_id": method_id,
            "rating": 0.75,
            "label": "useful",
            "note": "Feedback should be visible in the compact paper card.",
            "source": "paper-memory-test",
        },
    )
    assert feedback_response.status == 200
    conn.commit()
    return conn, api, paper_id, method_id, chunk_id, lesson_id, experiment_id



def _research_brief_fixture():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    paper_specs = [
        (
            "Alpha Research Brief Paper",
            "Alpha Research Brief Method",
            "alpha researchbriefsharedtoken compact evidence snippet",
            "ALPHA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL",
            "paper",
        ),
        (
            "Beta Research Brief Paper",
            "Beta Research Brief Method",
            "beta researchbriefsharedtoken compact evidence snippet",
            "BETA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL",
            "lab_note",
        ),
    ]
    paper_ids = []
    method_ids = []
    chunk_ids = []
    lesson_ids = []
    for title, method_name, evidence_text, raw_only_sentinel, source_type in paper_specs:
        ingest_response = api.dispatch(
            "POST",
            "/ingest",
            {
                "title": title,
                "authors": "Research Brief Lab",
                "abstract": f"A compressed paper memory source for {method_name}.",
                "url": f"https://example.com/{title.lower().replace(' ', '-')}",
                "publish_date": "2026-07-01",
                "content": (
                    f"{evidence_text}. "
                    f"{raw_only_sentinel} verbose full-paper text must not leak into "
                    "the compressed research brief."
                ),
                "source_type": source_type,
                "methods": [
                    {
                        "name": method_name,
                        "description": (
                            f"Use {evidence_text} when assembling bounded research briefs."
                        ),
                        "implementation_hint": (
                            "Reuse stored method search and paper memory cards only."
                        ),
                        "risks": "Calling a summarizer or including raw paper text breaks determinism.",
                        "category": "research_brief",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )
        assert ingest_response.status == 200
        paper_id = ingest_response.body["paper_id"]
        method_id = ingest_response.body["method_ids"][0]
        chunk_id = ingest_response.body["chunk_ids"][0]
        conn.execute(
            """
            UPDATE method_sources
            SET evidence_text = ?,
                evidence_start = ?,
                evidence_end = ?,
                evidence_strategy = ?
            WHERE method_id = ? AND chunk_id = ?
            """,
            (evidence_text, 0, len(evidence_text), "exact", method_id, chunk_id),
        )
        conn.execute(
            """
            UPDATE paper_extraction_status
            SET extractor_model = ?,
                extractor_prompt_version = ?
            WHERE paper_id = ?
            """,
            (
                f"{method_name.lower().replace(' ', '-')}-extractor",
                "research-brief-prompt-v3",
                paper_id,
            ),
        )
        lesson_id = rfs.log_lesson(
            f"{method_name} lesson",
            f"Lesson for {evidence_text} should appear in research briefs.",
            outcome="keep",
            method_id=method_id,
            conn=conn,
        )
        paper_ids.append(paper_id)
        method_ids.append(method_id)
        chunk_ids.append(chunk_id)
        lesson_ids.append(lesson_id)

    conn.commit()
    return conn, api, paper_ids, method_ids, chunk_ids, lesson_ids


def _write_research_brief_fixture_db(db_path):
    source_conn, _api, paper_ids, method_ids, chunk_ids, lesson_ids = _research_brief_fixture()
    target_conn = sqlite3.connect(str(db_path))
    try:
        source_conn.backup(target_conn)
    finally:
        target_conn.close()
        source_conn.close()
    return paper_ids, method_ids, chunk_ids, lesson_ids


def _run_memory_cli(argv):
    from io import StringIO
    from researchfs import memory as memory_cli

    stdout = StringIO()
    stderr = StringIO()
    code = memory_cli.main(argv, stdin=StringIO(""), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def _write_eval_fixture_db(db_path):
    source_conn = _eval_test_conn()
    paper_id = _insert_eval_paper(
        source_conn,
        [
            ("Memory CLI Eval Method", [0], {0}),
            ("Evidence Coverage Method", [1], {1}),
        ],
    )
    target_conn = sqlite3.connect(str(db_path))
    try:
        source_conn.backup(target_conn)
    finally:
        target_conn.close()
        source_conn.close()
    return paper_id


def test_memory_cli_query_returns_brief_json_and_excludes_raw_sentinels(tmp_path):
    db_path = tmp_path / "researchfs-memory-query.sqlite"
    paper_ids, method_ids, chunk_ids, lesson_ids = _write_research_brief_fixture_db(db_path)

    code, stdout, stderr = _run_memory_cli(
        [
            "query",
            "--query",
            "researchbriefsharedtoken",
            "--max-chars",
            "4000",
            "--max-papers",
            "2",
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 0
    assert stderr == ""
    body = json.loads(stdout)
    assert body["ok"] is True
    assert body["query"] == "researchbriefsharedtoken"
    assert body["max_chars"] == 4000
    assert body["max_papers"] == 2
    assert body["source_types"] is None
    assert body["paper_ids"] == paper_ids
    assert body["method_ids"] == method_ids
    assert body["chunk_ids"] == chunk_ids
    assert body["lesson_ids"] == lesson_ids
    assert "Alpha Research Brief Method" in body["context"]
    assert "Beta Research Brief Method" in body["context"]
    encoded = json.dumps(body)
    assert "ALPHA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL" not in encoded
    assert "BETA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL" not in encoded


def test_memory_cli_query_source_type_filter_passthrough(tmp_path):
    db_path = tmp_path / "researchfs-memory-query-filter.sqlite"
    paper_ids, method_ids, chunk_ids, lesson_ids = _write_research_brief_fixture_db(db_path)

    code, stdout, stderr = _run_memory_cli(
        [
            "query",
            "--query",
            "researchbriefsharedtoken",
            "--source-type",
            " lab_note ",
            "--max-papers",
            "1",
            "--max-chars",
            "4000",
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 0
    assert stderr == ""
    body = json.loads(stdout)
    assert body["source_types"] == ["lab_note"]
    assert body["paper_ids"] == [paper_ids[1]]
    assert body["method_ids"] == [method_ids[1]]
    assert body["chunk_ids"] == [chunk_ids[1]]
    assert body["lesson_ids"] == [lesson_ids[1]]
    assert "Paper Memory: Beta Research Brief Paper" in body["context"]
    assert "Paper Memory: Alpha Research Brief Paper" not in body["context"]


def test_memory_cli_query_logs_event_and_retrieved_method_ids(tmp_path):
    db_path = tmp_path / "researchfs-memory-query-log.sqlite"
    _paper_ids, method_ids, _chunk_ids, _lesson_ids = _write_research_brief_fixture_db(db_path)

    code, stdout, stderr = _run_memory_cli(
        [
            "query",
            "--query",
            "researchbriefsharedtoken",
            "--include-scores",
            "--log",
            "--event-type",
            "memory_cli_query_test",
            "--max-chars",
            "4000",
            "--max-papers",
            "2",
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 0
    assert stderr == ""
    body = json.loads(stdout)
    assert body["event_id"] is not None
    assert [score["method_id"] for score in body["scores"]] == method_ids

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        event = conn.execute(
            "SELECT event_type, query, retrieved_content FROM retrieval_events WHERE event_id = ?",
            (body["event_id"],),
        ).fetchone()
        assert event["event_type"] == "memory_cli_query_test"
        assert event["query"] == "researchbriefsharedtoken"
        assert event["retrieved_content"] == body["context"]
        retrieved_rows = conn.execute(
            "SELECT method_id FROM retrieved_methods WHERE event_id = ? ORDER BY method_id",
            (body["event_id"],),
        ).fetchall()
    finally:
        conn.close()
    assert [row["method_id"] for row in retrieved_rows] == method_ids


def test_memory_cli_query_rejects_empty_source_type_and_negative_bounds():
    invalid_cases = [
        (
            ["query", "--query", "researchbriefsharedtoken", "--source-type", "  "],
            "--source-type must not be empty",
        ),
        (
            ["query", "--query", "researchbriefsharedtoken", "--max-chars", "-1"],
            "must be non-negative",
        ),
        (
            ["query", "--query", "researchbriefsharedtoken", "--max-papers", "-1"],
            "must be positive",
        ),
    ]
    for argv, expected_error in invalid_cases:
        code, stdout, stderr = _run_memory_cli(argv)
        assert code == 2
        assert stdout == ""
        error_body = json.loads(stderr)
        assert error_body["ok"] is False
        assert expected_error in error_body["error"]


def test_memory_cli_evaluate_outputs_existing_evaluator_result(tmp_path):
    db_path = tmp_path / "researchfs-memory-evaluate.sqlite"
    paper_id = _write_eval_fixture_db(db_path)

    code, stdout, stderr = _run_memory_cli(
        [
            "evaluate",
            "--paper-id",
            str(paper_id),
            "--expected-methods",
            json.dumps(
                [
                    {"name": "Memory CLI Eval Method", "source_chunk_indices": [0]},
                    {"name": "Evidence Coverage Method", "source_chunk_indices": [1]},
                ]
            ),
            "--min-recall",
            "1.0",
            "--min-evidence-coverage",
            "1.0",
            "--notes",
            "cli lab eval",
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 0
    assert stderr == ""
    body = json.loads(stdout)
    assert body["ok"] is True
    evaluation = body["evaluation"]
    assert evaluation["paper_id"] == paper_id
    assert evaluation["matched_count"] == 2
    assert evaluation["recall"] == 1.0
    assert evaluation["evidence_coverage"] == 1.0
    assert evaluation["notes"] == "cli lab eval"


def test_memory_cli_evaluate_reads_expected_methods_file(tmp_path):
    db_path = tmp_path / "researchfs-memory-evaluate-file.sqlite"
    paper_id = _write_eval_fixture_db(db_path)
    expected_path = tmp_path / "expected-methods.json"
    expected_path.write_text(json.dumps(["Memory CLI Eval Method", "Evidence Coverage Method"]), encoding="utf-8")

    code, stdout, stderr = _run_memory_cli(
        [
            "evaluate",
            "--paper-id",
            str(paper_id),
            "--expected-methods-file",
            str(expected_path),
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 0
    assert stderr == ""
    body = json.loads(stdout)
    assert body["ok"] is True
    assert body["evaluation"]["matched_count"] == 2
    assert body["evaluation"]["passed"] is True


def test_memory_cli_evaluate_rejects_invalid_json_and_empty_methods(tmp_path):
    db_path = tmp_path / "researchfs-memory-evaluate-invalid.sqlite"
    paper_id = _write_eval_fixture_db(db_path)

    code, stdout, stderr = _run_memory_cli(
        [
            "evaluate",
            "--paper-id",
            str(paper_id),
            "--expected-methods",
            "[not-json]",
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 1
    assert stdout == ""
    error_body = json.loads(stderr)
    assert error_body["ok"] is False
    assert error_body["error"]["code"] == "invalid_json"

    code, stdout, stderr = _run_memory_cli(
        [
            "evaluate",
            "--paper-id",
            str(paper_id),
            "--expected-methods",
            "[]",
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 1
    assert stdout == ""
    error_body = json.loads(stderr)
    assert error_body["ok"] is False
    assert error_body["error"]["code"] == "invalid_evaluation"
    assert "expected_methods" in error_body["error"]["message"]


def test_memory_cli_evaluate_reports_missing_paper(tmp_path):
    db_path = tmp_path / "researchfs-memory-evaluate-missing.sqlite"
    _write_eval_fixture_db(db_path)

    code, stdout, stderr = _run_memory_cli(
        [
            "evaluate",
            "--paper-id",
            "9999",
            "--expected-methods",
            json.dumps(["Memory CLI Eval Method"]),
            "--db-path",
            str(db_path),
        ]
    )

    assert code == 1
    assert stdout == ""
    error_body = json.loads(stderr)
    assert error_body["ok"] is False
    assert error_body["error"]["code"] == "paper_not_found"


def test_research_brief_direct_get_research_brief_bounds_and_limits_papers():
    conn, _api, paper_ids, method_ids, chunk_ids, lesson_ids = _research_brief_fixture()

    full_body = rfs.get_research_brief(
        "researchbriefsharedtoken",
        conn=conn,
        max_chars=4000,
        max_papers=2,
    )

    assert full_body["ok"] is True
    assert full_body["query"] == "researchbriefsharedtoken"
    assert full_body["max_chars"] == 4000
    assert full_body["max_papers"] == 2
    assert full_body["source_types"] is None
    assert full_body["paper_ids"] == paper_ids
    assert full_body["method_ids"] == method_ids
    assert full_body["chunk_ids"] == chunk_ids
    assert full_body["lesson_ids"] == lesson_ids
    assert [item["paper_id"] for item in full_body["paper_memory_provenance"]] == paper_ids
    first_memory = full_body["paper_memory_provenance"][0]
    assert first_memory["extraction_status"]["extractor_model"] == (
        "alpha-research-brief-method-extractor"
    )
    assert first_memory["extraction_status"]["extractor_prompt_version"] == (
        "research-brief-prompt-v3"
    )
    assert first_memory["provenance_sources"] == [
        {
            "method_id": method_ids[0],
            "chunk_id": chunk_ids[0],
            "section": "extracted",
            "evidence_text": "alpha researchbriefsharedtoken compact evidence snippet",
            "evidence_start": 0,
            "evidence_end": len("alpha researchbriefsharedtoken compact evidence snippet"),
            "evidence_strategy": "exact",
        }
    ]
    assert len(full_body["context"]) <= 4000
    assert "Alpha Research Brief Method" in full_body["context"]
    assert "Beta Research Brief Method" in full_body["context"]
    assert "ALPHA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL" not in full_body["context"]
    assert "BETA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL" not in full_body["context"]

    one_paper_body = rfs.get_research_brief(
        "researchbriefsharedtoken",
        conn=conn,
        max_chars=4000,
        max_papers=1,
    )

    assert one_paper_body["max_papers"] == 1
    assert len(one_paper_body["paper_ids"]) == 1
    assert one_paper_body["paper_ids"][0] == paper_ids[0]
    assert one_paper_body["method_ids"] == method_ids
    assert one_paper_body["chunk_ids"] == [chunk_ids[0]]
    assert one_paper_body["lesson_ids"] == [lesson_ids[0]]
    assert "Paper Memory: Alpha Research Brief Paper" in one_paper_body["context"]
    assert "Paper Memory: Beta Research Brief Paper" not in one_paper_body["context"]

    bounded_body = rfs.get_research_brief(
        "researchbriefsharedtoken",
        conn=conn,
        max_chars=120,
        max_papers=2,
    )

    assert bounded_body["max_chars"] == 120
    assert len(bounded_body["context"]) <= 120

    try:
        rfs.get_research_brief("   ", conn=conn)
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("empty research brief query should be rejected")

    try:
        rfs.get_research_brief("researchbriefsharedtoken", conn=conn, max_chars=-1)
    except ValueError as exc:
        assert "max_chars" in str(exc)
    else:
        raise AssertionError("negative research brief max_chars should be rejected")


def test_research_brief_includes_standalone_lab_lessons_without_papers_or_methods():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)

    lesson_id = rfs.log_lesson(
        "Standalone protocol memory",
        "standalonelessonuniquetoken alpha should be reusable without linking a method or experiment.",
        outcome="keep",
        conn=conn,
    )
    rejected_lesson_id = rfs.log_lesson(
        "Substring-only protocol memory",
        "prealpha standalonelessonuniquetoken should not satisfy substring-only matching.",
        outcome="keep",
        conn=conn,
    )

    body = rfs.get_research_brief(
        "alpha standalonelessonuniquetoken",
        conn=conn,
        max_chars=1200,
    )

    assert body["ok"] is True
    assert body["paper_ids"] == []
    assert body["method_ids"] == []
    assert body["chunk_ids"] == []
    assert body["lesson_ids"] == [lesson_id]
    assert rejected_lesson_id not in body["lesson_ids"]
    assert "Standalone protocol memory" in body["context"]
    assert "standalonelessonuniquetoken alpha" in body["context"]
    assert "Substring-only protocol memory" not in body["context"]
    assert "No methods found matching query" in body["context"]
    assert "## Paper memory card" not in body["context"]


def test_research_brief_source_type_filter_selects_allowed_papers_before_limits():
    conn, _api, paper_ids, method_ids, chunk_ids, lesson_ids = _research_brief_fixture()

    # Same compact method can be supported by multiple source classes; filtering
    # must also restrict the rendered **Sources:** list, not just method IDs.
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Disallowed Paper Provenance Source",
            "Research Brief Lab",
            "A paper-only source linked to the lab-note method.",
            "https://example.com/disallowed-paper-provenance-source",
            "2026-07-02",
        ),
    )
    disallowed_paper_id = cur.lastrowid
    cur.execute(
        "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
        (
            disallowed_paper_id,
            "Paper-only provenance",
            "paper-only provenance should stay out of lab-note source lists",
        ),
    )
    disallowed_chunk_id = cur.lastrowid
    cur.execute(
        """
        INSERT INTO paper_extraction_status
            (paper_id, status, source_type, chunk_count, method_count, source_char_count, compressed_char_count)
        VALUES (?, 'extracted', 'paper', 1, 1, 62, 62)
        """,
        (disallowed_paper_id,),
    )
    cur.execute(
        "INSERT INTO method_sources (method_id, chunk_id) VALUES (?, ?)",
        (method_ids[1], disallowed_chunk_id),
    )
    conn.commit()

    lab_note_body = rfs.get_research_brief(
        "researchbriefsharedtoken",
        conn=conn,
        max_chars=4000,
        max_papers=1,
        source_types=[" lab_note ", "lab_note"],
    )

    assert lab_note_body["ok"] is True
    assert lab_note_body["source_types"] == ["lab_note"]
    assert lab_note_body["paper_ids"] == [paper_ids[1]]
    assert lab_note_body["method_ids"] == [method_ids[1]]
    assert lab_note_body["chunk_ids"] == [chunk_ids[1]]
    assert lab_note_body["lesson_ids"] == [lesson_ids[1]]
    assert "Alpha Research Brief Method" not in lab_note_body["context"]
    assert "Beta Research Brief Method" in lab_note_body["context"]
    assert "Paper Memory: Alpha Research Brief Paper" not in lab_note_body["context"]
    assert "Paper Memory: Beta Research Brief Paper" in lab_note_body["context"]
    assert "Disallowed Paper Provenance Source" not in lab_note_body["context"]
    assert "https://example.com/disallowed-paper-provenance-source" not in lab_note_body["context"]
    assert "ALPHA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL" not in _json_text(lab_note_body)
    assert "BETA_RESEARCH_BRIEF_RAW_ONLY_SENTINEL" not in _json_text(lab_note_body)

    lab_note_body_two = rfs.get_research_brief(
        "researchbriefsharedtoken",
        conn=conn,
        max_chars=4000,
        max_papers=2,
        source_types=["lab_note"],
    )
    assert lab_note_body_two["paper_ids"] == [paper_ids[1]]
    assert lab_note_body_two["chunk_ids"] == [chunk_ids[1]]
    assert disallowed_paper_id not in lab_note_body_two["paper_ids"]
    assert disallowed_chunk_id not in lab_note_body_two["chunk_ids"]
    assert "Disallowed Paper Provenance Source" not in lab_note_body_two["context"]

    paper_body = rfs.get_research_brief(
        "researchbriefsharedtoken",
        conn=conn,
        max_chars=4000,
        max_papers=1,
        source_type="paper",
    )
    assert paper_body["source_types"] == ["paper"]
    assert paper_body["paper_ids"] == [paper_ids[0]]
    assert paper_body["chunk_ids"] == [chunk_ids[0]]
    assert paper_body["method_ids"] == [method_ids[0], method_ids[1]]


def test_research_brief_rejects_invalid_source_types():
    conn, api, _paper_ids, _method_ids, _chunk_ids, _lesson_ids = _research_brief_fixture()

    invalid_values = ["   ", [], ["paper", ""], ["paper", 7], {"source": "paper"}]
    for invalid_source_types in invalid_values:
        try:
            rfs.get_research_brief(
                "researchbriefsharedtoken",
                conn=conn,
                source_types=invalid_source_types,
            )
        except ValueError as exc:
            assert "source_types" in str(exc)
        else:
            raise AssertionError(f"invalid source_types accepted: {invalid_source_types!r}")

    response = api.dispatch(
        "POST",
        "/brief",
        {"query": "researchbriefsharedtoken", "source_types": ["paper", 3]},
    )
    assert response.status == 400
    assert response.body["error"]["code"] == "invalid_field"
    assert "source_types" in response.body["error"]["message"]

def test_research_brief_provenance_omits_raw_chunk_content_when_evidence_text_is_missing():
    conn, _api, paper_ids, method_ids, chunk_ids, _lesson_ids = _research_brief_fixture()
    raw_only_sentinel = "BRIEF_NULL_EVIDENCE_RAW_ONLY_SENTINEL"
    conn.execute(
        "UPDATE chunks SET content = content || ? WHERE chunk_id = ?",
        (f" {raw_only_sentinel}", chunk_ids[0]),
    )
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = NULL,
            evidence_start = ?,
            evidence_end = ?,
            evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        (0, 52, "brief_span_without_excerpt", method_ids[0], chunk_ids[0]),
    )
    conn.commit()

    body = rfs.get_research_brief(
        "alpha researchbriefsharedtoken",
        conn=conn,
        max_chars=4000,
        max_papers=1,
    )

    assert body["ok"] is True
    assert body["paper_ids"] == [paper_ids[0]]
    assert raw_only_sentinel not in body["context"]
    encoded = _json_text(body)
    assert raw_only_sentinel not in encoded
    assert len(body["paper_memory_provenance"]) == 1
    memory = body["paper_memory_provenance"][0]
    assert memory["paper_id"] == paper_ids[0]
    assert memory["provenance_sources"] == [
        {
            "method_id": method_ids[0],
            "chunk_id": chunk_ids[0],
            "section": "extracted",
            "evidence_text": None,
            "evidence_start": 0,
            "evidence_end": 52,
            "evidence_strategy": "brief_span_without_excerpt",
        }
    ]
    assert memory["extraction_status"]["extractor_model"] == (
        "alpha-research-brief-method-extractor"
    )
    assert memory["extraction_status"]["extractor_prompt_version"] == (
        "research-brief-prompt-v3"
    )
    assert "evidence_text missing; no compressed excerpt stored" in body["context"]
    assert "strategy=brief_span_without_excerpt" in body["context"]


def test_research_brief_falls_back_to_chunk_fts_when_no_methods_match():
    conn, api, _paper_ids, _method_ids, _chunk_ids, _lesson_ids = _research_brief_fixture()
    raw_only_sentinel = "CHUNK_ONLY_RESEARCH_BRIEF_RAW_SENTINEL"
    chunk_only_token = "chunkonlybriefuniquetoken"

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Chunk Only Research Brief Paper",
            "authors": "Research Brief Lab",
            "abstract": "A compressed memory card for chunk-only fallback retrieval.",
            "url": "https://example.com/chunk-only-research-brief",
            "publish_date": "2026-07-01",
            "content": (
                f"The {chunk_only_token} appears only in stored paper chunk text. "
                f"{raw_only_sentinel} must not leak into research brief context."
            ),
            "methods": [],
        },
    )
    assert ingest_response.status == 200
    paper_id = ingest_response.body["paper_id"]
    chunk_id = ingest_response.body["chunk_ids"][0]
    assert ingest_response.body["method_ids"] == []

    body = rfs.get_research_brief(
        chunk_only_token,
        conn=conn,
        max_chars=1200,
        max_papers=1,
    )

    assert body["ok"] is True
    assert body["query"] == chunk_only_token
    assert body["paper_ids"] == [paper_id]
    assert body["chunk_ids"] == [chunk_id]
    assert body["method_ids"] == []
    assert body["lesson_ids"] == []
    assert body["source_types"] is None
    assert len(body["context"]) <= 1200
    assert "Paper Memory: Chunk Only Research Brief Paper" in body["context"]
    assert "- No linked methods." in body["context"]
    assert raw_only_sentinel not in body["context"]


def test_research_brief_chunk_fts_source_filter_skips_disallowed_before_max_papers():
    conn, api, _paper_ids, _method_ids, _chunk_ids, _lesson_ids = _research_brief_fixture()
    chunk_only_token = "sourcefilterchunkonlytoken"

    disallowed_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Disallowed Repo README Chunk Fallback Paper",
            "authors": "Research Brief Lab",
            "abstract": "A repo README artifact that must not consume filtered paper slots.",
            "url": "https://example.com/disallowed-repo-readme",
            "publish_date": "2026-07-01",
            "source_type": "repo_readme",
            "content": f"The {chunk_only_token} appears first but should be filtered out.",
            "methods": [],
        },
    )
    assert disallowed_response.status == 200
    disallowed_paper_id = disallowed_response.body["paper_id"]

    allowed_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Allowed Lab Note Chunk Fallback Paper",
            "authors": "Research Brief Lab",
            "abstract": "A lab note artifact that should fill the filtered paper slot.",
            "url": "https://example.com/allowed-lab-note",
            "publish_date": "2026-07-01",
            "source_type": "lab_note",
            "content": f"The {chunk_only_token} appears second and should be selected.",
            "methods": [],
        },
    )
    assert allowed_response.status == 200
    allowed_paper_id = allowed_response.body["paper_id"]
    allowed_chunk_id = allowed_response.body["chunk_ids"][0]

    body = rfs.get_research_brief(
        chunk_only_token,
        conn=conn,
        max_chars=1200,
        max_papers=1,
        source_types=["lab_note"],
    )

    assert body["ok"] is True
    assert body["source_types"] == ["lab_note"]
    assert body["paper_ids"] == [allowed_paper_id]
    assert disallowed_paper_id not in body["paper_ids"]
    assert body["chunk_ids"] == [allowed_chunk_id]
    assert "Paper Memory: Allowed Lab Note Chunk Fallback Paper" in body["context"]
    assert "Paper Memory: Disallowed Repo README Chunk Fallback Paper" not in body["context"]


def test_lab_api_retrieve_memory_mode_returns_chunk_only_paper_memory_and_logs():
    conn, api, _paper_ids, _method_ids, _chunk_ids, _lesson_ids = _research_brief_fixture()
    raw_only_sentinel = "RETRIEVE_MEMORY_CHUNK_ONLY_RAW_SENTINEL"
    chunk_only_token = "retrievememorychunkonlytoken"

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Retrieve Memory Chunk Only Paper",
            "authors": "Research Brief Lab",
            "abstract": "A compressed memory card for /retrieve memory mode chunk-only fallback.",
            "url": "https://example.com/retrieve-memory-chunk-only",
            "publish_date": "2026-07-01",
            "content": (
                f"The {chunk_only_token} appears only in stored paper chunk text. "
                f"{raw_only_sentinel} must not leak into /retrieve memory context."
            ),
            "methods": [],
        },
    )
    assert ingest_response.status == 200
    paper_id = ingest_response.body["paper_id"]
    chunk_id = ingest_response.body["chunk_ids"][0]
    assert ingest_response.body["method_ids"] == []

    response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": chunk_only_token,
            "mode": "memory",
            "max_chars": 1200,
            "max_papers": 1,
            "log": True,
            "event_type": "retrieve_memory_test",
        },
    )

    assert response.status == 200
    body = response.body
    assert body["ok"] is True
    assert body["query"] == chunk_only_token
    assert body["mode"] == "memory"
    assert body["paper_ids"] == [paper_id]
    assert body["chunk_ids"] == [chunk_id]
    assert body["method_ids"] == []
    assert body["lesson_ids"] == []
    assert body["max_papers"] == 1
    assert body["source_types"] is None
    assert len(body["context"]) <= 1200
    assert "Paper Memory: Retrieve Memory Chunk Only Paper" in body["context"]
    assert raw_only_sentinel not in body["context"]
    assert body["event_id"] is not None

    event = conn.execute(
        "SELECT event_type, query, retrieved_content FROM retrieval_events WHERE event_id = ?",
        (body["event_id"],),
    ).fetchone()
    assert event["event_type"] == "retrieve_memory_test"
    assert event["query"] == chunk_only_token
    assert event["retrieved_content"] == body["context"]

    retrieved_rows = conn.execute(
        "SELECT method_id FROM retrieved_methods WHERE event_id = ?",
        (body["event_id"],),
    ).fetchall()
    assert retrieved_rows == []


def test_lab_api_retrieve_memory_mode_rejects_non_positive_max_papers_and_method_mode_still_works():
    conn, api, _paper_ids, method_ids, _chunk_ids, _lesson_ids = _research_brief_fixture()

    invalid_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "researchbriefsharedtoken",
            "mode": "memory",
            "max_papers": 0,
        },
    )
    assert invalid_response.status == 400
    assert invalid_response.body["error"]["code"] == "invalid_field"
    assert "max_papers" in invalid_response.body["error"]["message"]

    method_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "alpha researchbriefsharedtoken",
            "max_chars": 1200,
            "include_scores": True,
        },
    )
    assert method_response.status == 200
    assert method_response.body["mode"] == "method"
    assert method_response.body["method_ids"] == [method_ids[0]]
    assert "scores" in method_response.body
    assert "paper_ids" not in method_response.body
    assert "chunk_ids" not in method_response.body
    assert "lesson_ids" not in method_response.body
    assert "max_papers" not in method_response.body



def test_research_brief_api_route_scores_and_logged_method_ids():
    conn, api, _paper_ids, method_ids, _chunk_ids, _lesson_ids = _research_brief_fixture()

    response = api.dispatch(
        "POST",
        "/brief",
        {
            "query": "researchbriefsharedtoken",
            "max_chars": 4000,
            "max_papers": 2,
            "include_scores": True,
            "log": True,
            "event_type": "research_brief_test",
        },
    )

    assert response.status == 200
    body = response.body
    assert body["ok"] is True
    assert body["query"] == "researchbriefsharedtoken"
    assert body["max_chars"] == 4000
    assert body["max_papers"] == 2
    assert body["method_ids"] == method_ids
    assert body["source_types"] is None
    assert "scores" in body
    assert body["scores"]
    assert [score["method_id"] for score in body["scores"]] == method_ids
    for score in body["scores"]:
        assert {"method_id", "name", "final_score", "base_score", "stats"} <= set(score)
        assert isinstance(score["final_score"], float)
        assert isinstance(score["stats"], dict)
    assert body["event_id"] is not None

    event = conn.execute(
        "SELECT event_type, query, retrieved_content FROM retrieval_events WHERE event_id = ?",
        (body["event_id"],),
    ).fetchone()
    assert event["event_type"] == "research_brief_test"
    assert event["query"] == "researchbriefsharedtoken"
    assert event["retrieved_content"] == body["context"]

    retrieved_rows = conn.execute(
        "SELECT method_id FROM retrieved_methods WHERE event_id = ? ORDER BY method_id",
        (body["event_id"],),
    ).fetchall()
    assert [row["method_id"] for row in retrieved_rows] == method_ids


def test_research_brief_api_routes_filter_source_types_and_preserve_methods():
    conn, api, paper_ids, method_ids, chunk_ids, lesson_ids = _research_brief_fixture()

    brief_response = api.dispatch(
        "POST",
        "/brief",
        {
            "query": "researchbriefsharedtoken",
            "max_chars": 4000,
            "max_papers": 1,
            "source_type": "lab_note",
        },
    )
    assert brief_response.status == 200
    brief_body = brief_response.body
    assert brief_body["source_types"] == ["lab_note"]
    assert brief_body["paper_ids"] == [paper_ids[1]]
    assert brief_body["method_ids"] == [method_ids[1]]
    assert brief_body["chunk_ids"] == [chunk_ids[1]]
    assert brief_body["lesson_ids"] == [lesson_ids[1]]
    assert "Alpha Research Brief Method" not in brief_body["context"]
    assert "Beta Research Brief Method" in brief_body["context"]
    assert "Paper Memory: Alpha Research Brief Paper" not in brief_body["context"]
    assert "Paper Memory: Beta Research Brief Paper" in brief_body["context"]

    retrieve_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "researchbriefsharedtoken",
            "mode": "memory",
            "max_chars": 4000,
            "max_papers": 1,
            "source_types": ["paper"],
        },
    )
    assert retrieve_response.status == 200
    retrieve_body = retrieve_response.body
    assert retrieve_body["source_types"] == ["paper"]
    assert retrieve_body["paper_ids"] == [paper_ids[0]]
    assert retrieve_body["method_ids"] == [method_ids[0]]
    assert retrieve_body["chunk_ids"] == [chunk_ids[0]]
    assert retrieve_body["lesson_ids"] == [lesson_ids[0]]
    assert "Alpha Research Brief Method" in retrieve_body["context"]
    assert "Beta Research Brief Method" not in retrieve_body["context"]
    assert "Paper Memory: Alpha Research Brief Paper" in retrieve_body["context"]
    assert "Paper Memory: Beta Research Brief Paper" not in retrieve_body["context"]


def test_research_brief_lab_client_helper_matches_route_contract():
    from researchfs import ResearchFSLabClient

    _conn, api, paper_ids, method_ids, chunk_ids, lesson_ids = _research_brief_fixture()
    client = ResearchFSLabClient(api=api)

    client_body = client.research_brief(
        "researchbriefsharedtoken",
        max_chars=4000,
        max_papers=2,
        include_scores=True,
    )
    route_body = api.dispatch(
        "POST",
        "/brief",
        {
            "query": "researchbriefsharedtoken",
            "max_chars": 4000,
            "max_papers": 2,
            "include_scores": True,
        },
    ).body

    assert client_body == route_body
    assert client_body["paper_ids"] == paper_ids
    assert client_body["method_ids"] == method_ids
    assert client_body["chunk_ids"] == chunk_ids
    assert client_body["lesson_ids"] == lesson_ids

    filtered_client_body = client.research_brief(
        "researchbriefsharedtoken",
        max_chars=4000,
        max_papers=1,
        source_types=("lab_note",),
    )
    assert filtered_client_body["source_types"] == ["lab_note"]
    assert filtered_client_body["paper_ids"] == [paper_ids[1]]
    assert filtered_client_body["method_ids"] == [method_ids[1]]
    assert filtered_client_body["chunk_ids"] == [chunk_ids[1]]
    assert filtered_client_body["lesson_ids"] == [lesson_ids[1]]


def test_lab_client_retrieve_memory_mode_forwards_max_papers():
    from researchfs import ResearchFSLabClient

    _conn, api, paper_ids, method_ids, chunk_ids, lesson_ids = _research_brief_fixture()
    client = ResearchFSLabClient(api=api)

    client_body = client.retrieve(
        "researchbriefsharedtoken",
        mode="memory",
        max_chars=4000,
        max_papers=2,
        include_scores=True,
    )

    assert client_body["ok"] is True
    assert client_body["mode"] == "memory"
    assert client_body["max_papers"] == 2
    assert client_body["paper_ids"] == paper_ids
    assert client_body["method_ids"] == method_ids
    assert client_body["chunk_ids"] == chunk_ids
    assert client_body["lesson_ids"] == lesson_ids
    assert "scores" in client_body

    filtered_client_body = client.retrieve(
        "researchbriefsharedtoken",
        mode="memory",
        max_chars=4000,
        max_papers=1,
        source_type="paper",
    )
    assert filtered_client_body["source_types"] == ["paper"]
    assert filtered_client_body["paper_ids"] == [paper_ids[0]]
    assert filtered_client_body["method_ids"] == [method_ids[0]]
    assert filtered_client_body["chunk_ids"] == [chunk_ids[0]]
    assert filtered_client_body["lesson_ids"] == [lesson_ids[0]]

def test_paper_memory_api_returns_compact_provenance_card():
    conn, api, paper_id, method_id, chunk_id, lesson_id, experiment_id = _paper_memory_fixture()

    response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=4000")

    assert response.status == 200
    body = response.body
    assert body["ok"] is True
    assert body["paper_id"] == paper_id
    assert body["max_chars"] == 4000
    assert body["method_ids"] == [method_id]
    assert body["chunk_ids"] == [chunk_id]
    assert body["lesson_ids"] == [lesson_id]
    assert body["experiment_ids"] == [experiment_id]
    assert body["extraction_status"]["status"] == "extracted"
    assert body["extraction_status"]["confidence_score"] == 0.88
    assert body["extraction_status"]["confidence_reason"] == (
        "stored method provenance covers the compact memory card"
    )
    assert body["extraction_status"]["compression_ratio"] is not None
    assert body["extraction_status"]["extractor_model"] == "paper-memory-test-extractor"
    assert body["extraction_status"]["extractor_prompt_version"] == "paper-memory-prompt-v2"
    assert body["provenance_sources"] == [
        {
            "method_id": method_id,
            "chunk_id": chunk_id,
            "section": "extracted",
            "evidence_text": (
                "exact evidence snippets, compact method metadata, lessons, experiments, and feedback"
            ),
            "evidence_start": 33,
            "evidence_end": 117,
            "evidence_strategy": "exact",
        }
    ]

    context = body["context"]
    assert len(context) <= 4000
    assert "Memory Card Contract Paper" in context
    assert "Deterministic Memory Card Assembly" in context
    assert "exact evidence snippets" in context
    assert "Prefer stored evidence" in context
    assert "Use stored provenance snippets" in context
    assert "Memory card experiment kept the deterministic assembly path." in context
    assert "Feedback should be visible in the compact paper card." in context

    # Keep the fixture connection alive until after assertions so route handlers
    # cannot accidentally depend on a closed external connection.
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1

def test_paper_memory_omits_raw_chunk_content_when_evidence_text_is_missing():
    conn, api, paper_id, method_id, chunk_id, _lesson_id, _experiment_id = _paper_memory_fixture()
    raw_only_sentinel = "PAPER_MEMORY_NULL_EVIDENCE_RAW_ONLY_SENTINEL"
    conn.execute(
        "UPDATE chunks SET content = content || ? WHERE chunk_id = ?",
        (f" {raw_only_sentinel}", chunk_id),
    )
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = NULL,
            evidence_start = ?,
            evidence_end = ?,
            evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        (33, 117, "stored_span_without_excerpt", method_id, chunk_id),
    )
    conn.commit()

    response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=4000")

    assert response.status == 200
    body = response.body
    assert raw_only_sentinel not in body["context"]
    encoded = _json_text(body)
    assert raw_only_sentinel not in encoded
    assert body["provenance_sources"] == [
        {
            "method_id": method_id,
            "chunk_id": chunk_id,
            "section": "extracted",
            "evidence_text": None,
            "evidence_start": 33,
            "evidence_end": 117,
            "evidence_strategy": "stored_span_without_excerpt",
        }
    ]
    assert "evidence_text missing; no compressed excerpt stored" in body["context"]
    assert "chars=33-117" in body["context"]
    assert "strategy=stored_span_without_excerpt" in body["context"]
    assert body["extraction_status"]["extractor_model"] == "paper-memory-test-extractor"
    assert body["extraction_status"]["extractor_prompt_version"] == "paper-memory-prompt-v2"


def test_paper_memory_api_max_chars_and_invalid_query_validation():
    _conn, api, paper_id, method_id, _chunk_id, _lesson_id, _experiment_id = _paper_memory_fixture()

    full_response = api.dispatch("GET", f"/paper/{paper_id}/memory")
    assert full_response.status == 200
    assert full_response.body["max_chars"] is None
    assert method_id in full_response.body["method_ids"]
    assert len(full_response.body["context"]) > 80

    truncated_response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=80")
    assert truncated_response.status == 200
    assert truncated_response.body["max_chars"] == 80
    assert len(truncated_response.body["context"]) <= 80

    empty_response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=0")
    assert empty_response.status == 200
    assert empty_response.body["max_chars"] == 0
    assert empty_response.body["context"] == ""

    invalid_response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=abc")
    assert invalid_response.status == 400
    assert invalid_response.body["error"]["code"] == "invalid_field"
    assert "max_chars" in invalid_response.body["error"]["message"]

    negative_response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=-1")
    assert negative_response.status == 400
    assert negative_response.body["error"]["code"] == "invalid_field"
    assert "max_chars" in negative_response.body["error"]["message"]


def test_lab_client_get_paper_memory_and_paper_route_ordering():
    from researchfs import ResearchFSLabClient

    conn, api, paper_id, method_id, chunk_id, lesson_id, experiment_id = _paper_memory_fixture()
    client = ResearchFSLabClient(api=api)

    detail_body = client.get_paper(paper_id)
    assert detail_body["ok"] is True
    assert detail_body["paper"]["paper_id"] == paper_id
    assert detail_body["paper"]["title"] == "Memory Card Contract Paper"
    assert [method["method_id"] for method in detail_body["methods"]] == [method_id]
    assert [chunk["chunk_id"] for chunk in detail_body["chunks"]] == [chunk_id]
    assert "context" not in detail_body

    memory_body = client.get_paper_memory(paper_id, max_chars=120)
    assert memory_body["ok"] is True
    assert memory_body["paper_id"] == paper_id
    assert memory_body["max_chars"] == 120
    assert memory_body["method_ids"] == [method_id]
    assert memory_body["chunk_ids"] == [chunk_id]
    assert memory_body["lesson_ids"] == [lesson_id]
    assert memory_body["experiment_ids"] == [experiment_id]
    assert len(memory_body["context"]) <= 120

    route_response = api.dispatch("GET", f"/paper/{paper_id}/memory?max_chars=120")
    assert route_response.status == 200
    assert route_response.body == memory_body
    assert conn.execute("SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()


def _json_text(value):
    return json.dumps(value, sort_keys=True)


def _bundle_test_source():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)
    raw_only_text = (
        "BUNDLE_RAW_CHUNK_ONLY_SENTINEL full paper paragraph must stay out of "
        "portable compressed bundles and imports."
    )
    evidence_text = (
        "bundleevidencetoken rotary cache gating evidence snippet for portable memory"
    )
    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Portable Bundle Memory Paper",
            "authors": "Bundle Lab",
            "abstract": "A paper about compressed portable ResearchFS memory.",
            "url": "https://example.com/portable-bundle-memory",
            "publish_date": "2026-07-01",
            "content": f"{evidence_text}\n\n{raw_only_text}",
            "extractor_model": "bundle-test-extractor",
            "extractor_prompt_version": "bundle-test-v1",
            "methods": [
                {
                    "name": "Portable Bundle Gating",
                    "description": "Use bundleevidencetoken gates from imported compressed memory.",
                    "implementation_hint": "Retrieve compact evidence snippets instead of raw papers.",
                    "risks": "Including full chunk content would make bundles non-portable.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert ingest_response.status == 200
    assert ingest_response.body["extraction_status"]["extractor_model"] == "bundle-test-extractor"
    assert ingest_response.body["extraction_status"]["extractor_prompt_version"] == "bundle-test-v1"
    paper_id = ingest_response.body["paper_id"]
    method_id = ingest_response.body["method_ids"][0]
    chunk_id = ingest_response.body["chunk_ids"][0]
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = ?,
            evidence_start = ?,
            evidence_end = ?,
            evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        (evidence_text, 0, len(evidence_text), "exact", method_id, chunk_id),
    )
    conn.execute(
        """
        UPDATE paper_extraction_status
        SET confidence_score = ?,
            confidence_reason = ?,
            extractor_model = ?,
            extractor_prompt_version = ?
        WHERE paper_id = ?
        """,
        (
            0.91,
            "bundle evidence snippets cover the imported method",
            "bundle-test-extractor",
            "bundle-test-v1",
            paper_id,
        ),
    )
    api.dispatch(
        "POST",
        "/feedback",
        {
            "method_id": method_id,
            "rating": 0.8,
            "label": "useful",
            "note": "Portable feedback summary should survive bundle export.",
            "source": "bundle-test",
        },
    )
    experiment_id = rfs.log_experiment(
        "bundle-lesson-carryover",
        val_bpb=1.234,
        memory_gb=2.5,
        status="keep",
        description="Bundle lesson experiment should not be required on import.",
        conn=conn,
    )
    method_lesson_id = rfs.log_lesson(
        "Prefer compact bundle lesson retrieval",
        "Carry over the reusable bundlelessonretrievaltoken lesson without repasting paper chunks.",
        outcome="adopted",
        method_id=method_id,
        experiment_id=experiment_id,
        conn=conn,
    )
    experiment_lesson_id = rfs.log_lesson(
        "Experiment-only bundle note",
        "This experiment-only lesson should stay local without a remapped method.",
        outcome="local-only",
        experiment_id=experiment_id,
        conn=conn,
    )
    conn.commit()
    return (
        conn,
        api,
        paper_id,
        method_id,
        chunk_id,
        evidence_text,
        raw_only_text,
        method_lesson_id,
        experiment_lesson_id,
    )


def _add_corpus_bundle_test_paper(
    conn,
    api,
    *,
    title,
    url,
    evidence_text,
    raw_only_text,
    method_name,
    lesson_label,
    lesson_content,
):
    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": title,
            "authors": "Corpus Lab",
            "abstract": f"{title} carries compressed local memory.",
            "url": url,
            "publish_date": "2026-07-01",
            "content": f"{evidence_text}\n\n{raw_only_text}",
            "extractor_model": "corpus-test-extractor",
            "extractor_prompt_version": "corpus-test-v1",
            "methods": [
                {
                    "name": method_name,
                    "description": f"Use {evidence_text} from imported compressed corpus memory.",
                    "implementation_hint": "Move compact evidence snippets between local ResearchFS databases.",
                    "risks": "Raw chunk text must not be serialized into the corpus bundle.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )
    assert ingest_response.status == 200
    paper_id = ingest_response.body["paper_id"]
    method_id = ingest_response.body["method_ids"][0]
    chunk_id = ingest_response.body["chunk_ids"][0]
    conn.execute(
        """
        UPDATE method_sources
        SET evidence_text = ?,
            evidence_start = ?,
            evidence_end = ?,
            evidence_strategy = ?
        WHERE method_id = ? AND chunk_id = ?
        """,
        (evidence_text, 0, len(evidence_text), "exact", method_id, chunk_id),
    )
    conn.execute(
        """
        UPDATE paper_extraction_status
        SET confidence_score = ?,
            confidence_reason = ?,
            extractor_model = ?,
            extractor_prompt_version = ?
        WHERE paper_id = ?
        """,
        (
            0.88,
            f"{method_name} evidence survives corpus export",
            "corpus-test-extractor",
            "corpus-test-v1",
            paper_id,
        ),
    )
    lesson_id = rfs.log_lesson(
        lesson_label,
        lesson_content,
        outcome="adopted",
        method_id=method_id,
        conn=conn,
    )
    conn.commit()
    return {
        "paper_id": paper_id,
        "method_id": method_id,
        "chunk_id": chunk_id,
        "lesson_id": lesson_id,
        "title": title,
        "evidence_text": evidence_text,
        "raw_only_text": raw_only_text,
        "method_name": method_name,
        "lesson_label": lesson_label,
        "lesson_content": lesson_content,
    }


def _corpus_bundle_test_source():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)
    papers = [
        _add_corpus_bundle_test_paper(
            conn,
            api,
            title="Corpus Bundle Alpha Memory Paper",
            url="https://example.com/corpus-bundle-alpha",
            evidence_text="corpusalphaevidencetoken compact alpha evidence for local bundle transfer",
            raw_only_text="CORPUS_ALPHA_RAW_SENTINEL alpha full paper prose must stay local only",
            method_name="Corpus Alpha Transfer",
            lesson_label="Prefer corpus alpha memory",
            lesson_content="corpusalphalessontoken should transfer as compressed lesson memory.",
        ),
        _add_corpus_bundle_test_paper(
            conn,
            api,
            title="Corpus Bundle Beta Memory Paper",
            url="https://example.com/corpus-bundle-beta",
            evidence_text="corpusbetaevidencetoken compact beta evidence for local bundle transfer",
            raw_only_text="CORPUS_BETA_RAW_SENTINEL beta full paper prose must stay local only",
            method_name="Corpus Beta Transfer",
            lesson_label="Prefer corpus beta memory",
            lesson_content="corpusbetalessontoken should transfer as compressed lesson memory.",
        ),
    ]
    return conn, api, papers


def _assert_corpus_bundle_is_compressed_only(bundle, raw_texts):
    assert bundle["kind"] == "researchfs.corpus_memory_bundle"
    assert bundle["version"] == 1
    assert "chunks" not in bundle
    encoded = _json_text(bundle)
    for raw_text in raw_texts:
        assert raw_text not in encoded
    for paper_bundle in bundle["papers"]:
        assert paper_bundle["kind"] == "researchfs.paper_memory_bundle"
        assert "content" not in paper_bundle["paper"]
        assert "chunks" not in paper_bundle
        assert all(
            "content" not in source
            for method in paper_bundle["methods"]
            for source in method["sources"]
        )

def test_memory_bundle_helper_export_is_compressed_only_with_provenance():
    from researchfs.bundle import export_paper_memory_bundle

    (
        conn,
        _api,
        paper_id,
        method_id,
        chunk_id,
        evidence_text,
        raw_only_text,
        method_lesson_id,
        experiment_lesson_id,
    ) = _bundle_test_source()

    bundle = export_paper_memory_bundle(paper_id, conn=conn, max_chars=900)
    encoded = _json_text(bundle)

    assert bundle["kind"] == "researchfs.paper_memory_bundle"
    assert bundle["version"] == 1
    assert bundle["paper"]["title"] == "Portable Bundle Memory Paper"
    assert bundle["memory"]["max_chars"] == 900
    assert len(bundle["memory"]["context"]) <= 900
    assert "Portable Bundle Gating" in bundle["memory"]["context"]
    assert evidence_text in encoded
    assert "bundle evidence snippets cover the imported method" in encoded
    assert "Portable feedback summary should survive bundle export." in encoded
    assert str(method_id) in encoded
    assert str(chunk_id) in encoded
    assert raw_only_text not in encoded
    assert "BUNDLE_RAW_CHUNK_ONLY_SENTINEL" not in encoded
    assert "chunks" not in bundle
    assert all("content" not in source for method in bundle["methods"] for source in method["sources"])
    assert bundle["extraction_status"]["extractor_model"] == "bundle-test-extractor"
    assert bundle["extraction_status"]["extractor_prompt_version"] == "bundle-test-v1"
    source = bundle["methods"][0]["sources"][0]
    assert source["source_method_id"] == method_id
    assert source["source_chunk_id"] == chunk_id
    assert source["evidence_text"] == evidence_text
    assert source["evidence_start"] == 0
    assert source["evidence_end"] == len(evidence_text)
    assert source["evidence_strategy"] == "exact"

    assert bundle["source_ids"]["lesson_ids"] == [method_lesson_id]
    lessons = bundle["lessons"]
    assert len(lessons) == 1
    lesson = lessons[0]
    assert lesson["source_lesson_id"] == method_lesson_id
    assert lesson["source_method_id"] == method_id
    assert lesson["label"] == "Prefer compact bundle lesson retrieval"
    assert lesson["content"] == (
        "Carry over the reusable bundlelessonretrievaltoken lesson without repasting paper chunks."
    )
    assert lesson["outcome"] == "adopted"
    assert lesson["created_at"]
    assert "source_experiment_id" in lesson
    assert "Experiment-only bundle note" not in encoded


def test_memory_bundle_helper_import_reconstructs_queryable_compressed_memory():
    from researchfs.bundle import export_paper_memory_bundle, import_paper_memory_bundle

    (
        source_conn,
        _api,
        paper_id,
        _method_id,
        _chunk_id,
        evidence_text,
        raw_only_text,
        method_lesson_id,
        experiment_lesson_id,
    ) = _bundle_test_source()
    bundle = export_paper_memory_bundle(paper_id, conn=source_conn, max_chars=1200)

    target_conn = sqlite3.connect(":memory:")
    target_conn.row_factory = sqlite3.Row
    initialize_db(target_conn)
    result = import_paper_memory_bundle(bundle, conn=target_conn)

    imported_paper_id = result["paper_id"]
    imported_method_ids = result["method_ids"]
    assert imported_method_ids
    assert result["lesson_ids"]
    assert result["source_id_map"]["lesson_ids"][str(method_lesson_id)] == result["lesson_ids"][0]
    assert str(experiment_lesson_id) not in result["source_id_map"]["lesson_ids"]
    assert target_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    stored_chunk = target_conn.execute(
        "SELECT content FROM chunks WHERE paper_id = ?",
        (imported_paper_id,),
    ).fetchone()
    assert stored_chunk is not None
    assert evidence_text in stored_chunk["content"]
    assert raw_only_text not in stored_chunk["content"]
    stored_source = target_conn.execute(
        """
        SELECT evidence_text, evidence_start, evidence_end, evidence_strategy
        FROM method_sources
        WHERE method_id = ?
        """,
        (imported_method_ids[0],),
    ).fetchone()
    assert stored_source["evidence_text"] == evidence_text
    assert stored_source["evidence_start"] == 0
    assert stored_source["evidence_end"] == len(evidence_text)
    assert stored_source["evidence_strategy"] == "exact"
    imported_status = target_conn.execute(
        """
        SELECT extractor_model, extractor_prompt_version
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (imported_paper_id,),
    ).fetchone()
    assert imported_status["extractor_model"] == "bundle-test-extractor"
    assert imported_status["extractor_prompt_version"] == "bundle-test-v1"

    retrieved_context, retrieved_ids = rfs.search_methods_and_ids(
        "bundleevidencetoken rotary", conn=target_conn, max_chars=1200
    )
    assert imported_method_ids[0] in retrieved_ids
    assert "Portable Bundle Gating" in retrieved_context
    assert "Portable Bundle Memory Paper" in retrieved_context
    lesson_context, lesson_retrieved_ids = rfs.search_methods_and_ids(
        "bundlelessonretrievaltoken", conn=target_conn, max_chars=1200
    )
    assert imported_method_ids[0] in lesson_retrieved_ids
    assert "Portable Bundle Gating" in lesson_context
    assert "Prefer compact bundle lesson retrieval" in lesson_context
    assert "bundlelessonretrievaltoken" in lesson_context

    imported_memory = rfs.get_paper_memory(imported_paper_id, conn=target_conn, max_chars=1200)
    assert "Portable Bundle Gating" in imported_memory["context"]
    assert "bundleevidencetoken" in imported_memory["context"]
    assert "BUNDLE_RAW_CHUNK_ONLY_SENTINEL" not in _json_text(imported_memory)
    lesson_rows = target_conn.execute(
        """
        SELECT method_id, experiment_id, label, content, outcome
        FROM lessons
        ORDER BY lesson_id
        """
    ).fetchall()
    assert len(lesson_rows) == 1
    lesson = lesson_rows[0]
    assert lesson["method_id"] == imported_method_ids[0]
    assert lesson["experiment_id"] is None
    assert lesson["label"] == "Prefer compact bundle lesson retrieval"
    assert lesson["content"] == (
        "Carry over the reusable bundlelessonretrievaltoken lesson without repasting paper chunks."
    )
    assert lesson["outcome"] == "adopted"
    assert "Prefer compact bundle lesson retrieval" in imported_memory["context"]
    assert "Carry over the reusable bundlelessonretrievaltoken lesson" in imported_memory["context"]


def test_memory_bundle_import_duplicate_is_noop_for_source_paper():
    from researchfs.bundle import export_paper_memory_bundle, import_paper_memory_bundle

    (
        conn,
        _api,
        paper_id,
        method_id,
        chunk_id,
        _evidence_text,
        _raw_only_text,
        method_lesson_id,
        _experiment_lesson_id,
    ) = _bundle_test_source()
    bundle = export_paper_memory_bundle(paper_id, conn=conn, max_chars=1200)
    before_counts = {
        "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "lessons": conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
        "method_sources": conn.execute("SELECT COUNT(*) FROM method_sources").fetchone()[0],
    }
    before_status = dict(
        conn.execute("SELECT * FROM paper_extraction_status WHERE paper_id = ?", (paper_id,)).fetchone()
    )

    result = import_paper_memory_bundle(bundle, conn=conn)

    after_counts = {
        "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "lessons": conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
        "method_sources": conn.execute("SELECT COUNT(*) FROM method_sources").fetchone()[0],
    }
    after_status = dict(
        conn.execute("SELECT * FROM paper_extraction_status WHERE paper_id = ?", (paper_id,)).fetchone()
    )
    assert result["duplicate"] is True
    assert result["paper_id"] == paper_id
    assert result["method_ids"] == [method_id]
    assert result["chunk_ids"] == [chunk_id]
    assert result["lesson_ids"] == [method_lesson_id]
    assert result["source_id_map"]["lesson_ids"] == {str(method_lesson_id): method_lesson_id}
    assert after_counts == before_counts
    assert after_status["source_type"] == before_status["source_type"]
    assert after_status["chunk_count"] == before_status["chunk_count"]
    assert after_status["method_count"] == before_status["method_count"]

def test_memory_bundle_duplicate_maps_each_source_chunk_once():
    from researchfs.bundle import export_paper_memory_bundle, import_paper_memory_bundle

    (
        conn,
        _api,
        paper_id,
        method_id,
        first_chunk_id,
        _evidence_text,
        _raw_only_text,
        _method_lesson_id,
        _experiment_lesson_id,
    ) = _bundle_test_source()
    second_chunk_text = "Second bundleevidencetoken proof for another implementation branch."
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chunks (paper_id, content) VALUES (?, ?)",
        (paper_id, second_chunk_text),
    )
    second_chunk_id = cur.lastrowid
    conn.execute(
        """
        INSERT INTO method_sources (
            method_id,
            chunk_id,
            evidence_text,
            evidence_start,
            evidence_end,
            evidence_strategy
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (method_id, second_chunk_id, second_chunk_text, 0, len(second_chunk_text), "exact"),
    )
    conn.execute(
        """
        UPDATE paper_extraction_status
        SET chunk_count = 2,
            source_char_count = source_char_count + ?,
            compressed_char_count = compressed_char_count + ?
        WHERE paper_id = ?
        """,
        (len(second_chunk_text), len(second_chunk_text), paper_id),
    )
    conn.commit()
    bundle = export_paper_memory_bundle(paper_id, conn=conn, max_chars=1600)

    result = import_paper_memory_bundle(bundle, conn=conn)

    assert result["duplicate"] is True
    assert result["chunk_ids"] == [first_chunk_id, second_chunk_id]
    chunk_map = result["source_id_map"]["chunk_ids"]
    assert chunk_map[str(first_chunk_id)] == [first_chunk_id]
    assert chunk_map[str(second_chunk_id)] == [second_chunk_id]


def test_memory_bundle_duplicate_import_maps_target_chunks_by_evidence():
    from researchfs.bundle import export_paper_memory_bundle, import_paper_memory_bundle

    (
        source_conn,
        _api,
        paper_id,
        method_id,
        _first_chunk_id,
        _evidence_text,
        _raw_only_text,
        method_lesson_id,
        experiment_lesson_id,
    ) = _bundle_test_source()
    second_chunk_text = "Independent bundleevidencetoken evidence for a transferred lab memory."
    cur = source_conn.cursor()
    cur.execute(
        "INSERT INTO chunks (paper_id, content) VALUES (?, ?)",
        (paper_id, second_chunk_text),
    )
    second_chunk_id = cur.lastrowid
    source_conn.execute(
        """
        INSERT INTO method_sources (
            method_id,
            chunk_id,
            evidence_text,
            evidence_start,
            evidence_end,
            evidence_strategy
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (method_id, second_chunk_id, second_chunk_text, 0, len(second_chunk_text), "exact"),
    )
    source_conn.commit()
    bundle = export_paper_memory_bundle(paper_id, conn=source_conn, max_chars=1600)
    target_conn = sqlite3.connect(":memory:")
    target_conn.row_factory = sqlite3.Row
    initialize_db(target_conn)

    first_import = import_paper_memory_bundle(bundle, conn=target_conn)
    second_import = import_paper_memory_bundle(bundle, conn=target_conn)

    assert first_import["duplicate"] is False
    assert second_import["duplicate"] is True
    assert second_import["paper_id"] == first_import["paper_id"]
    assert second_import["method_ids"] == first_import["method_ids"]
    assert second_import["chunk_ids"] == first_import["chunk_ids"]
    assert second_import["lesson_ids"] == first_import["lesson_ids"]
    assert first_import["source_id_map"]["lesson_ids"][str(method_lesson_id)] == first_import["lesson_ids"][0]
    assert second_import["source_id_map"]["lesson_ids"] == first_import["source_id_map"]["lesson_ids"]
    assert str(experiment_lesson_id) not in second_import["source_id_map"]["lesson_ids"]
    assert target_conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 1
    chunk_map = second_import["source_id_map"]["chunk_ids"]
    assert sorted(chunk_map) == sorted(str(chunk_id) for chunk_id in bundle["source_ids"]["chunk_ids"])
    assert sorted(mapped[0] for mapped in chunk_map.values()) == sorted(first_import["chunk_ids"])

def test_memory_bundle_import_rejects_raw_content_fields():
    from researchfs.bundle import export_paper_memory_bundle, import_paper_memory_bundle

    conn, _api, paper_id, _method_id, _chunk_id, _evidence_text, raw_only_text, *_lesson_ids = _bundle_test_source()
    bundle = export_paper_memory_bundle(paper_id, conn=conn)

    paper_content_bundle = {**bundle, "paper": {**bundle["paper"], "content": raw_only_text}}
    try:
        import_paper_memory_bundle(paper_content_bundle, conn=conn)
    except ValueError as exc:
        assert "paper.content" in str(exc)
    else:
        raise AssertionError("paper.content should be rejected")

    source_content_bundle = {
        **bundle,
        "methods": [
            {
                **bundle["methods"][0],
                "sources": [
                    {**bundle["methods"][0]["sources"][0], "content": raw_only_text}
                ],
            }
        ],
    }
    try:
        import_paper_memory_bundle(source_content_bundle, conn=conn)
    except ValueError as exc:
        assert "raw content" in str(exc)
    else:
        raise AssertionError("source content should be rejected")


def test_memory_bundle_export_skips_missing_evidence_without_raw_chunk_fallback():
    from researchfs.bundle import export_paper_memory_bundle

    conn, _api, paper_id, _method_id, _chunk_id, _evidence_text, raw_only_text, *_lesson_ids = _bundle_test_source()
    conn.execute(
        "UPDATE method_sources SET evidence_text = NULL, evidence_start = NULL, evidence_end = NULL, evidence_strategy = NULL"
    )
    conn.commit()

    bundle = export_paper_memory_bundle(paper_id, conn=conn)
    encoded = _json_text(bundle)

    assert raw_only_text not in encoded
    assert "BUNDLE_RAW_CHUNK_ONLY_SENTINEL" not in encoded
    assert bundle["methods"][0]["sources"] == []
    assert bundle["source_ids"]["chunk_ids"] == []


def test_memory_bundle_import_rejects_invalid_source_provenance():
    from researchfs.bundle import export_paper_memory_bundle, import_paper_memory_bundle

    conn, _api, paper_id, _method_id, _chunk_id, _evidence_text, _raw_only_text, *_lesson_ids = _bundle_test_source()
    bundle = export_paper_memory_bundle(paper_id, conn=conn)
    base_source = bundle["methods"][0]["sources"][0]
    invalid_cases = [
        ({**base_source, "evidence_text": "   "}, "evidence_text"),
        ({**base_source, "evidence_start": "0"}, "evidence_start"),
        ({**base_source, "evidence_end": base_source["evidence_start"]}, "greater than evidence_start"),
        ({**base_source, "evidence_strategy": "   "}, "evidence_strategy"),
    ]

    for invalid_source, expected_message in invalid_cases:
        invalid_bundle = {
            **bundle,
            "methods": [
                {
                    **bundle["methods"][0],
                    "sources": [invalid_source],
                }
            ],
        }
        try:
            import_paper_memory_bundle(invalid_bundle, conn=conn)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"invalid source provenance should be rejected: {expected_message}")


def test_memory_bundle_api_and_sdk_export_import_routes_round_trip():
    from researchfs import ResearchFSLabClient

    (
        _conn,
        source_api,
        paper_id,
        _method_id,
        _chunk_id,
        evidence_text,
        raw_only_text,
        method_lesson_id,
        _experiment_lesson_id,
    ) = _bundle_test_source()
    source_client = ResearchFSLabClient(api=source_api)

    route_response = source_api.dispatch("GET", f"/paper/{paper_id}/memory/export?max_chars=900")
    assert route_response.status == 200
    route_bundle = route_response.body["bundle"]
    assert route_response.body["ok"] is True
    assert route_bundle["memory"]["max_chars"] == 900
    assert evidence_text in _json_text(route_bundle)
    assert raw_only_text not in _json_text(route_bundle)
    assert route_bundle["source_ids"]["lesson_ids"] == [method_lesson_id]
    assert route_bundle["lessons"][0]["label"] == "Prefer compact bundle lesson retrieval"

    client_export = source_client.export_paper_memory_bundle(paper_id, max_chars=900)
    assert client_export["ok"] is True
    assert client_export["bundle"] == route_bundle

    target_conn = sqlite3.connect(":memory:")
    target_conn.row_factory = sqlite3.Row
    initialize_db(target_conn)
    target_api = ResearchFSLabAPI(conn=target_conn)
    target_client = ResearchFSLabClient(api=target_api)

    import_response = target_api.dispatch("POST", "/memory/import", {"bundle": route_bundle})
    assert import_response.status == 200
    assert import_response.body["ok"] is True
    imported_paper_id = import_response.body["paper_id"]
    imported_method_id = import_response.body["method_ids"][0]
    assert import_response.body["lesson_ids"]
    assert import_response.body["source_id_map"]["lesson_ids"][str(method_lesson_id)] == import_response.body["lesson_ids"][0]
    imported_lesson = target_conn.execute(
        "SELECT method_id, experiment_id, label FROM lessons WHERE lesson_id = ?",
        (import_response.body["lesson_ids"][0],),
    ).fetchone()
    assert imported_lesson["method_id"] == imported_method_id
    assert imported_lesson["experiment_id"] is None
    assert imported_lesson["label"] == "Prefer compact bundle lesson retrieval"

    retrieve_response = target_api.dispatch(
        "POST",
        "/retrieve",
        {"query": "bundleevidencetoken", "max_chars": 1200},
    )
    assert retrieve_response.status == 200
    assert imported_method_id in retrieve_response.body["method_ids"]
    assert "Portable Bundle Gating" in retrieve_response.body["context"]
    assert "BUNDLE_RAW_CHUNK_ONLY_SENTINEL" not in retrieve_response.body["context"]
    lesson_retrieve_response = target_api.dispatch(
        "POST",
        "/retrieve",
        {"query": "bundlelessonretrievaltoken", "max_chars": 1200, "include_scores": True},
    )
    assert lesson_retrieve_response.status == 200
    assert imported_method_id in lesson_retrieve_response.body["method_ids"]
    assert "Portable Bundle Gating" in lesson_retrieve_response.body["context"]
    assert "Prefer compact bundle lesson retrieval" in lesson_retrieve_response.body["context"]
    assert "bundlelessonretrievaltoken" in lesson_retrieve_response.body["context"]
    imported_lesson_score = next(
        score
        for score in lesson_retrieve_response.body["scores"]
        if score["method_id"] == imported_method_id
    )
    assert imported_lesson_score["lesson_match_count"] > 0
    assert imported_lesson_score["lesson_match_score"] > 0

    second_conn = sqlite3.connect(":memory:")
    second_conn.row_factory = sqlite3.Row
    initialize_db(second_conn)
    second_client = ResearchFSLabClient(conn=second_conn)
    client_import = second_client.import_paper_memory_bundle(route_bundle)
    assert client_import["ok"] is True
    assert client_import["lesson_ids"]
    assert client_import["source_id_map"]["lesson_ids"][str(method_lesson_id)] == client_import["lesson_ids"][0]
    client_retrieve = second_client.retrieve("bundlelessonretrievaltoken", max_chars=1200)
    assert client_import["method_ids"][0] in client_retrieve["method_ids"]
    assert "Portable Bundle Gating" in client_retrieve["context"]
    assert "Prefer compact bundle lesson retrieval" in client_retrieve["context"]
    assert "bundlelessonretrievaltoken" in client_retrieve["context"]

    imported_memory = target_api.dispatch("GET", f"/paper/{imported_paper_id}/memory?max_chars=1200")
    assert imported_memory.status == 200
    assert "bundleevidencetoken" in imported_memory.body["context"]
    assert "Carry over the reusable bundlelessonretrievaltoken lesson" in imported_memory.body["context"]


def test_corpus_memory_bundle_helper_and_sdk_export_multiple_papers_compressed_only():
    from researchfs import ResearchFSLabClient
    from researchfs.bundle import export_corpus_memory_bundle

    conn, api, papers = _corpus_bundle_test_source()
    paper_ids = [paper["paper_id"] for paper in papers]
    raw_texts = [paper["raw_only_text"] for paper in papers]

    bundle = export_corpus_memory_bundle(conn=conn, paper_ids=paper_ids, max_chars=700)

    _assert_corpus_bundle_is_compressed_only(bundle, raw_texts)
    assert [paper_bundle["paper"]["title"] for paper_bundle in bundle["papers"]] == [
        paper["title"] for paper in papers
    ]
    assert [paper_bundle["paper"]["source_paper_id"] for paper_bundle in bundle["papers"]] == paper_ids
    assert all(paper_bundle["memory"]["max_chars"] == 700 for paper_bundle in bundle["papers"])
    assert all(len(paper_bundle["memory"]["context"]) <= 700 for paper_bundle in bundle["papers"])
    encoded = _json_text(bundle)
    for paper in papers:
        assert paper["evidence_text"] in encoded
        assert paper["lesson_content"] in encoded
    for paper_bundle, paper in zip(bundle["papers"], papers):
        assert paper_bundle["extraction_status"]["extractor_model"] == "corpus-test-extractor"
        assert paper_bundle["extraction_status"]["extractor_prompt_version"] == "corpus-test-v1"
        source = paper_bundle["methods"][0]["sources"][0]
        assert source["evidence_text"] == paper["evidence_text"]
        assert source["evidence_start"] == 0
        assert source["evidence_end"] == len(paper["evidence_text"])
        assert source["evidence_strategy"] == "exact"

    client = ResearchFSLabClient(api=api)
    client_export = client.export_corpus_memory_bundle(paper_ids=paper_ids, max_chars=700)
    assert client_export["ok"] is True
    assert client_export["bundle"] == bundle


def test_corpus_memory_bundle_api_export_and_import_reuses_duplicates_with_aggregate_ids():
    from researchfs import ResearchFSLabClient

    _source_conn, source_api, papers = _corpus_bundle_test_source()
    paper_ids = [paper["paper_id"] for paper in papers]
    raw_texts = [paper["raw_only_text"] for paper in papers]

    export_response = source_api.dispatch("GET", f"/memory/export?paper_ids={paper_ids[0]},{paper_ids[1]}&max_chars=900")
    assert export_response.status == 200
    assert export_response.body["ok"] is True
    corpus_bundle = export_response.body["bundle"]
    _assert_corpus_bundle_is_compressed_only(corpus_bundle, raw_texts)
    assert len(corpus_bundle["papers"]) == 2

    target_conn = sqlite3.connect(":memory:")
    target_conn.row_factory = sqlite3.Row
    initialize_db(target_conn)
    target_api = ResearchFSLabAPI(conn=target_conn)
    import_response = target_api.dispatch("POST", "/memory/import", {"bundle": corpus_bundle})

    assert import_response.status == 200
    assert import_response.body["ok"] is True
    first_body = import_response.body
    assert len(first_body["paper_results"]) == 2
    assert len(first_body["paper_ids"]) == 2
    assert len(first_body["method_ids"]) == 2
    assert len(first_body["chunk_ids"]) == 2
    assert len(first_body["lesson_ids"]) == 2
    assert [result["paper_id"] for result in first_body["paper_results"]] == first_body["paper_ids"]
    assert [result["method_ids"][0] for result in first_body["paper_results"]] == first_body["method_ids"]
    assert [result["chunk_ids"][0] for result in first_body["paper_results"]] == first_body["chunk_ids"]
    assert [result["lesson_ids"][0] for result in first_body["paper_results"]] == first_body["lesson_ids"]
    assert all(result["duplicate"] is False for result in first_body["paper_results"])
    assert target_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
    assert target_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    stored_chunks = [
        row["content"]
        for row in target_conn.execute("SELECT content FROM chunks ORDER BY chunk_id").fetchall()
    ]
    for paper in papers:
        assert any(paper["evidence_text"] in chunk for chunk in stored_chunks)
        assert all(paper["raw_only_text"] not in chunk for chunk in stored_chunks)

    second_import = target_api.dispatch("POST", "/memory/import", {"bundle": corpus_bundle})
    assert second_import.status == 200
    assert second_import.body["ok"] is True
    assert second_import.body["paper_ids"] == first_body["paper_ids"]
    assert second_import.body["method_ids"] == first_body["method_ids"]
    assert second_import.body["chunk_ids"] == first_body["chunk_ids"]
    assert second_import.body["lesson_ids"] == first_body["lesson_ids"]
    assert all(result["duplicate"] is True for result in second_import.body["paper_results"])
    assert target_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
    assert target_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    assert target_conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 2

    client_conn = sqlite3.connect(":memory:")
    client_conn.row_factory = sqlite3.Row
    initialize_db(client_conn)
    client = ResearchFSLabClient(conn=client_conn)
    client_import = client.import_corpus_memory_bundle(corpus_bundle)
    assert client_import["ok"] is True
    assert len(client_import["paper_results"]) == 2
    assert client_import["paper_ids"] == [result["paper_id"] for result in client_import["paper_results"]]
    client_retrieve = client.retrieve("corpusbetaevidencetoken", max_chars=1200)
    assert client_import["method_ids"][1] in client_retrieve["method_ids"]
    assert "Corpus Beta Transfer" in client_retrieve["context"]
    assert "CORPUS_BETA_RAW_SENTINEL" not in client_retrieve["context"]


def test_corpus_memory_bundle_api_export_defaults_to_all_papers_and_import_rejects_raw_content():
    from researchfs.bundle import export_corpus_memory_bundle, import_corpus_memory_bundle

    conn, api, papers = _corpus_bundle_test_source()
    raw_texts = [paper["raw_only_text"] for paper in papers]
    all_export = api.dispatch("GET", "/memory/export?max_chars=800")
    assert all_export.status == 200
    all_bundle = all_export.body["bundle"]
    _assert_corpus_bundle_is_compressed_only(all_bundle, raw_texts)
    assert [paper_bundle["paper"]["source_paper_id"] for paper_bundle in all_bundle["papers"]] == [
        paper["paper_id"] for paper in papers
    ]

    helper_bundle = export_corpus_memory_bundle(conn=conn, max_chars=800)
    assert helper_bundle == all_bundle

    paper_content_bundle = {
        **helper_bundle,
        "papers": [
            {**helper_bundle["papers"][0], "paper": {**helper_bundle["papers"][0]["paper"], "content": raw_texts[0]}},
            helper_bundle["papers"][1],
        ],
    }
    try:
        import_corpus_memory_bundle(paper_content_bundle, conn=conn)
    except ValueError as exc:
        assert "paper.content" in str(exc)
    else:
        raise AssertionError("corpus paper.content should be rejected")

    chunk_content_bundle = {
        **helper_bundle,
        "papers": [
            {**helper_bundle["papers"][0], "chunks": [{"content": raw_texts[0]}]},
            helper_bundle["papers"][1],
        ],
    }
    try:
        import_corpus_memory_bundle(chunk_content_bundle, conn=conn)
    except ValueError as exc:
        assert "chunks" in str(exc) or "raw content" in str(exc)
    else:
        raise AssertionError("corpus chunks.content should be rejected")

    response = api.dispatch("POST", "/memory/import", {"bundle": paper_content_bundle})
    assert response.status == 400
    assert response.body["ok"] is False



def test_corpus_memory_bundle_import_rejects_invalid_later_paper_before_partial_import():
    from researchfs.bundle import export_corpus_memory_bundle

    source_conn, _, _ = _corpus_bundle_test_source()
    bundle = export_corpus_memory_bundle(conn=source_conn, max_chars=800)
    invalid_bundle = {
        **bundle,
        "papers": [
            bundle["papers"][0],
            {
                **bundle["papers"][1],
                "methods": [
                    {**bundle["papers"][1]["methods"][0], "name": "   "},
                ],
            },
        ],
    }

    target_conn = sqlite3.connect(":memory:")
    target_conn.row_factory = sqlite3.Row
    initialize_db(target_conn)
    target_api = ResearchFSLabAPI(conn=target_conn)

    response = target_api.dispatch("POST", "/memory/import", {"bundle": invalid_bundle})

    assert response.status == 400
    assert response.body["ok"] is False
    assert response.body["error"]["code"] == "invalid_bundle"
    assert "method.name" in response.body["error"]["message"]
    assert target_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    assert target_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert target_conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0] == 0
    assert target_conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] == 0
    assert target_conn.execute("SELECT COUNT(*) FROM method_sources").fetchone()[0] == 0
    assert target_conn.execute("SELECT COUNT(*) FROM paper_extraction_status").fetchone()[0] == 0


def test_corpus_memory_bundle_import_persists_through_file_backed_api_reopen():
    import tempfile
    from pathlib import Path
    from researchfs.bundle import export_corpus_memory_bundle

    source_conn, _, _ = _corpus_bundle_test_source()
    bundle = export_corpus_memory_bundle(conn=source_conn, max_chars=800)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "corpus-memory.sqlite"
        first_api = ResearchFSLabAPI(db_path=str(db_path))
        import_response = first_api.dispatch("POST", "/memory/import", {"bundle": bundle})
        assert import_response.status == 200
        assert import_response.body["ok"] is True
        assert len(import_response.body["paper_ids"]) == 2

        reopened_api = ResearchFSLabAPI(db_path=str(db_path))
        papers_response = reopened_api.dispatch("GET", "/papers")
        assert papers_response.status == 200
        assert [paper["title"] for paper in papers_response.body["papers"]] == [
            "Corpus Bundle Alpha Memory Paper",
            "Corpus Bundle Beta Memory Paper",
        ]

def test_lab_api_retrieve_can_log_event_and_retrieved_methods():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Auditable Retrieval Memory",
            "authors": "Audit Lab",
            "abstract": "A paper about auditable retrieval.",
            "url": "https://example.com/auditable-retrieval",
            "publish_date": "2026-06-30",
            "content": "Auditable retrieval methods should be logged for every lab inspection.",
            "methods": [
                {
                    "name": "Auditable Retrieval Alpha",
                    "description": "Log alpha retrieval memory for audit trails.",
                    "implementation_hint": "Persist the retrieval event.",
                    "risks": "Audits fail without event rows.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                },
                {
                    "name": "Auditable Retrieval Beta",
                    "description": "Log beta retrieval memory for audit trails.",
                    "implementation_hint": "Persist retrieved method links.",
                    "risks": "Audits fail without method rows.",
                    "category": "memory",
                    "source_chunk_indices": [0],
                },
            ],
        },
    )
    assert ingest_response.status == 200
    expected_method_ids = set(ingest_response.body["method_ids"])
    run_id = rfs.start_agent_run("loopgraph", "audit-model", conn=conn)
    experiment_id = rfs.log_experiment(
        "audit-commit",
        2.0,
        1.0,
        "keep",
        "auditable retrieval",
        agent_run_id=run_id,
        conn=conn,
    )

    retrieve_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "auditable retrieval",
            "max_chars": 1200,
            "log": True,
            "event_type": "lab_audit_search",
            "agent_run_id": run_id,
            "experiment_id": experiment_id,
        },
    )
    assert retrieve_response.status == 200
    assert retrieve_response.body["event_id"] is not None
    assert set(retrieve_response.body["method_ids"]) == expected_method_ids
    assert len(retrieve_response.body["context"]) <= 1200

    event_id = retrieve_response.body["event_id"]
    event = conn.execute(
        "SELECT * FROM retrieval_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    assert event["event_type"] == "lab_audit_search"
    assert event["query"] == "auditable retrieval"
    assert event["agent_run_id"] == run_id
    assert event["experiment_id"] == experiment_id
    assert event["retrieved_content"] == retrieve_response.body["context"]

    rows = conn.execute(
        "SELECT method_id FROM retrieved_methods WHERE event_id = ? ORDER BY method_id",
        (event_id,),
    ).fetchall()
    assert [row["method_id"] for row in rows] == sorted(expected_method_ids)



def test_lab_api_feedback_logs_summary_stats_and_reorders_retrieval():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    ingest_response = api.dispatch(
        "POST",
        "/ingest",
        {
            "title": "Feedback Ordering Memory",
            "authors": "Feedback Lab",
            "abstract": "A paper about feedbackorder retrieval ranking.",
            "url": "https://example.com/feedback-ordering",
            "publish_date": "2026-06-30",
            "content": "The feedbackorder token appears in comparable feedback ranking methods.",
            "methods": [
                {
                    "name": "Feedback Loop Alpha",
                    "description": "Use feedbackorder signals to rank comparable lab methods.",
                    "implementation_hint": "Apply a bounded human rating multiplier.",
                    "risks": "Ratings should not overwhelm empirical outcomes.",
                    "category": "feedback",
                    "source_chunk_indices": [0],
                },
                {
                    "name": "Feedback Loop Beta",
                    "description": "Use feedbackorder signals to rank comparable lab methods.",
                    "implementation_hint": "Apply a bounded human rating multiplier.",
                    "risks": "Ratings should not overwhelm empirical outcomes.",
                    "category": "feedback",
                    "source_chunk_indices": [0],
                },
            ],
        },
    )
    assert ingest_response.status == 200
    alpha_id, beta_id = ingest_response.body["method_ids"]

    initial_retrieve = api.dispatch(
        "POST",
        "/retrieve",
        {"query": "feedbackorder", "max_chars": 2000, "log": True},
    )
    assert initial_retrieve.status == 200
    assert initial_retrieve.body["method_ids"][:2] == [alpha_id, beta_id]
    assert "scores" not in initial_retrieve.body
    event_id = initial_retrieve.body["event_id"]

    missing_method = api.dispatch("POST", "/feedback", {"rating": 0.5})
    assert missing_method.status == 400
    assert missing_method.body["error"]["code"] == "missing_field"

    invalid_rating = api.dispatch(
        "POST",
        "/feedback",
        {"method_id": alpha_id, "rating": 1.5},
    )
    assert invalid_rating.status == 400
    assert invalid_rating.body["error"]["code"] == "invalid_field"

    unknown_method = api.dispatch(
        "POST",
        "/feedback",
        {"method_id": 999999, "rating": 0.5},
    )
    assert unknown_method.status == 400
    assert unknown_method.body["error"]["code"] == "invalid_feedback"

    negative_response = api.dispatch(
        "POST",
        "/feedback",
        {
            "event_id": event_id,
            "method_id": alpha_id,
            "rating": -1.0,
            "label": "less useful",
            "note": "Alpha was less relevant for this query.",
            "source": "api-test",
        },
    )
    assert negative_response.status == 200
    assert negative_response.body["feedback_id"] is not None

    positive_response = api.dispatch(
        "POST",
        "/feedback",
        {
            "event_id": event_id,
            "method_id": beta_id,
            "rating": 1.0,
            "label": "more useful",
            "note": "Beta was more relevant for this query.",
            "source": "api-test",
        },
    )
    assert positive_response.status == 200
    assert positive_response.body["feedback_id"] is not None

    detail_response = api.dispatch("GET", f"/method/{beta_id}")
    assert detail_response.status == 200
    feedback_summary = detail_response.body["feedback_summary"]
    assert feedback_summary["method_id"] == beta_id
    assert feedback_summary["feedback_count"] == 1
    assert feedback_summary["feedback_avg_rating"] == 1.0
    assert feedback_summary["positive_feedback_count"] == 1
    assert feedback_summary["negative_feedback_count"] == 0

    stats_response = api.dispatch("GET", "/stats")
    assert stats_response.status == 200
    assert stats_response.body["counts"]["method_feedback"] == 2
    assert stats_response.body["feedback"]["count"] == 2
    assert stats_response.body["feedback"]["average_rating"] == 0.0

    default_reranked_response = api.dispatch(
        "POST",
        "/retrieve",
        {"query": "feedbackorder", "max_chars": 2000},
    )
    assert default_reranked_response.status == 200
    assert default_reranked_response.body["method_ids"][:2] == [beta_id, alpha_id]
    assert "scores" not in default_reranked_response.body

    scored_method_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "feedbackorder",
            "max_chars": 2000,
            "include_scores": True,
            "log": True,
        },
    )
    assert scored_method_response.status == 200
    assert scored_method_response.body["method_ids"][:2] == [beta_id, alpha_id]
    assert scored_method_response.body["event_id"] is not None
    assert "Feedback Loop Beta" in scored_method_response.body["context"]
    method_scores = scored_method_response.body["scores"]
    assert [score["method_id"] for score in method_scores[:2]] == [beta_id, alpha_id]
    assert method_scores[0]["final_score"] >= method_scores[1]["final_score"]
    assert method_scores[0]["feedback_multiplier"] > method_scores[1]["feedback_multiplier"]
    assert method_scores[0]["feedback_count"] == 1
    assert method_scores[0]["feedback_avg_rating"] == 1.0
    assert method_scores[1]["feedback_count"] == 1
    assert method_scores[1]["feedback_avg_rating"] == -1.0

    scored_graph_response = api.dispatch(
        "POST",
        "/retrieve",
        {
            "query": "feedbackorder",
            "max_chars": 2000,
            "mode": "graph",
            "include_scores": True,
            "log": True,
        },
    )
    assert scored_graph_response.status == 200
    assert scored_graph_response.body["mode"] == "graph"
    assert scored_graph_response.body["event_id"] is not None
    assert scored_graph_response.body["method_ids"][:2] == [beta_id, alpha_id]
    assert [score["method_id"] for score in scored_graph_response.body["scores"][:2]] == [
        beta_id,
        alpha_id,
    ]
    assert scored_graph_response.body["context"].startswith("# ResearchFS Graph Context")


def test_lab_api_ingest_file_loads_markdown_tex_and_pdf():
    import tempfile
    from pathlib import Path

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    with tempfile.TemporaryDirectory() as tmpdir:
        note_path = Path(tmpdir) / "api-file-memory.md"
        note_path.write_text(
            "# API File Memory\n\n"
            "The apifiletoken route loads local lab notes into paper memory.",
            encoding="utf-8",
        )
        response = api.dispatch(
            "POST",
            "/ingest/file",
            {
                "path": str(note_path),
                "authors": "Local Lab",
                "methods": [
                    {
                        "name": "API File Memory Route",
                        "description": "Load apifiletoken notes through the API.",
                        "implementation_hint": "Use POST /ingest/file for local notes.",
                        "risks": "Requires a local readable text file.",
                        "category": "ingestion",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )
        assert response.status == 200
        assert response.body["ingestion"]["extraction_status"] == "extracted"
        assert "confidence_score" in response.body["ingestion"]
        assert "confidence_reason" in response.body["ingestion"]
        assert response.body["extraction_status"]["source_type"] == "file"
        status = conn.execute(
            "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
            (response.body["paper_id"],),
        ).fetchone()
        assert status["source_type"] == "file"
        method_id = response.body["method_ids"][0]

        paper = conn.execute(
            "SELECT title, authors, url FROM papers WHERE paper_id = ?",
            (response.body["paper_id"],),
        ).fetchone()
        assert paper["title"] == "Api File Memory"
        assert paper["authors"] == "Local Lab"
        assert paper["url"].startswith("file://")

        retrieve_response = api.dispatch(
            "POST",
            "/retrieve",
            {"query": "apifiletoken", "max_chars": 800, "log": True},
        )
        assert retrieve_response.status == 200
        assert method_id in retrieve_response.body["method_ids"]
        assert retrieve_response.body["event_id"] is not None

        tex_path = Path(tmpdir) / "api-tex-memory.tex"
        tex_path.write_text(
            "\\section{API TeX Memory}\n"
            "The apitextoken route loads TeX lab notes into paper memory.",
            encoding="utf-8",
        )
        tex_response = api.dispatch(
            "POST",
            "/ingest/file",
            {
                "path": str(tex_path),
                "authors": "API TeX Lab",
                "methods": [
                    {
                        "name": "API TeX Memory Route",
                        "description": "Load apitextoken TeX notes through the API.",
                        "implementation_hint": "Use POST /ingest/file with a TeX path.",
                        "risks": "TeX is treated as UTF-8 source text, not rendered PDF.",
                        "category": "ingestion",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )
        assert tex_response.status == 200
        assert tex_response.body["ingestion"]["extraction_status"] == "extracted"
        tex_method_id = tex_response.body["method_ids"][0]

        tex_paper = conn.execute(
            "SELECT title, authors, url FROM papers WHERE paper_id = ?",
            (tex_response.body["paper_id"],),
        ).fetchone()
        assert tex_paper["title"] == "Api Tex Memory"
        assert tex_paper["authors"] == "API TeX Lab"
        assert tex_paper["url"] == tex_path.resolve().as_uri()

        tex_retrieve_response = api.dispatch(
            "POST",
            "/retrieve",
            {"query": "apitextoken", "max_chars": 800, "log": True},
        )
        assert tex_retrieve_response.status == 200
        assert tex_method_id in tex_retrieve_response.body["method_ids"]
        assert tex_retrieve_response.body["event_id"] is not None

        pdf_path = Path(tmpdir) / "api-pdf-memory.pdf"
        _write_minimal_pdf(
            pdf_path,
            "The apipdftoken route loads actual PDF text into paper memory.",
        )
        pdf_response = api.dispatch(
            "POST",
            "/ingest/file",
            {
                "path": str(pdf_path),
                "authors": "API PDF Lab",
                "methods": [
                    {
                        "name": "API PDF Memory Route",
                        "description": "Load apipdftoken PDFs through the API.",
                        "implementation_hint": "Use POST /ingest/file with a PDF path.",
                        "risks": "Malformed PDFs should return explicit extraction errors.",
                        "category": "ingestion",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )
        assert pdf_response.status == 200
        assert pdf_response.body["ingestion"]["extraction_status"] == "extracted"
        pdf_method_id = pdf_response.body["method_ids"][0]

        pdf_paper = conn.execute(
            "SELECT title, authors, url FROM papers WHERE paper_id = ?",
            (pdf_response.body["paper_id"],),
        ).fetchone()
        assert pdf_paper["title"] == "Api Pdf Memory"
        assert pdf_paper["authors"] == "API PDF Lab"
        assert pdf_paper["url"] == pdf_path.resolve().as_uri()

        pdf_retrieve_response = api.dispatch(
            "POST",
            "/retrieve",
            {"query": "apipdftoken", "max_chars": 800, "log": True},
        )
        assert pdf_retrieve_response.status == 200
        assert pdf_method_id in pdf_retrieve_response.body["method_ids"]
        assert pdf_retrieve_response.body["event_id"] is not None

        malformed_path = Path(tmpdir) / "malformed-api-paper.pdf"
        malformed_path.write_bytes(b"%PDF-1.7\nbroken local API PDF\n")
        malformed_response = api.dispatch("POST", "/ingest/file", {"path": str(malformed_path)})
        assert malformed_response.status == 400
        assert malformed_response.body["ok"] is False
        assert malformed_response.body["error"]["code"] == "extraction_failed"
        assert "Could not extract text from PDF" in malformed_response.body["error"]["message"]



def test_lab_api_ingest_file_preserves_source_type_override():
    import tempfile
    from pathlib import Path

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    with tempfile.TemporaryDirectory() as tmpdir:
        note_path = Path(tmpdir) / "lab-note-source-type.md"
        note_path.write_text(
            "# Lab Note Source Type\n\n"
            "The apisourcetypetoken local note preserves its lab_note artifact label.",
            encoding="utf-8",
        )
        response = api.dispatch(
            "POST",
            "/ingest/file",
            {
                "path": str(note_path),
                "authors": "Source Type Lab",
                "source_type": "lab_note",
                "methods": [
                    {
                        "name": "API Source Type Route",
                        "description": "Preserve apisourcetypetoken lab note labels through ingestion.",
                        "implementation_hint": "Pass source_type through local file overrides.",
                        "risks": "Dropping the label hides the artifact provenance.",
                        "category": "ingestion",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )

        assert response.status == 200
        assert response.body["extraction_status"]["source_type"] == "lab_note"
        status = conn.execute(
            "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
            (response.body["paper_id"],),
        ).fetchone()
        assert status["source_type"] == "lab_note"

def test_lab_api_ingest_arxiv_uses_stub_fetcher_and_temp_db_path():
    import tempfile
    from pathlib import Path

    fetched_ids = []

    def stub_fetcher(arxiv_id):
        fetched_ids.append(arxiv_id)
        return {
            "title": "Arxiv Gate Scaling Note",
            "authors": "Arxiv Author",
            "abstract": "Gate scaling can stabilize small transformer runs.",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "publish_date": "2026-06-30",
            "content": "Arxiv gate scaling dampens residual branches for stable training.",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "researchfs_lab.sqlite"
        api = ResearchFSLabAPI(db_path=str(db_path), arxiv_fetcher=stub_fetcher)
        ingest_response = api.dispatch(
            "POST",
            "/ingest/arxiv",
            {
                "id": "2606.12345",
                "methods": [
                    {
                        "name": "Arxiv Gate Scaling",
                        "description": "Scale gate outputs to stabilize compact runs.",
                        "implementation_hint": "Multiply gate outputs by a small learned scalar.",
                        "risks": "Can slow convergence if initialized too low.",
                        "category": "normalization",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )
        assert ingest_response.status == 200
        assert fetched_ids == ["2606.12345"]
        assert db_path.exists()
        assert "confidence_score" in ingest_response.body["ingestion"]
        assert "confidence_reason" in ingest_response.body["ingestion"]
        method_id = ingest_response.body["method_ids"][0]

        reopened_api = ResearchFSLabAPI(db_path=str(db_path), arxiv_fetcher=stub_fetcher)
        retrieve_response = reopened_api.dispatch(
            "POST",
            "/retrieve",
            {"query": "arxiv gate scaling", "max_chars": 600},
        )
        assert retrieve_response.status == 200
        assert method_id in retrieve_response.body["method_ids"]
        assert "Arxiv Gate Scaling" in retrieve_response.body["context"]


def test_lab_api_ingest_repo_uses_stub_fetcher_and_temp_db_path():
    import tempfile
    from pathlib import Path

    fetched_repos = []

    def stub_fetcher(repo):
        fetched_repos.append(repo)
        return {
            "title": "GitHub: example/gate-repo",
            "authors": "example",
            "abstract": "Repository README describes repogatetoken scaling.",
            "url": "https://github.com/example/gate-repo",
            "publish_date": "n/a",
            "content": "Repogatetoken README content dampens residual branches for stable training.",
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "researchfs_lab.sqlite"
        api = ResearchFSLabAPI(db_path=str(db_path), repo_fetcher=stub_fetcher)
        ingest_response = api.dispatch(
            "POST",
            "/ingest/repo",
            {
                "repo": "example/gate-repo",
                "methods": [
                    {
                        "name": "Repo Gate Scaling",
                        "description": "Scale repository gate outputs to stabilize compact runs.",
                        "implementation_hint": "Multiply gate outputs by a small learned scalar.",
                        "risks": "Can slow convergence if initialized too low.",
                        "category": "normalization",
                        "source_chunk_indices": [0],
                    }
                ],
            },
        )
        assert ingest_response.status == 200
        assert fetched_repos == ["example/gate-repo"]
        assert ingest_response.body["repo"] == "example/gate-repo"
        assert db_path.exists()
        assert ingest_response.body["extraction_status"]["source_type"] == "repo_readme"
        assert ingest_response.body["ingestion"]["extraction"]["source_type"] == "repo_readme"
        method_id = ingest_response.body["method_ids"][0]

        db_conn = sqlite3.connect(db_path)
        status = db_conn.execute(
            "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
            (ingest_response.body["paper_id"],),
        ).fetchone()
        db_conn.close()
        assert status[0] == "repo_readme"

        reopened_api = ResearchFSLabAPI(db_path=str(db_path), repo_fetcher=stub_fetcher)
        retrieve_response = reopened_api.dispatch(
            "POST",
            "/retrieve",
            {"query": "repogatetoken", "max_chars": 600},
        )
        assert retrieve_response.status == 200
        assert method_id in retrieve_response.body["method_ids"]
        assert "Repo Gate Scaling" in retrieve_response.body["context"]


def test_lab_api_ingest_repo_preserves_source_type_override():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)

    def stub_fetcher(repo):
        return {
            "title": "GitHub: example/artifact-repo",
            "authors": "example",
            "abstract": "Artifact README describes repoartifacttoken scheduling.",
            "url": "https://github.com/example/artifact-repo",
            "publish_date": "n/a",
            "content": "The repoartifacttoken schedule is documented in this README.",
        }

    api = ResearchFSLabAPI(conn=conn, repo_fetcher=stub_fetcher)
    response = api.dispatch(
        "POST",
        "/ingest/repo",
        {
            "repo": "example/artifact-repo",
            "source_type": " lab_artifact ",
            "methods": [
                {
                    "name": "Repo Artifact Schedule",
                    "description": "Use repoartifacttoken scheduling from a lab artifact.",
                    "implementation_hint": "Apply the schedule from the README.",
                    "risks": "Artifact README may drift from code behavior.",
                    "category": "schedule",
                    "source_chunk_indices": [0],
                }
            ],
        },
    )

    assert response.status == 200
    assert response.body["extraction_status"]["source_type"] == "lab_artifact"
    status = conn.execute(
        "SELECT source_type FROM paper_extraction_status WHERE paper_id = ?",
        (response.body["paper_id"],),
    ).fetchone()
    assert status["source_type"] == "lab_artifact"


def test_lab_api_ingest_repo_requires_repo():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn, repo_fetcher=lambda repo: {})

    response = api.dispatch("POST", "/ingest/repo", {})

    assert response.status == 400
    assert response.body["error"]["code"] == "missing_field"


def test_lab_api_ingest_repo_reports_malformed_repo_as_client_error():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)

    def malformed_repo_fetcher(repo):
        raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")

    api = ResearchFSLabAPI(conn=conn, repo_fetcher=malformed_repo_fetcher)
    response = api.dispatch("POST", "/ingest/repo", {"repo": "not-a-repo"})

    assert response.status == 400
    assert response.body["error"]["code"] == "invalid_repo"
    assert "owner/repo" in response.body["error"]["message"]


def test_fetch_github_readme_repo_path_parser_is_strict_and_preserves_valid_cases():
    assert _parse_github_repo_path("owner/repo") == ("owner", "repo")
    assert _parse_github_repo_path("owner/repo.git") == ("owner", "repo")
    assert _parse_github_repo_path("https://github.com/owner/repo/") == ("owner", "repo")

    invalid_values = [
        "owner/repo/extra",
        "https://github.com/owner/repo/tree/main",
        "https://github.com/owner/repo?tab=readme",
        "https://github.com/owner/repo#readme",
    ]
    for value in invalid_values:
        try:
            _parse_github_repo_path(value)
        except ValueError as exc:
            assert "owner/repo" in str(exc)
        else:
            raise AssertionError(f"Expected malformed repo path to fail: {value}")


def test_lab_api_ingest_repo_rejects_extra_path_with_default_fetcher():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    api = ResearchFSLabAPI(conn=conn)

    response = api.dispatch("POST", "/ingest/repo", {"repo": "owner/repo/extra"})

    assert response.status == 400
    assert response.body["error"]["code"] == "invalid_repo"

if __name__ == "__main__":
    import sys
    try:
        import pytest
        sys.exit(pytest.main([__file__] + sys.argv[1:]))
    except ImportError:
        print("pytest not installed, running tests manually...")
        test_funcs = [g for name, g in globals().items() if name.startswith("test_") and callable(g)]
        passed = 0
        failed = 0
        for func in test_funcs:
            try:
                func()
                print(f"  {func.__name__}: PASSED")
                passed += 1
            except Exception:
                print(f"  {func.__name__}: FAILED")
                import traceback
                traceback.print_exc()
                failed += 1
        print(f"\nManual test summary: {passed} passed, {failed} failed")
        sys.exit(1 if failed > 0 else 0)
