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

def test_extract_code_block_fallback():
    response = "def baz():\n    return 1"
    code = extract_code_block(response)
    assert code == "def baz():\n    return 1"

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

import sqlite3
from researchfs.db import initialize_db
from researchfs.ingest import seed_initial_data, ingest_paper
from researchfs import retrieve as rfs
from researchfs.extractor import chunk_text, extract_json_block
from researchfs.fetcher import clean_latex
from evals.nanochat_bpb import metrics
from agents.editor import apply_hunks, check_syntax, HunkError


def _in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    seed_initial_data(conn)
    return conn


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


def test_extract_json_block_variants():
    block = '```json\n{"methods": [{"name": "x"}]}\n```'
    assert extract_json_block(block) == '{"methods": [{"name": "x"}]}'

    raw = '{"methods": [{"name": "y"}]}'
    assert extract_json_block(raw) == raw

    fallback = "plain text"
    assert extract_json_block(fallback) == fallback


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
        "search", "fatiguetest", "content", agent_run_id=prev_run, experiment_id=exp_prev, conn=conn, method_ids=[tainted_id]
    )
    rfs.update_experiment(exp_prev, status="keep", conn=conn)

    # Current run: tainted method is retrieved then discarded.
    current_run = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", current_run, conn)
    exp_cur = rfs.log_experiment("abc456", 2.1, 0.5, "pending", "test", current_run, conn)
    rfs.log_retrieval_event(
        "search", "fatiguetest", "content", agent_run_id=current_run, experiment_id=exp_cur, conn=conn, method_ids=[tainted_id]
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
        "search", "fatiguetest", "content", agent_run_id=prev_run_neutral, experiment_id=exp_n_keep, conn=conn, method_ids=[neutral_id]
    )
    rfs.update_experiment(exp_n_keep, status="keep", conn=conn)
    exp_n_discard = rfs.log_experiment("abc124", 2.1, 0.5, "pending", "test", prev_run_neutral, conn)
    rfs.log_retrieval_event(
        "search", "fatiguetest", "content", agent_run_id=prev_run_neutral, experiment_id=exp_n_discard, conn=conn, method_ids=[neutral_id]
    )
    rfs.update_experiment(exp_n_discard, status="discard", conn=conn)

    prev_run_boosted = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", prev_run_boosted, conn)
    exp_b_discard = rfs.log_experiment("abc123", 2.1, 0.5, "pending", "test", prev_run_boosted, conn)
    rfs.log_retrieval_event(
        "search", "fatiguetest", "content", agent_run_id=prev_run_boosted, experiment_id=exp_b_discard, conn=conn, method_ids=[boosted_id]
    )
    rfs.update_experiment(exp_b_discard, status="discard", conn=conn)

    # Current run: boosted method is retrieved and kept.
    current_run = rfs.start_agent_run("loopgraph", "gpt-4o-mini", conn)
    rfs.log_experiment("baseline", 2.0, 0.5, "keep", "baseline", current_run, conn)
    exp_cur = rfs.log_experiment("abc456", 2.0, 0.5, "pending", "test", current_run, conn)
    rfs.log_retrieval_event(
        "search", "fatiguetest", "content", agent_run_id=current_run, experiment_id=exp_cur, conn=conn, method_ids=[boosted_id]
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
        "content": "This is the content. It needs enough text to create at least one chunk.",
    }
    result = ingest_paper(meta, llm_fn=stub_llm, conn=conn)

    assert result["paper_id"] is not None
    assert len(result["chunk_ids"]) > 0
    assert len(result["method_ids"]) == 1

    pack = rfs.search_methods("stub keyword", conn)
    assert "Stub Test Method" in pack



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
    cur.execute("SELECT 1 FROM method_sources WHERE method_id = ? AND chunk_id = ?", (existing_method_id, new_chunk_id))
    assert cur.fetchone() is not None, "Expected method_sources to link the existing method to the new chunk"
    
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


def test_cost_estimate_uses_model_rates():
    cost = metrics.estimate_cost(1_000_000, 200_000, "gpt-4o-mini")
    assert cost > 0
    assert cost == (1_000_000 * 0.00015 + 200_000 * 0.0006) / 1000


def test_apply_hunks_success():
    content = "line 1\nline 2\nline 3"
    response = "some text\n<<<<<<< SEARCH\nline 2\n=======\nline 2 modified\n>>>>>>>"
    res = apply_hunks(content, response)
    assert res == "line 1\nline 2 modified\nline 3"


def test_apply_hunks_not_found():
    content = "line 1\nline 2\nline 3"
    response = "<<<<<<< SEARCH\nmissing line\n=======\nreplacement\n>>>>>>>"
    try:
        apply_hunks(content, response)
        assert False, "Should raise HunkError"
    except HunkError:
        pass


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
                if len(args) >= 3 and args[:3] == ["git", "diff", "--quiet"]:
                    res.returncode = 1
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
                if len(args) >= 3 and args[:3] == ["git", "diff", "--quiet"]:
                    res.returncode = 1
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
            except AssertionError as e:
                print(f"  {func.__name__}: FAILED")
                import traceback
                traceback.print_exc()
                failed += 1
        print(f"\nManual test summary: {passed} passed, {failed} failed")
        sys.exit(1 if failed > 0 else 0)
