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
