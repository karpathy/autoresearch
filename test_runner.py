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
from researchfs.ingest import seed_initial_data
from researchfs import retrieve as rfs
from evals.nanochat_bpb import metrics
from agents.editor import apply_hunks, check_syntax, HunkError


def _in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    seed_initial_data(conn)
    return conn


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
