from run_agent import (
    extract_code_block,
    extract_description,
    parse_log,
    assemble_system_prompt,
    assemble_user_prompt
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
