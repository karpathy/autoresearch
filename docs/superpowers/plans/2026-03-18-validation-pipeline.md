# Validation pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a minimal validation pipeline that executes gated commands, boots apps, and orchestrates smoke checks using subprocesses and short-circuiting logic.

**Architecture:** A focused `autosaas.validation_pipeline` module drives required gates by mapping gate names to commands or helper utilities; it records GateResult objects, stops at the first failure, and reports a run status. Helpers in `autosaas.utils` abstract subprocess execution and simple HTTP readiness probing.

**Tech Stack:** Python 3, standard library (`subprocess`, `http.client`), pytest for behavior verification, uv for command execution.

---

### Task 1: Minimal validation pipeline and gate utilities

**Files:**
- Create: `autosaas/validation_pipeline.py`
- Modify: `autosaas/utils.py`
- Test: `tests/test_validation_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from autosaas.validation_pipeline import run_required_gates
from autosaas.state import SliceRun


def test_run_required_gates_returns_fail_on_first_required_error(tmp_path):
    run = SliceRun(slice_name="demo", branch="autosaas/demo", base_commit="abc1234")
    result = run_required_gates(
        run,
        required_gates=["lint", "typecheck"],
        command_map={"lint": "python -c 'raise SystemExit(1)'", "typecheck": "python -c 'print(1)'"},
        cwd=tmp_path,
    )
    assert result.status == "revert"
    assert any(g.name == "lint" and not g.passed for g in result.gate_results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validation_pipeline.py::test_run_required_gates_returns_fail_on_first_required_error -v`
Expected: FAIL with import errors due to missing validation pipeline module.

- [ ] **Step 3: Write minimal implementation**

```python
# autosaas/validation_pipeline.py
from autosaas.state import GateResult
from autosaas.utils import run_command, wait_for_app_ready


def run_required_gates(run, required_gates, command_map, cwd):
    gate_results = []
    for gate in required_gates:
        command = command_map.get(gate)
        passed = False
        output = None
        if gate in ("lint", "typecheck", "test") and command:
            passed, output = run_command(command, cwd=cwd)
        elif gate == "app_boot":
            passed = wait_for_app_ready(command_map.get("app_boot"), cwd=cwd)
        elif gate == "smoke":
            passed, output = run_command(command_map.get("smoke"), cwd=cwd)
        gate_results.append(GateResult(name=gate, passed=passed, details=output))
        if not passed:
            run.status = "revert"
            run.gate_results = gate_results
            return run
    run.status = "pass"
    run.gate_results = gate_results
    return run
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validation_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autosaas/validation_pipeline.py autosaas/utils.py tests/test_validation_pipeline.py docs/superpowers/plans/2026-03-18-validation-pipeline.md
git commit -m "feat: add validation pipeline with required gates"
```
