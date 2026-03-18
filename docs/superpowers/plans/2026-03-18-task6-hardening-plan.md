# Task 6 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the outstanding Task 6 issues covering secret redaction, gate contracts, status semantics, config validation, and supporting documentation.

**Architecture:** The plan tightens the orchestration entry point (`autosaas/main.py`), validation pipeline, and config loader around deterministic artifacts, while keeping docs and templates aligned.

**Tech Stack:** Python 3.10+, pytest/uv for testing, YAML configs, simple TSV templating.

---

### Task 1: Harden redaction and crash semantics

**Files:**
- Modify: `autosaas/main.py`
- Test: `tests/test_main_loop.py`

- [ ] **Step 1: Write failing regression tests**

  Add one additional pytest case that monkeypatches `reporter.format_slice_run` to return `"sk_live_123"` and asserts the final `RunResult.report` contains neither `sk_live` nor `_123`. Add another test case that raises an exception from within `run_once` (e.g. stub `load_repo_context` to raise) and assert the returned `status == "crash"` instead of propagating.

- [ ] **Step 2: Run failing test**

  Run `uv run pytest tests/test_main_loop.py::test_run_once_redacts_full_sk_live -v` expecting failure before implementation, then rerun full `tests/test_main_loop.py::test_run_once_handles_exceptions -v` for added crash test.

- [ ] **Step 3: Implement minimal fix**

  Update `run_once` to wrap the orchestration in try/except, returning a `RunResult(..., status="crash", report=<redacted summary/exception hint>)`. Extend redaction logic to scrub `sk_live_[A-Za-z0-9]+` tokens via a regex pattern and ensure summary always redacted.

- [ ] **Step 4: Run tests to confirm**

  Run `uv run pytest tests/test_main_loop.py -v` and expect both new tests to pass along with existing ones.

- [ ] **Step 5: Commit**

  `git add autosaas/main.py tests/test_main_loop.py` then `git commit -m "fix: harden run loop redaction and crash"`

### Task 2: Extend config contract with app boot URL

**Files:**
- Modify: `autosaas/config.py`
- Modify: `tests/test_config.py`
- Modify: `docs/target-repo-contract.md`, `docs/validation-protocol.md`, `templates/program.private.md`
- Modify: `templates/project.autosaas.yaml`

- [ ] **Step 1: Write failing test**

  Add a pytest case that writes `project.autosaas.yaml` with `app_boot_url: http://localhost` and asserts `load_target_config` returns it accessible, along with existing command entries.

- [ ] **Step 2: Run failing test**

  Run `uv run pytest tests/test_config.py::test_load_target_config_optionally_reads_app_boot -v` expecting failure.

- [ ] **Step 3: Implement minimal fix**

  Update `config.py` to recognize optional `app_boot_url` (default None) and include it in the returned dataclass. Update `RunResult` flow to include this URL when populating `command_map` or gating logic as `app_boot_url` entry used by `validation_pipeline._execute_gate`. Extend docs/template to mention optional `app_boot_url` and the ability to configure `app_boot` gate via this key.

- [ ] **Step 4: Run tests**

  Run `uv run pytest tests/test_config.py::test_load_target_config_optionally_reads_app_boot -v` to confirm pass.

- [ ] **Step 5: Commit**

  `git add autosaas/config.py tests/test_config.py docs/target-repo-contract.md docs/validation-protocol.md templates/project.autosaas.yaml templates/program.private.md` and `git commit -m "feat: add optional app boot url"`

### Task 3: Align validation status semantics

**Files:**
- Modify: `autosaas/validation_pipeline.py`
- Modify: `autosaas/branch_keeper.py`
- Modify: `tests/test_validation_pipeline.py` (maybe new tests) and `tests/test_branch_keeper.py`
- Modify: `docs/validation-protocol.md`, `docs/architecture.md` (if needed)

- [ ] **Step 1: Write failing tests**

  Update `tests/test_validation_pipeline.py` to expect success sets `status == "keep"`; for empty required gates, stub to ensure `run_required_gates` returns `status == "revert"`. Update branch keeper tests to check empty map returns `revert`.

- [ ] **Step 2: Run failing tests**

  Run combined pytest commands for the touched tests (subset) and confirm failures.

- [ ] **Step 3: Implement fix**

  Change `run_required_gates` to set `run.status = "keep"` instead of `pass`. Modify `decide_keep_or_revert` to return `revert` when the gate-result map is empty and to default to `revert` when there's any failure. Update docs to describe `keep`/`revert` language.

- [ ] **Step 4: Run tests**

  Run targeted `uv run pytest tests/test_validation_pipeline.py tests/test_branch_keeper.py -v` to confirm.

- [ ] **Step 5: Commit**

  `git add autosaas/validation_pipeline.py autosaas/branch_keeper.py tests/test_validation_pipeline.py tests/test_branch_keeper.py docs/validation-protocol.md docs/architecture.md` and `git commit -m "fix: align validation status semantics"`

### Task 4: Harden config loader and docs cleanup

**Files:**
- Modify: `autosaas/config.py`
- Modify: `tests/test_config.py`
- Possibly adjust docs if necessary

- [ ] **Step 1: Write failing test**

  Add pytest case that writes YAML root as a list and asserts `load_target_config` raises `ValueError` with clear message.

- [ ] **Step 2: Run test**

  Run `uv run pytest tests/test_config.py::test_load_target_config_rejects_non_mapping -v` expecting failure.

- [ ] **Step 3: Implement fix**

  Update `load_target_config` to check `data` is dict-like and raise `ValueError` if not. Avoid broad except, allow YAML errors to bubble.

- [ ] **Step 4: Run tests**

  Run targeted test and ensure pass.

- [ ] **Step 5: Commit**

  `git add autosaas/config.py tests/test_config.py` and `git commit -m "fix: validate config mapping root"`

### Task 5: Low-risk cleanup

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `templates/autosaas-results.tsv`

- [ ] **Step 1: Update pyproject**

  Add `requires-python = ">=3.10"` under `[project]` metadata.

- [ ] **Step 2: Sync lockfile**

  Run `uv sync` to regenerate `uv.lock` so `requires-python` matches.

- [ ] **Step 3: Align TSV template header**

  Ensure `templates/autosaas-results.tsv` uses `RESULTS_TSV_HEADER` exactly (same newline and header order). If header missing newline, replace accordingly.

- [ ] **Step 4: Run necessary regen/test**

  No tests needed beyond verifying lock file, but rerun key suites if impacted.

- [ ] **Step 5: Commit**

  `git add pyproject.toml uv.lock templates/autosaas-results.tsv` and `git commit -m "chore: update metadata"`

