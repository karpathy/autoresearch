---
phase: 02-experiment-infrastructure
verified: 2026-03-25T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 2: Experiment Infrastructure Verification Report

**Phase Goal:** A complete experiment loop harness exists that can run train.py, log results, manage git state, and recover from crashes -- ready for an agent to drive
**Verified:** 2026-03-25
**Status:** PASSED
**Re-verification:** No -- initial verification

---

## Scope Clarification: INFRA-01 and Git Management

INFRA-01 ("Each experiment is a git commit; improvement = keep, regression = git reset --hard") is classified as **agent-behavioral** by the project's own research document (02-RESEARCH.md, "Validation Architecture" section: "INFRA-01 (git workflow) and the full agent loop are agent-behavioral and cannot be unit tested in isolation. They are validated in Phase 4 (VALD-01, VALD-02)"). Decision D-01 establishes that the agent drives git directly via shell commands -- there is no git wrapper code to deliver in Phase 2. The Phase 2 deliverable for INFRA-01 is: train.py produces structured metrics.json that enables the agent to make keep/discard decisions, and results.tsv format is fully specified and tested so the agent can write it correctly. Both are verified below.

---

## Goal Achievement

### Observable Truths (from Plan frontmatter)

#### Plan 02-01 Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | train.py writes metrics.json on success with all sub-metrics (combined_metric, recall_at_1, recall_at_5, mean_cosine, distill_loss, arc_loss, vat_loss, sep_loss, peak_vram_mb, epochs, elapsed_seconds) | VERIFIED | train.py lines 716-731: full metrics dict with all 12 fields written via json.dump |
| 2 | train.py writes metrics.json on OOM crash with status=oom and peak_vram_mb | VERIFIED | train.py lines 757-763: OOM except block writes status=oom, peak_vram_mb, error |
| 3 | train.py writes metrics.json on general crash with status=crash and error message | VERIFIED | train.py lines 774-780: general except block writes status=crash, peak_vram_mb, error |
| 4 | train.py prints greppable summary block to stdout starting with --- separator | VERIFIED | train.py lines 734-746: print("---") followed by key:value pairs for all sub-metrics |
| 5 | train.py uses exactly EPOCHS iterations (imported from prepare.py) with no early stopping | VERIFIED | train.py line 614: for epoch in range(EPOCHS); grep confirms no "early_stop" or "patience" in train.py |
| 6 | peak_vram_mb is recorded via torch.cuda.max_memory_allocated() in both success and crash paths | VERIFIED | train.py line 707 (success), line 756 (OOM), lines 770-771 (crash) |
| 7 | metrics.json and run.log are in .gitignore | VERIFIED | .gitignore lines 26-27: metrics.json, run.log under "Experiment artifacts" comment |

#### Plan 02-02 Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 8 | Test suite validates metrics.json success schema has all required fields | VERIFIED | test_infrastructure.py::TestMetricsJsonSchema::test_success_schema_has_all_required_fields -- PASSES |
| 9 | Test suite validates metrics.json OOM schema has status=oom and peak_vram_mb | VERIFIED | test_infrastructure.py::TestMetricsJsonSchema::test_oom_schema_has_required_fields -- PASSES |
| 10 | Test suite validates metrics.json crash schema has status=crash and error | VERIFIED | test_infrastructure.py::TestMetricsJsonSchema::test_crash_schema_has_required_fields -- PASSES |
| 11 | Test suite validates results.tsv format (7 tab-separated columns with correct header) | VERIFIED | test_infrastructure.py::TestResultsTsvFormat (5 tests) -- all PASS |
| 12 | Test suite validates 3-consecutive-crash detection from results.tsv rows | VERIFIED | test_infrastructure.py::TestCrashStreakDetection (2 tests) -- all PASS |
| 13 | Test suite validates EPOCHS is imported from prepare and equals 10 | VERIFIED | test_infrastructure.py::TestTrainPyContract::test_epochs_value_is_10 -- AST-parses prepare.py, confirms EPOCHS=10 |
| 14 | All tests pass with python -m pytest tests/test_infrastructure.py -x -q | VERIFIED | 27 passed in 0.05s |

**Score: 7/7 requirements fully verified (14/14 specific truths verified)**

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `train.py` | Agent-editable training script with metrics.json output and crash handling | VERIFIED | 783 lines; contains main(), __main__ guard, three metrics.json write paths, greppable stdout |
| `.gitignore` | Git ignore rules including metrics.json and run.log | VERIFIED | Both entries present under "Experiment artifacts" section |
| `tests/test_infrastructure.py` | Unit tests for all INFRA requirements | VERIFIED | 218 lines (minimum 120 required), 27 tests |
| `tests/conftest.py` | Shared fixtures for mock metrics.json and sample results.tsv | VERIFIED | 86 lines (minimum 30 required), 6 fixtures |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| train.py main() | metrics.json | json.dump at end of main() success path | WIRED | Lines 730-731: `with open("metrics.json", "w") as f: json.dump(metrics, f, indent=2)` |
| train.py __main__ OOM except | metrics.json | json.dump in OOM except block | WIRED | Lines 762-763: confirmed via `except torch.cuda.OutOfMemoryError` at line 755 |
| train.py __main__ general except | metrics.json | json.dump in general except block | WIRED | Lines 779-780: confirmed via `except Exception as e` at line 768 |
| train.py | prepare.py EPOCHS | import statement | WIRED | Line 24: `EPOCHS` in `from prepare import (..., EPOCHS, ...)` |
| tests/test_infrastructure.py | train.py | AST parsing and import verification | WIRED | Lines 136-202: `open("train.py").read()` + ast.parse for contract verification |
| tests/test_infrastructure.py | metrics.json schema | JSON schema validation of expected fields | WIRED | Lines 16-66: required field sets validated against fixture dicts |

---

## Data-Flow Trace (Level 4)

Not applicable to this phase. Phase 2 delivers a training script and test suite -- no UI components, dashboards, or pages that render dynamic data from a data source. The metrics.json file IS the data output; its production is verified in the key link section above.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 27 infrastructure tests pass | `python -m pytest tests/test_infrastructure.py -v --tb=short` | 27 passed in 0.05s | PASS |
| EPOCHS=10 in prepare.py | `grep -n "EPOCHS" prepare.py` | Line 46: `EPOCHS = 10` | PASS |
| metrics.json written in 3+ places | `grep -c "metrics.json" train.py` | 4 (open calls at lines 730, 762, 779 + comment line 713) | PASS |
| No early stopping | `grep -c "early_stop\|patience" train.py` | 0 | PASS |
| train.py has main() | `python -c "import ast; ..."` | main() present | PASS |
| Phase commits exist in git | `git log --oneline` | 0e1d5c9, 07a5980, 615a015, 8a2a5a9 all present | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| INFRA-01 | 02-02 | Each experiment is a git commit; improvement = keep, regression = git reset --hard | SATISFIED (infrastructure layer) | Phase 2 provides the metrics.json output that enables the agent to make keep/discard decisions. Git execution is agent-behavioral per D-01 and RESEARCH.md; full verification deferred to Phase 4 (VALD-01, VALD-02) |
| INFRA-02 | 02-01, 02-02 | results.tsv logs every experiment: commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status, description | SATISFIED | metrics.json success path contains all log fields; results.tsv format validated by TestResultsTsvFormat (7-column tab-separated schema confirmed) |
| INFRA-03 | 02-01, 02-02 | OOM and runtime crashes caught, logged as "crash", git reset performed, loop continues | SATISFIED | train.py catches OutOfMemoryError and Exception separately, writes metrics.json in both paths; git reset is agent-behavioral per D-01 |
| INFRA-04 | 02-02 | After 3 consecutive crashes on the same idea, agent skips that direction | SATISFIED (data layer) | crash_streak_results_tsv fixture + TestCrashStreakDetection confirms the results.tsv data structure enables streak detection; agent decision logic deferred to Phase 3 (program.md) per RESEARCH.md |
| INFRA-05 | 02-01, 02-02 | Peak VRAM tracked and logged per experiment | SATISFIED | torch.cuda.max_memory_allocated() at lines 707, 756, 770-771; peak_vram_mb in all three metrics.json schemas; test_vram_always_present PASSES |
| INFRA-06 | 02-01, 02-02 | Run output includes decomposed sub-metrics in greppable format | SATISFIED | metrics.json contains all 8 sub-metrics; stdout summary block prints all with --- separator at lines 734-746; test_train_py_prints_summary PASSES |
| INFRA-07 | 02-01, 02-02 | Fixed budget of 10 epochs, teacher cache build time excluded | SATISFIED | EPOCHS=10 in prepare.py line 46 (immutable); train.py imports EPOCHS; no early_stop/patience in train.py; test_epochs_value_is_10 and test_no_early_stopping PASS |

All 7 INFRA requirements satisfied. No orphaned requirements (REQUIREMENTS.md traceability table maps exactly INFRA-01 through INFRA-07 to Phase 2).

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None found | -- | -- | -- |

Scanned train.py, tests/test_infrastructure.py, tests/conftest.py, .gitignore for: TODO/FIXME/HACK, placeholder returns, empty implementations, hardcoded empty data, early stopping. Zero matches found.

---

## Human Verification Required

### 1. INFRA-01 -- Git commit/reset cycle end-to-end

**Test:** With the current train.py, run: `git add train.py && git commit -m "test: verify git workflow" && python train.py > run.log 2>&1 && cat metrics.json`. Then verify: (a) metrics.json has status="success", (b) git log shows the commit, (c) run `git reset --hard HEAD~1` and confirm train.py returns to prior state.
**Expected:** Full commit-run-read-reset cycle completes without errors. metrics.json contains valid JSON with all 12 fields.
**Why human:** Requires actual GPU and training data. Cannot verify live training execution programmatically without the full environment.

### 2. INFRA-03 -- OOM crash recovery

**Test:** Modify train.py to set BATCH_SIZE = 99999 (intentional OOM), run it, confirm metrics.json has status="oom" and peak_vram_mb is populated, then run `git reset --hard HEAD~1` and confirm train.py is restored.
**Expected:** OOM triggers the OutOfMemoryError except block, metrics.json written, process exits 1, git reset restores previous train.py.
**Why human:** Requires GPU to trigger actual CUDA OOM. The code path is verified by AST, but live execution on hardware is unverifiable without the GPU environment.

---

## Gaps Summary

No gaps. All must-haves verified.

Phase 2's scope is precisely: equip train.py with structured metrics output and crash handling, and prove the schema contracts with a test suite. The git management and results.tsv writing are agent behaviors documented in program.md (Phase 3), validated live in Phase 4. This split is explicitly specified in 02-RESEARCH.md and 02-CONTEXT.md decision D-01.

---

_Verified: 2026-03-25_
_Verifier: Claude (gsd-verifier)_
