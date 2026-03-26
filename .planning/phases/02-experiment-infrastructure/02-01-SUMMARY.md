---
phase: 02-experiment-infrastructure
plan: 01
subsystem: infra
tags: [metrics-json, crash-handling, vram-tracking, experiment-harness]

# Dependency graph
requires:
  - phase: 01-codebase-split
    provides: train.py with main() function, prepare.py with constants and evaluation
provides:
  - "metrics.json output on success with all sub-metrics"
  - "metrics.json output on OOM crash with status=oom and peak_vram_mb"
  - "metrics.json output on general crash with status=crash and error message"
  - "Greppable stdout summary block with all decomposed sub-metrics"
  - "EPOCHS imported from prepare.py (immutable budget)"
  - "peak_vram_mb tracked in all exit paths"
  - ".gitignore entries for metrics.json and run.log"
affects: [02-02, 03-agent-instructions]

# Tech tracking
tech-stack:
  added: []
  patterns: [metrics-json-contract, try-except-crash-handling, greppable-stdout-summary]

key-files:
  created: []
  modified: [train.py, prepare.py, .gitignore]

key-decisions:
  - "EPOCHS moved from train.py local constant to prepare.py import -- enforces immutable budget"
  - "recall_at_5 added to metrics.json alongside recall_at_1 for richer evaluation signal"
  - "json import placed inside main() and __main__ block to keep train.py lightweight when imported as module"

patterns-established:
  - "metrics.json contract: success/oom/crash status with structured fields"
  - "Three-path exit: success writes full metrics, OOM writes status+vram, crash writes status+error"
  - "Greppable stdout: --- separator followed by key:value pairs"

requirements-completed: [INFRA-02, INFRA-03, INFRA-05, INFRA-06, INFRA-07]

# Metrics
duration: 3min
completed: 2026-03-24
---

# Phase 02 Plan 01: Metrics Output and Crash Handling Summary

**metrics.json output with success/OOM/crash paths, VRAM tracking via max_memory_allocated, greppable stdout summary with all sub-metrics, and fixed EPOCHS budget imported from prepare.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-24T20:18:37Z
- **Completed:** 2026-03-24T20:21:39Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- train.py writes metrics.json on success with all sub-metrics (combined_metric, recall_at_1, recall_at_5, mean_cosine, distill_loss, arc_loss, vat_loss, sep_loss, peak_vram_mb, epochs, elapsed_seconds)
- train.py writes metrics.json on OOM crash (status=oom) and general crash (status=crash) with peak_vram_mb and error info
- Greppable stdout summary prints all decomposed sub-metrics after --- separator
- EPOCHS constant moved to prepare.py and imported, enforcing immutable 10-epoch budget

## Task Commits

Each task was committed atomically:

1. **Task 1: Add metrics.json output and crash handling to train.py** - `0e1d5c9` (feat)
2. **Task 2: Update .gitignore for experiment artifacts** - `07a5980` (chore)

## Files Created/Modified
- `train.py` - Added metrics.json output in success/OOM/crash paths, enhanced __main__ try/except, greppable stdout summary with all sub-metrics, recall_at_5 tracking, EPOCHS import from prepare.py
- `prepare.py` - Added EPOCHS=10 constant for immutable experiment budget
- `.gitignore` - Added metrics.json and run.log entries

## Decisions Made
- Moved EPOCHS from train.py local constant to prepare.py -- ensures the experiment budget is immutable (agent cannot accidentally change it)
- Added recall_at_5 to metrics.json even though it is not in the combined metric -- provides richer evaluation signal for agent analysis
- Kept json/traceback imports inside __main__ block (not at module top) -- train.py stays lightweight when imported as module

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added EPOCHS constant to prepare.py**
- **Found during:** Task 1 (metrics.json output and crash handling)
- **Issue:** Plan required `from prepare import ... EPOCHS` but EPOCHS was not exported by prepare.py (it was a local constant in train.py)
- **Fix:** Added `EPOCHS = 10` to prepare.py constants section and updated train.py import to include EPOCHS
- **Files modified:** prepare.py, train.py
- **Verification:** `grep 'EPOCHS' train.py` shows import and usage, no local definition
- **Committed in:** 0e1d5c9 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for enforcing immutable epoch budget per INFRA-07. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- train.py now produces structured metrics.json that the agent loop (Phase 3) can read reliably
- Crash handling ensures the loop can recover from OOM and runtime errors
- .gitignore prevents experiment artifacts from polluting git history
- Ready for 02-02 (results.tsv logging and agent git workflow)

---
*Phase: 02-experiment-infrastructure*
*Completed: 2026-03-24*
