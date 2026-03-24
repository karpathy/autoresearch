---
phase: 01-core-refactoring
plan: 01
subsystem: data-pipeline
tags: [onnxruntime, timm, torchvision, teacher-cache, retrieval-eval, reid]

# Dependency graph
requires: []
provides:
  - "prepare.py: immutable data/teacher/evaluation infrastructure"
  - "TrendyolEmbedder ONNX teacher with disk+memory cache"
  - "CombinedDistillDataset and CombinedArcFaceDataset"
  - "run_retrieval_eval with duck-typed model parameter"
  - "compute_combined_metric (0.5*recall@1 + 0.5*mean_cosine)"
  - "Builder functions: build_distill_dataset, build_arcface_dataset, build_val_dataset"
affects: [01-02, 01-03, 02-infrastructure, 03-agent]

# Tech tracking
tech-stack:
  added: []
  patterns: [verbatim-extraction, builder-functions-for-DI, duck-typed-model-interface]

key-files:
  created: [prepare.py]
  modified: []

key-decisions:
  - "Used second PadToSquare (TF.pad) not first (broken tf.pad) per Pitfall 1"
  - "DINOv2Teacher included but not instantiated per D-03"
  - "transformers import deferred to function scope to avoid top-level dependency"
  - "Builder functions accept transform and quality_degradation as params to avoid circular imports per Pitfall 4"

patterns-established:
  - "prepare.py is the trust boundary: agent cannot edit evaluation or data loading"
  - "Duck-typed model interface: any object with .encode(images) -> Tensor[B, 256]"
  - "Builder functions own data paths; train.py owns augmentations"

requirements-completed: [REFAC-01, REFAC-02, REFAC-03, REFAC-04, REFAC-07]

# Metrics
duration: 6min
completed: 2026-03-25
---

# Phase 01 Plan 01: Create prepare.py Summary

**Extracted all immutable ReID infrastructure (teacher, datasets, evaluation, caching, metrics) from monolith into prepare.py with builder functions for dependency injection**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-24T19:54:00Z
- **Completed:** 2026-03-24T20:00:12Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created prepare.py (864 lines) with all immutable infrastructure extracted verbatim from finetune_trendyol_arcface3.py
- All 27 required exports verified present and importable
- compute_combined_metric formula verified: 0.5 * recall@1 + 0.5 * mean_cosine
- Zero argparse, zero training code, zero imports from train.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Create prepare.py with all immutable infrastructure** - `f178dd9` (feat)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `prepare.py` - Immutable data/teacher/evaluation infrastructure (864 lines): constants, PadToSquare, set_seed, TrendyolEmbedder, DINOv2Teacher, DistillImageFolder, SampledImageFolder, CombinedDistillDataset, CombinedArcFaceDataset, collation functions, teacher cache, retrieval evaluation, metric computation, and builder functions

## Decisions Made
- Used second PadToSquare definition (line 226 with TF.pad) not first (line 57 with broken tf.pad)
- Deferred transformers import inside _patch_transformers_compat and DINOv2Teacher to avoid top-level import cost when only ONNX teacher is used
- Builder functions accept transform/quality_degradation as parameters, keeping augmentation control in train.py while data paths stay immutable in prepare.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing Python dependencies in venv**
- **Found during:** Task 1 verification
- **Issue:** loguru, timm, and other dependencies were not installed in the project .venv
- **Fix:** Installed via `uv pip install -r requirements.txt`
- **Files modified:** None (venv packages only)
- **Verification:** `python -c "import prepare"` succeeds
- **Committed in:** Not committed (runtime environment, not source code)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for verification to run. No scope creep.

## Issues Encountered
None beyond the dependency installation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- prepare.py is complete and importable, ready for train.py (Plan 01-02) to import from
- All builder functions, evaluation, and teacher infrastructure are in place
- DINOv2Teacher kept for future use but not instantiated in v1

## Self-Check: PASSED

- prepare.py: FOUND
- Commit f178dd9: FOUND
- SUMMARY.md: FOUND

---
*Phase: 01-core-refactoring*
*Completed: 2026-03-25*
