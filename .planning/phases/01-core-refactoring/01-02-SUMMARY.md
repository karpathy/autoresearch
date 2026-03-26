---
phase: 01-core-refactoring
plan: 02
subsystem: training-pipeline
tags: [timm, arcface, distillation, vat, reid, lcnet050, sgd, cosine-annealing]

# Dependency graph
requires:
  - phase: 01-01
    provides: "prepare.py: immutable data/teacher/evaluation infrastructure"
provides:
  - "train.py: agent-editable training script with module-level constants"
  - "FrozenBackboneWithHead with .encode(images) -> Tensor[B, 256] L2-normalized"
  - "ProjectionHead, ArcMarginProduct, RandomQualityDegradation classes"
  - "run_train_epoch function for full training loop"
  - "vat_embedding_loss for VAT regularization"
  - "Greppable summary block (combined_metric, recall@1, mean_cosine, peak_vram_mb)"
  - "OOM catch handler printing status: OOM"
affects: [01-03, 02-infrastructure, 03-agent]

# Tech tracking
tech-stack:
  added: []
  patterns: [module-level-constants-as-config, differential-lr-optimizer, arcface-phaseout-schedule]

key-files:
  created: [train.py]
  modified: []

key-decisions:
  - "EPOCHS defaults to 10 (not 80 from monolith) per fixed budget requirement"
  - "UNFREEZE_EPOCH defaults to 0 (backbone unfrozen from start per D-02)"
  - "matplotlib imports deferred inside save_batch_visualization to avoid headless server issues"
  - "Removed comment words matching forbidden patterns (patience, early_stop) to pass strict grep checks"

patterns-established:
  - "All hyperparameters are module-level UPPER_SNAKE_CASE constants (no argparse, no config files)"
  - "train.py imports from prepare.py, never the reverse"
  - "model.encode(images) returns Tensor[B, 256] L2-normalized as the evaluation interface"
  - "Greppable summary block printed to stdout with --- separator"

requirements-completed: [REFAC-01, REFAC-05, REFAC-06]

# Metrics
duration: 5min
completed: 2026-03-25
---

# Phase 01 Plan 02: Create train.py Summary

**Extracted all agent-editable ReID training components (model, losses, augmentations, training loop) from monolith into train.py with 26 module-level constants replacing argparse**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-24T20:02:39Z
- **Completed:** 2026-03-24T20:08:29Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created train.py (722 lines) with all agent-editable components extracted verbatim from finetune_trendyol_arcface3.py
- Converted all 26 argparse arguments to module-level UPPER_SNAKE_CASE constants
- FrozenBackboneWithHead.encode() verified to return Tensor[B, 256] L2-normalized
- Greppable summary block prints combined_metric, recall@1, mean_cosine, peak_vram_mb
- OOM handler catches torch.cuda.OutOfMemoryError and prints status: OOM
- Zero argparse, zero early stopping, zero checkpoint saving, zero ONNX export

## Task Commits

Each task was committed atomically:

1. **Task 1: Create train.py with all agent-editable components** - `a45bf56` (feat)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `train.py` - Agent-editable training script (722 lines): module-level constants, ProjectionHead, ArcMarginProduct, FrozenBackboneWithHead, RandomQualityDegradation, build_train_transform, save_batch_visualization, vat_embedding_loss, EpochStats, run_train_epoch, main()

## Decisions Made
- Set EPOCHS=10 (not 80) per fixed budget requirement for autoresearch experiments
- Set UNFREEZE_EPOCH=0 (not 5) so backbone is unfrozen from start per D-02
- Deferred matplotlib imports inside save_batch_visualization function to avoid import issues on headless servers
- Included dynamic unfreeze logic in main() even though default is epoch 0, so agent can experiment with delayed unfreezing

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed forbidden pattern words from comment**
- **Found during:** Task 1 verification
- **Issue:** Comment "NO early stopping, NO patience" contained words that fail the strict grep acceptance criteria check
- **Fix:** Changed comment to "fixed epoch budget" which conveys the same meaning without triggering the grep
- **Files modified:** train.py
- **Verification:** `grep -cE "patience|early_stop|no_improve" train.py` returns 0
- **Committed in:** a45bf56 (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial comment wording change to pass verification. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- train.py is complete and importable, ready for integration testing (Plan 01-03)
- prepare.py + train.py split is now complete: `from prepare import` bridge verified working
- FrozenBackboneWithHead.encode() contract verified: Tensor[B, 256] L2-normalized
- All module-level constants ready for agent editing in Phase 3

## Self-Check: PASSED

- train.py: FOUND
- Commit a45bf56: FOUND
- SUMMARY.md: FOUND

---
*Phase: 01-core-refactoring*
*Completed: 2026-03-25*
