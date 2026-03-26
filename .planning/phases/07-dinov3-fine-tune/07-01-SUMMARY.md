---
phase: 07-dinov3-fine-tune
plan: 01
subsystem: training
tags: [dinov3, lora, peft, contrastive-learning, infonce, vit, fine-tuning]

# Dependency graph
requires:
  - phase: 06-multi-teacher-infrastructure
    provides: "Multi-teacher cache infrastructure and teacher interface pattern"
provides:
  - "dino_finetune/prepare_dino.py -- immutable DINOv3 ViT-H+ infrastructure (model, data, eval, adapter save/load)"
  - "dino_finetune/train_dino.py -- agent-editable LoRA training with InfoNCE loss"
affects: [07-dinov3-fine-tune, 08-radio-integration]

# Tech tracking
tech-stack:
  added: [peft]
  patterns: [autoresearch-sub-project, lora-fine-tuning, supervised-infonce]

key-files:
  created:
    - dino_finetune/prepare_dino.py
    - dino_finetune/train_dino.py
  modified: []

key-decisions:
  - "DINOv3 ViT-H+ (840M, 1280d) chosen over DINOv2 ViT-g (1.1B, 1536d) per research: better retrieval perf, cleaner PEFT integration"
  - "Gradient accumulation 16x with batch_size=8 for effective batch=128 (contrastive learning needs large batches)"
  - "CLS token at index 0 of last_hidden_state (register tokens at indices 1-4 are skipped)"

patterns-established:
  - "Autoresearch sub-project pattern: prepare_X.py (immutable) + train_X.py (agent-editable)"
  - "PEFT LoRA injection via get_peft_model with separate q_proj/v_proj targeting"

requirements-completed: [DINO3-01, DINO3-02]

# Metrics
duration: 3min
completed: 2026-03-25
---

# Phase 7 Plan 1: DINOv3 Fine-tune Infrastructure Summary

**DINOv3 ViT-H+ (840M) LoRA fine-tuning sub-project with prepare_dino.py (immutable infra) and train_dino.py (agent-editable InfoNCE training)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-25T13:20:45Z
- **Completed:** 2026-03-25T13:23:56Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Created prepare_dino.py with 9 functions: model loading (bf16+SDPA), data pipeline, CLS embedding extraction, recall@1+mean_cosine evaluation, adapter save/load, VRAM smoke test
- Created train_dino.py with all agent-tunable constants at module level: LoRA config (r=16, alpha=32, q_proj/v_proj), InfoNCE loss, gradient accumulation, cosine LR schedule
- Both files follow the proven autoresearch pattern from the main project (prepare.py/train.py)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create prepare_dino.py** - `895a1ba` (feat)
2. **Task 2: Create train_dino.py** - `982b626` (feat)

## Files Created/Modified
- `dino_finetune/prepare_dino.py` - Immutable infrastructure: DINOv3 model loading, data pipeline, evaluation, adapter save/load
- `dino_finetune/train_dino.py` - Agent-editable training: LoRA config, InfoNCE loss, optimizer, training loop with gradient accumulation

## Decisions Made
- DINOv3 ViT-H+ (840M) selected as base model. Research confirmed DINOv3 ViT-g does not exist; ViT-H+ outperforms DINOv2 ViT-g on retrieval (+10.9 GAP) despite fewer params.
- Batch size 8 with 16x gradient accumulation for effective batch 128. Physical batch safe for 24GB VRAM.
- CLS token extracted at index 0 of last_hidden_state (documented register token offset at indices 1-4).
- GradScaler used with bf16 autocast for mixed-precision stability.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. `peft` library must be installed before running (`pip install peft`).

## Known Stubs
None - both files are fully implemented with all functions and constants.

## Next Phase Readiness
- Ready for Plan 02 (program_dino.md agent instructions) and Plan 03 (DINOv3FTTeacher integration)
- VRAM profiling recommended on first real run to validate batch_size=8 estimate

---
*Phase: 07-dinov3-fine-tune*
*Completed: 2026-03-25*
