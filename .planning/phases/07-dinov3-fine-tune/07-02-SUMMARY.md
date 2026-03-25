---
phase: 07-dinov3-fine-tune
plan: 02
subsystem: training
tags: [dinov3, lora, agent-instructions, autoresearch, contrastive-learning]

# Dependency graph
requires:
  - phase: 07-dinov3-fine-tune
    plan: 01
    provides: "prepare_dino.py + train_dino.py infrastructure files"
provides:
  - "dino_finetune/program_dino.md -- agent instructions for autonomous DINOv3 LoRA fine-tuning experiments"
  - "Complete autoresearch sub-project pattern: prepare + train + program (per D-07)"
affects: [07-dinov3-fine-tune, 08-radio-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [autoresearch-sub-project-complete, agent-instructions-pattern]

key-files:
  created:
    - dino_finetune/program_dino.md
  modified: []

key-decisions:
  - "Temperature tuning prioritized as Priority 1 in experiment strategy -- most impactful parameter for contrastive learning"
  - "BATCH_SIZE hard-capped at 16 with gradient accumulation for effective batch size scaling"
  - "LoRA target module expansion documented as Priority 5 (after fundamentals are tuned)"

patterns-established:
  - "Autoresearch sub-project complete: prepare_X.py (immutable) + train_X.py (agent-editable) + program_X.md (agent instructions)"

requirements-completed: [DINO3-02]

# Metrics
duration: 3min
completed: 2026-03-25
---

# Phase 7 Plan 2: DINOv3 Agent Instructions Summary

**program_dino.md with prioritized experiment strategy (temperature > LR > LoRA rank > batch size > target modules > augmentation) completing the autoresearch sub-project pattern**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-25T13:26:21Z
- **Completed:** 2026-03-25T13:28:52Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Created program_dino.md with all 7 required sections: overview, hard constraints, search space, experiment strategy, workflow, pitfalls, output format
- Documented all 10+ tunable constants from train_dino.py with default values, safe ranges, and expected effects
- Prioritized experiment strategy across 6 priority levels for systematic hyperparameter exploration
- Completed the autoresearch sub-project pattern: prepare_dino.py + train_dino.py + program_dino.md (per D-07)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create program_dino.md** - `a4d39c9` (feat)

## Files Created/Modified
- `dino_finetune/program_dino.md` - Agent instructions for autonomous DINOv3 LoRA fine-tuning with full search space, prioritized strategy, and workflow documentation

## Decisions Made
- Temperature tuning as Priority 1: most impactful parameter in contrastive learning, small changes produce large metric swings
- Hard cap BATCH_SIZE at 16 (not just "watch VRAM"): explicit constraint prevents OOM from physical batch size, agents use gradient accumulation instead
- LoRA target module expansion as Priority 5: fundamental hyperparameters (temperature, LR, rank) should be tuned before expanding adapter capacity

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - program_dino.md is complete with all required sections and content.

## Next Phase Readiness
- DINOv3 autoresearch sub-project is fully complete (prepare + train + program)
- Ready for Plan 03 (DINOv3FTTeacher integration into main autoresearch pipeline)
- An agent can now start autonomous fine-tuning experiments immediately

---
*Phase: 07-dinov3-fine-tune*
*Completed: 2026-03-25*
