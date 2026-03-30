---
phase: quick
plan: 260330-e2u
subsystem: dino-finetune
tags: [early-stopping, training, collapse-guard]
dependency_graph:
  requires: []
  provides: [guarded-early-stopping, collapse-counter]
  affects: [dino-training-loop]
tech_stack:
  added: []
  patterns: [consecutive-collapse-detection, guard-before-save]
key_files:
  modified:
    - dino_finetune/train_dino.py
decisions:
  - "3 consecutive collapsed epochs as threshold (not 1) to allow transient recovery"
  - "Patience counter excluded during collapse (collapsed combined metric is meaningless)"
  - "Recall drop check remains active during collapse (recall drop is still meaningful)"
metrics:
  duration_seconds: 59
  completed: 2026-03-30T02:11:25Z
  tasks_completed: 1
  tasks_total: 1
---

# Quick Task 260330-e2u: Fix Early Stopping to Prevent Saving Collapsed Adapters

Guarded early stopping with consecutive collapse counter so collapsed adapters are never saved and training can recover from transient cosine collapse.

## Changes

### Task 1: Restructure eval block with collapse guard and consecutive counter

**Commit:** `8a436a7`

Four changes to `train_dino.py`:

1. **New constant** `EARLY_STOP_COLLAPSE_CONSECUTIVE = 3` -- training tolerates up to 2 consecutive collapsed epochs before stopping.

2. **New counter** `collapse_counter = 0` initialized alongside `patience_counter` and `best_combined`.

3. **Updated config log** to include `collapse_consecutive=` for experiment traceability.

4. **Restructured eval block** with new ordering:
   - Step 1: Collapse check runs FIRST (before any save decision)
   - Step 2: If collapsed -- increment counter, log warning, skip save entirely; stop if 3 consecutive
   - Step 3: If not collapsed -- reset counter, then normal improved/save/patience logic
   - Step 4: Recall drop check runs regardless of collapse state

Key invariant: `save_adapter(model)` is only reachable inside the `else` (not-collapsed) branch, making it impossible to persist a collapsed adapter.

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED
