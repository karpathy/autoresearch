---
phase: 06-multi-teacher-infrastructure
plan: 02
subsystem: train.py multi-teacher training loop
tags: [multi-teacher, weighted-loss, projection-heads, backward-compatible]
dependency_graph:
  requires: [TEACHER_REGISTRY, init_teachers, build_all_teacher_caches, per-teacher-cache]
  provides: [TEACHER-constant, TEACHERS-constant, per-teacher-proj-heads, multi-teacher-distill-loss]
  affects: [agent experiment loop (teacher selection as tunable parameter)]
tech_stack:
  added: []
  patterns: [per-teacher ModuleDict projection heads, weighted cosine distillation, forward_backbone for raw features]
key_files:
  created: []
  modified: [train.py]
decisions:
  - "LCNet replaces FrozenBackboneWithHead: adapted plan's proj_heads concept to LCNet architecture"
  - "forward_backbone returns 1280d summary features for per-teacher projection in multi-teacher mode"
  - "First teacher's projection head assigned to self.proj for encode() backward compatibility"
  - "SWA evaluation uses first/default teacher for mean_cosine (trust boundary preserved)"
metrics:
  duration: 4min
  completed: "2026-03-25T13:15:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 06 Plan 02: Multi-Teacher Constants and Training Loop Summary

TEACHER/TEACHERS constants with per-teacher projection heads (ModuleDict) and weighted distillation loss in train.py, backward-compatible single-teacher default mode.

## Changes Made

### Task 1: Add TEACHER/TEACHERS constants and update imports
**Commit:** d89478a

- Added TEACHER_REGISTRY, init_teachers, build_all_teacher_caches imports from prepare.py
- Added TEACHER="trendyol_onnx" constant (single-teacher default, backward compatible)
- Added TEACHERS: dict[str, float] | None = None constant (multi-teacher mode, overrides TEACHER)
- Removed TEACHER_CACHE_DIR constant (cache dirs now come from TEACHER_REGISTRY per D-08)
- Added _get_active_teachers() helper resolving TEACHER/TEACHERS into names + weights
- Added teacher_dims parameter to LCNet.__init__ for multi-teacher projection heads
- Added proj_heads: nn.ModuleDict mapping teacher names to per-teacher Linear+BN projection heads (per D-06)
- Added forward_backbone() method returning raw 1280d features before any projection
- self.proj points to first teacher's head in multi-teacher mode (encode() contract preserved)

### Task 2: Wire multi-teacher training loop and cache building into main()
**Commit:** 94b6093

- Replaced run_train_epoch signature: `teacher: TrendyolEmbedder` -> `teachers: dict[str, object], teacher_weights: dict[str, float]`
- Multi-teacher distillation loss: per-teacher forward_backbone -> proj_heads[t_name] -> cosine loss, weighted sum (per D-07)
- main() calls build_all_teacher_caches for VRAM-safe sequential caching (per D-11)
- main() calls init_teachers for multi-teacher initialization
- Model creation passes teacher_dims when len(teacher_names) > 1
- Optimizer includes all proj_heads parameters in multi-teacher mode
- SWA evaluation uses first/default teacher for mean_cosine computation
- Removed unused imports: TrendyolEmbedder, init_teacher, DEFAULT_TEACHER_CACHE_DIR
- Evaluation path (run_retrieval_eval, compute_combined_metric) completely unchanged -- trust boundary intact

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan references FrozenBackboneWithHead, actual class is LCNet**
- **Found during:** Task 1
- **Issue:** Plan was written against an older codebase version with FrozenBackboneWithHead class. Current code uses custom LCNet backbone.
- **Fix:** Adapted all plan instructions to LCNet architecture: teacher_dims param on LCNet.__init__, proj_heads as ModuleDict, forward_backbone returning 1280d summary from forward_features.
- **Files modified:** train.py
- **Commit:** d89478a

## Decisions Made

1. **LCNet adaptation**: Plan's FrozenBackboneWithHead concept mapped to LCNet -- proj_heads created inside LCNet.__init__ when teacher_dims provided, forward_backbone exposes raw 1280d features
2. **Projection head structure**: Each teacher gets nn.Sequential(Linear(1280, t_dim), BatchNorm1d(t_dim)) matching single-teacher proj structure
3. **encode() backward compatibility**: self.proj assigned to first teacher's projection head so encode() returns (B, 256) in both modes
4. **SWA mean_cosine**: Uses first/default teacher (teacher_names[0]) for evaluation consistency with single-teacher baseline

## Known Stubs

None -- all functionality is fully wired. DINOv3Teacher and RADIOTeacher stubs exist in prepare.py (from Plan 01) but are not activated by default (TEACHER="trendyol_onnx", TEACHERS=None).

## Verification Results

- TEACHER="trendyol_onnx" constant exists: PASSED
- TEACHERS=None constant exists: PASSED
- _get_active_teachers() helper works: PASSED
- TEACHER_REGISTRY imported: PASSED
- init_teachers imported: PASSED
- build_all_teacher_caches imported: PASSED
- proj_heads ModuleDict works in multi-teacher mode: PASSED
- forward_backbone returns (B, 1280): PASSED
- encode() returns (B, 256) in both modes: PASSED
- run_train_epoch has teachers/teacher_weights params (AST check): PASSED
- TEACHER_CACHE_DIR constant removed (count=0): PASSED
- teacher_weights referenced in code (count=5): PASSED
- run_retrieval_eval preserved: PASSED
- compute_combined_metric preserved: PASSED
- train.py syntax valid: PASSED
- prepare.py syntax valid: PASSED
