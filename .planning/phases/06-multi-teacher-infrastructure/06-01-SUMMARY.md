---
phase: 06-multi-teacher-infrastructure
plan: 01
subsystem: prepare.py teacher infrastructure
tags: [multi-teacher, registry, cache, dinov2-fix]
dependency_graph:
  requires: []
  provides: [TEACHER_REGISTRY, init_teachers, build_all_teacher_caches, per-teacher-cache]
  affects: [train.py multi-teacher mode (06-02)]
tech_stack:
  added: []
  patterns: [registry pattern, per-teacher memory cache, sequential GPU cleanup]
key_files:
  created: []
  modified: [prepare.py]
decisions:
  - "DINOv2 CLS token fix: out.last_hidden_state[:, 0, :] for (B, 256) shape"
  - "Per-teacher memory cache keyed by teacher_name prevents cross-teacher collision"
  - "Existing Trendyol cache reused by writing metadata.json from .npy file count"
  - "Stub classes raise NotImplementedError for Phase 7/8 implementation"
metrics:
  duration: 2min
  completed: "2026-03-25T13:08:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 06 Plan 01: Multi-Teacher Registry and Cache Infrastructure Summary

Multi-teacher registry with 5 teachers, per-teacher cache isolation, DINOv2 CLS token bug fix, and VRAM-safe sequential cache building in prepare.py.

## Changes Made

### Task 1: TEACHER_REGISTRY, stub teachers, DINOv2 fix
**Commit:** 49700bb

- Fixed DINOv2Teacher.encode_batch to extract CLS token (`[:, 0, :]`) returning 1D (256,) arrays instead of 3D tensors
- Added DINOv3Teacher stub class (NotImplementedError for Phase 7) with embedding_dim=256
- Added RADIOTeacher stub class (NotImplementedError for Phase 8) with embedding_dim=None
- Added TEACHER_REGISTRY dict with 5 entries: trendyol_onnx, dinov2, dinov3_ft, radio_so400m, radio_h
- Added init_teachers() function for multi-teacher initialization by name
- Preserved backward-compatible init_teacher() function

### Task 2: Per-teacher cache infrastructure
**Commit:** 260ccf6

- Replaced global `_TEACHER_MEM_CACHE` with per-teacher `_TEACHER_MEM_CACHES` dict keyed by teacher_name
- Added `_write_cache_metadata()` writing metadata.json with teacher_name, embedding_dim, num_samples, cache_date
- Added `_is_cache_valid()` checking metadata.json against expected dim and sample count
- Added `build_all_teacher_caches()` processing teachers sequentially with explicit `gc.collect()` + `torch.cuda.empty_cache()` between teachers
- Existing Trendyol cache (no metadata.json) preserved by counting .npy files and writing metadata
- Updated `load_teacher_embeddings()` with optional `teacher_name` param (defaults to "default" for backward compat)

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

1. **DINOv2 CLS token extraction**: Used `[:, 0, :]` indexing plus `.flatten()` to guarantee 1D output regardless of model internals
2. **Per-teacher cache key**: teacher_name string used as outer dict key in _TEACHER_MEM_CACHES, with "default" fallback for backward compat
3. **Trendyol cache preservation**: Counts existing .npy files to write metadata rather than triggering expensive rebuild
4. **Stub init_kwargs forwarding**: Stubs accept **kwargs in __init__ so registry init_kwargs pass through before raising NotImplementedError

## Known Stubs

| File | Location | Stub | Resolution |
|------|----------|------|------------|
| prepare.py | DINOv3Teacher class | NotImplementedError | Phase 7 |
| prepare.py | RADIOTeacher class | NotImplementedError | Phase 8 |

These stubs are intentional placeholders in the registry -- they define the interface contract but defer implementation to their respective phases. The registry correctly includes them so train.py can reference teacher names consistently.

## Verification Results

- TEACHER_REGISTRY has 5 entries with correct names: PASSED
- DINOv2Teacher uses CLS token (last_hidden_state[:, 0, :]): PASSED
- Per-teacher _TEACHER_MEM_CACHES is dict: PASSED
- build_all_teacher_caches, _write_cache_metadata, _is_cache_valid importable: PASSED
- Old global _TEACHER_MEM_CACHE removed: PASSED
- grep TEACHER_REGISTRY count > 0: PASSED (7 occurrences)
