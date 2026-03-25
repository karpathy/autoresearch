---
phase: 08-radio-integration
plan: 01
subsystem: teacher
tags: [radio, c-radio, nvidia, distillation, adaptor, summary-cache, torch-hub]

# Dependency graph
requires:
  - phase: 06-multi-teacher
    provides: "Teacher registry, per-teacher cache dirs, load_teacher_embeddings"
  - phase: 05-ssl-lcnet
    provides: "Custom LCNet with forward_backbone, proj_heads ModuleDict"
provides:
  - "RADIOTeacher class loading C-RADIOv4 models from local RADIO/ clone"
  - "RADIO_VERSION_MAP constant for SO400M and H variants"
  - "Per-adaptor summary caching with metadata.json"
  - "load_radio_teacher_embeddings for train.py consumption"
  - "RADIO_VARIANT and RADIO_ADAPTORS agent-tunable constants in train.py"
  - "Per-adaptor projection heads (nn.ModuleDict) in train.py"
affects: [08-radio-integration, train.py, prepare.py]

# Tech tracking
tech-stack:
  added: [torch.hub (local source), C-RADIOv4]
  patterns: [runtime dim discovery via dummy forward pass, per-adaptor projection heads]

key-files:
  created: []
  modified:
    - prepare.py
    - train.py

key-decisions:
  - "RADIO backbone summary_dim=2304 for SO400M (embed_dim=1152 * 2 summary_idxs), discovered at runtime"
  - "RADIO teachers use pre-cached embeddings only during training (no online inference needed)"
  - "Per-adaptor distillation loss equally weighted within a RADIO teacher's weight allocation"

patterns-established:
  - "Runtime dim discovery: always use dummy forward pass, never hardcode RADIO dimensions"
  - "Per-adaptor caching: separate cache dir per adaptor with metadata.json"

requirements-completed: [RADIO-01, RADIO-02, RADIO-03, RADIO-06]

# Metrics
duration: 5min
completed: 2026-03-25
---

# Phase 8 Plan 1: RADIO Summary Integration Summary

**RADIOTeacher class with adaptor-aware summary caching and per-adaptor projection heads wired into distillation loop**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-25T13:37:56Z
- **Completed:** 2026-03-25T13:43:54Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- RADIOTeacher loads C-RADIOv4-SO400M/H from local RADIO/ clone via torch.hub.load with adaptor selection
- Summary features cached as per-sample .npy files in per-adaptor directories with metadata.json
- train.py has RADIO_VARIANT and RADIO_ADAPTORS as agent-tunable constants with per-adaptor projection heads
- Feature dimensions discovered at runtime (SO400M backbone=2304d), no hardcoded dims

## Task Commits

Each task was committed atomically:

1. **Task 1: Add RADIOTeacher class to prepare.py with adaptor-aware summary caching** - `4d97acc` (feat)
2. **Task 2: Wire RADIO constants and summary loading into train.py** - `b8cbbc5` (feat)

## Files Created/Modified
- `prepare.py` - RADIOTeacher class, RADIO_VERSION_MAP, build_radio_summary_cache, load_radio_teacher_embeddings
- `train.py` - RADIO_VARIANT/RADIO_ADAPTORS constants, _load_radio_metadata helper, per-adaptor projection heads, RADIO distillation loss wiring

## Decisions Made
- SO400M backbone summary_dim is 2304 (not 1152 or 3456) -- 1152 embed_dim * 2 summary_idxs. Always determined at runtime via dummy forward pass.
- RADIO teachers do not need online inference during training -- all embeddings are pre-cached, saving significant VRAM and compute.
- Per-adaptor distillation distributes a RADIO teacher's weight equally across all active adaptors (e.g., if RADIO weight=0.5 and 2 adaptors, each adaptor gets 0.25 weight).
- RADIO init_kwargs changed from `version=` to `variant=` to match the short-name interface (so400m/h).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Worktree was on wrong branch (upstream karpathy autoresearch, not ReID)**
- **Found during:** Initial file read
- **Issue:** Worktree branch was based on master (GPT language model codebase) instead of gsd/v2.0-expanded-search-space (ReID codebase)
- **Fix:** `git reset --hard gsd/v2.0-expanded-search-space` to get correct source files
- **Files modified:** All (branch reset)
- **Verification:** prepare.py contains ReID code with DINOv2Teacher, TEACHER_REGISTRY

**2. [Rule 3 - Blocking] RADIO/ directory not present in worktree**
- **Found during:** Task 1 preparation
- **Issue:** RADIO/ is a cloned external repo not tracked by git, missing from worktree
- **Fix:** Created symlink from worktree to main repo's RADIO/ directory
- **Verification:** `ls RADIO/hubconf.py` succeeds

---

**Total deviations:** 2 auto-fixed (both blocking)
**Impact on plan:** Both fixes were environment setup issues, not code changes. No scope creep.

## Issues Encountered
- RADIO checkpoint downloads ~1.12GB on first load (cached for subsequent runs)
- Minor UserWarning about unexpected keys in state dict (patch_generator._vis_cond.norm_mean/std) -- harmless

## Known Stubs
None -- all RADIO functionality is fully wired. The existing Phase 8 stub in the teacher registry has been replaced with a working implementation.

## Next Phase Readiness
- RADIO summary distillation is complete and ready for agent experimentation
- Plan 02 (spatial distillation) can build on this foundation
- All 3 adaptors (backbone, dino_v3, siglip2-g) are supported

---
*Phase: 08-radio-integration*
*Completed: 2026-03-25*
