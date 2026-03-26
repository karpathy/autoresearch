---
phase: 08-radio-integration
plan: 02
subsystem: spatial-distillation
tags: [radio, spatial, distillation, conv1x1, on-the-fly, adapter]

# Dependency graph
requires:
  - phase: 08-radio-integration
    plan: 01
    provides: "RADIOTeacher class with adaptor-aware summary caching"
  - phase: 05-ssl-lcnet
    provides: "Custom LCNet with forward_features returning (spatial, summary)"
provides:
  - "RADIOTeacher.extract_spatial_batch() for on-the-fly spatial feature extraction"
  - "RADIOTeacher.get_spatial_info() for spatial grid dimensions"
  - "SpatialAdapter (Conv1x1 + BN) for student-to-teacher spatial alignment"
  - "spatial_distillation_loss function with bilinear interpolation and L2 normalization"
  - "SPATIAL_DISTILL_WEIGHT agent-tunable constant (default 0.0 = disabled)"
affects: [train.py, prepare.py]

# Tech tracking
tech-stack:
  added: [on-the-fly RADIO spatial inference]
  patterns: [ImageNet un-normalization for RADIO input, L2-normalized spatial MSE loss]

key-files:
  created: []
  modified:
    - prepare.py
    - train.py

key-decisions:
  - "On-the-fly spatial extraction instead of disk caching (417GB per adaptor vs 329GB available)"
  - "L2-normalize both student and teacher spatial features before MSE loss to handle scale mismatch"
  - "Un-normalize ImageNet images to [0,1] for RADIO input (reverse mean/std) rather than storing raw images"
  - "Student spatial channels from last LCNet stage (make_divisible(512*scale)) discovered programmatically"

patterns-established:
  - "Reverse ImageNet normalization pattern: images_01 = images * std + mean for RADIO input"
  - "Spatial distillation conditional on SPATIAL_DISTILL_WEIGHT > 0 (zero-cost when disabled)"

requirements-completed: [RADIO-04, RADIO-05]

# Metrics
duration: 5min
completed: 2026-03-25
---

# Phase 8 Plan 2: Spatial Distillation Summary

**On-the-fly RADIO spatial distillation with Conv1x1+BN adapter, bilinear interpolation, and L2-normalized MSE loss**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-25T13:48:06Z
- **Completed:** 2026-03-25T13:53:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- RADIOTeacher.extract_spatial_batch() runs RADIO inference on-the-fly per training batch, returning [B, N_tokens, D] spatial features
- RADIOTeacher.get_spatial_info() provides spatial grid dimensions for the loss function
- SpatialAdapter (Conv1x1 + BatchNorm2d) projects student pre-GAP spatial channels to RADIO spatial dim
- spatial_distillation_loss bilinear-interpolates student spatial to teacher grid, L2-normalizes both, computes MSE
- SPATIAL_DISTILL_WEIGHT defaults to 0.0 (disabled) -- agent sets positive value to enable
- Training loop conditionally computes spatial loss only when weight > 0 (zero overhead when disabled)
- Images reverse-normalized from ImageNet mean/std to [0,1] for RADIO input
- Spatial adapter parameters added to optimizer (and optimizer rebuild at unfreeze)
- Spatial loss logged as separate sub-metric in progress bar and epoch summary

## Task Commits

Each task was committed atomically:

1. **Task 1: Add on-the-fly spatial feature extraction to RADIOTeacher in prepare.py** - `c0de316` (feat)
2. **Task 2: Add SpatialAdapter, spatial distillation loss, and SPATIAL_DISTILL_WEIGHT to train.py** - `530af9f` (feat)

## Files Created/Modified
- `prepare.py` - RADIOTeacher.extract_spatial_batch(), RADIOTeacher.get_spatial_info()
- `train.py` - SPATIAL_DISTILL_WEIGHT constant, SpatialAdapter class, spatial_distillation_loss function, training loop wiring, EpochStats.spatial_loss field, metrics.json output

## Decisions Made
- On-the-fly spatial extraction chosen over disk caching because spatial features for 495k samples at 196 tokens x 1152d float32 = 417GB per adaptor, exceeding 329GB available disk. On-the-fly trades GPU compute for zero disk usage.
- L2-normalization applied to both student and teacher spatial feature maps before MSE loss. This handles the scale mismatch between student LCNet features and RADIO teacher features without requiring explicit scale tuning.
- Reverse ImageNet normalization (images * std + mean) used to recover [0,1] images for RADIO input. This is simpler than modifying the dataset/collate to return raw images and avoids storing duplicate tensors.
- Student spatial channels computed programmatically as make_divisible(512 * LCNET_SCALE) to match the last LCNet stage output, avoiding hardcoded values that would break if scale changes.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs
None -- all spatial distillation functionality is fully wired. When SPATIAL_DISTILL_WEIGHT = 0.0 (default), the spatial distillation path is completely inactive with zero compute overhead. When set to a positive value, the full pipeline (RADIO inference, spatial adapter, loss computation) executes.

---
*Phase: 08-radio-integration*
*Completed: 2026-03-25*
