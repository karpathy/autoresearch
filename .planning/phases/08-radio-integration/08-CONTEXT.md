# Phase 8: RADIO Integration - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Integrate RADIO models as teachers with adaptor selection (backbone, dino_v3, siglip2-g), summary + spatial feature caching, and spatial distillation loss. All agent-tunable via constants.

</domain>

<decisions>
## Implementation Decisions

### RADIO Adaptor Strategy
- **D-01:** Cache ALL 3 adaptor outputs (backbone, dino_v3, siglip2-g) for each RADIO variant. Disk is cheap (~10GB per adaptor), re-running RADIO inference is expensive.
- **D-02:** Summary features cached as .npy files per sample, native dimension (1152d for SO400M, 1280d for H). Same pattern as other teachers.
- **D-03:** Spatial features cached as memory-mapped .npy files. Format: `{sample_id}_spatial.npy` shape `(N, D)` where N=num_spatial_tokens, D=feature_dim. Memory-mapped to avoid loading entire cache into RAM.
- **D-04:** `RADIO_VARIANT = "so400m"` constant in train.py (or "h"). `RADIO_ADAPTORS = ["backbone"]` list constant — agent selects which adaptor outputs to distill from.
- **D-05:** Each adaptor gets its own projection head in train.py (because dims may differ across adaptors). Agent can tune projection architectures.

### Spatial Distillation
- **D-06:** Spatial distillation loss aligns student's pre-GAP spatial features (from Phase 5 custom LCNet) with RADIO's spatial features. Student spatial is bilinear-interpolated to match RADIO's spatial resolution.
- **D-07:** `SPATIAL_DISTILL_WEIGHT = 0.0` default (disabled). Agent enables by setting positive value. Can be used alongside summary distillation.
- **D-08:** Spatial adapter in train.py: Conv1x1 projecting student channels to RADIO spatial dim, followed by BatchNorm.

### Integration with Multi-Teacher
- **D-09:** RADIO teachers register as `"radio_so400m"` and `"radio_h"` in the Phase 6 teacher registry. Each variant × adaptor combination is a separate cache.
- **D-10:** Cache directory structure: `workspace/output/teacher_cache/radio_so400m/backbone/`, `workspace/output/teacher_cache/radio_so400m/dino_v3/`, etc.
- **D-11:** RADIO teacher class loads from cloned `RADIO/` directory (user already cloned NVlabs/RADIO).

### Claude's Discretion
- Spatial feature interpolation method (bilinear vs nearest)
- Whether to L2-normalize spatial features before loss computation
- Batch loading strategy for memory-mapped spatial features (pre-load batch or lazy per-sample)
- RADIO model initialization (torch.hub.load vs direct import from cloned repo)
- Cache building batch size for RADIO (balance VRAM vs speed)

</decisions>

<canonical_refs>
## Canonical References

### Source Code
- `RADIO/` — Cloned NVlabs/RADIO repository (local)
- `train.py` — Phase 5 custom LCNet with spatial feature API
- `prepare.py` — Phase 6 multi-teacher infrastructure

### Research
- `.planning/research/ARCHITECTURE.md` — RADIO integration analysis
- `.planning/research/PITFALLS.md` — VRAM constraints, spatial cache sizing
- `.planning/research/STACK.md` — einops dependency, RADIO summary dim verification

### External
- RADIO/RADIOv2.5_tech_report.pdf — Architecture details, adaptor design
- RADIO/RADIOv4.0_tech_report.pdf — C-RADIOv4 improvements, adaptor heads
- RADIO/amradio.pdf — Original AM-RADIO paper (CVPR 2024)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 6 teacher registry — extend with RADIO entries
- Phase 6 cache architecture — extend for per-adaptor caching
- Phase 5 custom LCNet `.forward_features()` — provides spatial features for distillation

### Established Patterns
- Per-teacher cache directories with metadata.json
- Sequential cache building (one teacher on GPU at a time)
- Native-dim caching with train.py projection

### Integration Points
- prepare.py RADIOTeacher class → teacher registry
- train.py RADIO_VARIANT/RADIO_ADAPTORS constants
- train.py spatial distillation loss → uses LCNet.encode_with_spatial()
- Cache dirs → workspace/output/teacher_cache/radio_*/

</code_context>

<specifics>
## Specific Ideas

User specifically wants:
- All RADIO adaptor outputs available (backbone, dino_v3, siglip2-g)
- Both C-RADIOv4-SO400M and C-RADIOv4-H supported
- Agent can choose which adaptors to distill from at experiment time

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 08-radio-integration*
*Context gathered: 2026-03-25*
