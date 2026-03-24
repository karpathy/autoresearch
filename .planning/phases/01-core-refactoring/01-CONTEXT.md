# Phase 1: Core Refactoring - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Split the monolithic `finetune_trendyol_arcface3.py` (~1400 lines) into two files:
- `prepare.py` — immutable: data loading, teacher inference/caching, evaluation, metric calculation
- `train.py` — agent-editable: student model, losses, optimizer, scheduler, augmentations, training loop

The trust boundary is the critical output: agent can ONLY edit train.py, evaluation is locked in prepare.py.

</domain>

<decisions>
## Implementation Decisions

### Split Boundary
- **D-01:** Training augmentations live entirely in train.py. prepare.py provides base dataset classes only. Agent has full control over augmentation pipeline. Eval transforms remain fixed in prepare.py.
- **D-02:** Backbone freezing/unfreezing strategy lives in train.py as an agent-tunable experiment variable (number of stages to unfreeze, epoch to unfreeze, per-stage LR multiplier).

### Teacher Model
- **D-03:** Default teacher is ONNX (Trendyol) — `distill_qwen_lcnet050_retail_2.onnx`. Faster inference, existing disk cache at `workspace/output/trendyol_teacher_cache2/`. DINOv2 is not supported in v1.
- **D-04:** Teacher selection is a prepare.py concern (immutable). Agent cannot switch teachers.

### Execution Model
- **D-05:** `python train.py` runs independently, importing from prepare.py. Follows original autoresearch pattern. No prepare.py wrapper/orchestrator.
- **D-06:** train.py has zero CLI flags — all configuration via module-level Python constants that the agent edits directly.

### Contract
- **D-07:** train.py's model must implement `.encode(images) -> Tensor[B, 256]` (L2-normalized). This is the interface prepare.py uses for evaluation.
- **D-08:** train.py prints final metrics to stdout in a greppable format. prepare.py provides the evaluation function that train.py calls.

### Claude's Discretion
- Exact split of helper functions (PadToSquare, collate functions, etc.) — Claude decides based on whether they're experiment variables or fixed infrastructure
- Whether to keep ArcFace classification head setup in prepare.py or train.py

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Autoresearch Pattern
- `prepare.py` — Original autoresearch prepare.py (data prep + runtime utilities for GPT)
- `train.py` — Original autoresearch train.py (model + optimizer + training loop for GPT)

### Source Monolith
- `finetune_trendyol_arcface3.py` — The 1400-line monolith to be split

### Research
- `.planning/research/ARCHITECTURE.md` — Detailed split boundary analysis with component mapping
- `.planning/research/FEATURES.md` — Feature landscape and dependency graph

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TrendyolEmbedder` class (ONNX inference) → prepare.py
- `DINOv2Teacher` class → NOT used in v1 (ONNX only)
- `FrozenBackboneWithHead` → train.py (agent-editable model wrapper)
- `RandomQualityDegradation` → train.py (augmentation, agent-tunable)
- `CombinedDistillDataset`, `CombinedArcFaceDataset` → prepare.py (data loading)
- `SampledImageFolder`, `DistillImageFolder` → prepare.py (base datasets)
- Evaluation functions (retrieval recall, cosine alignment) → prepare.py

### Established Patterns
- Module-level constants (no argparse) — matches original autoresearch train.py
- Teacher embedding cache: in-memory dict + disk .npy files — already implemented
- Two-phase training: distillation warmup then ArcFace + distill joint training

### Integration Points
- train.py imports from prepare.py: datasets, teacher cache, evaluation function, constants
- prepare.py exports: `EPOCHS`, `EMBEDDING_DIM`, `IMAGE_SIZE`, evaluation functions, data loaders

</code_context>

<specifics>
## Specific Ideas

- Follow original autoresearch pattern as closely as possible for the split
- Teacher cache at `workspace/output/trendyol_teacher_cache2/` should be reused, not rebuilt

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-core-refactoring*
*Context gathered: 2026-03-25*
