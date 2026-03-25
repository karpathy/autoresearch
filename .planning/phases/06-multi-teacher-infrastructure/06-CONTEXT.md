# Phase 6: Multi-Teacher Infrastructure - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Expand prepare.py to support 5+ teachers with independent caches, and add constants in train.py so the agent can select which teacher(s) to distill from. This is the foundation for RADIO and DINOv3 integration.

</domain>

<decisions>
## Implementation Decisions

### Teacher Registry
- **D-01:** Dict-based teacher registry in prepare.py. Each teacher is a dict entry: `{"name": str, "class": TeacherClass, "embedding_dim": int, "cache_dir": str}`.
- **D-02:** Teacher names are simple strings: `"trendyol_onnx"`, `"dinov2"`, `"dinov3_ft"`, `"radio_so400m"`, `"radio_h"`. These are the keys used in train.py constants.
- **D-03:** `TEACHER = "trendyol_onnx"` in train.py — single teacher selection. Maps to registry entry. Default unchanged from v1.
- **D-04:** `init_teachers()` function in prepare.py returns a dict of initialized teacher objects. Only initializes the teachers specified by train.py's TEACHER or TEACHERS constant.

### Multi-Teacher Loss
- **D-05:** `TEACHERS = {"trendyol_onnx": 1.0}` dict in train.py for multi-teacher mode. Keys are teacher names, values are loss weights. When TEACHERS is set, TEACHER is ignored.
- **D-06:** Each teacher gets its own projection head in train.py (necessary because dims differ: Trendyol=256, DINOv2=256, RADIO=1152). Projection heads are part of the model, agent-tunable.
- **D-07:** Total distillation loss = `sum(weight * distill_loss(student, teacher_i) for teacher_i, weight in TEACHERS)`. Agent can tune per-teacher weights.

### Cache Architecture
- **D-08:** Per-teacher cache directory: `workspace/output/teacher_cache/{teacher_name}/`. Each contains .npy embedding files and a `metadata.json`.
- **D-09:** `metadata.json` schema: `{teacher_name, embedding_dim, num_samples, cache_date, model_version, input_resolution}`.
- **D-10:** All caches store native dimension (no projection at cache time). Projection happens in train.py. This allows reusing caches across different student projection experiments.
- **D-11:** Cache building is sequential — one teacher loaded to GPU at a time, then unloaded before next. Progress bar per teacher with ETA.
- **D-12:** Cache validation on load: check metadata.json embedding_dim matches expected, check sample count matches dataset size. Warn + rebuild if mismatch.

### Claude's Discretion
- Exact cache file naming convention (by image path hash, by index, etc.)
- Whether to use memory-mapped numpy for large caches (defer to Phase 8 for RADIO spatial)
- CombinedDistillDataset modifications to support multi-teacher loading
- Teacher initialization error handling (model not found, download failed)

</decisions>

<canonical_refs>
## Canonical References

### Source Code
- `prepare.py` — Current teacher infrastructure (TrendyolEmbedder, DINOv2Teacher stub, load_teacher_embeddings)
- `train.py` — Current TEACHER constant and distillation loss
- `finetune_trendyol_arcface3.py` — Original multi-teacher support patterns

### Research
- `.planning/research/ARCHITECTURE.md` — Multi-teacher integration analysis
- `.planning/research/PITFALLS.md` — VRAM constraints, embedding dimension mismatch risks

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TrendyolEmbedder` class — existing ONNX teacher, becomes one registry entry
- `DINOv2Teacher` class (stub) — exists but may have encode_batch bug (3D vs 2D output). Verify and fix.
- `load_teacher_embeddings()` — generic cache builder, extend for multi-teacher
- `_TEACHER_MEM_CACHE` — in-memory cache dict pattern, reuse per teacher

### Established Patterns
- Teacher cache: disk .npy + memory dict hybrid
- `CombinedDistillDataset.__getitem__` returns `(image, teacher_embedding, label)` — extend to return dict of teacher embeddings

### Integration Points
- prepare.py teacher initialization — add registry and init_teachers()
- prepare.py cache building — add per-teacher sequential build
- train.py constants — add TEACHER/TEACHERS/TEACHER_WEIGHTS
- train.py distillation_loss — extend for multi-teacher weighted combination

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-multi-teacher-infrastructure*
*Context gathered: 2026-03-25*
