# Phase 6: Multi-Teacher Infrastructure - Research

**Researched:** 2026-03-25
**Domain:** Multi-teacher knowledge distillation infrastructure for ReID
**Confidence:** HIGH

## Summary

Phase 6 expands prepare.py to support 5+ teachers with independent caches, and adds TEACHER/TEACHERS constants to train.py so the agent can select which teacher(s) to distill from. This is pure infrastructure -- no new model architectures or loss functions, just the plumbing that later phases (7: DINOv3, 8: RADIO) will plug into.

The core work is: (1) a teacher registry dict in prepare.py mapping teacher names to their class/config, (2) per-teacher cache directories with metadata.json, (3) sequential cache building with GPU memory safety, (4) extending CombinedDistillDataset to return multi-teacher embeddings, and (5) TEACHER/TEACHERS constants plus multi-projection-head support in train.py.

**Primary recommendation:** Build the registry and cache infrastructure first (prepare.py side), then wire up train.py constants and multi-teacher loss. The DINOv2Teacher bug (3D output) must be fixed as part of this phase since it is currently broken and blocks any multi-teacher usage.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Dict-based teacher registry in prepare.py. Each teacher is a dict entry: `{"name": str, "class": TeacherClass, "embedding_dim": int, "cache_dir": str}`.
- **D-02:** Teacher names are simple strings: `"trendyol_onnx"`, `"dinov2"`, `"dinov3_ft"`, `"radio_so400m"`, `"radio_h"`. These are the keys used in train.py constants.
- **D-03:** `TEACHER = "trendyol_onnx"` in train.py -- single teacher selection. Maps to registry entry. Default unchanged from v1.
- **D-04:** `init_teachers()` function in prepare.py returns a dict of initialized teacher objects. Only initializes the teachers specified by train.py's TEACHER or TEACHERS constant.
- **D-05:** `TEACHERS = {"trendyol_onnx": 1.0}` dict in train.py for multi-teacher mode. Keys are teacher names, values are loss weights. When TEACHERS is set, TEACHER is ignored.
- **D-06:** Each teacher gets its own projection head in train.py (necessary because dims differ: Trendyol=256, DINOv2=256, RADIO=1152). Projection heads are part of the model, agent-tunable.
- **D-07:** Total distillation loss = `sum(weight * distill_loss(student, teacher_i) for teacher_i, weight in TEACHERS)`. Agent can tune per-teacher weights.
- **D-08:** Per-teacher cache directory: `workspace/output/teacher_cache/{teacher_name}/`. Each contains .npy embedding files and a `metadata.json`.
- **D-09:** `metadata.json` schema: `{teacher_name, embedding_dim, num_samples, cache_date, model_version, input_resolution}`.
- **D-10:** All caches store native dimension (no projection at cache time). Projection happens in train.py. This allows reusing caches across different student projection experiments.
- **D-11:** Cache building is sequential -- one teacher loaded to GPU at a time, then unloaded before next. Progress bar per teacher with ETA.
- **D-12:** Cache validation on load: check metadata.json embedding_dim matches expected, check sample count matches dataset size. Warn + rebuild if mismatch.

### Claude's Discretion
- Exact cache file naming convention (by image path hash, by index, etc.)
- Whether to use memory-mapped numpy for large caches (defer to Phase 8 for RADIO spatial)
- CombinedDistillDataset modifications to support multi-teacher loading
- Teacher initialization error handling (model not found, download failed)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TEACH-01 | prepare.py supports 5+ teachers: Trendyol ONNX, DINOv2, DINOv3-ft, all C-RADIO variants | Teacher registry dict (D-01/D-02); DINOv2Teacher bug fix; RADIOTeacher and DINOv3Teacher as stubs with correct interface |
| TEACH-02 | Each teacher has independent cache directory with metadata | Per-teacher cache dirs (D-08); metadata.json schema (D-09); cache validation (D-12) |
| TEACH-03 | Cache building sequential per teacher (VRAM safety) | Sequential init/inference/unload pattern (D-11); explicit torch.cuda.empty_cache() + gc.collect() between teachers |
| TEACH-04 | TEACHER is a module-level constant -- agent can switch teachers | TEACHER constant in train.py (D-03); init_teachers() selects based on TEACHER or TEACHERS (D-04) |
| TEACH-05 | Multi-teacher mode with per-teacher loss weights as tunable constants | TEACHERS dict in train.py (D-05); per-teacher projection heads (D-06); weighted loss combination (D-07) |
</phase_requirements>

## Architecture Patterns

### Recommended Project Structure

No new files are created. This phase modifies two existing files:

```
prepare.py  (IMMUTABLE side -- add registry + cache infra)
  +-- TEACHER_REGISTRY dict
  +-- init_teachers() function
  +-- load_teacher_embeddings() refactored for multi-teacher
  +-- DINOv2Teacher bug fix (3D -> 2D output)
  +-- DINOv3Teacher stub class
  +-- RADIOTeacher stub class (Phase 8 fills in RADIO-specific logic)
  +-- build_teacher_cache() per-teacher sequential builder
  +-- validate_teacher_cache() metadata checker

train.py  (AGENT-EDITABLE side -- add constants + multi-head)
  +-- TEACHER = "trendyol_onnx" constant
  +-- TEACHERS = None constant (multi-teacher mode)
  +-- Per-teacher ProjectionHead instances
  +-- Multi-teacher distillation loss computation

workspace/output/teacher_cache/  (NEW directory structure)
  +-- trendyol_onnx/
  |     +-- metadata.json
  |     +-- {hash}.npy ...
  +-- dinov2/
  |     +-- metadata.json
  |     +-- {hash}.npy ...
  +-- (etc for each teacher)
```

### Pattern 1: Teacher Registry

**What:** A module-level dict in prepare.py that maps teacher name strings to configuration dicts.
**When:** Always -- this is the central lookup for all teacher operations.

```python
# prepare.py

TEACHER_REGISTRY: dict[str, dict] = {
    "trendyol_onnx": {
        "class": TrendyolEmbedder,
        "embedding_dim": 256,
        "cache_dir": "workspace/output/teacher_cache/trendyol_onnx",
        "init_kwargs": {},
    },
    "dinov2": {
        "class": DINOv2Teacher,
        "embedding_dim": 256,
        "cache_dir": "workspace/output/teacher_cache/dinov2",
        "init_kwargs": {"model_name": "Trendyol/trendyol-dino-v2-ecommerce-256d"},
    },
    "dinov3_ft": {
        "class": DINOv3Teacher,  # stub -- Phase 7 provides model
        "embedding_dim": 256,  # TBD after fine-tuning
        "cache_dir": "workspace/output/teacher_cache/dinov3_ft",
        "init_kwargs": {},
    },
    "radio_so400m": {
        "class": RADIOTeacher,  # stub -- Phase 8 provides full implementation
        "embedding_dim": 1280,  # TBD -- needs runtime verification
        "cache_dir": "workspace/output/teacher_cache/radio_so400m",
        "init_kwargs": {"version": "c-radio_v4-so400m"},
    },
    "radio_h": {
        "class": RADIOTeacher,  # stub -- Phase 8
        "embedding_dim": 1280,  # TBD -- needs runtime verification
        "cache_dir": "workspace/output/teacher_cache/radio_h",
        "init_kwargs": {"version": "c-radio_v4-h"},
    },
}
```

**Confidence:** HIGH for registry pattern, MEDIUM for RADIO embedding dims (1152 or 1280 -- needs runtime verification, see Open Questions).

### Pattern 2: Sequential Cache Building with VRAM Safety

**What:** Build caches one teacher at a time, explicitly freeing GPU memory between teachers.
**When:** At startup when any requested teacher's cache is missing or invalid.

```python
def build_all_teacher_caches(
    teacher_names: list[str],
    image_paths: list[str],
    device: str = "cuda",
) -> None:
    """Build caches sequentially, one teacher at a time for VRAM safety."""
    for name in teacher_names:
        entry = TEACHER_REGISTRY[name]
        cache_dir = Path(entry["cache_dir"])

        # Check if cache is valid
        if _is_cache_valid(cache_dir, entry["embedding_dim"], len(image_paths)):
            logger.info(f"Cache valid for {name}, skipping build")
            continue

        logger.info(f"Building cache for {name}...")
        teacher = entry["class"](**entry["init_kwargs"], device=device)

        # Build cache with progress bar
        _build_single_teacher_cache(teacher, name, image_paths, cache_dir, entry["embedding_dim"])

        # Explicit cleanup
        del teacher
        torch.cuda.empty_cache()
        import gc; gc.collect()

        logger.info(f"Cache built for {name}, GPU memory freed")
```

### Pattern 3: Multi-Teacher Dataset Return

**What:** CombinedDistillDataset returns a dict of teacher embeddings instead of a single tensor.
**When:** In multi-teacher mode (TEACHERS dict is set).

```python
# Option A: Dataset returns dict of embeddings (recommended)
# CombinedDistillDataset.__getitem__ returns (image, label, path) -- unchanged
# load_teacher_embeddings() called per-teacher in the training loop

# In train.py training loop:
for images, labels, paths in distill_loader:
    teacher_embs = {}
    for teacher_name, teacher_obj in teachers.items():
        teacher_embs[teacher_name] = load_teacher_embeddings(
            paths, teacher_obj, device,
            TEACHER_REGISTRY[teacher_name]["cache_dir"]
        )

    # Per-teacher projection + loss
    total_distill_loss = 0.0
    for teacher_name, weight in TEACHERS.items():
        proj_head = projection_heads[teacher_name]
        student_proj = proj_head(backbone_features)
        student_proj = F.normalize(student_proj, p=2, dim=1)
        teacher_emb = teacher_embs[teacher_name]
        cos = F.cosine_similarity(student_proj, teacher_emb, dim=1)
        total_distill_loss += weight * (1.0 - cos).mean()
```

**Key design choice:** Do NOT modify CombinedDistillDataset to return teacher embeddings directly. Keep it returning `(image, label, path)` and call `load_teacher_embeddings()` in the training loop. This keeps the dataset generic and allows different teachers per training configuration without rebuilding the dataset.

### Pattern 4: Backward-Compatible init_teacher() Wrapper

**What:** Keep the existing `init_teacher()` function working for single-teacher mode while adding `init_teachers()` for multi-teacher.

```python
def init_teacher(device: str = "cuda") -> TrendyolEmbedder:
    """Legacy single-teacher init. Returns TrendyolEmbedder."""
    return TrendyolEmbedder(device=device)

def init_teachers(
    teacher_names: list[str],
    device: str = "cuda",
) -> dict[str, object]:
    """Initialize multiple teachers. Only loads the requested ones."""
    teachers = {}
    for name in teacher_names:
        if name not in TEACHER_REGISTRY:
            raise ValueError(f"Unknown teacher: {name}. Available: {list(TEACHER_REGISTRY.keys())}")
        entry = TEACHER_REGISTRY[name]
        try:
            teachers[name] = entry["class"](**entry["init_kwargs"], device=device)
            logger.info(f"Initialized teacher: {name}")
        except Exception as e:
            logger.error(f"Failed to initialize teacher {name}: {e}")
            raise
    return teachers
```

### Anti-Patterns to Avoid

- **Loading all teachers to GPU simultaneously:** RADIO-H alone may consume 10+ GB VRAM. Never keep multiple large teachers in memory. Cache-then-unload pattern.
- **Projecting embeddings before caching:** D-10 explicitly says store native dims. Projection is in train.py, not prepare.py. This allows reuse across experiments.
- **Modifying CombinedDistillDataset to embed teacher lookups:** Keep dataset returning (image, label, path). Teacher embedding lookup happens separately via load_teacher_embeddings.
- **Coupling RADIO-specific logic into this phase:** Phase 6 creates stub classes. Phase 8 fills in RADIO-specific adaptor selection, spatial features, etc.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cache file naming | Custom naming scheme | MD5 hash of image path (existing pattern) | Already proven in current codebase, collision-free for practical purposes |
| Progress bars | Custom progress printing | `tqdm` (already available via torch ecosystem) or manual progress like current code | Existing pattern works fine |
| Teacher model loading | Custom download/checkpoint logic | `transformers.AutoModel` for DINOv2, `torch.hub` for RADIO, ONNX runtime for Trendyol | Each teacher has its own canonical loading mechanism |
| JSON metadata | Custom serialization | `json.dump`/`json.load` with standard Python dicts | stdlib is sufficient |

## Common Pitfalls

### Pitfall 1: DINOv2Teacher 3D Output Bug (CONFIRMED)

**What goes wrong:** DINOv2Teacher.encode_batch() returns 3D tensors `(num_tokens, hidden_dim)` per sample instead of 1D `(256,)` vectors. This causes `np.stack` in `load_teacher_embeddings` to fail or produce wrong-shaped embeddings.
**Why it happens:** Line 251 in prepare.py uses `out.last_hidden_state` which is `(B, num_tokens, hidden_dim)`. Iterating over batch dimension gives 2D matrices per sample, not 1D vectors.
**How to avoid:** Fix to use CLS token: `emb = out.last_hidden_state[:, 0, :]` giving `(B, 256)`. Then `[e.cpu().numpy() for e in emb]` produces list of `(256,)` arrays.
**Warning signs:** Shape mismatch errors during cache building. DINOv2 cache files being 197x larger than expected (197 tokens * 256 dim instead of just 256).
**Confidence:** HIGH -- verified by reading the code.

### Pitfall 2: RADIO Summary Dimension Uncertainty

**What goes wrong:** RADIO embedding dimension is assumed to be 1152 (from STATE.md) but could be different. If the registry has wrong `embedding_dim`, cache validation will incorrectly flag valid caches as corrupt.
**Why it happens:** RADIO's `summary_dim` property depends on `embed_dim * summary_idxs.shape[0]`. Without loading the model, the exact dimension is unknown. Different RADIO variants (SO400M vs H) may have different embed_dim values.
**How to avoid:** When building the RADIOTeacher stub in this phase, make `embedding_dim` lazy -- read it from the model at initialization time rather than hardcoding in the registry. Or defer exact dims to Phase 8 when RADIO is fully integrated.
**Warning signs:** Dimension mismatch between cached embeddings and what train.py projection head expects.
**Confidence:** MEDIUM -- needs runtime verification.

### Pitfall 3: Existing Cache Migration

**What goes wrong:** The existing Trendyol cache lives at `/data/training/reid/workspace/output/trendyol_teacher_cache2/` (a flat directory with no metadata.json). The new system expects caches at `workspace/output/teacher_cache/trendyol_onnx/` with metadata.json. First run rebuilds the entire Trendyol cache unnecessarily.
**Why it happens:** Cache directory path changed. No migration path provided.
**How to avoid:** Either (a) symlink the old cache into the new path, or (b) treat `DEFAULT_TEACHER_CACHE_DIR` as the Trendyol cache path in the registry (use existing location), or (c) accept one-time rebuild cost. Option (b) is simplest -- override the cache_dir for trendyol_onnx to point to the existing cache location.
**Warning signs:** First multi-teacher run takes hours rebuilding a cache that already exists.
**Confidence:** HIGH -- verified existing cache location and format.

### Pitfall 4: Memory Cache Collision Between Teachers

**What goes wrong:** The current `_TEACHER_MEM_CACHE` is a single global dict keyed by image path. If two teachers cache embeddings for the same image path, one overwrites the other.
**Why it happens:** Cache key is just the image path, not teacher_name + image_path.
**How to avoid:** Either (a) make the cache key `f"{teacher_name}:{path}"`, or (b) use a per-teacher memory cache dict: `_TEACHER_MEM_CACHES: dict[str, dict[str, np.ndarray]] = {}`.
**Warning signs:** Cosine similarity suddenly becomes 1.0 (student perfectly matches teacher) -- because the "teacher" embedding being loaded is actually from a different teacher with identical image path key.
**Confidence:** HIGH -- obvious from code inspection.

### Pitfall 5: OOM During Multi-Teacher Cache Building

**What goes wrong:** Building caches for DINOv2 + RADIO back-to-back exhausts GPU memory even with sequential loading, because PyTorch CUDA memory is not fully released by `del` alone.
**Why it happens:** CUDA memory fragmentation. `torch.cuda.empty_cache()` releases unused memory but doesn't defragment. Large models leave fragmented allocations.
**How to avoid:** Between teachers: `del teacher; gc.collect(); torch.cuda.empty_cache()`. If still OOM, can run cache building in subprocess per teacher (nuclear option). Also set `max_split_size_mb` if fragmentation is severe.
**Warning signs:** Second teacher OOMs despite first being deleted. `torch.cuda.memory_summary()` shows high fragmentation.
**Confidence:** MEDIUM -- depends on model sizes.

### Pitfall 6: Train.py Import Bloat

**What goes wrong:** train.py imports grow unwieldy as it needs to import TEACHER_REGISTRY, init_teachers, load_teacher_embeddings, and all the new constants.
**Why it happens:** Natural growth of the prepare.py -> train.py import bridge.
**How to avoid:** Keep the import block organized. Only export what train.py actually needs. The registry itself is an implementation detail of prepare.py -- train.py should call `init_teachers(["trendyol_onnx"])` without needing to know the registry structure.
**Confidence:** HIGH -- design pattern.

## Code Examples

### Example 1: Teacher Registry and Initialization

```python
# prepare.py -- teacher registry

TEACHER_REGISTRY: dict[str, dict] = {
    "trendyol_onnx": {
        "class": TrendyolEmbedder,
        "embedding_dim": 256,
        "cache_dir": DEFAULT_TEACHER_CACHE_DIR,  # reuse existing cache!
        "init_kwargs": {},
    },
    "dinov2": {
        "class": DINOv2Teacher,
        "embedding_dim": 256,
        "cache_dir": "workspace/output/teacher_cache/dinov2",
        "init_kwargs": {"model_name": "Trendyol/trendyol-dino-v2-ecommerce-256d"},
    },
    # stubs for future phases
    "dinov3_ft": {
        "class": None,  # Phase 7
        "embedding_dim": 256,
        "cache_dir": "workspace/output/teacher_cache/dinov3_ft",
        "init_kwargs": {},
    },
    "radio_so400m": {
        "class": None,  # Phase 8
        "embedding_dim": None,  # determined at init time
        "cache_dir": "workspace/output/teacher_cache/radio_so400m",
        "init_kwargs": {"version": "c-radio_v4-so400m"},
    },
    "radio_h": {
        "class": None,  # Phase 8
        "embedding_dim": None,  # determined at init time
        "cache_dir": "workspace/output/teacher_cache/radio_h",
        "init_kwargs": {"version": "c-radio_v4-h"},
    },
}
```

### Example 2: DINOv2Teacher Bug Fix

```python
# prepare.py -- BEFORE (buggy)
@torch.no_grad()
def encode_batch(self, images: list[np.ndarray | Image.Image]) -> list[np.ndarray | None]:
    # ...
    out = self.model(batch)
    emb = out.last_hidden_state  # BUG: shape (B, num_tokens, 256) -- 3D!
    return [e.cpu().numpy() for e in emb]  # each e is (num_tokens, 256) -- WRONG

# prepare.py -- AFTER (fixed)
@torch.no_grad()
def encode_batch(self, images: list[np.ndarray | Image.Image]) -> list[np.ndarray | None]:
    # ...
    out = self.model(batch)
    emb = out.last_hidden_state[:, 0, :]  # CLS token only: shape (B, 256)
    return [e.cpu().numpy().flatten() for e in emb]  # each e is (256,) -- correct
```

### Example 3: Per-Teacher Memory Cache

```python
# prepare.py -- per-teacher memory cache (fixes collision pitfall)

_TEACHER_MEM_CACHES: dict[str, dict[str, np.ndarray]] = {}

def load_teacher_embeddings(
    image_paths: Sequence[str],
    teacher,
    device: torch.device,
    cache_dir: str | None = None,
    teacher_name: str = "default",
) -> torch.Tensor:
    """Load teacher embeddings with per-teacher in-memory + disk caching."""
    if teacher_name not in _TEACHER_MEM_CACHES:
        _TEACHER_MEM_CACHES[teacher_name] = {}
    mem_cache = _TEACHER_MEM_CACHES[teacher_name]

    embeddings: list[np.ndarray] = [None] * len(image_paths)
    uncached_indices: list[int] = []

    for i, path in enumerate(image_paths):
        if path in mem_cache:
            embeddings[i] = mem_cache[path]
            continue
        if cache_dir:
            cache_path = Path(cache_dir) / f"{hashlib.md5(path.encode()).hexdigest()}.npy"
            if cache_path.exists():
                emb = np.load(cache_path)
                mem_cache[path] = emb
                embeddings[i] = emb
                continue
        uncached_indices.append(i)

    if uncached_indices:
        pil_images = [Image.open(image_paths[i]).convert("RGB") for i in uncached_indices]
        emb_list = teacher.encode_batch(pil_images)
        for j, i in enumerate(uncached_indices):
            emb = emb_list[j]
            embeddings[i] = emb
            mem_cache[image_paths[i]] = emb
            if cache_dir and emb is not None:
                cache_path = Path(cache_dir) / f"{hashlib.md5(image_paths[i].encode()).hexdigest()}.npy"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, emb)

    return torch.tensor(np.stack(embeddings), device=device, dtype=torch.float32)
```

### Example 4: metadata.json Creation and Validation

```python
# prepare.py -- cache metadata

import json
from datetime import datetime

def _write_cache_metadata(
    cache_dir: Path,
    teacher_name: str,
    embedding_dim: int,
    num_samples: int,
    model_version: str = "",
    input_resolution: int = 224,
) -> None:
    metadata = {
        "teacher_name": teacher_name,
        "embedding_dim": embedding_dim,
        "num_samples": num_samples,
        "cache_date": datetime.now().isoformat(),
        "model_version": model_version,
        "input_resolution": input_resolution,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def _is_cache_valid(
    cache_dir: Path,
    expected_dim: int,
    expected_samples: int,
) -> bool:
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["embedding_dim"] != expected_dim:
            logger.warning(f"Cache dim mismatch: expected {expected_dim}, got {meta['embedding_dim']}")
            return False
        if meta["num_samples"] != expected_samples:
            logger.warning(f"Cache sample count mismatch: expected {expected_samples}, got {meta['num_samples']}")
            return False
        return True
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Cache metadata corrupted: {e}")
        return False
```

### Example 5: train.py Multi-Teacher Constants and Loss

```python
# train.py -- new constants

# Single teacher mode (default, backward compatible)
TEACHER = "trendyol_onnx"

# Multi-teacher mode (set to enable, overrides TEACHER)
# Example: TEACHERS = {"trendyol_onnx": 0.5, "dinov2": 0.5}
TEACHERS: dict[str, float] | None = None

# In main():
if TEACHERS is not None:
    teacher_names = list(TEACHERS.keys())
else:
    teacher_names = [TEACHER]

# Build caches sequentially
build_all_teacher_caches(teacher_names, all_image_paths, device=str(device))

# Initialize only needed teachers (for online inference during training)
teachers = init_teachers(teacher_names, device=str(device))

# Per-teacher projection heads
projection_heads = {}
for name in teacher_names:
    teacher_dim = TEACHER_REGISTRY[name]["embedding_dim"]
    projection_heads[name] = ProjectionHead(backbone_out_dim, teacher_dim).to(device)
```

## Existing Cache Compatibility

The current Trendyol cache at `/data/training/reid/workspace/output/trendyol_teacher_cache2/` contains:
- ~5.5 million files (6.3GB total based on ls output)
- Each file: MD5 hash of image path + `.npy`, shape `(256,)`, dtype `float32`
- No metadata.json

**Recommendation:** Set `trendyol_onnx` registry entry's `cache_dir` to `DEFAULT_TEACHER_CACHE_DIR` (the existing path). Add a metadata.json to the existing directory without moving files. This avoids a multi-hour cache rebuild.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single teacher (TrendyolEmbedder) | Multi-teacher registry | Phase 6 | Enables 5+ teacher experimentation |
| Global memory cache | Per-teacher memory caches | Phase 6 | Prevents cross-teacher cache pollution |
| Flat cache dir | Per-teacher dirs with metadata.json | Phase 6 | Enables cache validation and management |
| `init_teacher()` returns one | `init_teachers()` returns dict | Phase 6 | Agent selects teacher(s) via constants |

## Open Questions

1. **RADIO summary embedding dimension**
   - What we know: RADIO uses `embed_dim * summary_idxs.shape[0]` for summary_dim. STATE.md says "expected 1152d" for SO400M.
   - What's unclear: Without loading the model, exact dims for C-RADIOv4-SO400M and C-RADIOv4-H are unconfirmed. The code shows `summary_idxs` may multiply the base embed_dim.
   - Recommendation: In Phase 6, set RADIO embedding_dim as `None` in registry. Phase 8 will determine the actual dim at model init time and write it to metadata.json. This avoids hardcoding a wrong value.

2. **Trendyol cache sample count for validation**
   - What we know: The existing cache has ~5.5M files but no metadata to record expected sample count.
   - What's unclear: Exact sample count needed for metadata.json validation.
   - Recommendation: When adding metadata.json to existing cache, count the .npy files and record that as num_samples. Future validation compares against dataset size.

3. **Memory budget for multi-teacher memory caches**
   - What we know: Each 256-dim float32 embedding = 1KB. 500K images * 5 teachers = ~2.5GB in-memory. For RADIO (1152-dim) this increases.
   - What's unclear: Whether the machine has enough RAM for all teacher caches simultaneously in memory.
   - Recommendation: In Phase 6, only the active teachers (specified by TEACHER/TEACHERS) are loaded into memory cache. Inactive teacher caches stay on disk. Memory-mapped numpy is deferred to Phase 8 per CONTEXT.md.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual validation (no pytest infrastructure exists) |
| Config file | None |
| Quick run command | `python -c "from prepare import TEACHER_REGISTRY; print(TEACHER_REGISTRY.keys())"` |
| Full suite command | `python train.py` (end-to-end with single teacher, verify backward compat) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TEACH-01 | Registry has 5+ teacher entries | smoke | `python -c "from prepare import TEACHER_REGISTRY; assert len(TEACHER_REGISTRY) >= 5"` | N/A (inline) |
| TEACH-02 | Per-teacher cache with metadata | integration | Build Trendyol cache, verify metadata.json exists and is valid JSON | Wave 0 |
| TEACH-03 | Sequential cache building | manual | Run with 2 teachers, observe VRAM stays below threshold between builds | manual-only (VRAM monitoring) |
| TEACH-04 | TEACHER constant switches teacher | smoke | Set `TEACHER="dinov2"` in train.py, verify DINOv2Teacher is initialized | Wave 0 |
| TEACH-05 | Multi-teacher loss computation | integration | Set `TEACHERS={"trendyol_onnx": 0.5, "dinov2": 0.5}`, run 1 epoch, verify loss is computed | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -c "from prepare import TEACHER_REGISTRY, init_teachers; print('OK')"`
- **Per wave merge:** `python train.py` (full run with default TEACHER to verify backward compat)
- **Phase gate:** Full suite green before /gsd:verify-work

### Wave 0 Gaps
- No formal test files exist. Validation is via import checks and end-to-end runs.
- The project uses manual testing through the autoresearch loop, not pytest.

## Sources

### Primary (HIGH confidence)
- `prepare.py` lines 100-253 -- TrendyolEmbedder and DINOv2Teacher implementation
- `prepare.py` lines 660-702 -- load_teacher_embeddings and _TEACHER_MEM_CACHE
- `train.py` lines 1-70 -- current imports and constants
- `RADIO/radio/common.py` -- RESOURCE_MAP with all C-RADIO variant URLs and configs
- `RADIO/radio/radio_model.py` lines 136-263 -- RADIOModel forward and summary extraction
- `RADIO/radio/adaptor_base.py` -- RadioOutput(summary, features) NamedTuple

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- prepare.py/train.py split patterns
- `.planning/research/PITFALLS.md` -- teacher cache invalidation, VRAM safety
- `.planning/research/STACK.md` -- available libraries and version constraints
- `.planning/STATE.md` -- RADIO SO400M dim uncertainty, DINOv2 bug flagged

### Tertiary (LOW confidence)
- RADIO embedding dimensions for SO400M/H variants (needs runtime verification)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all libraries already available
- Architecture: HIGH -- registry pattern well-understood, existing code patterns clear
- Pitfalls: HIGH -- DINOv2 bug confirmed by code reading, cache collision obvious from code
- RADIO dimensions: MEDIUM -- needs runtime model loading to confirm

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable infrastructure code, no external dependency changes expected)
