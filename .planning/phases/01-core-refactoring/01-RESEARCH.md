# Phase 1: Core Refactoring - Research

**Researched:** 2026-03-25
**Domain:** Python monolith splitting -- ML training pipeline refactoring
**Confidence:** HIGH

## Summary

Phase 1 splits the 1576-line `finetune_trendyol_arcface3.py` monolith into two files: `prepare.py` (immutable data/teacher/evaluation infrastructure) and `train.py` (agent-editable model/losses/optimizer/training loop). This is a pure refactoring task -- no new functionality, just reorganization following the Karpathy autoresearch pattern.

The monolith is well-structured and maps cleanly to the two-file split. The main complexity is (1) getting the split boundary exactly right so the agent has full control over training but zero control over evaluation, (2) converting argparse arguments to module-level constants in train.py, and (3) handling the circular dependency where prepare.py needs the model for evaluation but train.py imports from prepare.py.

**Primary recommendation:** Extract verbatim from monolith -- do not rewrite logic. The code is production-tested. Split along the boundary defined in CONTEXT.md decisions, convert argparse to constants, and verify the combined metric matches the original.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Training augmentations live entirely in train.py. prepare.py provides base dataset classes only. Agent has full control over augmentation pipeline. Eval transforms remain fixed in prepare.py.
- **D-02:** Backbone freezing/unfreezing strategy lives in train.py as an agent-tunable experiment variable (number of stages to unfreeze, epoch to unfreeze, per-stage LR multiplier).
- **D-03:** Default teacher is ONNX (Trendyol) -- `distill_qwen_lcnet050_retail_2.onnx`. Faster inference, existing disk cache at `workspace/output/trendyol_teacher_cache2/`. DINOv2 is not supported in v1.
- **D-04:** Teacher selection is a prepare.py concern (immutable). Agent cannot switch teachers.
- **D-05:** `python train.py` runs independently, importing from prepare.py. Follows original autoresearch pattern. No prepare.py wrapper/orchestrator.
- **D-06:** train.py has zero CLI flags -- all configuration via module-level Python constants that the agent edits directly.
- **D-07:** train.py's model must implement `.encode(images) -> Tensor[B, 256]` (L2-normalized). This is the interface prepare.py uses for evaluation.
- **D-08:** train.py prints final metrics to stdout in a greppable format. prepare.py provides the evaluation function that train.py calls.

### Claude's Discretion
- Exact split of helper functions (PadToSquare, collate functions, etc.) -- Claude decides based on whether they're experiment variables or fixed infrastructure
- Whether to keep ArcFace classification head setup in prepare.py or train.py

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REFAC-01 | Monolith split into prepare.py (immutable) and train.py (agent-editable) | Component mapping in Architecture Patterns section; line-by-line split boundary defined |
| REFAC-02 | Evaluation logic lives exclusively in prepare.py | `run_retrieval_eval` (lines 1078-1140) and combined metric (line 1479) extracted to prepare.py; grep verification pattern provided |
| REFAC-03 | Teacher embedding cache extracted to prepare.py with disk + memory caching | `TrendyolEmbedder` (lines 79-171), `_TEACHER_MEM_CACHE` (line 757), `load_teacher_embeddings` (lines 760-798) all go to prepare.py verbatim |
| REFAC-04 | Dataset loading extracted to prepare.py with fixed train/val splits | `CombinedDistillDataset`, `CombinedArcFaceDataset`, `DistillImageFolder`, `SampledImageFolder` all go to prepare.py; dataset construction happens in prepare.py exports |
| REFAC-05 | train.py exposes all tunable parameters as module-level constants | Argparse conversion table provided; every `args.*` becomes a `SCREAMING_SNAKE_CASE` constant |
| REFAC-06 | train.py model implements `.encode(images) -> Tensor[B, 256]` | Already implemented in `FrozenBackboneWithHead.encode()` at line 748; no changes needed |
| REFAC-07 | prepare.py computes combined metric: `0.5 * recall@1 + 0.5 * mean_cosine` | Extract from main() line 1479 into a standalone `compute_combined_metric()` function in prepare.py |

</phase_requirements>

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Verified |
|---------|---------|---------|----------|
| torch | 2.11.0+cu130 | Training framework | `python3 -c "import torch"` |
| timm | 1.0.26 | Backbone models (lcnet_050) | `python3 -c "import timm"` |
| onnxruntime | 1.19.2 | ONNX teacher inference | `python3 -c "import onnxruntime"` |
| torchvision | (bundled with torch) | Transforms, datasets | Implicit |
| loguru | installed | Logging | Import in monolith |
| numpy | installed | Array ops, cache storage | Import in monolith |
| Pillow | installed | Image loading | Import in monolith |
| transformers | installed | DINOv2 (not used in v1 but imported) | Import in monolith |
| matplotlib | installed | Batch visualization | Import in monolith |

### No New Dependencies
Per REQUIREMENTS.md: "No new pip dependencies" is an explicit out-of-scope constraint. The split uses only libraries already imported in the monolith.

## Architecture Patterns

### Target Project Structure
```
autoresearch/
  prepare.py          # IMMUTABLE -- data, teacher, evaluation, caching
  train.py            # AGENT-EDITABLE -- model, losses, optimizer, training loop
  workspace/
    output/
      trendyol_teacher_cache2/   # Existing teacher embedding cache (.npy files)
```

### Pattern 1: Component-to-File Mapping

**prepare.py contents (extracted verbatim from monolith):**

| Component | Lines in Monolith | Notes |
|-----------|-------------------|-------|
| CUDA lib path setup | 14-24 | Must be at top of prepare.py (before onnxruntime import) |
| `PadToSquare` | 226-244 (second, correct version) | Fixed infrastructure; uses `TF.pad` correctly |
| `TrendyolEmbedder` | 79-171 | ONNX teacher -- immutable |
| `_patch_transformers_compat` | 173-185 | Needed if DINOv2 ever used |
| `DINOv2Teacher` | 188-223 | Keep for completeness but v1 uses ONNX only |
| `set_seed` | 247-251 | Reproducibility -- immutable |
| `DistillImageFolder` | 311-330 | Base dataset class |
| `SampledImageFolder` | 333-380 | Base dataset class |
| `CombinedDistillDataset` | 383-534 | Multi-source distillation data |
| `CombinedArcFaceDataset` | 537-666 | Multi-source ArcFace data |
| `collate_distill` | 669-678 | Collation function |
| `collate_arcface` | 681-690 | Collation function |
| `_TEACHER_MEM_CACHE` + `load_teacher_embeddings` | 757-798 | Cache-aware teacher inference |
| `run_retrieval_eval` | 1078-1140 | Retrieval evaluation (recall@1, recall@k) |
| `compute_combined_metric` | NEW function | `0.5 * recall@1 + 0.5 * mean_cosine` (currently inline at line 1479) |
| `build_val_transform` | Extracted from `build_transform` (lines 1168-1176, is_training=False branch) | Eval transforms -- immutable |
| Dataset path constants | Lines 1276-1279 | `/data/mnt/...` paths as named constants |
| `NUM_ARCFACE_CLASSES` | Computed at dataset init, exported | train.py needs this for ArcMarginProduct |

**train.py contents (extracted verbatim from monolith):**

| Component | Lines in Monolith | Notes |
|-----------|-------------------|-------|
| `ProjectionHead` | 254-260 | Agent-editable model component |
| `ArcMarginProduct` | 263-282 | Agent-editable (s, m are experiment variables) |
| `RandomQualityDegradation` | 285-308 | Agent-editable augmentation |
| `FrozenBackboneWithHead` | 693-754 | Student model with `.encode()` contract |
| `vat_embedding_loss` | 864-898 | Agent-editable loss |
| `EpochStats` | 901-908 | Dataclass for epoch results |
| `run_train_epoch` | 911-1075 | Full training loop -- agent-editable |
| `build_train_transform` | Extracted from `build_transform` (lines 1155-1167, is_training=True branch) | Agent-editable augmentations |
| `save_batch_visualization` | 801-861 | Debugging utility (agent may want to modify) |
| `main()` | Rewritten | Imports from prepare.py, runs training, calls evaluation, prints summary |
| Module-level constants | Converted from argparse | All `args.*` become `UPPER_SNAKE` constants |

### Pattern 2: Argparse-to-Constants Conversion

Every argparse argument becomes a module-level constant in train.py:

```python
# train.py -- Module-level constants (agent edits these)
MODEL_NAME = "hf-hub:timm/lcnet_050.ra2_in1k"
IMAGE_SIZE = 224
EMBEDDING_DIM = 256
BATCH_SIZE = 256
ARCFACE_BATCH_SIZE = 128
LR = 1e-1
WEIGHT_DECAY = 1e-5
EPOCHS = 10                    # Fixed budget per REQUIREMENTS
NUM_WORKERS = 16
SEED = 42
DEVICE = "cuda"
QUALITY_DEGRADATION_PROB = 0.5
DROP_HARD_RATIO = 0.2
USE_ARCFACE = True
ARCFACE_S = 32.0
ARCFACE_M = 0.50
ARCFACE_LOSS_WEIGHT = 0.05
VAT_WEIGHT = 0.0
VAT_EPSILON = 8.0
SEP_WEIGHT = 1.0
UNFREEZE_EPOCH = 0             # 0 = unfreeze from start (current behavior)
BACKBONE_LR_MULT = 0.1        # Backbone LR = LR * this
TEACHER_CACHE_DIR = "workspace/output/trendyol_teacher_cache2"
OUTPUT_DIR = "workspace/output/distill_trendyol_lcnet050_retail"
RETRIEVAL_MAX_SAMPLES = 5000
RETRIEVAL_TOPK = 5
```

Note: `EPOCHS` changes from default 80 to 10 per the fixed budget requirement.

### Pattern 3: Import Bridge (train.py imports from prepare.py)

```python
# train.py
from prepare import (
    # Data
    CombinedDistillDataset, CombinedArcFaceDataset,
    collate_distill, collate_arcface,
    PadToSquare,
    # Teacher
    TrendyolEmbedder, load_teacher_embeddings,
    # Evaluation
    run_retrieval_eval, compute_combined_metric,
    # Transforms
    build_val_transform,
    # Constants
    TRAIN_DIR, VAL_DIR, ARCFACE_DIR, REID_ROOT,
    BLACKLIST_ROOT, SKIP_CLASSES,
    # Utility
    set_seed,
)
```

### Pattern 4: Avoiding Circular Imports

prepare.py's `run_retrieval_eval` accepts the model as a parameter -- it does NOT import from train.py:

```python
# prepare.py
def run_retrieval_eval(
    model,          # Any object with .encode(images) -> Tensor[B, 256]
    dataset,        # datasets.ImageFolder
    device,
    amp,
    max_samples,
    topk,
    seed,
    batch_size,
    num_workers,
) -> dict[str, float]:
    ...
```

### Pattern 5: Summary Block (greppable output)

```python
# End of train.py main()
print("---")
print(f"combined_metric:  {combined:.6f}")
print(f"recall@1:         {recall_1:.6f}")
print(f"mean_cosine:      {mean_cos:.6f}")
print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
print(f"total_seconds:    {elapsed:.1f}")
print(f"epochs:           {EPOCHS}")
```

### Anti-Patterns to Avoid
- **Rewriting logic during split:** Extract verbatim. Do not "improve" code. This is a refactoring, not a feature change.
- **Putting training loop in prepare.py:** The full training loop is agent-editable. Agent must be able to change gradient accumulation, warmup logic, epoch structure.
- **Config files:** No YAML/JSON/TOML. Code IS the config.
- **Checkpoint/resume logic:** Each experiment starts fresh from pretrained backbone weights.
- **argparse anywhere:** train.py has zero CLI flags. prepare.py has zero CLI flags.

### Discretion Decisions (Claude's choices)

**PadToSquare:** Goes to prepare.py. It is fixed image preprocessing infrastructure, not an experiment variable. Both datasets and eval transforms use it.

**collate_distill / collate_arcface:** Go to prepare.py. They are data infrastructure, not experiment variables. They match the dataset classes which are in prepare.py.

**ArcMarginProduct:** Goes to train.py. The ArcFace head parameters (s, m, num_classes) are experiment variables the agent should be able to modify. train.py gets `NUM_ARCFACE_CLASSES` from prepare.py to initialize it.

**save_batch_visualization:** Goes to train.py. It is a debugging aid during training that the agent may want to modify or remove.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Argument parsing | argparse replacement | Module-level constants | Agent edits constants directly; argparse adds indirection |
| Teacher inference | Custom ONNX wrapper | Existing `TrendyolEmbedder` verbatim | Already handles batch inference, provider selection, error handling |
| Embedding cache | Custom cache system | Existing `_TEACHER_MEM_CACHE` + disk .npy pattern | Already handles memory + disk + batch inference fallback |
| Retrieval evaluation | Custom metric computation | Existing `run_retrieval_eval` verbatim | Handles subset sampling, similarity matrix, recall@k correctly |
| ArcFace loss | Custom angular margin | Existing `ArcMarginProduct` verbatim | Correct gradient computation with angular margin |

**Key insight:** This phase is pure extraction, not creation. Every component already exists and works. The risk is introducing bugs while moving code, not building new functionality.

## Common Pitfalls

### Pitfall 1: Duplicate PadToSquare Class (BUG IN MONOLITH)
**What goes wrong:** The monolith has TWO `PadToSquare` classes -- line 57 (uses `tf.pad` which is undefined) and line 226 (uses `TF.pad` correctly). The second one shadows the first. During the split, using the wrong version causes a NameError at runtime.
**Prevention:** Use ONLY the second `PadToSquare` (line 226) which references `TF` (the imported `torchvision.transforms.functional`). Delete the first one entirely.
**Confidence:** HIGH -- verified by reading the source code.

### Pitfall 2: Circular Import (prepare.py <-> train.py)
**What goes wrong:** prepare.py evaluation needs the model to call `.encode()`. train.py imports from prepare.py. If prepare.py also imports from train.py, Python raises ImportError.
**Prevention:** prepare.py NEVER imports from train.py. Evaluation functions accept the model as a parameter. train.py passes its model to prepare.py's evaluation function.
**How to avoid:** `run_retrieval_eval(model, ...)` takes a duck-typed model argument, not an import of FrozenBackboneWithHead.

### Pitfall 3: ArcFace Class Count Mismatch
**What goes wrong:** `ArcMarginProduct` needs `out_features = len(arcface_dataset.classes)`. If the dataset is constructed in prepare.py but ArcFace head in train.py, the count must be passed correctly.
**Prevention:** prepare.py exports a function that builds the dataset AND returns the class count. Or export `NUM_ARCFACE_CLASSES` as a computed constant. train.py uses this to initialize ArcMarginProduct.
**Recommended approach:** prepare.py provides a `build_arcface_dataset(transform, quality_degradation)` function that returns `(dataset, num_classes)`. train.py calls it.

### Pitfall 4: Dataset Construction Needs train.py Augmentations
**What goes wrong:** `CombinedDistillDataset` and `CombinedArcFaceDataset` take `transform` and `quality_degradation` as constructor arguments. These transforms are agent-editable (live in train.py). But the datasets are prepare.py concerns.
**Prevention:** prepare.py provides builder functions that accept transform and quality_degradation as parameters:
```python
# prepare.py
def build_distill_dataset(transform, quality_degradation) -> CombinedDistillDataset:
    ...
def build_arcface_dataset(transform, quality_degradation) -> tuple[CombinedArcFaceDataset, int]:
    ...
def build_val_dataset() -> datasets.ImageFolder:
    ...
```
train.py constructs its augmentations, then passes them to these builders.

### Pitfall 5: Early Stopping Removed Without Replacement
**What goes wrong:** Monolith has early stopping (patience=10, lines 1486-1492). In autoresearch, fixed 10 epochs means no early stopping. Forgetting to remove it silently shortens runs.
**Prevention:** train.py runs exactly `EPOCHS` iterations. No early stopping. No patience parameter.

### Pitfall 6: ONNX Export Left in train.py
**What goes wrong:** Monolith exports ONNX at end of training (lines 1523-1570). In autoresearch, ONNX export is a future concern (EXPRT-01, v2). Including it wastes time per experiment.
**Prevention:** Do NOT include ONNX export in train.py. It is out of scope for Phase 1.

### Pitfall 7: Forgetting to Remove DINOv2 Teacher Path
**What goes wrong:** Monolith has `--use-dino-teacher` flag. Per D-03, DINOv2 is NOT supported in v1. Leaving the flag or conditional logic in the code creates confusion.
**Prevention:** prepare.py always uses `TrendyolEmbedder`. No teacher selection flag. `DINOv2Teacher` class can remain in prepare.py for future use but is never instantiated.

### Pitfall 8: LD_LIBRARY_PATH CUDA Setup Scope
**What goes wrong:** The CUDA lib path setup (lines 14-24) must run before `import onnxruntime`. If it lives only in prepare.py and train.py imports prepare.py late, onnxruntime may already be imported without CUDA support.
**Prevention:** This code block goes at the very top of prepare.py, before any other imports. Since train.py imports from prepare.py, it will execute when prepare.py is first imported.

## Code Examples

### prepare.py -- Public API Surface

```python
# prepare.py -- Public exports

# Constants (immutable)
EMBEDDING_DIM = 256
IMAGE_SIZE = 224
TRAIN_DIR = "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/train"
VAL_DIR = "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/val"
ARCFACE_DIR = "/data/mnt/mnt_ml_shared/Vic/retail_product_checkout_crop"
REID_ROOT = Path("/data/mnt/mnt_ml_shared/joesu/reid/data/reid_train/train")
REID_PRODUCTS = str(REID_ROOT / "products")
REID_COMMODITY = str(REID_ROOT / "commodity")
REID_NEGATIVES = str(REID_ROOT / "negatives")
SKIP_CLASSES = {"0000000000"}
DEFAULT_TEACHER_CACHE_DIR = "workspace/output/trendyol_teacher_cache2"

# Functions
def set_seed(seed: int) -> None: ...
def build_val_transform(model_name: str, image_size: int) -> transforms.Compose: ...
def build_distill_dataset(transform, quality_degradation, blacklist_ratio=0.10) -> CombinedDistillDataset: ...
def build_arcface_dataset(transform, quality_degradation, skip_extra_classes=None, max_per_class=100) -> tuple[CombinedArcFaceDataset, int]: ...
def build_val_dataset(model_name: str, image_size: int) -> datasets.ImageFolder | None: ...
def init_teacher(device: str = "cuda") -> TrendyolEmbedder: ...
def load_teacher_embeddings(paths, teacher, device, cache_dir=None) -> torch.Tensor: ...
def run_retrieval_eval(model, dataset, device, amp, max_samples, topk, seed, batch_size, num_workers) -> dict[str, float]: ...
def compute_combined_metric(recall_at_1: float, mean_cosine: float) -> float: ...

# Classes (exported for type hints, not typically constructed directly)
class TrendyolEmbedder: ...
class PadToSquare: ...
class CombinedDistillDataset: ...
class CombinedArcFaceDataset: ...
```

### train.py -- Main Structure

```python
# train.py -- Agent-editable training script
from prepare import (
    set_seed, build_val_transform, build_distill_dataset, build_arcface_dataset,
    build_val_dataset, init_teacher, load_teacher_embeddings,
    run_retrieval_eval, compute_combined_metric,
    collate_distill, collate_arcface, PadToSquare,
    EMBEDDING_DIM, IMAGE_SIZE, DEFAULT_TEACHER_CACHE_DIR,
    SKIP_CLASSES, VAL_DIR,
)
import time, sys, torch, timm, numpy as np
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ============================================================
# EXPERIMENT VARIABLES (agent edits these)
# ============================================================
MODEL_NAME = "hf-hub:timm/lcnet_050.ra2_in1k"
BATCH_SIZE = 256
ARCFACE_BATCH_SIZE = 128
LR = 1e-1
WEIGHT_DECAY = 1e-5
EPOCHS = 10
NUM_WORKERS = 16
SEED = 42
DEVICE = "cuda"
# ... (all other constants)

# ============================================================
# MODEL (agent edits architecture)
# ============================================================
class ProjectionHead(nn.Module): ...
class ArcMarginProduct(nn.Module): ...
class FrozenBackboneWithHead(nn.Module): ...

# ============================================================
# LOSSES (agent edits loss functions)
# ============================================================
def vat_embedding_loss(...): ...

# ============================================================
# AUGMENTATIONS (agent edits transforms)
# ============================================================
class RandomQualityDegradation: ...
def build_train_transform(...): ...

# ============================================================
# TRAINING LOOP (agent edits everything here)
# ============================================================
class EpochStats: ...
def run_train_epoch(...): ...

def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    # Build augmentations (agent controls these)
    train_transform = build_train_transform(...)
    quality_degradation = RandomQualityDegradation(prob=QUALITY_DEGRADATION_PROB)

    # Build datasets (prepare.py controls data, train.py controls transforms)
    distill_dataset = build_distill_dataset(train_transform, quality_degradation)
    arcface_dataset, num_arcface_classes = build_arcface_dataset(train_transform, quality_degradation)
    val_dataset = build_val_dataset(MODEL_NAME, IMAGE_SIZE)

    # Build model, optimizer, etc.
    model = FrozenBackboneWithHead(MODEL_NAME, EMBEDDING_DIM, DEVICE).to(device)
    teacher = init_teacher(DEVICE)
    ...

    # Training loop
    t_start = time.time()
    for epoch in range(EPOCHS):
        stats = run_train_epoch(...)

        # Evaluation (prepare.py controls this)
        if val_dataset is not None:
            retrieval_metrics = run_retrieval_eval(model, val_dataset, ...)
        combined = compute_combined_metric(retrieval_metrics["recall@1"], stats.mean_cosine)

    # Print greppable summary
    elapsed = time.time() - t_start
    print("---")
    print(f"combined_metric:  {combined:.6f}")
    print(f"recall@1:         {recall_1:.6f}")
    print(f"mean_cosine:      {mean_cos:.6f}")
    print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print(f"total_seconds:    {elapsed:.1f}")
    print(f"epochs:           {EPOCHS}")

if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("---")
        print("status: OOM")
        print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
        sys.exit(1)
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (needs install) |
| Config file | none -- see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REFAC-01 | prepare.py and train.py both exist and are importable | smoke | `python -c "import prepare; import train"` | No -- Wave 0 |
| REFAC-02 | Zero evaluation code in train.py | grep check | `! grep -E "recall@|retrieval_eval|combined_metric.*0\.5" train.py` | No -- Wave 0 |
| REFAC-03 | Teacher cache loads and returns correct shape | unit | `pytest tests/test_prepare.py::test_teacher_cache -x` | No -- Wave 0 |
| REFAC-04 | Dataset builders return correct types and sizes | unit | `pytest tests/test_prepare.py::test_dataset_builders -x` | No -- Wave 0 |
| REFAC-05 | train.py has no argparse, only module-level constants | grep check | `! grep -E "argparse|parse_args|add_argument" train.py` | No -- Wave 0 |
| REFAC-06 | model.encode() returns Tensor[B, 256] L2-normalized | unit | `pytest tests/test_train.py::test_encode_contract -x` | No -- Wave 0 |
| REFAC-07 | compute_combined_metric returns 0.5*r1 + 0.5*cos | unit | `pytest tests/test_prepare.py::test_combined_metric -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q` (if tests exist) or grep/import smoke checks
- **Per wave merge:** Full suite
- **Phase gate:** All grep checks pass + import smoke test + combined metric formula verified

### Wave 0 Gaps
- [ ] `tests/test_prepare.py` -- covers REFAC-02, REFAC-03, REFAC-04, REFAC-07
- [ ] `tests/test_train.py` -- covers REFAC-05, REFAC-06
- [ ] `tests/conftest.py` -- shared fixtures (mock teacher, temp directories)
- [ ] Framework install: `pip install pytest` -- pytest not currently installed
- [ ] Note: Full integration tests (actually training for 1 epoch) require GPU and real data; these are manual verification, not CI tests

## State of the Art

| Old Approach (Monolith) | Current Approach (Split) | Impact |
|--------------------------|--------------------------|--------|
| argparse for all config | Module-level constants | Agent edits Python source directly |
| 80 epochs default | 10 epochs fixed budget | Fair experiment comparison |
| Early stopping (patience) | Fixed epoch count, no early stopping | Consistent experiment duration |
| ONNX export in training | No export (deferred to v2) | Faster per-experiment turnaround |
| DINOv2 teacher option | ONNX-only teacher | Simpler, faster teacher inference |
| Checkpoint saving every epoch | No checkpoints (fresh start each experiment) | Simpler, no state carryover |

## Open Questions

1. **Val barcode exclusion from ArcFace**
   - What we know: Monolith excludes val barcodes from ArcFace training set (lines 1304-1308) to prevent data leakage
   - What's unclear: Should this logic live in prepare.py's `build_arcface_dataset` or should train.py pass the exclusion set?
   - Recommendation: prepare.py handles this automatically -- it knows VAL_DIR and can compute the exclusion set. Agent should not need to worry about data leakage.

2. **Whitelist centroid EMA state**
   - What we know: `wl_centroid_ema` dict (line 1410) tracks running average of whitelist embeddings for separation loss
   - What's unclear: This is mutable state that persists across epochs. It is training state, not evaluation state.
   - Recommendation: Lives in train.py. It is part of the training loop's running state, initialized in `main()`.

3. **ArcFace phaseout schedule**
   - What we know: Monolith has `--arcface-phaseout-epoch` (lines 1417-1422) for linear decay of ArcFace weight
   - What's unclear: Whether this should be a train.py constant or removed entirely
   - Recommendation: Keep as `ARCFACE_PHASEOUT_EPOCH = 0` constant in train.py (0 = disabled, matching current effective behavior since unfreeze happens from epoch 0).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | Everything | Yes | 3.12.11 | -- |
| PyTorch + CUDA | Training | Yes | 2.11.0+cu130 | -- |
| timm | Backbone models | Yes | 1.0.26 | -- |
| onnxruntime | Teacher inference | Yes | 1.19.2 | -- |
| ONNX teacher model | Teacher cache | Yes | at `/data/mnt/...` | -- |
| Teacher cache | Fast startup | Yes | `workspace/output/trendyol_teacher_cache2/` | Rebuild (slow first run) |
| Training data | Datasets | Assumed | at `/data/mnt/...` | -- |
| pytest | Testing | No | -- | Install: `pip install pytest` |
| GPU | Training | Assumed | RTX 4090 24GB | -- |

**Missing dependencies with no fallback:** None blocking.
**Missing dependencies with fallback:** pytest (install trivially).

## Sources

### Primary (HIGH confidence)
- `finetune_trendyol_arcface3.py` -- direct line-by-line analysis of 1576-line monolith
- `.planning/phases/01-core-refactoring/01-CONTEXT.md` -- user decisions D-01 through D-08
- `.planning/research/ARCHITECTURE.md` -- detailed split boundary analysis
- `.planning/research/FEATURES.md` -- feature landscape and dependencies
- `.planning/research/PITFALLS.md` -- catalogued pitfalls with prevention strategies
- `.planning/REQUIREMENTS.md` -- REFAC-01 through REFAC-07 specifications

### Secondary (MEDIUM confidence)
- Runtime environment probing (Python 3.12.11, torch 2.11.0, timm 1.0.26, onnxruntime 1.19.2) -- verified via direct import
- Teacher cache existence at `workspace/output/trendyol_teacher_cache2/` -- verified via filesystem check
- ONNX model existence at `/data/mnt/...` path -- verified via filesystem check

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and version-verified
- Architecture: HIGH -- monolith read line-by-line, split boundary clear from CONTEXT.md decisions
- Pitfalls: HIGH -- duplicate PadToSquare bug confirmed by source reading; circular import pattern well-understood
- Validation: MEDIUM -- pytest not installed yet; test patterns are standard but untested

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable -- no external dependency changes expected)
