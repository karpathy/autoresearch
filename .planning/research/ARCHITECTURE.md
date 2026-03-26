# Architecture Patterns

**Domain:** Autonomous ML experimentation for ReID model training
**Researched:** 2026-03-25

## Recommended Architecture

The autoresearch pattern is architecturally simple by design. Three files, one loop, one metric.

```
                     +--------------+
                     |  program.md  |  (human writes agent instructions)
                     +--------------+
                            |
                            v
                     +--------------+
                     |  AI Agent    |  (Claude Code / Codex)
                     |  (the loop)  |
                     +--------------+
                       |    |    |
            edit       |    |    |   read results
                       v    |    v
+------------------+   |  +------------------+
|   train.py       |<--+  |  results.tsv     |
|  (AGENT EDITS)   |      |  (APPEND ONLY)   |
+------------------+      +------------------+
        |                         ^
        | imports from            | logs to
        v                         |
+------------------+      +------------------+
|   prepare.py     |      |    run.log       |
|  (IMMUTABLE)     |      |  (per-run)       |
+------------------+      +------------------+
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| prepare.py | Data loading, teacher init+caching, evaluation, metric calculation, constants | Imported by train.py |
| train.py | Student model definition, loss functions, optimizer, scheduler, augmentations, training loop | Imports prepare.py; writes run.log; agent edits this |
| program.md | Agent instructions: what to search, constraints, ReID domain knowledge | Read by agent at start |
| results.tsv | Experiment log: commit hash, metric, VRAM, status, description | Written by agent after each run |
| run.log | Per-experiment stdout/stderr capture | Written by train.py, read by agent |

### Data Flow

1. Agent reads program.md and results.tsv for context
2. Agent modifies train.py with an experimental idea
3. Agent commits the change to git
4. Agent runs `python train.py > run.log 2>&1`
5. train.py imports evaluation/data from prepare.py
6. prepare.py loads teacher, builds/uses cache, runs evaluation
7. train.py prints summary metrics to stdout (captured in run.log)
8. Agent reads metrics from run.log
9. Agent records in results.tsv
10. If improved: keep commit. If not: `git reset --hard HEAD~1`
11. GOTO 2

## prepare.py Internal Architecture

```python
# Constants (immutable)
EPOCHS = 10
IMAGE_SIZE = 224
EMBEDDING_DIM = 256

# Teacher (immutable)
class TrendyolEmbedder:       # ONNX teacher inference
class DINOv2Teacher:           # HuggingFace teacher inference
def load_teacher_embeddings(): # Cache-aware batch inference

# Data (immutable)
class PadToSquare:             # Image preprocessing
class DistillImageFolder:      # Dataset with path return
class SampledImageFolder:      # Dataset with class sampling
class CombinedDistillDataset:  # Multi-source distillation data
class CombinedArcFaceDataset:  # Multi-source ArcFace data
def build_val_transform():     # Validation transforms
def collate_distill():         # Custom collation
def collate_arcface():         # Custom collation

# Evaluation (immutable)
def evaluate_retrieval():      # recall@1, recall@k computation
def compute_combined_metric(): # 0.5 * recall@1 + 0.5 * mean_cosine
```

## train.py Internal Architecture

```python
# Import from prepare.py
from prepare import (
    TrendyolEmbedder, load_teacher_embeddings,
    CombinedDistillDataset, CombinedArcFaceDataset,
    evaluate_retrieval, compute_combined_metric,
    EPOCHS, IMAGE_SIZE, EMBEDDING_DIM,
)

# Model (agent-editable)
class ProjectionHead:          # Linear projection to embedding space
class ArcMarginProduct:        # ArcFace classification head
class FrozenBackboneWithHead:  # Student model (backbone + projection)

# Losses (agent-editable)
def vat_embedding_loss():      # VAT regularization
# Distillation loss (cosine similarity)
# ArcFace loss (cross-entropy on angular margin)
# Separation loss

# Training (agent-editable)
def build_train_transform():   # Training augmentations
def run_train_epoch():         # Single epoch training loop
def main():                    # Orchestrator: setup, train, evaluate, print summary
```

## Patterns to Follow

### Pattern 1: Import Bridge Between prepare.py and train.py
**What:** train.py imports constants, data loaders, evaluation functions from prepare.py
**When:** Always -- this is the core contract
**Example:**
```python
# train.py
from prepare import (
    build_distill_loader, build_arcface_loader, build_val_loader,
    init_teacher, load_teacher_embeddings,
    evaluate_retrieval, compute_combined_metric,
    EPOCHS, EMBEDDING_DIM,
)
```
**Why:** Clean separation. Agent can change anything in train.py without affecting evaluation integrity.

### Pattern 2: Summary Block at End of Run
**What:** Print a machine-readable summary block that the agent can grep
**When:** End of every training run
**Example:**
```python
print("---")
print(f"combined_metric:  {metric:.6f}")
print(f"recall@1:         {recall_1:.6f}")
print(f"mean_cosine:      {mean_cos:.6f}")
print(f"peak_vram_mb:     {peak_vram:.1f}")
print(f"total_seconds:    {elapsed:.1f}")
print(f"epochs:           {EPOCHS}")
```
**Why:** Agent parses this with grep. Must be stable format even as train.py changes.

### Pattern 3: Graceful OOM Handling
**What:** Wrap training in try/except for CUDA OOM
**When:** Every run -- agent experiments will sometimes exceed 24GB
**Example:**
```python
try:
    main()
except torch.cuda.OutOfMemoryError:
    print("---")
    print("status: OOM")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    sys.exit(1)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Shared Mutable State Between prepare.py and train.py
**What:** Global variables or singletons that train.py modifies and prepare.py reads
**Why bad:** Breaks the immutability contract. Evaluation could depend on training state.
**Instead:** One-way dependency: train.py imports from prepare.py, never the reverse.

### Anti-Pattern 2: Config Files
**What:** Using YAML/JSON/TOML config files for hyperparameters
**Why bad:** Agent must edit code directly. Config files add indirection, are harder to diff, and create ambiguity about what the agent should modify.
**Instead:** Hardcode hyperparameters in train.py. The agent changes them by editing code.

### Anti-Pattern 3: Checkpoint/Resume Logic
**What:** Saving checkpoints and resuming training across runs
**Why bad:** Each experiment must start fresh for fair comparison. Resume logic adds complexity and creates hidden state.
**Instead:** Every run starts from pretrained backbone weights (from timm). 10 epochs from scratch.

### Anti-Pattern 4: Dynamic Imports or Plugin Systems
**What:** Loading loss functions or models from config strings
**Why bad:** Agent should see all code inline. Dynamic loading hides what is being used.
**Instead:** Everything is explicit Python in train.py.

## Scalability Considerations

| Concern | Current (single agent) | Future (multi-agent) |
|---------|----------------------|---------------------|
| GPU contention | Not an issue -- one agent, one GPU | Would need GPU allocation per agent |
| Git conflicts | Not an issue -- single branch | Each agent needs its own branch |
| Teacher cache | Shared across runs (disk + memory) | Shared cache works if agents read-only |
| results.tsv | Append-only, no conflicts | Would need per-agent or locking |
| Experiment rate | ~4-6 per hour (10 epochs each) | Linear scaling with GPUs |

Multi-agent is explicitly out of scope. The architecture supports it naturally (separate branches, shared teacher cache) but no effort should be spent on it.

## Sources

- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) -- canonical three-file pattern
- `finetune_trendyol_arcface3.py` -- current monolith to refactor
- `program.md` -- original autoresearch agent instructions
