# Technology Stack

**Project:** ReID Autoresearch
**Researched:** 2026-03-25

## Critical Constraint: No New Dependencies

The project explicitly forbids adding new pip dependencies (see PROJECT.md "Out of Scope"). The entire stack must be built from what already exists in `requirements.txt` and `pyproject.toml`. This is not a limitation — it is a design feature of the autoresearch pattern. The agent modifies *code*, not infrastructure.

## Existing Stack (from requirements.txt + pyproject.toml)

### Core Framework
| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| PyTorch | 2.9.1 (pinned, CUDA 12.8) | Training framework, autograd, AMP | Already installed |
| timm | latest (unpinned) | Student backbone (LCNet050), pretrained models, data config | Already installed |
| torchvision | latest (unpinned) | Transforms, ImageFolder datasets | Already installed |
| transformers | latest (unpinned) | DINOv2 teacher model loading (HuggingFace) | Already installed |
| onnxruntime-gpu | >=1.17,<1.20 | Trendyol ONNX teacher inference | Already installed |

### Supporting Libraries
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| numpy | >=2.2.6 | Array ops, teacher embedding cache | Already installed |
| Pillow | latest | Image loading and manipulation | Already installed |
| loguru | latest | Structured logging | Already installed |
| matplotlib | >=3.10.8 | Batch visualization, training plots | Already installed |

### Autoresearch Infrastructure (from pyproject.toml, not in requirements.txt)
| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| pandas | >=2.3.3 | Results TSV parsing and analysis | Available but unused in current script |
| pyarrow | >=21.0.0 | Parquet/data format support for pandas | Available |
| requests | >=2.32.0 | HTTP requests (model downloads) | Available |
| kernels | >=0.11.7 | Optimized CUDA kernels (from nanochat) | Available but likely irrelevant for ReID |
| rustbpe / tiktoken | latest | Tokenizer (from nanochat, not needed for ReID) | Irrelevant |

**Confidence:** HIGH — versions verified from `pyproject.toml` and `requirements.txt` in the repo.

## What the Agent Edits vs What Is Fixed

This is the most important architectural decision in the stack. Following Karpathy's pattern:

### Fixed (prepare.py) — Agent CANNOT touch
| Component | Technology | Why Fixed |
|-----------|-----------|-----------|
| Data loading | torchvision.datasets.ImageFolder + custom Dataset classes | Ground truth data pipeline must be immutable |
| Teacher inference | onnxruntime (TrendyolEmbedder) or transformers (DINOv2Teacher) | Teacher is the reference signal |
| Teacher caching | numpy .npy files + in-memory dict | Cache is infrastructure, not an experiment variable |
| Evaluation | Retrieval recall@1/k via cosine similarity matrix | Metric must be immutable for fair comparison |
| Transforms (val) | PadToSquare + Resize + Normalize | Val transforms must match across experiments |

### Editable (train.py) — Agent CAN modify
| Component | Technology | What Can Change |
|-----------|-----------|-----------------|
| Student model | timm (LCNet050 backbone) + ProjectionHead | Architecture, embedding dim, which stages to unfreeze |
| Loss functions | torch.nn.functional (cosine, cross-entropy) | Loss weights, new loss combinations, margin values |
| ArcFace head | Custom ArcMarginProduct | Scale (s), margin (m), feature dim |
| VAT regularization | Custom vat_embedding_loss | Epsilon, xi, power iterations, weight |
| Separation loss | Custom implementation | Weight, centroid EMA |
| Optimizer | torch.optim.SGD | Optimizer type (AdamW, etc.), LR, weight decay, schedule |
| Scheduler | CosineAnnealingLR | Schedule type, warmup, T_max |
| AMP / GradScaler | torch.amp | Enabled/disabled, dtype |
| Augmentations (train) | torchvision.transforms | Augmentation pipeline, probabilities |
| Batch size | DataLoader params | Batch size (within 24GB VRAM) |
| Training loop | Pure Python | Gradient accumulation, epoch structure |

## What NOT to Use (and Why)

### Hyperparameter Search Frameworks (Optuna, Ray Tune, etc.)
**Do NOT use.** The autoresearch pattern replaces these entirely. The LLM agent IS the search algorithm. Traditional HPO tools:
- Require pip install (forbidden)
- Use statistical search (Bayesian, random) which is less flexible than LLM reasoning
- Cannot modify architecture, only tune predefined parameter spaces
- Add complexity without fitting the "agent edits code" paradigm

### ReID-Specific Frameworks (FastReID, Torchreid)
**Do NOT use.** These are monolithic frameworks with their own training loops, configs, and dependencies. They conflict with:
- The single-file train.py constraint
- The no-new-dependencies rule
- The agent-editable simplicity requirement

The current script already implements the key ReID techniques (ArcFace, distillation, retrieval eval) in ~1400 lines of self-contained code. That is the right approach.

### Neural Architecture Search (NAS) Libraries (AutoML, NNI, etc.)
**Do NOT use.** Same reasoning as HPO frameworks. The agent performs architecture search by directly editing model code — changing backbone stages, projection dimensions, loss combinations. This is more flexible than NAS search spaces.

### Weights & Biases / MLflow / Experiment Trackers
**Do NOT use.** The autoresearch pattern uses:
- `results.tsv` for experiment tracking (simple, git-friendly)
- Git commits for versioning each experiment
- `run.log` for per-experiment details
This is deliberate minimalism. Adding W&B would require a dependency and add latency to the loop.

### torch.compile
**Consider carefully.** Available in PyTorch 2.9.1 but:
- Compilation overhead (~30-60s) is significant in 10-epoch runs
- May interfere with agent's ability to modify model architecture dynamically
- Benefits mainly appear at scale (many iterations)
- **Recommendation:** Do NOT enable by default in train.py. The agent can experiment with it as one of its changes.

## Stack Architecture for Autoresearch Split

```
finetune_trendyol_arcface3.py (current monolith, ~1400 lines)
        |
        v
+------------------+     +------------------+
|   prepare.py     |     |    train.py      |
|   (IMMUTABLE)    |     |   (AGENT EDITS)  |
+------------------+     +------------------+
| - Data loading   |     | - Student model  |
| - Teacher init   |     | - Loss functions |
| - Teacher cache  |     | - Optimizer      |
| - Evaluation     |     | - Scheduler      |
| - Val transforms |     | - Train augments |
| - Metric calc    |     | - Training loop  |
| - Results I/O    |     | - Batch size     |
+------------------+     +------------------+
        |                         |
        v                         v
+------------------+     +------------------+
|   program.md     |     |  results.tsv     |
| (HUMAN EDITS)    |     | (APPEND ONLY)    |
+------------------+     +------------------+
```

## Key PyTorch 2.9.1 Features to Leverage

| Feature | Use Case | Confidence |
|---------|----------|------------|
| `torch.amp.autocast` + `torch.amp.GradScaler` | Mixed precision training (already used) | HIGH |
| `torch.nn.functional.normalize` | L2 normalization for embeddings | HIGH |
| `torch.optim.lr_scheduler.CosineAnnealingLR` | LR scheduling (already used) | HIGH |
| Fused AdamW (`torch.optim.AdamW(fused=True)`) | Faster optimizer step on CUDA | HIGH — agent should experiment with this |
| `torch.backends.cudnn.benchmark = True` | Auto-tune convolution algorithms | HIGH — should be set in prepare.py |

## Versions to Pin

The `requirements.txt` is too loose. For reproducibility in autoresearch (where the agent needs stable baselines), these should be tighter:

| Package | Current Pin | Recommended | Reason |
|---------|-------------|-------------|--------|
| torch | ==2.9.1 (pyproject.toml) | ==2.9.1 | Already pinned, good |
| timm | unpinned | Pin to whatever is installed | Backbone behavior must not drift |
| torchvision | unpinned | Pin to match torch 2.9.1 | Transform behavior must be stable |
| onnxruntime-gpu | >=1.17,<1.20 | Pin exact version installed | Teacher inference must be reproducible |

**Confidence:** MEDIUM — version pinning is best practice but the current loose pins haven't caused issues yet.

## Installation

No new installations needed. The existing environment has everything required:

```bash
# Already set up via:
pip install -r requirements.txt
# OR via uv:
uv sync
```

The only "installation" work is refactoring the monolith into prepare.py + train.py, which is a code task, not a dependency task.

## Sources

- [Karpathy autoresearch repo](https://github.com/karpathy/autoresearch) — canonical pattern reference
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) — GradScaler API
- `pyproject.toml` and `requirements.txt` in repo — actual dependency versions
- `finetune_trendyol_arcface3.py` — current implementation showing all used libraries
