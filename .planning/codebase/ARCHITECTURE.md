# Architecture

**Analysis Date:** 2026-03-25

## Pattern Overview

**Overall:** Single-file monolithic scripts with a two-stage pipeline (prepare then train)

**Key Characteristics:**
- Minimal file count by design: 3 core files (`prepare.py`, `train.py`, `program.md`)
- No module hierarchy or package structure; everything lives at project root
- `train.py` is the sole mutable file in the AI-agent experiment loop; `prepare.py` is read-only at runtime
- Script-level execution (no CLI framework beyond argparse in `prepare.py`; `train.py` has zero CLI flags)
- Configuration via module-level constants, not config files or environment variables
- `finetune_trendyol_arcface3.py` is a separate, unrelated fine-tuning script (computer vision, not LLM pretraining)

## Layers

**Data Preparation Layer:**
- Purpose: One-time download of training data shards from HuggingFace and BPE tokenizer training
- Location: `prepare.py` (lines 56-203, the `download_data()` and `train_tokenizer()` functions, plus the `__main__` block)
- Contains: Parquet shard download with retries, rustbpe tokenizer training, tiktoken encoding creation
- Depends on: HuggingFace dataset `karpathy/climbmix-400b-shuffle`, `rustbpe`, `tiktoken`, `pyarrow`
- Used by: `train.py` imports runtime utilities from this module
- Cache location: `~/.cache/autoresearch/` (data shards in `data/`, tokenizer in `tokenizer/`)

**Runtime Utilities Layer:**
- Purpose: Tokenizer wrapper, dataloader, and evaluation function used during training
- Location: `prepare.py` (lines 209-365, the `Tokenizer` class, `make_dataloader()`, `evaluate_bpb()`)
- Contains: `Tokenizer` class (wraps tiktoken), `make_dataloader()` (BOS-aligned best-fit document packing), `evaluate_bpb()` (bits-per-byte metric)
- Depends on: Cached tokenizer pickle, cached parquet data shards, PyTorch
- Used by: `train.py` imports `MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `make_dataloader`, `evaluate_bpb`

**Model Layer:**
- Purpose: GPT language model definition with modern architecture features
- Location: `train.py` (lines 30-291)
- Contains: `GPTConfig` dataclass, `GPT` model class, `CausalSelfAttention`, `MLP`, `Block` modules
- Depends on: PyTorch, Flash Attention 3 via `kernels` package
- Used by: Training loop in `train.py`

**Optimizer Layer:**
- Purpose: Custom combined optimizer (Muon for 2D matrices, AdamW for everything else)
- Location: `train.py` (lines 293-427)
- Contains: `MuonAdamW` class, `adamw_step_fused()` and `muon_step_fused()` compiled kernels, polar express orthogonalization coefficients
- Depends on: PyTorch, `torch.compile`
- Used by: Training loop in `train.py`

**Training Loop Layer:**
- Purpose: Single-GPU training with time-budget termination, gradient accumulation, LR scheduling
- Location: `train.py` (lines 429-631)
- Contains: Hyperparameter constants, model/optimizer construction, training loop, evaluation, logging
- Depends on: Model layer, optimizer layer, runtime utilities from `prepare.py`
- Used by: Direct execution (`uv run train.py`)

## Data Flow

**Preparation Flow (one-time):**

1. `prepare.py __main__` parses args (num shards)
2. `download_data()` downloads parquet shards from HuggingFace to `~/.cache/autoresearch/data/`
3. `train_tokenizer()` reads text from parquet, trains BPE via rustbpe, saves tiktoken pickle + token_bytes tensor to `~/.cache/autoresearch/tokenizer/`

**Training Flow (each experiment):**

1. `train.py` module-level code loads Flash Attention 3 kernel via `kernels.get_kernel()` (selects repo based on GPU capability)
2. `Tokenizer.from_directory()` loads cached tiktoken pickle
3. `build_model_config(DEPTH)` derives model dimensions from depth and aspect ratio constants
4. `GPT` model constructed on `meta` device, moved to CUDA, weights initialized via `init_weights()`
5. `model.setup_optimizer()` creates `MuonAdamW` with per-parameter-group learning rates (embeddings via AdamW, matrices via Muon)
6. `torch.compile(model)` compiles the model
7. `make_dataloader()` creates infinite iterator yielding `(inputs, targets, epoch)` tensors via best-fit document packing into fixed-size batches
8. Training loop runs for `TIME_BUDGET` seconds (300s), with gradient accumulation, LR warmup/warmdown schedule
9. After time budget expires, `evaluate_bpb()` computes validation bits-per-byte on pinned validation shard
10. Summary stats printed to stdout

**State Management:**
- No checkpointing or model saving; training is ephemeral (5-minute runs)
- Experiment results tracked externally in `results.tsv` (untracked by git)
- Git branch state serves as the experiment version control (`git reset` to discard failed experiments)

## Key Abstractions

**GPTConfig:**
- Purpose: Immutable model configuration (dimensions, heads, layers, window pattern)
- Examples: `train.py` line 33
- Pattern: Python `@dataclass` with defaults; constructed by `build_model_config(depth)` at line 469

**Tokenizer:**
- Purpose: Wraps tiktoken encoding with BOS token support and batch encoding
- Examples: `prepare.py` line 209
- Pattern: Class with `from_directory()` classmethod factory; delegates to tiktoken `Encoding`

**MuonAdamW:**
- Purpose: Hybrid optimizer dispatching Muon (matrix orthogonalization) for 2D params and AdamW for others
- Examples: `train.py` line 356
- Pattern: Subclasses `torch.optim.Optimizer`; parameter groups tagged with `kind='adamw'` or `kind='muon'`; uses 0-D CPU tensors for `torch.compile` compatibility

**make_dataloader:**
- Purpose: Infinite generator yielding packed, BOS-aligned training batches with zero padding waste
- Examples: `prepare.py` line 276
- Pattern: Generator function with best-fit bin-packing; pre-allocates pinned CPU and GPU buffers; uses `non_blocking` async transfer

## Entry Points

**`prepare.py` (standalone execution):**
- Location: `prepare.py` line 371
- Triggers: Manual one-time execution (`uv run prepare.py`)
- Responsibilities: Download data shards, train BPE tokenizer, save to cache directory

**`train.py` (standalone execution):**
- Location: `train.py` line 1 (module-level, no `if __name__` guard)
- Triggers: Each experiment run (`uv run train.py`)
- Responsibilities: Build model, train for TIME_BUDGET seconds, evaluate, print results

**`finetune_trendyol_arcface3.py` (standalone execution):**
- Location: `finetune_trendyol_arcface3.py`
- Triggers: Manual execution for computer vision fine-tuning (separate from autoresearch LLM pipeline)
- Responsibilities: Qwen distillation + ArcFace fine-tuning on retail product dataset

## Error Handling

**Strategy:** Fail-fast with minimal error handling

**Patterns:**
- Training loop checks for NaN/exploding loss at `train.py` line 570: `if math.isnan(train_loss_f) or train_loss_f > 100: exit(1)`
- Data download retries with exponential backoff in `prepare.py` line 66
- Assertions used liberally for invariant checking (dimension divisibility, shard existence, split validity)
- No try/except in the training loop; crashes propagate to the AI agent which reads `tail -n 50 run.log`
- `prepare.py` line 261: asserts training shards exist to guard against infinite loop

## Cross-Cutting Concerns

**Logging:** Print statements to stdout; no structured logging framework. Training progress uses `\r` carriage return for in-place updates (line 590). The AI agent redirects output to `run.log` and greps for metrics.

**Validation:** Assertions throughout for shape/dimension invariants. No input validation beyond what assertions provide.

**Authentication:** None. Data downloaded from public HuggingFace dataset without auth.

**Compilation:** `torch.compile` used for both the model (line 508) and optimizer step functions (lines 305-306, 316-317) with `dynamic=False, fullgraph=True`.

**Memory Management:** Manual GC management in training loop (lines 593-598): GC disabled after first step, explicit collect every 5000 steps to avoid ~500ms stalls. `PYTORCH_ALLOC_CONF=expandable_segments:True` set at module import time.

---

*Architecture analysis: 2026-03-25*
