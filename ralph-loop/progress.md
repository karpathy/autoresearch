# Experiment Progress

## Current State

- **Best val_bpb**: 1.154745
- **Best commit**: 1d4122d
- **Baseline val_bpb**: 1.192676
- **Total improvement**: -0.037931 (-3.2%)
- **Experiments run**: 28

## GPU Environment

- **GPU**: NVIDIA RTX 4070 Ti SUPER (16GB VRAM)
- **Server**: nigel.birs.ca
- **Max DEVICE_BATCH_SIZE**: 32 (at depth=5). 64 OOMs at depth 8.
- **Working dir**: ~/autoresearch/
- **Run command**: `source ~/.local/bin/env && cd ~/autoresearch && timeout 700 uv run train.py > run.log 2>&1`
- **Steps per 5min**: ~358 (at depth=5, batch=32)
- **MFU**: ~6.7%

## Current Best Hyperparameters

```
DEPTH = 5
DEVICE_BATCH_SIZE = 32
ASPECT_RATIO = 64       # model_dim = 384 (3 heads of 128)
HEAD_DIM = 128
MATRIX_LR = 0.08
EMBEDDING_LR = 1.2
UNEMBEDDING_LR = 0.008
SCALAR_LR = 1.0
WEIGHT_DECAY = 0.2
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.2
ADAM_BETAS = (0.8, 0.95)
TOTAL_BATCH_SIZE = 2**19  # 524K tokens
WINDOW_PATTERN = "S"     # all short, last layer forced long by code
softcap = 15
```

## Strategic Insights

### What works on this GPU
- **Smaller models, more steps**: Depth 5 >> depth 6 >> depth 8. On this GPU, 356 steps at 24.6M params beats 169 steps at 50.3M params.
- **Higher LRs across the board**: All LR defaults need ~2x for short schedules. Matrix: 0.08, Embedding: 1.2, Unembedding: 0.008, Scalar: 1.0.
- **Non-zero final LR**: FINAL_LR_FRAC=0.2 keeps learning through warmdown.
- **Warmdown 0.3**: Sweet spot (0.15 and 0.2 both worse).
- **All-short windows**: Pattern "S" better than "SSSL" (last layer is forced long anyway).

### What fails on this GPU
- **Warmup**: Any warmup wastes steps.
- **Deeper/wider models**: Depth 7+ or wider dims reduce steps too much.
- **Smaller total batch**: 2**18 noisier, worse than 2**19.
- **Larger device batch**: Batch 64 at depth 6 worse (unclear why — same total batch).
- **SwiGLU**: Marginally worse than ReLU² at same param count.
- **All-long windows**: Slower and worse.
- **Softcap 30**: Worse than 15.

### Key principles
1. **Speed > capacity** on this GPU. Throughput (steps) matters more than model size.
2. **All defaults need LR rescaling** for ~350-step regime.
3. **Architecture simplifications help** when they increase throughput.

### Remaining high-value ideas
- Remove value embeddings (simplification + speed)
- Tied embeddings (fewer params)
- x0_lambdas tuning (initial value 0.1)
- Weight decay 0.3 (try increasing instead of decreasing)
- Muon momentum warmup schedule changes

## Experiment History

| # | Experiment | val_bpb | Status | Insight |
|---|-----------|---------|--------|---------|
| 0 | Baseline (batch=16) | 1.191909 | keep | Initial |
| 0b | Baseline (batch=32) | 1.192676 | keep | Max batch, same perf |
| 1 | 5% LR warmup | 1.244263 | discard | Warmup wastes steps |
| 2a | Depth 12, batch=32 | crash | crash | OOM |
| 2b | Depth 10, batch=32 | crash | crash | OOM |
| 2c | Depth 10, batch=16 | 1.373499 | discard | Too few steps |
| 3 | Matrix LR 0.06, Emb LR 0.9 | 1.181332 | keep | Higher LR helps |
| 4 | Matrix LR 0.08, Emb LR 1.2 | 1.178592 | keep | Even higher |
| 5 | Matrix LR 0.12, Emb LR 1.8 | 1.183606 | discard | Too high |
| 6 | Warmdown 0.5→0.3 | 1.177074 | keep | Less cooldown |
| 7 | Warmdown 0.3→0.15 | 1.180232 | discard | Too aggressive |
| 8 | FINAL_LR_FRAC 0.0→0.1 | 1.174359 | keep | Non-zero floor |
| 9 | FINAL_LR_FRAC 0.1→0.2 | 1.174092 | keep | Marginal improvement |
| 10 | FINAL_LR_FRAC 0.2→0.3 | 1.174884 | discard | Too high |
| 11 | Weight decay 0.2→0.1 | 1.175010 | discard | Slightly worse |
| 12 | Unembedding LR 0.004→0.008 | 1.170299 | keep | LR scaling win |
| 13 | Unembedding LR 0.008→0.016 | 1.176043 | discard | Too high |
| 14 | Softcap 15→30 | 1.177892 | discard | Worse |
| 15 | Scalar LR 0.5→1.0 | 1.169471 | keep | LR scaling win |
| 16 | Adam beta1 0.8→0.85 | 1.169895 | discard | Marginally worse |
| 17 | Depth 8→6 | 1.157848 | keep | MASSIVE win: fewer params, more steps |
| 18 | Depth 6→4 | 1.184572 | discard | Too small (11.5M) |
| 19 | Batch 32→64 at depth 6 | 1.184304 | discard | Worse (unclear why) |
| 20 | Total batch 2**19→2**18 | 1.175643 | discard | Noisier gradients |
| 21 | SwiGLU activation | 1.159808 | discard | Marginally worse |
| 22 | Window SSSL→SSLL | 1.159303 | discard | Marginally worse |
| 23 | Matrix LR 0.08→0.10 | 1.157981 | discard | No improvement |
| 24 | Embedding LR 1.2→1.5 | 1.159335 | discard | Worse |
| 25 | Depth 6→5 | 1.157145 | keep | Even smaller better |
| 26 | Depth 5→7 | 1.215677 | discard | Too deep too slow |
| 27 | Aspect 64→80 at depth 5 | 1.178817 | discard | Wider = slower |
| 28 | Aspect 64→48 at depth 5 | 1.166972 | discard | Too small (14.4M) |
| 29 | HEAD_DIM 128→64 | 1.172555 | discard | Too small (19.3M) |
| 30 | Warmdown 0.3→0.2 | 1.159932 | discard | Worse |
| 31 | Window all-long L | 1.160776 | discard | Slower |
| 32 | Window all-short S | 1.154745 | keep | Faster + better quality |
