# Rust/CUDA Training Backend

A high-performance Rust replacement for `train.py`, targeting H100 GPUs with Flash Attention 3.

## Results

Trained for 1000 steps on the same dataset and seed (42) as the Python baseline:

| | Python baseline | This backend |
|---|---|---|
| Implementation | Python / PyTorch | Rust / custom CUDA |
| Attention | Standard | Flash Attention 3 (bf16, sm90) |
| Attention pattern | Full | Sliding window (SSSSL) |
| Depth | 12 layers | 30 layers |
| d_model | 768 | 512 |
| Params | ~124M | ~120M |
| val_bpb (1000 steps) | 0.9921 | **0.8673** |
| Improvement | — | **−0.1248 bpb (−12.6%)** |

## Architecture

The key insight is that with the same parameter budget, depth scales better than width at this model size. Going from 12 layers at d_model=768 to 30 layers at d_model=512 consistently improves validation BPB.

Additional modifications validated through ablation:

| Component | Change | Effect |
|-----------|--------|--------|
| Attention | Sliding window SSSSL pattern (256,256,256,256,2048) | Reduces memory, allows depth=30 |
| Position | RoPE (base=200K, softcap=15.0) | Stable long-context training |
| MLP | Value Enhancement gating on alternating layers | +0.003 bpb |
| Optimizer | Muon (matrix weights) + AdamW (embeddings) | Standard Muon config |
| Cooldown | 50% of total steps | Optimal vs 25%/75% |
| Embedding LR | 0.9× peak LR | −0.007 bpb vs default |
| Warmdown ratio | 0.5 | Optimal LR decay shape |

**On neuron rinsing**: We tested dynamic layer reinit at 0%, 2%, 5%, and 10% dead-neuron thresholds. Zero reinit events fired across all thresholds. Muon's Newton-Schulz orthogonalization continuously redistributes gradient energy, preventing neuron death entirely. Rinsing code is present but disabled.

## Model config

```
N_LAYER=30, d_model=512, n_head=4, head_dim=128, vocab=8192, seq_len=2048
Attention window pattern: [256, 256, 256, 256, 2048] x 6  (SSSSL repeating)
RoPE base=200K, softcap=15.0
Value Enhancement on alternating layers (gate_ch=32)
119.5M parameters
Optimizer: Muon (matrix weights) + AdamW (embeddings/biases/scalars)
peak_lr=0.04, embedding_lr=0.9, warmdown_ratio=0.5, weight_decay=0.2
```

## Requirements

- CUDA 12.x, H100 GPU (sm90) for Flash Attention 3
- Rust nightly (edition 2024)
- Python 3.10+ for feeder.py

## Build

```bash
# Optional: build Flash Attention 3 first (requires H100, ~10 min)
bash brain/fa3/build_fa3.sh

# Build the binary
FLASH_ATTN_V3_BUILD_DIR=brain/fa3/build cargo build --release
```

## Run

```bash
# Stream training data from HuggingFace and pipe into the binary
NUM_TRAIN_SHARDS=794 python3 feeder.py --stream --prefetch 4 2>feeder.log \
  | MAX_STEPS=1000 COOLDOWN_STEPS=500 WARMDOWN_RATIO=0.5 \
    ./target/release/autoresearch-brain train \
      --stream-input \
      --data-dir /path/to/packed/val/shards \
      --seed 42 \
    > train.log 2>&1
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| MAX_STEPS | inf | Total training steps |
| COOLDOWN_STEPS | 500 | Steps in WSD cooldown phase |
| WARMDOWN_RATIO | 0.75 | LR decay shape during cooldown (use 0.5) |
| BATCH_SIZE | 64 | Device batch size |
| TOTAL_BATCH | 524288 | Total tokens per step |
| PEAK_LR | 0.04 | Peak learning rate |
| CHECKPOINT_EVERY | 100 | Checkpoint interval (steps) |

## Data pipeline

Training data streams from HuggingFace parquets via `feeder.py` (best-fit bin-packing, no padding waste). Val data is read from pre-packed local shard files. See `prepare.py` for shard preparation.

Val shards use shard index 6542 as the fixed validation set (same as Python baseline).
