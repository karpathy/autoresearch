# Zero-to-One Guide: From Karpathy's Videos to Running autoresearch

## Context

You're a beginner learning deep learning through Karpathy's "Neural Networks: Zero to Hero" series, currently at **makemore 3** (activations, gradients, BatchNorm). The goal is to create a comprehensive guide that maps every concept in the `autoresearch` codebase to Karpathy's teaching, identifies what you still need to learn, and gives you a clear path from where you are now to running your own model training sessions.

The `autoresearch` codebase is a single-GPU LLM training framework created by Karpathy himself. It trains a GPT-style transformer on a text dataset in **5-minute time-budgeted experiments**. The entire trainable code lives in two files: `train.py` (model + optimizer + training loop) and `prepare.py` (data loading + tokenizer + evaluation).

---

## Part 1: Karpathy's Video Series — Complete Curriculum

| # | Video | Duration | Status for You |
|---|-------|----------|---------------|
| 1 | **Building Micrograd** — backpropagation, autograd engine from scratch | 2h25m | Done |
| 2 | **Makemore 1** — bigram language model, PyTorch tensors, loss functions | 1h57m | Done |
| 3 | **Makemore 2: MLP** — multilayer perceptrons, learning rates, train/val/test splits | 1h15m | Done |
| 4 | **Makemore 3: Activations & Gradients, BatchNorm** — activation statistics, gradient flow, batch normalization | 1h55m | **You are here** |
| 5 | **Makemore 4: Becoming a Backprop Ninja** — manual backprop through cross-entropy, matrix ops | 1h56m | Next |
| 6 | **Makemore 5: WaveNet** — deeper architectures, hierarchical convolutions, torch.nn | 56m | Next |
| 7 | **Let's Build GPT** — transformer from scratch, self-attention, multi-head attention | 1h56m | **Critical for autoresearch** |
| 8 | **Let's Build the GPT Tokenizer** — BPE algorithm, encode/decode, tiktoken | 2h13m | **Critical for autoresearch** |
| 9 | **Reproducing GPT-2 (124M)** — full training pipeline, AdamW, gradient accumulation, mixed precision, distributed training | ~4h | **Most directly relevant** |

---

## Part 2: Concept Map — Codebase to Videos

### Concepts You Already Know (Videos 1-4)

| Concept | Where in Codebase | Which Video |
|---------|-------------------|-------------|
| Backpropagation / `loss.backward()` | `train.py:549` | Video 1 (Micrograd) |
| Cross-entropy loss | `train.py:289` `F.cross_entropy(...)` | Video 2 (Makemore 1) |
| Negative log likelihood | Used for BPB metric in `prepare.py:356` | Video 2 |
| PyTorch tensors & operations | Everywhere | Video 2 |
| Train/val splits | `prepare.py` train vs val shards | Video 3 (Makemore 2: MLP) |
| Learning rate tuning | `train.py:432-440` multiple LR params | Video 3 |
| Weight initialization | `train.py:149-182` | Video 3 |
| Activation statistics / gradient health | The entire design of BatchNorm → replaced by RMSNorm here | Video 4 (Makemore 3) **You are here** |
| Batch Normalization concept | Not used in codebase (replaced by RMSNorm), but understanding it helps | Video 4 |

### Concepts You'll Learn Next (Videos 5-6)

| Concept | Where in Codebase | Which Video |
|---------|-------------------|-------------|
| Manual backprop through complex ops | Understanding what `loss.backward()` does internally | Video 5 (Backprop Ninja) |
| Deeper network architectures | `train.py` 8-layer transformer | Video 6 (WaveNet) |
| `torch.nn` module system | `nn.Embedding`, `nn.Linear`, `nn.Module` throughout `train.py` | Video 6 |
| Hierarchical feature extraction | Attention layers building representations | Video 6 |

### Concepts in "Let's Build GPT" (Video 7) — The Big Leap

This is the **most important video** for understanding autoresearch. Nearly every architectural concept maps here:

| Concept | Where in Codebase | Notes |
|---------|-------------------|-------|
| **Self-attention mechanism** | `train.py:61-96` `CausalSelfAttention` class | Q, K, V projections, attention scores |
| **Multi-head attention** | `train.py:66-68` `n_head`, `n_kv_head` | Multiple attention heads in parallel |
| **Causal masking** | `train.py:93` `causal=True` in flash_attn | Prevents attending to future tokens |
| **MLP (feed-forward) block** | `train.py:99-109` `MLP` class | Expansion → activation → projection |
| **Residual connections** | `train.py:118-120` `x = x + attn(...)` | Skip connections around attention & MLP |
| **Layer normalization** | `train.py:43` (RMSNorm variant) | Pre-norm architecture |
| **Token embeddings** | `train.py:130` `nn.Embedding` | Converting token IDs to vectors |
| **Position information** | `train.py:52-58` (RoPE, more advanced than video) | How model knows token position |
| **Autoregressive generation** | Forward pass predicts next token | Core GPT concept |
| **Transformer block stacking** | `train.py:112-121` `TransformerBlock` | Multiple layers of attention+MLP |

### Concepts in "GPT Tokenizer" (Video 8)

| Concept | Where in Codebase | Notes |
|---------|-------------------|-------|
| **Byte Pair Encoding (BPE)** | `prepare.py:157-175` tokenizer training | Core tokenization algorithm |
| **Encode/decode functions** | `prepare.py:141-152` Tokenizer class | Text ↔ token IDs |
| **Vocabulary size choice** | `prepare.py:46` `VOCAB_SIZE = 8192` | Smaller than GPT-2's 50257 |
| **Special tokens** | `prepare.py:50-51` BOS token | Reserved tokens for control |
| **tiktoken library** | `prepare.py:170` builds tiktoken.Encoding | OpenAI's fast tokenizer |
| **Token-to-bytes mapping** | `prepare.py:184-196` for BPB metric | Understanding token granularity |

### Concepts in "Reproducing GPT-2" (Video 9) — Training at Scale

| Concept | Where in Codebase | Notes |
|---------|-------------------|-------|
| **AdamW optimizer** | `train.py:305-314` fused AdamW | Weight decay decoupled from gradient |
| **Fused optimizer kernels** | `train.py:306` `@torch.compile` | GPU-optimized optimizer steps |
| **Mixed precision (bfloat16)** | `train.py:462-463` `autocast(..., dtype=torch.bfloat16)` | Half-precision for speed |
| **Gradient accumulation** | `train.py:546-551` micro-step loop | Simulating large batches on limited VRAM |
| **Learning rate warmup** | `train.py:516-525` `get_lr_multiplier()` | Gradual LR increase at start |
| **Learning rate cosine decay** | `train.py:521-525` warmdown schedule | Smooth LR decrease |
| **Weight decay** | `train.py:441` `WEIGHT_DECAY = 0.2` | Regularization |
| **Batch size (token-based)** | `train.py:430` `TOTAL_BATCH_SIZE = 2**19` (~524K tokens) | Large effective batch |
| **Model FLOPs Utilization** | `train.py:208-222` MFU calculation | Measuring GPU efficiency |
| **torch.compile** | `train.py:503` model compilation | JIT optimization |
| **Weight tying** | Not used here (separate embed/unembed) | Discussed in video |
| **Loss sanity checks** | `train.py:569-572` NaN/explosion detection | Verifying training stability |
| **Evaluation during training** | `prepare.py:343-365` `evaluate_bpb()` | Validation metric computation |

---

## Part 3: Advanced Concepts BEYOND Karpathy's Videos

These concepts in the codebase are **not covered** in the Zero to Hero series. You'll need to learn them from papers, blog posts, or additional resources:

### Architecture Advances

| Concept | Where in Codebase | What to Learn |
|---------|-------------------|---------------|
| **RMSNorm** (instead of LayerNorm) | `train.py:43` `F.rms_norm()` | Simpler normalization: just divides by RMS, no mean subtraction. Paper: "Root Mean Square Layer Normalization" (2019) |
| **Rotary Position Embeddings (RoPE)** | `train.py:52-58` | Encodes position by rotating Q/K vectors. Much better than learned positional embeddings from the GPT video. Paper: "RoFormer" (2021) |
| **Flash Attention 3** | `train.py:93` `fa3.flash_attn_func()` | Memory-efficient exact attention. Doesn't change the math, just makes it fast. Paper: "FlashAttention" (Dao 2022) |
| **Grouped Query Attention (GQA)** | `train.py:66-68` `n_kv_head` | Shares K/V heads across Q heads to save memory. Paper: "GQA" (2023) |
| **Sliding Window Attention** | `train.py:195-206` SSSL pattern | Some layers only attend to nearby tokens. Used in Mistral. |
| **ReLU squared activation** | `train.py:107` `F.relu(x).square()` | Sharper activation than GELU/ReLU. Paper: "Primer" (2021) |
| **Logit softcap** | `train.py:285` `softcap * tanh(logits/softcap)` | Prevents extreme logit values. Used in Gemma 2. |
| **Value Embeddings (ResFormer)** | `train.py:74-87` | Adds input-dependent values from a separate embedding table. Paper: "ResFormer" |
| **Per-layer residual scaling** | `train.py:134-135` `resid_lambdas`, `x0_lambdas` | Learned scaling for residual connections + skip to input. Stabilizes deep networks. |

### Optimizer Advances

| Concept | Where in Codebase | What to Learn |
|---------|-------------------|---------------|
| **Muon Optimizer** | `train.py:316-418` | Novel optimizer using matrix orthogonalization for weight updates. Much faster than Adam for matrix params. Blog: "Muon optimizer" by Jordan Cheun |
| **Polar Express orthogonalization** | `train.py:323-335` | Approximates SVD to orthogonalize gradient updates |
| **NorMuon variance reduction** | `train.py:337-348` | Channel-wise variance normalization for adaptive step sizes |
| **Nesterov momentum** | `train.py:320-322` | "Look-ahead" momentum variant (slightly covered in theory, not in videos) |
| **Cautious weight decay** | `train.py:352-353` | Only decay when gradient agrees with parameter sign |
| **Per-parameter-group LRs** | `train.py:432-440` | Different learning rates for embeddings vs matrices vs scalars |
| **Momentum scheduling** | `train.py:527-529` | Muon momentum increases from 0.85→0.95 over training |

### Training Infrastructure

| Concept | Where in Codebase | What to Learn |
|---------|-------------------|---------------|
| **`uv` package manager** | `pyproject.toml` | Modern Python package manager (replaces pip/conda) |
| **Parquet data format** | `prepare.py` data loading | Columnar storage for efficient data loading |
| **Best-fit document packing** | `prepare.py:276-337` | Packs documents into fixed-length sequences with zero waste |
| **Bits per byte (BPB) metric** | `prepare.py:343-365` | Vocab-size-independent evaluation metric |
| **Time-budgeted training** | `train.py:593-598` 5-minute budget | Fixed compute budget for comparable experiments |

---

## Part 4: Your Learning Path — Step by Step

### Phase 1: Finish the Video Series (Where You Are Now)

**Step 1: Complete Makemore 3** (you're here)
- Focus on: understanding why activations and gradients can explode/vanish
- This directly explains why autoresearch uses RMSNorm and careful initialization

**Step 2: Makemore 4 — Backprop Ninja**
- Solidifies your understanding of what `loss.backward()` computes
- After this, gradient accumulation in `train.py:546-551` will make sense

**Step 3: Makemore 5 — WaveNet**
- Learn `torch.nn` module system (Module, Linear, Embedding)
- After this, you can read `train.py`'s class structure

**Step 4: Let's Build GPT** (THE key video)
- This is where everything clicks for autoresearch
- After this video, you'll understand ~70% of `train.py`
- Pay special attention to: self-attention, multi-head attention, transformer blocks, residual connections

**Step 5: Let's Build the GPT Tokenizer**
- After this, you'll understand `prepare.py`'s tokenizer code completely
- Focus on BPE algorithm and the encode/decode pipeline

**Step 6: Reproducing GPT-2** (THE training video)
- This is the closest to what autoresearch does
- Covers: AdamW, gradient accumulation, mixed precision, learning rate scheduling
- After this, you'll understand ~85% of the training loop in `train.py`

### Phase 2: Bridge the Gap (Advanced Concepts)

After finishing all videos, these are the concepts to study before diving into autoresearch. Ordered by importance:

1. **RMSNorm** — Read a short blog post. It's just LayerNorm without the mean subtraction. 5 minutes.
2. **Rotary Position Embeddings (RoPE)** — Watch a YouTube explainer (e.g., "RoPE explained"). The key idea: rotate Q and K vectors by position-dependent angles so their dot product encodes relative position. 30 minutes.
3. **Flash Attention** — You don't need to understand the algorithm, just know it computes exact attention with O(N) memory instead of O(N^2). It's a drop-in replacement. 10 minutes.
4. **Grouped Query Attention** — Simple: instead of separate K,V per head, share them across groups. Saves memory. 10 minutes.
5. **Muon Optimizer** — This is the most novel concept. Read Jordan Cheun's blog post. Key idea: for matrix parameters, orthogonalize the gradient update direction. 1 hour.
6. **Mixed Precision (bfloat16)** — Covered in GPT-2 video. Use half-precision for forward/backward, full precision for optimizer state. 15 minutes.

### Phase 3: Hands-On with autoresearch

**Step 1: Set up the environment**
```bash
# Install uv (Python package manager, like pip but faster)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the repo
cd /home/user/autoresearch

# Install all dependencies (PyTorch + CUDA, Flash Attention, etc.)
uv sync

# Download data and train tokenizer (~2 minutes, one-time)
uv run prepare.py
```

**Step 2: Read the code in this order**
1. `prepare.py` lines 46-51 — constants (vocab size, sequence length)
2. `prepare.py` lines 141-203 — tokenizer (maps to GPT Tokenizer video)
3. `prepare.py` lines 254-365 — dataloader and evaluation
4. `train.py` lines 29-42 — model config dataclass
5. `train.py` lines 61-96 — attention (maps to "Let's Build GPT")
6. `train.py` lines 99-109 — MLP block
7. `train.py` lines 112-121 — transformer block (residual connections)
8. `train.py` lines 125-291 — full GPT model (forward pass)
9. `train.py` lines 296-427 — optimizer (MuonAdamW — the advanced part)
10. `train.py` lines 429-452 — hyperparameters (what you'll experiment with)
11. `train.py` lines 537-605 — training loop (maps to GPT-2 reproduction video)

**Step 3: Run your first training experiment**
```bash
# Run training (takes ~5 minutes + startup overhead)
uv run train.py
```
This will output metrics like:
```
val_bpb:          0.997900    # Lower is better
training_seconds: 300.1       # Fixed 5-minute budget
peak_vram_mb:     45060.2     # GPU memory used
mfu_percent:      39.80       # How efficiently you're using the GPU
total_tokens_M:   499.6       # Millions of tokens processed
num_params_M:     50.3        # Model size
```

**Step 4: Start experimenting**
The framework is designed for you to modify hyperparameters in `train.py` lines 429-452 and observe the effect on `val_bpb`. Try:
- Changing `DEPTH` (number of layers): 4 vs 8 vs 12
- Changing `DEVICE_BATCH_SIZE`: affects gradient accumulation steps
- Changing learning rates: `MATRIX_LR`, `EMBEDDING_LR`
- Changing `WEIGHT_DECAY`

---

## Part 5: Key Files Reference

| File | Purpose | Editable? |
|------|---------|-----------|
| `train.py` | Model architecture, optimizer, training loop, hyperparameters | Yes — this is what you modify |
| `prepare.py` | Data download, tokenizer, dataloader, evaluation function | No — read-only infrastructure |
| `program.md` | Instructions for autonomous agent experiments | Reference only |
| `README.md` | Project documentation | Reference only |
| `analysis.ipynb` | Jupyter notebook for analyzing experiment results | Yes — for analysis |
| `pyproject.toml` | Python dependencies | Reference only |

---

## Part 6: Concept Difficulty Ladder

From what you know now to what's in the codebase, ranked by conceptual distance:

```
WHAT YOU KNOW (Videos 1-4)
  |  Backpropagation, loss functions, MLPs, batch norm, activations
  |
  v  [Complete Videos 5-6: ~3 hours]
TORCH.NN & DEEPER NETWORKS
  |  Module system, deeper architectures, convolutions
  |
  v  [Complete Video 7: ~2 hours]  ← BIGGEST LEAP
TRANSFORMER / GPT ARCHITECTURE
  |  Self-attention, multi-head attention, residual connections
  |  transformer blocks, autoregressive generation
  |
  v  [Complete Video 8: ~2 hours]
TOKENIZATION (BPE)
  |  Byte pair encoding, encode/decode, vocabulary
  |
  v  [Complete Video 9: ~4 hours]
FULL TRAINING PIPELINE
  |  AdamW, gradient accumulation, mixed precision, LR scheduling
  |
  v  [Self-study: ~3-5 hours]
ADVANCED CONCEPTS (beyond videos)
  |  RMSNorm, RoPE, Flash Attention, GQA, Muon optimizer
  |  Sliding window attention, value embeddings, logit softcap
  |
  v
READY TO RUN AND MODIFY AUTORESEARCH
```

**Estimated total learning time from where you are: ~15-20 hours of video + study**

---

## Verification

After completing the learning path and setup:
1. Run `uv run prepare.py` — should download data and create tokenizer in `~/.cache/autoresearch/`
2. Run `uv run train.py` — should complete a 5-minute training run and output `val_bpb`
3. Modify a hyperparameter (e.g., change `DEPTH = 8` to `DEPTH = 4` in `train.py:451`)
4. Run again and compare `val_bpb` — you're now doing the same thing the autonomous agent does
