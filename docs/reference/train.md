# Reference: `train.py`

`train.py` is the only file the agent edits. It defines the model, the optimizer, the schedules, and the training loop. This page documents the surface as it exists today (i.e., the *baseline* the agent starts from). Internals — what each piece *does* — are split into [internals/model.md](../internals/model.md) and [internals/optimizer.md](../internals/optimizer.md).

## CLI

```bash
uv run train.py
```

No flags. Configuration is by editing the constants block (see below). On startup, `train.py`:

1. Imports `MAX_SEQ_LEN`, `TIME_BUDGET`, `Tokenizer`, `make_dataloader`, `evaluate_bpb` from `prepare.py`.
2. Loads the tokenizer from `~/.cache/autoresearch/tokenizer/`.
3. Builds the model on CUDA via `to_empty` + `init_weights`.
4. Compiles it with `torch.compile(model, dynamic=False)`.
5. Trains for `TIME_BUDGET` seconds (compilation/warmup excluded).
6. Runs `evaluate_bpb` once.
7. Prints a fixed summary block.

## Hyperparameter block

The block of module-level constants that controls a run. Lives near the bottom of `train.py` (around line 432). Everything in this block is fair game for the agent; the values listed are the H100 baseline.

```python
# Model architecture
ASPECT_RATIO = 64        # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128           # target head dimension for attention
WINDOW_PATTERN = "SSSL"  # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6       # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004   # learning rate for lm_head (Adam)
MATRIX_LR = 0.04         # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5          # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2       # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0       # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5     # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0      # final LR as fraction of initial

# Model size
DEPTH = 8                # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
```

### Architecture knobs

| Constant | Default | Effect |
|---|---|---|
| `ASPECT_RATIO` | 64 | Sets `n_embd = DEPTH * ASPECT_RATIO`, then rounds up to a multiple of `HEAD_DIM`. With `DEPTH=8` and `HEAD_DIM=128` this yields `n_embd = 512` (depth × 64 = 512, already a multiple of 128). |
| `HEAD_DIM` | 128 | Target attention head dimension. `n_head = n_embd // HEAD_DIM`. |
| `WINDOW_PATTERN` | `"SSSL"` | One char per layer (cycled). `S` = half context (`MAX_SEQ_LEN/2 = 1024`), `L` = full context (`MAX_SEQ_LEN = 2048`). Last layer is always overridden to full. |
| `DEPTH` | 8 | Number of transformer blocks. |
| `DEVICE_BATCH_SIZE` | 128 | Sequences per micro-batch on the GPU. |

### Optimization knobs

| Constant | Default | Effect |
|---|---|---|
| `TOTAL_BATCH_SIZE` | `2**19` (524,288) tokens | Tokens per optimizer step. Must be divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`. Determines `grad_accum_steps`. |
| `EMBEDDING_LR` | 0.6 | LR for `wte` and value embeddings (AdamW group). Multiplied by `(n_embd/768)^-0.5`. |
| `UNEMBEDDING_LR` | 0.004 | LR for `lm_head` (AdamW group). Same scaling. |
| `MATRIX_LR` | 0.04 | LR for transformer matrices (Muon groups). Per-shape scaling described in [internals/optimizer.md](../internals/optimizer.md). |
| `SCALAR_LR` | 0.5 | Base LR for the per-layer scalar groups. `resid_lambdas` uses `SCALAR_LR * 0.01`; `x0_lambdas` uses `SCALAR_LR`. |
| `WEIGHT_DECAY` | 0.2 | Initial cautious weight decay for Muon groups. Decays linearly to zero over training. |
| `ADAM_BETAS` | `(0.8, 0.95)` | `(β1, β2)` for AdamW groups (except `x0_lambdas`, which uses `(0.96, 0.95)`). |

### Schedule knobs

| Constant | Default | Effect |
|---|---|---|
| `WARMUP_RATIO` | 0.0 | Fraction of `TIME_BUDGET` used for linear LR warmup from 0 → 1. |
| `WARMDOWN_RATIO` | 0.5 | Fraction of `TIME_BUDGET` used at the end for linear cooldown to `FINAL_LR_FRAC`. |
| `FINAL_LR_FRAC` | 0.0 | LR multiplier at the very end of training. |

Schedules are time-based, not step-based. `progress = total_training_time / TIME_BUDGET`. See `get_lr_multiplier`, `get_muon_momentum`, `get_weight_decay` below.

## `GPTConfig` (dataclass)

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768       # overridden by tokenizer.get_vocab_size() at build
    n_layer: int = 12             # overridden by DEPTH
    n_head: int = 6               # overridden by build_model_config
    n_kv_head: int = 6            # overridden by build_model_config
    n_embd: int = 768             # overridden by build_model_config
    window_pattern: str = "SSSL"  # overridden by WINDOW_PATTERN
```

Built at runtime by `build_model_config(depth)`:

```python
base_dim = depth * ASPECT_RATIO
model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
num_heads = model_dim // HEAD_DIM
GPTConfig(
    sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
    n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
    n_embd=model_dim, window_pattern=WINDOW_PATTERN,
)
```

`n_kv_head == n_head` by default — no GQA. Setting `n_kv_head < n_head` (with `n_head % n_kv_head == 0`) enables grouped-query attention; the asserts in `CausalSelfAttention.__init__` are the only constraint.

## `GPT` model

| Method | Purpose |
|---|---|
| `__init__(config)` | Builds modules and registers RoPE buffers. Allocates value-embedding tables only for the layers selected by `has_ve`. |
| `init_weights()` | Initializes everything in the prescribed scheme (see [internals/model.md](../internals/model.md)). Casts `wte` and value embeddings to `bfloat16`. |
| `_precompute_rotary_embeddings(seq_len, head_dim, base=10000, device=None)` | Returns `(cos, sin)` of shape `(1, seq_len, 1, head_dim/2)`, in bf16. Called with `seq_len = config.sequence_len * 10` so the model can extrapolate. |
| `_compute_window_sizes(config)` | Translates `window_pattern` into a per-layer `(window, lookback)` tuple consumed by Flash Attention 3. Last layer is forced to `(long_window, 0)`. |
| `estimate_flops()` | Returns FLOPs/token for forward+backward, used for MFU. |
| `num_scaling_params()` | Returns a dict counting parameters by group (`wte`, `value_embeds`, `lm_head`, `transformer_matrices`, `scalars`, `total`). |
| `setup_optimizer(unembedding_lr, embedding_lr, matrix_lr, weight_decay, adam_betas, scalar_lr)` | Builds and returns a `MuonAdamW` optimizer with the documented param-group split. Records each group's `initial_lr`. |
| `forward(idx, targets=None, reduction='mean')` | Standard LM forward. Returns logits when `targets is None`, otherwise the cross-entropy loss. Logits are softcapped at 15 (`15 * tanh(logits / 15)`). |

## `MuonAdamW` optimizer

```python
optimizer = MuonAdamW(param_groups)
optimizer.step()  # dispatches each group to _step_adamw or _step_muon
```

Param groups have an extra `kind` key: `"adamw"` or `"muon"`. AdamW groups carry `(lr, betas, eps, weight_decay)`. Muon groups carry `(lr, momentum, ns_steps, beta2, weight_decay)`.

The `setup_optimizer` factory creates:

- One AdamW group per scalar/embedding role (`lm_head`, `wte`, `value_embeds`, `resid_lambdas`, `x0_lambdas`).
- One Muon group **per unique parameter shape** in `transformer.h` — each block exposes the same set of shapes, so this stacks parameters across blocks for vectorized updates.

The `lr` field on every group is overwritten each step by `lr * lrm` where `lrm = get_lr_multiplier(progress)`. The `initial_lr` stored at construction time is what the schedules multiply.

`muon_step_fused` and `adamw_step_fused` are `@torch.compile(dynamic=False, fullgraph=True)` and take 0-D CPU tensors for hyperparameters to avoid recompilation when those values change between steps.

## Schedule functions

```python
def get_lr_multiplier(progress: float) -> float:
    # piecewise linear: 0 → 1 over [0, WARMUP_RATIO),
    #                   1 over [WARMUP_RATIO, 1 - WARMDOWN_RATIO),
    #                   1 → FINAL_LR_FRAC over [1 - WARMDOWN_RATIO, 1].

def get_muon_momentum(step: int) -> float:
    # 0.85 → 0.95 linearly over the first 300 steps, constant afterwards.

def get_weight_decay(progress: float) -> float:
    # WEIGHT_DECAY * (1 - progress). Linear decay to zero.
```

`progress` is `min(total_training_time / TIME_BUDGET, 1.0)`. `total_training_time` is wall-clock training time (per-iteration `dt`), accumulated only after step 10 — so compilation and the first few warmup iterations don't count.

## Training loop

```python
while True:
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:        # bf16 autocast
            loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = ...
    update group lrs, muon momentum, muon weight decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    if NaN or loss > 100: print("FAIL"); exit(1)

    if step > 10:
        total_training_time += dt
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

    step += 1

# eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
```

Notes:

- `train_loader` is consumed before the optimizer step so the next batch is overlapping with the current step's compute (the dataloader prefetches via pinned + non-blocking copy).
- `gc.collect(); gc.freeze(); gc.disable()` runs once at step 0 to suppress the ~500 ms pause Python's GC otherwise causes mid-step. A safety `gc.collect()` runs every 5,000 steps.
- The fast-fail check on NaN / `loss > 100` is `train.py`-side, not in the harness.

## Output summary

After training, `train.py` prints exactly:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

| Key | Source |
|---|---|
| `val_bpb` | `evaluate_bpb` return value. The metric. |
| `training_seconds` | Sum of per-step `dt` after warmup. |
| `total_seconds` | `time() - t_start`, includes startup, compilation, eval. |
| `peak_vram_mb` | `torch.cuda.max_memory_allocated() / 1024 / 1024`. |
| `mfu_percent` | `100 * flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / training_seconds / H100_BF16_PEAK_FLOPS`. Always relative to H100 peak — not a "this GPU's MFU". |
| `total_tokens_M` | `step * TOTAL_BATCH_SIZE / 1e6`. |
| `num_steps` | Total optimizer steps taken. |
| `num_params_M` | `num_scaling_params()['total'] / 1e6`. |
| `depth` | The `DEPTH` constant. |

The agent reads only the `val_bpb` and `peak_vram_mb` lines via grep. Other keys are for human inspection.
