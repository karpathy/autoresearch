# Forking for smaller hardware

The default `train.py` is tuned for a single H100 (80 GB, BF16) with Flash Attention 3. On smaller GPUs, MacBooks (MPS / MLX), or AMD cards, you'll need to change either *just the constants* or, in some cases, the kernel imports as well.

If a maintained fork already exists for your platform, prefer it — see the [README's notable forks](../../README.md#notable-forks). Otherwise, the playbook below.

## What stays the same

- The contract from `prepare.py`: `MAX_SEQ_LEN` (handled per-platform — see below), `EVAL_TOKENS`, `TIME_BUDGET`, `evaluate_bpb`, the pinned val shard. If you change `MAX_SEQ_LEN` you accept that your `val_bpb` is no longer comparable to runs at 2048.
- The agent contract from `program.md`: results.tsv schema, keep/discard rules, the LOOP FOREVER pattern.

## What you tune

These are all in `train.py` unless noted, plus three constants in `prepare.py` that are sometimes worth changing for *very* small setups.

### Smaller-but-still-CUDA GPU (e.g., RTX 3090, 4090, 4080)

Often you only need to drop batch and depth a bit:

```python
DEPTH = 6                # was 8
DEVICE_BATCH_SIZE = 64   # was 128 — halve until peak VRAM fits
TOTAL_BATCH_SIZE = 2**18 # was 2**19 — keep DEVICE_BATCH_SIZE * MAX_SEQ_LEN | TOTAL_BATCH_SIZE
WINDOW_PATTERN = "L"     # was "SSSL" — banded attention is FA3-tuned, simpler full attention often runs faster on consumer cards
```

`TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0` is asserted in `train.py`. With `DEVICE_BATCH_SIZE=64` and `MAX_SEQ_LEN=2048` the per-fwd token count is `131_072`, so `TOTAL_BATCH_SIZE=2**18` gives `grad_accum_steps=2`.

### Apple Silicon / MacOS

Two options:

1. **Use a maintained fork.** [`miolini/autoresearch-macos`](https://github.com/miolini/autoresearch-macos) and [`trevin-creator/autoresearch-mlx`](https://github.com/trevin-creator/autoresearch-mlx) handle the kernel and device-selection rewrites.
2. **Adapt yourself.** You'll need to:
   - Swap `kernels.get_kernel(...)` and `fa3.flash_attn_func` for `torch.nn.functional.scaled_dot_product_attention` (no FA3 on MPS/MLX).
   - Replace the `cuda` device, pinned-memory, and `non_blocking` calls in `make_dataloader` with MPS/MLX equivalents.
   - Drop the `H100_BF16_PEAK_FLOPS` constant or replace it with the platform's peak so `mfu_percent` is meaningful (it's a cosmetic stat — the metric is `val_bpb`).
   - Use `--num-shards 1` or `2` since the data is mostly streaming I/O.

### AMD ROCm

[`andyluo7/autoresearch`](https://github.com/andyluo7/autoresearch) handles the ROCm path. You'll typically:

- Install PyTorch with the ROCm wheel index instead of `cu128`.
- Confirm Flash Attention 3 is available via `kernels-community/flash-attn3` for your GPU capability tuple.
- Otherwise tune as for any smaller CUDA card.

### CPU-only or tiny laptops

The repo isn't designed for this, but it's instructive. The smallest sensible config:

In `prepare.py` (you accept losing comparability):

```python
MAX_SEQ_LEN = 256          # was 2048 — total tokens/fwd shrinks 8×
EVAL_TOKENS = 4 * 524288   # was 40 * 524288 — eval is shorter
VOCAB_SIZE = 1024          # was 8192 — smaller vocab, faster tokenizer
```

Then re-run `prepare.py` so the tokenizer rebuilds at the new vocab.

In `train.py`:

```python
DEPTH = 4
ASPECT_RATIO = 32
DEVICE_BATCH_SIZE = 8
TOTAL_BATCH_SIZE = 2**14
WINDOW_PATTERN = "L"
```

You'll also want to swap the FA3 import for `F.scaled_dot_product_attention`. Expect mediocre `val_bpb` because the model is genuinely small — the goal of a CPU run is to verify the pipeline works, not to win.

### Tinier datasets for tiny models

For very small models, ClimbMix is too entropic to learn anything in 5 minutes. Karpathy recommends `karpathy/tinystories-gpt4-clean`:

- Replace the `BASE_URL` and `MAX_SHARD` in `prepare.py`.
- Update `_document_batches`/`text_iterator` if the parquet schema differs (TinyStories typically has the same `text` column, so this often works as-is).
- Re-run `prepare.py` to retokenize.

Tinystories at small `MAX_SEQ_LEN` (256–512) and small `DEPTH` (2–4) actually produces coherent samples, which makes for a more rewarding tiny-hardware setup.

## Quick reference: what each knob actually controls

| Knob | Lives in | Effect on memory | Effect on throughput | Effect on metric |
|---|---|---|---|---|
| `MAX_SEQ_LEN` | `prepare.py` | linear (acts on K/V cache and activations) | linear | breaks comparability if changed |
| `EVAL_TOKENS` | `prepare.py` | none | shifts overhead per run | smaller eval = noisier metric |
| `VOCAB_SIZE` | `prepare.py` | linear in embedding tables | mild | breaks comparability; requires re-prep |
| `DEPTH` | `train.py` | linear | linear | usually monotonic up to a point |
| `ASPECT_RATIO` | `train.py` | quadratic (via `n_embd`) | quadratic | usually monotonic |
| `HEAD_DIM` | `train.py` | small | depends on FA3 path | small unless head count gets weird |
| `WINDOW_PATTERN` | `train.py` | none | "L" is simpler, "SSSL" saves attention compute on long-context layers | small but real |
| `DEVICE_BATCH_SIZE` | `train.py` | linear | sub-linear (kernel utilization) | indirect (via `TOTAL_BATCH_SIZE` choice) |
| `TOTAL_BATCH_SIZE` | `train.py` | none | none (just changes grad_accum) | classic LR/batch tradeoff |

## Sanity check after any platform port

Run the baseline once manually:

```bash
uv run train.py
```

Make sure:

1. The script prints a `val_bpb:` line (not `FAIL`).
2. Peak VRAM matches your card.
3. `mfu_percent` is reasonable for your platform (the H100 baseline is irrelevant on other hardware).

If those three are fine, the agent loop will work.
