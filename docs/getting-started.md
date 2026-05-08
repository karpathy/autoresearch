# Getting Started

This page takes you from a fresh clone to "the agent is running experiments". It assumes a Linux box with one NVIDIA GPU. macOS / AMD users should jump to [operations/forking.md](operations/forking.md) and pick a fork from the [README's notable-forks list](../README.md#notable-forks).

## Requirements

- A single NVIDIA GPU. Default hyperparameters in `train.py` are tuned for an H100 (80 GB, BF16). Smaller cards work but need lowered `DEPTH`, `DEVICE_BATCH_SIZE`, `TOTAL_BATCH_SIZE`, and `MAX_SEQ_LEN` (see [forking](operations/forking.md)).
- Python 3.10 or newer.
- `uv` (project manager). Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- ~20 GB free disk for the default 10 training shards plus the validation shard. `prepare.py --num-shards -1` downloads all 6,542 shards (much larger).

## Install dependencies

```bash
git clone https://github.com/karpathy/autoresearch
cd autoresearch
uv sync
```

`uv sync` resolves `pyproject.toml` against the cu128 PyTorch index. The `kernels` package fetches a Flash Attention 3 wheel on first use; on Hopper (H100) it pulls `varunneal/flash-attention-3`, on other capabilities `kernels-community/flash-attn3`.

## One-time data preparation

```bash
uv run prepare.py
```

Default behavior:

- Downloads 10 training shards plus the pinned validation shard `shard_06542.parquet` from `karpathy/climbmix-400b-shuffle` on Hugging Face.
- Trains an 8,192-token BPE tokenizer with [`rustbpe`](https://crates.io/crates/rustbpe), wraps it as a `tiktoken` encoding, and pickles it.
- Builds a `token_bytes.pt` lookup that records the UTF-8 byte length of every token id (used by `evaluate_bpb`).

Everything lands under `~/.cache/autoresearch/`:

```
~/.cache/autoresearch/data/shard_00000.parquet ... shard_00009.parquet, shard_06542.parquet
~/.cache/autoresearch/tokenizer/tokenizer.pkl
~/.cache/autoresearch/tokenizer/token_bytes.pt
```

Common adjustments:

- `--num-shards 4` to download fewer shards (faster, smaller disk).
- `--num-shards -1` to download all 6,542 shards (only useful for very long-running setups).
- `--download-workers 16` to parallelize downloads.

Re-running `prepare.py` is safe; it skips shards that already exist and skips tokenizer training if `tokenizer.pkl` and `token_bytes.pt` are both present.

## Verify the harness with one manual run

Before pointing an agent at the repo, run training once yourself to confirm the GPU + kernels + dataloader all work:

```bash
uv run train.py
```

You should see compilation messages, then a streaming step counter, then a final summary:

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

If `val_bpb` prints, the harness is healthy. The exact number depends on the GPU. On H100 expect ~0.99 with the baseline.

If the run crashes, look at the stack trace. Common issues:

- *Flash Attention 3 wheel mismatch* — the `kernels` package downloads a wheel keyed on the GPU capability; non-Hopper cards get `kernels-community/flash-attn3`. If neither is available for your driver/CUDA combo, you'll need a fork that swaps in regular SDPA.
- *OOM on smaller GPUs* — drop `DEVICE_BATCH_SIZE` (default 128) until peak VRAM fits.
- *No parquet files found* — `prepare.py` didn't finish; re-run it.

## Start an autonomous run

Open the agent of your choice in this directory with permissions disabled (so it can edit/commit/run without prompts), then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent reads [`program.md`](../program.md), proposes a tag, creates `autoresearch/<tag>`, verifies the cache, initializes `results.tsv`, runs the baseline, then loops forever until you stop it.

What to expect:

- One experiment ≈ 5 minutes of training plus ~30 s of compilation, eval, and overhead.
- ~12 experiments per hour, ~100 over an 8-hour sleep.
- The agent appends to `results.tsv` after every run — keep, discard, or crash. The file is left untracked so resets don't lose history.
- `run.log` is overwritten every run.

When you wake up:

- `cat results.tsv` to see what was tried.
- `git log --oneline autoresearch/<tag>` to see the kept changes (one commit each).
- Open `analysis.ipynb` and run all cells to regenerate `progress.png` (running-best plot).

See [agent-workflow.md](agent-workflow.md) for the full loop semantics and [operations/analysis.md](operations/analysis.md) for the review workflow.

## Next steps

- Iterate on `program.md` to change how the agent picks ideas, weights complexity, or rewinds. The default is intentionally minimal.
- Try multiple agents on the same compute by giving each its own tag (`autoresearch/mar5-gpu0`, `autoresearch/mar5-gpu1`) and `CUDA_VISIBLE_DEVICES=N`.
- Adapt for smaller hardware via [operations/forking.md](operations/forking.md).
