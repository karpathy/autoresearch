# Contributing to autoresearch

This repository is a research platform for iterating on `train.py` — a single-GPU GPT training script. Contributions should focus on experiments that improve **val_bpb** (validation bits per byte) within a fixed compute budget.

## Prerequisites

- A single NVIDIA GPU (H100 primary; RTX 3090/4090 works with reduced batch sizes)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py   # one-time setup (~2 min)
uv run train.py     # baseline run (~5 min)
```

## Architecture context

The codebase has three files — do not add new modules:

- **`prepare.py`** — Data preparation, tokenizer training, evaluation. **Do not modify.**
- **`train.py`** — GPT model (`GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`), optimizer (Muon/AdamW), and training loop. This is where changes go.
- **`program.md`** — Agent instructions for autonomous experiments.

Key model defaults in `train.py`:
```python
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

Any architectural change must be compatible with these defaults and the existing rotary embedding + sliding window attention setup.

## What to contribute

Useful contributions are experiments that improve val_bpb:

- **Architecture changes** — attention variants, layer types, embeddings in `train.py`
- **Optimizer changes** — Muon/AdamW modifications, learning rate schedules
- **Hyperparameter discoveries** — better default configs backed by `results.tsv` data

Less useful for this repo:
- CI/CD, linting configs, contribution templates, or workflow documentation

## Evaluation

All changes are judged by **val_bpb** (lower = better) on the same 5-minute training budget. Log your results in `results.tsv` with columns: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`.

## Questions?

Open an issue before starting large changes.
