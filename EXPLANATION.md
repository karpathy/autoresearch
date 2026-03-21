# How Autoresearch Works

## Overview

Autoresearch is an **autonomous AI research platform** that lets an AI agent conduct ML research overnight without human supervision. It gives the agent a small but real LLM training setup and lets it experiment autonomously — modifying code, training, evaluating, and iterating.

## The Core Loop

The agent runs an infinite experiment loop:

1. **Propose an idea** — e.g., change learning rate, tweak architecture, adjust hyperparameters
2. **Edit `train.py`** — the only file the agent is allowed to modify
3. **Git commit** the change
4. **Train for exactly 5 minutes** (`uv run train.py`)
5. **Evaluate** using `val_bpb` (validation bits per byte — lower is better)
6. **Keep or discard** — if the metric improved, keep the commit; otherwise `git reset` to revert
7. **Log** the result to `results.tsv` and repeat

This runs approximately 12 experiments per hour, or around 100 overnight.

## Key Design Principles

- **Single file to modify**: The agent only touches `train.py`. This keeps diffs reviewable and prevents the agent from breaking infrastructure.
- **Fixed time budget**: Every experiment gets exactly 5 minutes of training, making results directly comparable regardless of model size or architecture changes.
- **One metric**: `val_bpb` is vocab-size-independent, so even architectural changes (like changing vocab size) can be fairly compared.
- **Instructions via Markdown**: Humans program `program.md` with research directions and constraints — the agent reads this instead of receiving interactive commands.

## The Three Critical Files

| File | Role | Who edits it? |
|------|------|---------------|
| `prepare.py` | Data downloading, tokenization, dataloader, evaluation | Nobody (read-only) |
| `train.py` | GPT model architecture, optimizer, training loop, hyperparams | AI agent |
| `program.md` | Research instructions and constraints | Human |

## What's in `train.py`

The training code is a single-GPU GPT implementation (derived from Karpathy's nanochat) with:

- **Transformer model** with RoPE, Flash Attention 3, sliding window attention, value embeddings, ReLU² MLP, logit softcapping
- **MuonAdamW optimizer** — Muon (orthogonalization-based) for matrix params, AdamW for everything else
- **Training loop** with gradient accumulation, dynamic LR scheduling, and a hard 5-minute wall-clock cutoff

## What's in `prepare.py`

Fixed infrastructure the agent cannot modify:

- Downloads data shards from HuggingFace (`climbmix-400b-shuffle`)
- Trains a BPE tokenizer (8192 vocab)
- Provides a best-fit packing dataloader (100% token utilization, no padding)
- Implements `evaluate_bpb()` — the single evaluation metric

## Experiment Tracking

Results are logged to `results.tsv` with columns: commit hash, val_bpb, memory usage, status (keep/discard/crash), and a description. Humans review results in the morning using `analysis.ipynb`.

## In Short

Human writes research directions in Markdown → AI agent autonomously experiments with `train.py` all night → human reviews `results.tsv` in the morning. It's designed for simplicity (one GPU, one file, one metric) and full autonomy.
