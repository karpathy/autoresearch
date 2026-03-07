# autoresearch

Autonomous LLM pretraining research, driven by AI agents.

The idea: give an AI agent a small but real LLM training setup and let it run experiments overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

This particular implementation is trying to be the least fancy baseline, but it's clear how one would adjust the `program.md` file to run more sophisticated research programs with more elaborate instructions. For example, the agent can actively do little experiments on research while the job is running.

The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat).

A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069). This code in particular is a simpler, more self-contained version that I thought people might like to play with.

## How it works

The repo is deliberately small and only has a few files:

- **`constants.py`** — fixed rules: sequence length, time budget, eval tokens. Not modified.
- **`prepare.py`** — one-time data prep (downloads training data, trains a BPE tokenizer) and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc.
- **`program.md`** — instructions for the agent. Point your agent here and let it go.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
constants.py    — fixed constants (do not modify)
prepare.py      — data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## License

MIT
