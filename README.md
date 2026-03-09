# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching only the Python files like you normally would as a researcher. You are also programming the Markdown files that define the research org: `program.md` for the main loop, `soul_architect.md` and `soul_oracle.md` for the two subagent voices, and `spiritualguidance.md` for the running debate/synthesis memory. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and the core moving parts are:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — the top-level autonomous research loop and decision policy.
- **`soul_architect.md`** — the structural, skeptical subagent voice.
- **`soul_oracle.md`** — the intuitive, pattern-sensitive subagent voice.
- **`spiritualguidance.md`** — the running cycle log where Architect and Oracle disagree, synthesize, and leave the next directive.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

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

## Research org additions

This branch adds a simple multi-voice research loop:

- `program.md` now requires a consult/synthesis cycle before each experiment.
- `soul_architect.md` pushes for bottleneck clarity, reversible experiments, and structural discipline.
- `soul_oracle.md` pushes for pattern recognition, bolder search moves, and escaping stale local search.
- `spiritualguidance.md` stores the cycle-by-cycle state, disagreement, joint directive, and outcome memory.

The point is not roleplay. The point is forcing the agent to externalize the tradeoff between conservative search and bolder exploration before it edits `train.py`.

## Runtime compatibility improvements

This branch also adds practical host-compatibility improvements in `train.py`:

- a runtime fallback from FlashAttention 3 to native PyTorch SDPA on unsupported CUDA capability paths
- optional disabling of `torch.compile` and Triton-heavy fused paths on hosts where those fail
- env-controlled overrides for batch sizing and evaluation batching so smaller GPUs can still run useful experiments
- timing behavior that stops wasting fake "compile warmup" steps when the model is running eagerly

These changes were motivated by getting the repo to run on multiple non-H100 machines without rewriting `prepare.py` or the evaluation harness.

### Useful env overrides

For lower-VRAM or quirky hosts, the following overrides are useful:

```bash
AUTORESEARCH_DEVICE_BATCH_SIZE=32 AUTORESEARCH_EVAL_BATCH_SIZE=64 uv run train.py
AUTORESEARCH_DISABLE_TORCH_COMPILE=1 AUTORESEARCH_DISABLE_MODEL_COMPILE=1 uv run train.py
```

The first pattern reduces training/eval memory pressure. The second forces eager execution when a host trips over Dynamo, Triton, or custom-kernel compile issues.

**Platforms support**. This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. The code is just a demonstration and I don't know how much I'll support it going forward. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

In this branch the agent is expected to read `program.md`, `soul_architect.md`, `soul_oracle.md`, and `spiritualguidance.md` together before it starts making research decisions.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — top-level autonomous loop
soul_architect.md — structural subagent voice
soul_oracle.md  — intuitive subagent voice
spiritualguidance.md — living guidance log
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## License

MIT
