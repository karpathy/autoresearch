# autoresearch-commons

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that adds **collaborative knowledge sharing** between autonomous research agents.

The original autoresearch gives an AI agent a real LLM training setup and lets it experiment autonomously overnight. This fork adds the missing piece: a shared knowledge base so agents **learn from each other's experiments** — across sessions, across hardware, across people.

> *"The next step for autoresearch is that it has to be asynchronously massively collaborative for agents (think SETI@home style). The goal is not to emulate a single PhD student, it's to emulate a research community of them."* — @karpathy, March 2026

## What's new

This fork adds a **knowledge protocol** and an **orchestration layer** on top of the original autoresearch:

| File | Purpose | Who uses it |
|------|---------|-------------|
| `platform_utils.py` | Auto-detect CUDA / MPS / CPU | `train.py` imports it |
| `commons.py` | Knowledge base read/write CLI | Agents call it |
| `director.py` | Experiment orchestration (optional) | Human runs it |
| `director.md` | Director agent instructions | Director agent |
| `knowledge/` | Shared experiment knowledge | Everyone |
| `program.md` | Worker instructions (modified) | Worker agents |

Everything from the original is preserved: `prepare.py` (untouched), `train.py` (same structure, now platform-portable), the 5-minute time budget, and val_bpb as the sole metric.

## Quick start

**Requirements:** NVIDIA GPU, Apple Silicon Mac, or CPU. Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# For CUDA users with Flash Attention 3:
uv sync --extra cuda

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

## Running an agent (single-agent mode)

Same as the original, but now agents read/write to the knowledge base:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The modified `program.md` adds two steps to the experiment loop:
1. **Before experimenting:** `uv run commons.py read-brief` — read what's been tried
2. **After each experiment:** `uv run commons.py write-card ...` — record what was learned

## Running the director (multi-agent mode)

The director plans experiment strategy from the knowledge base:

```bash
# See what experiments the knowledge base suggests
uv run director.py plan

# Add a custom experiment idea
uv run director.py add --hypothesis "Try SwiGLU activation" --category architecture --priority 1

# Check queue status
uv run director.py status
```

## Knowledge base CLI

```bash
# Read knowledge summary (what agents see before planning)
uv run commons.py read-brief

# Record an experiment result
uv run commons.py write-card \
  --commit abc1234 \
  --hypothesis "Halve batch size" \
  --result 0.986 \
  --delta -0.012 \
  --status keep \
  --lesson "More steps in fixed time budget" \
  --tags "batch_size,optimization"

# View experiment coverage
uv run commons.py coverage

# Generate session synthesis report
uv run commons.py synthesize --session mar8

# Update rolling meta-synthesis
uv run commons.py update-meta
```

## Project structure

```
prepare.py          — constants, data prep + runtime utilities (do not modify)
train.py            — model, optimizer, training loop (agent modifies this)
platform_utils.py   — CUDA/MPS/CPU auto-detection and abstraction
commons.py          — knowledge base interface (library + CLI)
director.py         — experiment orchestration (optional)
program.md          — worker agent instructions
director.md         — director agent instructions
knowledge/
  cards/            — one JSON per experiment (the raw data)
  synthesis/        — session reports + meta-synthesis (the summaries)
  index.json        — fast-query experiment index
  queue.json        — experiment queue (director-managed)
```

## How the knowledge protocol works

Every experiment produces an **experiment card** (JSON):
- What was tried (hypothesis) and what changed (config diff)
- The result (val_bpb, delta, memory, steps)
- What was learned (lesson) and categorical tags
- Platform info (so agents know which findings are hardware-specific)

Periodically, cards are **synthesized** into session reports and a rolling **meta-synthesis** that includes:
- Confirmed findings and dead ends
- An **experiment coverage map** showing what areas are saturated vs. under-explored
- Open questions for future experiments

The next agent reads the meta-synthesis before planning, so it doesn't repeat dead ends and focuses on high-value directions.

## Platform support

This fork auto-detects your hardware:
- **NVIDIA GPU (CUDA):** Flash Attention 3, torch.compile, bfloat16 autocast
- **Apple Silicon (MPS):** SDPA attention with sliding window mask, no compile, smaller default batch size
- **CPU:** SDPA attention, bfloat16 autocast, slowest but works anywhere

## Design philosophy

- **The knowledge protocol is the contribution.** Platform ports let more people run experiments. The knowledge protocol lets agents *learn from each other*. That's the real multiplier.
- **Separation of concerns.** The protocol (knowledge format) is git-native and works standalone. The orchestration (director) is an optional convenience layer.
- **Minimal upstream diff.** We add files, not complexity. `train.py` gets one import swap. Everything else is additive.

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — MPS adaptation patterns
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MLX reference

## License

MIT
