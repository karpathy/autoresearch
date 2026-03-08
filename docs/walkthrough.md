# Autonomous LLM Research with autoresearch-commons: A Complete Walkthrough

*A detailed guide to running AI agents that experiment with LLM training overnight — and learn from each other.*

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [The Big Idea](#2-the-big-idea)
3. [Architecture Overview](#3-architecture-overview)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Part 1: Running Your First Agent (Single-Agent Mode)](#6-part-1-running-your-first-agent)
7. [Part 2: Understanding What the Agent Does](#7-part-2-understanding-what-the-agent-does)
8. [Part 3: The Knowledge Protocol](#8-part-3-the-knowledge-protocol)
9. [Part 4: Multi-Agent Mode with the Director](#9-part-4-multi-agent-mode-with-the-director)
10. [Part 5: Cross-Machine Collaboration via Git](#10-part-5-cross-machine-collaboration-via-git)
11. [Part 6: Interpreting Results](#11-part-6-interpreting-results)
12. [Platform-Specific Notes](#12-platform-specific-notes)
13. [Troubleshooting](#13-troubleshooting)
14. [How This Differs from Upstream autoresearch](#14-how-this-differs-from-upstream-autoresearch)

---

## 1. What Is This?

In March 2026, Andrej Karpathy released [autoresearch](https://github.com/karpathy/autoresearch) — a minimal repo where you point an AI coding agent at a small LLM training script and let it experiment autonomously. The agent modifies the model architecture, optimizer, and hyperparameters, trains for exactly 5 minutes, checks if validation loss improved, keeps or discards the change, and repeats. You go to sleep, wake up to 100 experiments and a better model.

Karpathy then [tweeted](https://x.com/karpathy/status/2030705271627284816):

> *"The next step for autoresearch is that it has to be asynchronously massively collaborative for agents (think: SETI@home style). The goal is not to emulate a single PhD student, it's to emulate a research community of them."*

**autoresearch-commons** is a fork that implements this vision. It adds:

- A **knowledge protocol** so agents record what they learn and read what others have found
- A **director** that plans experiments and coordinates multiple agents
- **Multi-platform support** so you can run on NVIDIA GPUs, Apple Silicon Macs, or CPU
- **Multi-agent safety** so agents don't corrupt shared state

The result: multiple AI agents, potentially on different machines, autonomously running experiments and learning from each other's results — like a research lab that never sleeps.

---

## 2. The Big Idea

Traditional ML research looks like this:

```
Human thinks of idea → Human codes it → Human runs experiment → Human reads results → Repeat
```

Karpathy's autoresearch automates this:

```
Agent thinks of idea → Agent codes it → Agent runs 5-min experiment → Agent reads results → Repeat forever
```

autoresearch-commons extends this to multiple agents:

```
Agent reads what all agents have learned so far
  → Agent picks an unexplored direction
  → Agent codes and runs the experiment
  → Agent records what it learned to the shared knowledge base
  → Next agent (on any machine) reads the updated knowledge
  → Repeat across all agents
```

The knowledge base is the key innovation. Without it, each agent starts from zero. With it, the 50th experiment benefits from everything learned in the first 49 — even if those experiments ran on a different machine, in a different session, days apart.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Human (You)                       │
│  - Launches agents                                   │
│  - Optionally runs the director                      │
│  - Reviews results in the morning                    │
└─────────────┬───────────────────────┬───────────────┘
              │                       │
              ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│   Agent (Machine A)  │   │   Agent (Machine B)  │
│   e.g. H100 GPU      │   │   e.g. MacBook M2    │
│                       │   │                       │
│  1. Read brief        │   │  1. Read brief        │
│  2. Modify train.py   │   │  2. Modify train.py   │
│  3. Train 5 min       │   │  3. Train 5 min       │
│  4. Write card        │   │  4. Write card        │
│  5. Repeat            │   │  5. Repeat            │
└───────────┬───────────┘   └───────────┬───────────┘
            │                           │
            ▼                           ▼
┌─────────────────────────────────────────────────────┐
│              Shared Knowledge Base (Git)              │
│                                                       │
│  knowledge/                                           │
│    cards/         ← one JSON per experiment           │
│    synthesis/     ← session reports + meta-synthesis  │
│    index.json     ← fast-query index                  │
│    queue.json     ← experiment queue (director)       │
└─────────────────────────────────────────────────────┘
```

The knowledge base lives in the `knowledge/` directory and is shared via git. Agents read it before planning and write to it after each experiment.

### File Roles

| File | Who writes it | Who reads it | Purpose |
|------|--------------|-------------|---------|
| `prepare.py` | Nobody (read-only) | `train.py` | Data prep, tokenizer, evaluation, constants |
| `train.py` | The agent | The agent | Model, optimizer, training loop — the thing being experimented on |
| `platform_utils.py` | Nobody (read-only) | `train.py` | Hardware detection, attention backends, memory tracking |
| `commons.py` | Nobody (read-only) | The agent (via CLI) | Knowledge base operations |
| `director.py` | Nobody (read-only) | Human or director agent | Experiment planning and queue management |
| `program.md` | Human (you iterate on this) | The agent | Agent instructions — the "research methodology" |
| `director.md` | Human | Director agent | Director strategy instructions |
| `knowledge/` | Agents + director | Everyone | The shared knowledge base |

---

## 4. Prerequisites

### Hardware

One of:
- **NVIDIA GPU** (any modern card; tested on H100, works on consumer GPUs too)
- **Apple Silicon Mac** (M1/M2/M3/M4 — any variant)
- **CPU** (any machine — slowest but works everywhere)

### Software

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **git** — for version control and cross-machine sync
- **An AI coding agent** — Claude Code, Codex, Cursor, or similar

### Disk Space

About 2 GB for the FineWeb training data (downloaded once by `prepare.py`).

---

## 5. Installation

```bash
# Clone the repository
git clone https://github.com/shehral/autoresearch-commons.git
cd autoresearch-commons

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For NVIDIA GPUs with Flash Attention 3 (optional, faster attention):
uv sync --extra cuda

# Download training data and train tokenizer (one-time, ~2 minutes)
uv run prepare.py
```

Verify the setup works:

```bash
# Run a single training experiment (~5 minutes)
uv run train.py
```

You should see output ending with something like:

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

The numbers will vary by hardware. On a MacBook you'll see fewer steps and different memory numbers. That's expected — the 5-minute budget is fixed, but throughput varies.

---

## 6. Part 1: Running Your First Agent

### Step 1: Launch your AI agent

Open Claude Code (or your preferred AI coding agent) in the repository directory.

### Step 2: Give it the starting prompt

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

### Step 3: The agent takes over

The agent will:

1. Read `program.md` for its instructions
2. Read `README.md`, `prepare.py`, `train.py`, and `commons.py` for context
3. Propose a run tag (e.g., `mar8`)
4. Create a branch: `git checkout -b autoresearch/mar8`
5. Check that training data exists
6. Create `results.tsv` with the baseline
7. Read the knowledge base: `uv run commons.py read-brief`
8. Ask you to confirm, then begin experimenting

### Step 4: Go to sleep

Once the agent confirms setup and starts the loop, you can walk away. The agent will:

- Run an experiment every ~5 minutes (plus a few seconds overhead)
- That's ~12 experiments/hour, ~100 overnight
- Each experiment modifies `train.py`, trains, evaluates, and decides to keep or discard
- Every experiment gets recorded as a knowledge card
- Every 20 experiments, the agent generates a synthesis report

### Step 5: Check results in the morning

```bash
# See the knowledge summary
uv run commons.py read-brief

# See what's been tried in each category
uv run commons.py coverage

# Generate a progress chart
uv run commons.py plot

# Look at the experiment branch
git log --oneline autoresearch/mar8
```

---

## 7. Part 2: Understanding What the Agent Does

### The Experiment Loop

Here's exactly what happens in each iteration:

```
┌─────────────────────────────────────────────┐
│ 1. READ: uv run commons.py read-brief       │
│    - See coverage map (what's been tried)    │
│    - See recent findings                     │
│    - See queued experiments from director     │
├─────────────────────────────────────────────┤
│ 2. PLAN: Choose an experiment                │
│    - Pick an under-explored area             │
│    - Or build on a promising finding         │
│    - Or take a queued experiment             │
├─────────────────────────────────────────────┤
│ 3. CODE: Modify train.py                     │
│    - Change architecture, optimizer, etc.    │
│    - Git commit the change                   │
├─────────────────────────────────────────────┤
│ 4. RUN: uv run train.py > run.log 2>&1      │
│    - Fixed 5-minute training budget          │
│    - Redirected to run.log (not stdout)      │
├─────────────────────────────────────────────┤
│ 5. EVALUATE: grep "^val_bpb:" run.log       │
│    - Lower val_bpb = better                  │
│    - Also check peak memory usage            │
├─────────────────────────────────────────────┤
│ 6. RECORD: uv run commons.py write-card ...  │
│    - Hypothesis, result, lesson, tags        │
│    - Links to prior cards (lineage)          │
├─────────────────────────────────────────────┤
│ 7. DECIDE: Keep or discard?                  │
│    - Improved → keep the commit              │
│    - Worse → git reset to previous best      │
├─────────────────────────────────────────────┤
│ 8. REPEAT                                    │
└─────────────────────────────────────────────┘
```

### What the Agent Can Change

The agent modifies `train.py` and only `train.py`. Within that file, everything is fair game:

- **Model architecture**: number of layers, embedding dimension, attention heads, activation functions, normalization, positional encoding
- **Optimizer**: learning rate, weight decay, scheduler, optimizer type (Muon, AdamW, etc.)
- **Training loop**: batch size, gradient accumulation, sequence length
- **Hyperparameters**: anything in the config dataclass

### What the Agent Cannot Change

- `prepare.py` — the evaluation metric and data loading are fixed
- `commons.py`, `director.py`, `platform_utils.py` — infrastructure is read-only
- Cannot install new packages
- Cannot change the 5-minute time budget

### The Metric: val_bpb

**val_bpb** stands for "validation bits per byte." It measures how well the model predicts the next token on held-out validation data, normalized by byte count so it's independent of vocabulary size. Lower is better.

This is important: because the time budget is fixed at 5 minutes, the agent is essentially optimizing for "best model quality achievable in 5 minutes of training." This means:

- Bigger models might be better per-step but get fewer steps — there's a sweet spot
- Faster training tricks (better batch size, compiled ops) translate directly to lower val_bpb
- The optimal configuration depends on your hardware

---

## 8. Part 3: The Knowledge Protocol

This is what makes autoresearch-commons different from upstream autoresearch. Every experiment produces a structured record that future agents can learn from.

### Experiment Cards

After each experiment, the agent records a **card** — a JSON file in `knowledge/cards/`:

```json
{
  "id": "card_20260308_143052_abc1234",
  "timestamp": "2026-03-08T14:30:52",
  "commit_id": "abc1234",
  "hypothesis": "Halve batch size from 64 to 32 to get more optimization steps in the fixed time budget",
  "config_diff": {
    "BATCH_SIZE": "64 → 32",
    "model": "unchanged"
  },
  "results": {
    "val_bpb": 0.986,
    "delta": -0.012,
    "peak_vram_mb": 22400,
    "training_seconds": 300.1,
    "num_steps": 1906,
    "estimated_flops": 1.2e15,
    "num_params": 50300000
  },
  "status": "keep",
  "lesson": "Doubling the number of steps by halving batch size more than compensates for the smaller batch. The model sees the same data but with more frequent weight updates.",
  "tags": ["batch_size", "optimization"],
  "platform": {
    "gpu": "NVIDIA H100",
    "ram_gb": 80,
    "framework": "pytorch-cuda"
  },
  "prior_knowledge_used": ["card_20260308_140000_baseline"]
}
```

Key fields:
- **hypothesis**: What the agent tried and why — the human-readable experiment description
- **status**: `keep` (improved), `revert` (worse), `inconclusive` (unclear), `crash` (failed), `retracted` (later invalidated)
- **lesson**: What the agent learned — this is the most valuable field for future agents
- **tags**: Categorical labels for the coverage map
- **prior_knowledge_used**: Which earlier cards informed this experiment (lineage)

### The Brief

When an agent runs `uv run commons.py read-brief`, it sees a formatted summary:

```
=== Knowledge Brief ===

Recent experiments (last 10):
  card_007 [keep]  val_bpb=0.986 Δ=-0.012 (batch_size, optimization) prior=1
  card_006 [revert] val_bpb=1.005 Δ=+0.007 (architecture, activation)
  card_005 [keep]  val_bpb=0.998 Δ=-0.000 (learning_rate)
  ...

Coverage Map:
  batch_size:     3 experiments, 2 kept, best Δ=-0.012, best bpb=0.986
  architecture:   5 experiments, 1 kept, best Δ=-0.003, best bpb=0.995
  learning_rate:  2 experiments, 1 kept, best Δ=-0.000, best bpb=0.998
  optimizer:      0 experiments  ← UNDER-EXPLORED
  regularization: 0 experiments  ← UNDER-EXPLORED

Experiment Queue:
  [P1] Try SwiGLU activation (architecture)
  [P2] Increase model depth to 12 layers (architecture)
```

This tells the agent:
- What's been tried recently and what worked
- Which areas are well-explored vs. under-explored
- What the director has queued up as strategic priorities

### Synthesis

Every 20 experiments, the agent generates a **synthesis report** — a markdown summary of what's been learned:

```bash
uv run commons.py synthesize --session mar8
```

Periodically, a **meta-synthesis** rolls up all session reports into a single document:

```bash
uv run commons.py update-meta
```

The meta-synthesis is what agents read for the big picture — confirmed findings, dead ends, and open questions.

### Card Retraction

Sometimes an experiment result is wrong — maybe the baseline was misconfigured, or a bug inflated the numbers. Rather than deleting the card (which would lose history), you can retract it:

```bash
uv run commons.py retract --id card_003 --reason "Baseline was misconfigured, delta is invalid"
```

Retracted cards are excluded from the brief and coverage map but remain in the knowledge base for audit purposes.

### Experiment Lineage

Cards can reference which prior cards inspired them via `--prior-cards`. This creates a knowledge graph:

```
baseline → batch_size_halved → batch_size_16_with_grad_accum → ...
baseline → swiglu_activation → swiglu_with_deeper_model → ...
```

Future agents can trace chains of reasoning: "This finding led to that experiment, which led to this breakthrough."

---

## 9. Part 4: Multi-Agent Mode with the Director

The **director** is an optional orchestration layer for running multiple agents. It plans experiment strategy and manages a shared queue.

### Why Use the Director?

Without the director, agents read the knowledge base and independently decide what to try. This works but can lead to duplication — two agents might try the same thing simultaneously.

The director adds:
- **Strategic planning**: Automatically identifies under-explored areas and generates experiment ideas
- **A shared queue**: Agents can claim experiments from the queue to avoid duplication
- **Priority system**: High-priority experiments get tried first

### Using the Director CLI

```bash
# Generate experiment ideas from the knowledge base
uv run director.py plan

# Manually add an experiment to the queue
uv run director.py add --hypothesis "Try SwiGLU activation" --category architecture --priority 1

# Check queue status
uv run director.py status
```

### How Agents See the Queue

When an agent runs `read-brief`, any pending experiments from the queue appear in an "Experiment Queue" section. The agent can choose to implement a queued experiment or pursue its own idea. The queue is advisory, not mandatory — agents retain creative freedom.

### Priority Levels

| Priority | Meaning | When to use |
|----------|---------|-------------|
| 1 | High | Under-explored areas (< 3 experiments), critical gaps |
| 2 | Medium | Promising areas showing improvement, worth deepening |
| 3 | Low | Open questions, exploratory ideas |
| 5 | Default | Manual additions without strong priority signal |

### Multi-Agent Safety

When multiple agents share a knowledge base, things can go wrong — two agents writing to the same file, a crashed agent leaving a queue item stuck, etc. autoresearch-commons handles this:

- **Atomic writes**: JSON files are written via `tempfile` + `os.replace()` so a crash mid-write doesn't corrupt data
- **File locking**: Queue operations use `fcntl.LOCK_EX` for process-level mutual exclusion
- **Stale claim timeout**: If an agent claims a queue item and crashes, the item is automatically released after 15 minutes

---

## 10. Part 5: Cross-Machine Collaboration via Git

The knowledge base lives in `knowledge/` and is committed to git. This means agents on different machines can share findings by pushing and pulling.

### Setup: Two Machines

**Machine A** (e.g., H100 server):

```bash
git clone https://github.com/yourname/autoresearch-commons.git
cd autoresearch-commons
uv sync --extra cuda
uv run prepare.py
```

**Machine B** (e.g., MacBook):

```bash
git clone https://github.com/yourname/autoresearch-commons.git
cd autoresearch-commons
uv sync
uv run prepare.py
```

### The Workflow

1. Machine A's agent runs experiments and writes cards
2. Machine A commits and pushes: `git add knowledge/ && git commit && git push`
3. Machine B pulls: `git pull`
4. Machine B's agent reads the brief — it now sees Machine A's findings
5. Machine B's agent runs its own experiments, informed by Machine A's results
6. Machine B pushes its cards
7. Machine A pulls and benefits from Machine B's findings

### Important: Platform-Specific Results

Because the training budget is fixed at 5 minutes, the optimal configuration depends on hardware. An H100 gets ~950 steps in 5 minutes; an M2 MacBook might get ~50. This means:

- Results are **not directly comparable** across platforms
- But lessons often **transfer**: "SwiGLU helps" or "smaller batch size is better" may hold across hardware
- Cards include platform info so agents can weight findings appropriately

---

## 11. Part 6: Interpreting Results

### The Coverage Map

```bash
uv run commons.py coverage
```

This shows how many experiments have been run in each category. Look for:

- **Under-explored areas** (0-2 experiments): These are opportunities — there might be easy wins nobody has tried
- **Saturated areas** (10+ experiments): Diminishing returns — the agent has probably found the local optimum here
- **High-delta categories**: Areas where experiments have produced the biggest improvements

### The Progress Chart

```bash
uv run commons.py plot
```

This generates a scatter plot of val_bpb over time with a running-best line. You want to see:

- The running-best line trending downward (the model is getting better)
- A mix of keep and revert points (the agent is exploring, not just making safe changes)
- Eventually, the curve flattening (diminishing returns)

### Reading results.tsv

The TSV file shows every experiment in chronological order:

```
commit    val_bpb    memory_gb  status   description
a1b2c3d   0.997900   44.0       keep     baseline
b2c3d4e   0.993200   44.2       keep     increase LR to 0.04
c3d4e5f   1.005000   44.0       discard  switch to GeLU activation
d4e5f6g   0.000000   0.0        crash    double model width (OOM)
```

### Reading the Git Log

```bash
git log --oneline autoresearch/mar8
```

The branch only contains commits that improved val_bpb (discarded experiments are reverted). So the git history reads like a success story — each commit is a validated improvement.

---

## 12. Platform-Specific Notes

### NVIDIA GPU (CUDA)

- **Best experience**. Fastest training, most steps per experiment.
- Uses Flash Attention 3 (via the `kernels` package) for fast attention.
- `torch.compile` is enabled for additional speedup.
- bfloat16 autocast for mixed precision.
- Peak memory tracked natively via `torch.cuda.max_memory_allocated()`.
- Install with `uv sync --extra cuda` for Flash Attention.

### Apple Silicon (MPS)

- **Good for experimentation**, but slower than CUDA.
- Uses PyTorch's `scaled_dot_product_attention` (SDPA) with a manually constructed sliding-window causal mask.
- `torch.compile` is disabled (not well supported on MPS).
- No autocast (not well supported on MPS).
- Peak memory tracked via a background polling thread at 100ms intervals (MPS lacks native peak tracking).
- Expect fewer training steps per experiment — the 5-minute budget still applies.

### CPU

- **Slowest but works anywhere**. Good for testing the setup or running on a server without a GPU.
- Uses SDPA attention.
- bfloat16 autocast on supported CPUs.
- Very few training steps per experiment — results will be different from GPU runs.

---

## 13. Troubleshooting

### `uv` not found

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add this to your shell profile (`~/.zshrc` or `~/.bashrc`).

### `prepare.py` fails to download data

Check your internet connection. The data is downloaded from HuggingFace. If you're behind a proxy, set `HF_HUB_DISABLE_PROGRESS_BARS=1` and ensure HTTPS access.

### `train.py` crashes with OOM

The default configuration is tuned for an H100 with 80GB VRAM. On smaller GPUs or Apple Silicon, the agent should reduce batch size and/or model size. If running manually, edit the config in `train.py`:

```python
# Reduce these for smaller GPUs
BATCH_SIZE = 16  # or 8, or 4
```

### Agent stops after one experiment

Check `program.md` — the instructions say "NEVER STOP." Some AI agents have safety features that make them want to pause. Make sure your agent is configured for autonomous operation (no confirmation prompts).

### Knowledge base seems empty

Run `uv run commons.py read-brief`. If it shows no cards, make sure the agent is actually running the `write-card` command after each experiment. Check if `knowledge/cards/` has any JSON files.

### Tests fail after moving files

If you've reorganized the repo, run:

```bash
uv run pytest tests/ -v
```

All 145 tests should pass. If subprocess-based tests fail, it's usually a `cwd` issue — the CLI tests need to run from the project root.

---

## 14. How This Differs from Upstream autoresearch

| Feature | Upstream (karpathy/autoresearch) | autoresearch-commons |
|---------|--------------------------------|---------------------|
| Platform | CUDA only | CUDA + MPS + CPU |
| Memory between sessions | None — each session starts fresh | Knowledge base persists across sessions |
| Multi-agent | Not supported | File locking, queue, stale claim timeout |
| Experiment tracking | results.tsv (flat file) | Structured JSON cards + synthesis reports |
| Experiment planning | Agent decides independently | Director can plan strategy and queue experiments |
| Experiment lineage | None | Cards link to prior findings |
| Bad experiment handling | Delete the commit | Retract the card (preserves history) |
| Progress visualization | None | Scatter plot with running-best line |
| Validation | None | Schema validation on card creation |
| Tests | None | 145 tests |

### What's Preserved from Upstream

- `prepare.py` is untouched
- `train.py` has the same structure (model, optimizer, loop) — just with platform-portable imports
- The 5-minute time budget
- val_bpb as the sole metric
- The agent-modifies-one-file philosophy
- The `program.md` instruction format

---

## Further Reading

- [Karpathy's autoresearch announcement](https://x.com/karpathy/status/2030371219518931079)
- [Karpathy's vision for collaborative agents](https://x.com/karpathy/status/2030705271627284816)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the upstream repo
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — the full training pipeline autoresearch is derived from
