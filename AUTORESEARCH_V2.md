# Autoresearch v2 — Multi-Agent Autonomous ML Research Platform

Built on top of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): an AI agent autonomously experiments on a small GPT training setup, modifying code, training for a fixed time budget, checking if the result improved, and repeating. We extended this from a single-agent/single-file loop into a multi-agent, multi-scale, self-improving research platform.

## What's New

### Variable Time Scales

Instead of a fixed 5-minute budget, experiments run at different scales:

| Scale | Duration | Use |
|-------|----------|-----|
| probe | 30s | Memory/compilation check |
| quick | 2min | Rough signal, kill bad ideas fast |
| standard | 5min | Real evaluation (original behavior) |
| long | 15min | Confirm promising results |
| deep | 30min | Final validation |

Agents start with quick runs and escalate promising ideas to longer scales. All configurable via `AR_*` environment variables.

### Research Memory

Replaces the flat `results.tsv` with a shared knowledge base in `results/`:

- **experiments.jsonl** — All experiment records from all agents (append-only, file-locked)
- **lessons.jsonl** — Discovered patterns and insights (e.g., "depth>12 always OOMs")
- **journal.md** — Research summaries and the director's agenda

Backward-compatible `results.tsv` is auto-generated for the analysis notebook.

### Model Checkpoints

Promising runs save full model + optimizer state to `checkpoints/`. This enables the scaling ladder: a good 5-minute result can be resumed and trained for 15 more minutes without starting from scratch.

### Multi-Agent Research Org

Five specialized agent roles, each with tailored instructions:

| Role | What it does |
|------|-------------|
| **Explorer** | Bold architectural experiments — new activations, attention variants, normalization |
| **Optimizer** | Systematic hyperparameter tuning — LR sweeps, batch size, schedules |
| **Analyst** | Reads all results, finds patterns, writes lessons, suggests combinations |
| **Reviewer** | Validates improvements at longer scales, runs ablations, catches false positives |
| **Director** | Coordinates research agenda, cherry-picks across branches, redirects stuck agents |

Each agent runs in its own git worktree with its own branch, sharing results through the knowledge base.

### GPU Queue

When multiple agents share one GPU, a priority queue manages access. Probe jobs (30s) jump ahead so agents get fast feedback. Agents can plan their next experiment while waiting.

## File Structure

```
# New infrastructure
config.py              — Configurable constants (env var overrides)
knowledge.py           — JSONL research memory with file locking
checkpoint.py          — Model checkpoint save/load
run_experiment.py      — Experiment runner wrapper
gpu_queue.py           — Priority queue for GPU sharing
launch.py              — Multi-agent orchestrator

# Agent role instructions
programs/explorer.md
programs/optimizer.md
programs/analyst.md
programs/reviewer.md
programs/director.md

# Original files (modified)
prepare.py             — Constants now configurable via config.py
train.py               — Checkpoint save added (env-var gated)
program.md             — Multi-agent awareness section added

# Shared directories (gitignored, created at runtime)
results/               — Knowledge base (experiments, lessons, journal)
queue/                 — GPU job queue
checkpoints/           — Saved model states
worktrees/             — Per-agent git worktrees
```

## Quick Start

### Solo Mode (enhanced single agent)

```bash
python launch.py launch --tag mar10 --preset solo
```

One explorer agent with the full knowledge base, variable time scales, and checkpointing.

### Duo (explorer + optimizer sharing one GPU)

```bash
python launch.py launch --tag mar10 --preset duo --single-gpu
```

### Full Research Org in tmux

```bash
python launch.py launch --tag mar10 --preset full --single-gpu --tmux
```

All five agents running in tmux panes. Attach with `tmux attach -t autoresearch-mar10`.

### Custom Agent Mix

```bash
python launch.py launch --tag mar10 --agents explorer:2,optimizer,director --single-gpu
```

Two explorers, one optimizer, one director.

### Cleanup

```bash
python launch.py cleanup --tag mar10
```

Removes worktrees and branches for a tag.

## Standalone Tools

```bash
# View research briefing (what agents read before each experiment)
python run_experiment.py --briefing

# Run a single experiment manually
python run_experiment.py --scale quick --description "try GLU activation"

# Log a lesson
python run_experiment.py --lesson architecture medium "GLU helps at depth=8"

# Check GPU queue
python gpu_queue.py status

# List checkpoints
python checkpoint.py

# Sync knowledge base to legacy results.tsv
python knowledge.py sync-tsv
```

## Requirements

- Single NVIDIA GPU (tested on H100)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Claude Code](https://claude.com/claude-code) CLI (`claude` command)
