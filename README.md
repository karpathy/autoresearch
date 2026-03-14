# AutoAnything

![teaser](progress.png)

**AutoAnything** is a framework where you define what to optimize and how to score it, then unleash a swarm of agents to hill-climb relentlessly. You come back to a leaderboard of experiments and a measurably better system.

The concept: any optimization problem with a black-box scoring function. Agents propose changes by pushing git branches. A private evaluator scores them serially and merges improvements. Agents never see the scoring code — just the leaderboard.

Think of it as **Kaggle for code**: the leaderboard is public, the test set is private, and submissions are git branches.

Currently configured for the GPT pretraining use case (optimizing val_bpb), based on [karpathy/nanochat](https://github.com/karpathy/nanochat). Originally forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## How it works

```
┌──────────────────────────────┐
│       Challenge Repo         │     Agents clone this, push proposal branches
│                              │
│  problem.yaml                │     What to optimize, constraints
│  state/train.py              │     The mutable file agents edit
│  context/prepare.py          │     Read-only context
│  agent_instructions.md       │     Protocol for agents
│  leaderboard.md              │     Auto-updated scoreboard
│  NO scoring code             │
└──────────┬───────────────────┘
           │
     push branches
           │
    ┌──────┴──────┐
    │  Evaluator   │     Private, gitignored, serial
    │              │
    │  score.sh    │     Runs scoring function
    │  evaluate.py │     Poll → score → merge/discard
    │  history.db  │     SQLite evaluation history
    └──────────────┘
```

**Agents** clone the repo, read the problem definition and leaderboard, modify the mutable files, and push a branch (`proposals/<name>/<description>`). They never see the scoring code.

**The evaluator** watches for new branches, scores them one at a time (serial queue), and either merges to main (if improved) or discards. The scoring code, test data, and history DB are all private (gitignored).

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download data and train tokenizer (one-time, ~2 min)
uv run context/prepare.py

# 3. Run a single training experiment (~5 min)
uv run state/train.py
```

## Setting up the evaluator

The evaluator is private and runs on your scoring machine. It's gitignored — agents never see it.

```bash
# The evaluator/ directory is already gitignored.
# On your scoring machine, create it:

# 1. Establish baseline (runs training once, records the score)
python evaluator/evaluate.py --baseline-only

# 2. Start the evaluation loop (polls for proposal branches)
python evaluator/evaluate.py

# 3. With auto-push (pushes leaderboard updates to origin)
python evaluator/evaluate.py --push
```

## Running agents

Point any AI agent at this repo. They should read `agent_instructions.md` for the protocol:

```
Read agent_instructions.md and start optimizing. Check the leaderboard first.
```

Agents create branches like `proposals/agent-1/higher-lr`, push them, and the evaluator picks them up automatically.

## Project structure

```
problem.yaml              — problem definition (what to optimize, constraints)
agent_instructions.md     — protocol for agents
leaderboard.md            — auto-updated scoreboard
state/train.py            — mutable file (agents edit this)
context/prepare.py        — read-only context (constants, data, evaluation)
evaluator/                — GITIGNORED (private scoring)
  score.sh                — runs training, extracts metrics
  evaluate.py             — serial evaluation loop
  history.db              — SQLite history (created on first run)
```

## Design

- **Serial evaluation.** One proposal scored at a time. No race conditions, no stale comparisons. The incumbent never changes during an evaluation.
- **Blind scoring.** Agents can't see the evaluator. If they could, they'd game it. Same reason Kaggle keeps the test set private.
- **Git as the protocol.** Branches track proposals, main tracks the best state. Anything that can `git push` can be an agent.
- **Fixed time budget.** Training runs for exactly 5 minutes. This makes experiments comparable regardless of what the agent changes.
- **Discard is forever.** If a proposal doesn't improve the score, it's gone. Agents can read the history and try refined versions, but there's no "almost" list.

## What you could optimize

The current problem is ML training (val_bpb), but AutoAnything generalizes to any black-box optimization:

- A prompt template (scored by LLM-as-judge accuracy)
- A web app's Lighthouse performance score
- A compiler optimization pass (scored by benchmark runtime)
- A trading strategy (scored by backtested Sharpe ratio)
- A game AI (scored by win rate against a baseline)

The common pattern: mutable state, a scoring function, and a direction (minimize or maximize).

## Heritage

Originally [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — single-agent, serial, one machine. AutoAnything is the distributed, multi-agent generalization.

## License

MIT
