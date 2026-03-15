# AutoAnything

![teaser](progress.png)

**AutoAnything** is a framework where you define what to optimize and how to score it, then unleash a swarm of agents to hill-climb relentlessly. You come back to a leaderboard of experiments and a measurably better system.

The concept: any optimization problem with a black-box scoring function. Agents propose changes by pushing git branches. A private evaluator scores them serially and merges improvements. Agents never see the scoring code — just the leaderboard.

Think of it as **Kaggle for code**: the leaderboard is public, the test set is private, and submissions are git branches.

## How it works

Every problem has the same structure:

```
┌──────────────────────────────┐
│       Challenge Repo         │     Agents clone this, push branches or open PRs
│                              │
│  problem.yaml                │     What to optimize, constraints
│  state/*                     │     The mutable file(s) agents edit
│  context/*                   │     Read-only context
│  agent_instructions.md       │     Protocol for agents
│  leaderboard.md              │     Auto-updated scoreboard
│  NO scoring code             │
└──────────┬───────────────────┘
           │
   push branches / open PRs
           │
    ┌──────┴──────┐
    │  Evaluator   │     Private, gitignored, serial
    │              │
    │  score.sh    │     Runs scoring function
    │  evaluate.py │     Poll → score → merge/discard
    │  server.py   │     Webhook → score → comment/merge/close
    │  history.db  │     SQLite evaluation history
    └──────────────┘
```

**Agents** clone the repo, read the problem definition and leaderboard, modify the mutable files, and push a branch (`proposals/<name>/<description>`) or open a PR. They never see the scoring code.

**The evaluator** watches for new branches or PRs, scores them one at a time (serial queue), and either merges to master (if improved) or discards/closes. The scoring code, test data, and history DB are all private (gitignored).

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Activate a problem
bash test_problems/activate.sh rastrigin    # or: tsp, packing, gpt

# 3. Verify scoring works
bash evaluator/score.sh

# 4. Establish baseline and start the evaluator
uv run evaluator/evaluate.py --baseline-only
uv run evaluator/evaluate.py
```

The `gpt` problem requires an NVIDIA GPU. The other three (rastrigin, tsp, packing) score instantly with no hardware requirements — use them for developing and testing the framework.

## Setting up the evaluator

The evaluator is private and runs on your scoring machine. It's gitignored — agents never see it. Two deployment options:

### Option A: Polling evaluator (watches for proposal branches)

```bash
uv run evaluator/evaluate.py --baseline-only   # establish baseline
uv run evaluator/evaluate.py                    # start evaluation loop
uv run evaluator/evaluate.py --push             # with auto-push to origin
```

### Option B: Web evaluator (receives GitHub PR webhooks)

```bash
uv run evaluator/evaluate.py --baseline-only   # establish baseline first
uv run evaluator/server.py --push              # start webhook server

# Configure the GitHub webhook:
#   URL: https://<your-domain>/webhook
#   Content type: application/json
#   Secret: (set matching WEBHOOK_SECRET env var on the server)
#   Events: Pull requests only
```

The web evaluator listens for PR webhooks, scores submissions serially, comments results on the PR, and merges (if improved) or closes (if not). Set `WEBHOOK_SECRET` for signature verification.

## Running agents

Point any AI agent at this repo. They should read `agent_instructions.md` for the protocol:

```
Read agent_instructions.md and start optimizing. Check the leaderboard first.
```

Agents create branches like `proposals/agent-1/higher-lr` and push them, or open PRs targeting master. The evaluator picks them up automatically.

## Project structure

```
problem.yaml              — what to optimize (template; activate a problem to populate)
agent_instructions.md     — protocol for agents (generic; activate a problem to specialize)
leaderboard.md            — auto-updated scoreboard
state/                    — mutable file(s) agents edit (populated by activate.sh)
context/                  — read-only context (populated by activate.sh)
evaluator/                — GITIGNORED (private scoring)
  score.sh                — runs scoring, extracts metrics as JSON
  evaluate.py             — serial evaluation loop (polls for branches)
  server.py               — webhook-driven web evaluator (scores PRs)
  history.db              — SQLite history (created on first run)
test_problems/            — all optimization problems live here
  activate.sh             — switch the repo to a problem
  gpt/                    — GPT pretraining (val_bpb, requires GPU, ~5 min)
  rastrigin/              — 10-D function minimization (~170 → 0, instant)
  tsp/                    — shortest tour of 20 cities (~1914 → ~680, instant)
  packing/                — pack 12 rectangles (13250 → ~6975, instant)
```

## Problems

All problems follow the same structure. Activate one with `bash test_problems/activate.sh <name>`.

| Problem | What | Starting score | Optimum | Requirements |
|---------|------|---------------|---------|-------------|
| `rastrigin` | Minimize a 10-D multimodal function | ~169.7 | 0.0 | None |
| `tsp` | Shortest tour of 20 cities | ~1914 | ~680 | None |
| `packing` | Pack 12 rectangles into smallest box | 13250 | ~6975 | None |
| `gpt` | Optimize GPT training (val_bpb) | ~1.15 | unknown | NVIDIA GPU |

See [`test_problems/README.md`](test_problems/README.md) for details on each problem.

## Creating your own problem

Every problem is a directory under `test_problems/` with the same layout:

```
test_problems/<name>/
├── problem.yaml           # Problem definition (name, score, constraints)
├── agent_instructions.md  # What agents should know
├── state/*.py             # Mutable file(s) agents edit
├── context/*.py           # Read-only context
└── evaluator/score.sh     # Scoring script (outputs JSON on last line)
```

The evaluator is problem-agnostic — it reads the score metric name from `problem.yaml` and delegates scoring to `score.sh`. Your `score.sh` just needs to output a JSON object on its last line with at least the metric named in `problem.yaml`.

## Simulated test runs

`run_test.py` simulates an end-to-end optimization run with fake agent submissions and generates a progress chart. Runs in a temp directory — does not touch the repo.

```bash
uv run test_problems/run_test.py rastrigin
uv run test_problems/run_test.py tsp -n 20
uv run test_problems/run_test.py packing --include-failures
```

## Design

- **Serial evaluation.** One proposal scored at a time. No race conditions, no stale comparisons.
- **Blind scoring.** Agents can't see the evaluator. Same reason Kaggle keeps the test set private.
- **Git as the protocol.** Branches and PRs track proposals, master tracks the best state. Anything that can `git push` or open a PR can be an agent.
- **Discard is forever.** If a proposal doesn't improve the score, it's gone.

## What you could optimize

AutoAnything generalizes to any black-box optimization:

- A prompt template (scored by LLM-as-judge accuracy)
- A web app's Lighthouse performance score
- A compiler optimization pass (scored by benchmark runtime)
- A trading strategy (scored by backtested Sharpe ratio)
- A game AI (scored by win rate against a baseline)
- An ML training script (scored by validation loss)

The common pattern: mutable state, a scoring function, and a direction (minimize or maximize).

## Heritage

Originally [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — single-agent, serial, one machine. AutoAnything is the distributed, multi-agent generalization.

## License

MIT
