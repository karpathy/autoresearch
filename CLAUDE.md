# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AutoAnything — a framework for autonomous optimization via AI agents. Agents propose changes, an evaluator scores them against a black-box metric, and only improvements are kept. Any optimization problem with a scoring function can be plugged in.

## Repository Structure

```
autoanything/
├── problem.yaml             # Problem definition (template; populated by activate.sh)
├── agent_instructions.md    # Protocol for agents (generic; populated by activate.sh)
├── leaderboard.md           # Auto-updated scoreboard
├── state/                   # MUTABLE — file(s) agents modify (populated by activate.sh)
├── context/                 # READ-ONLY — background for agents (populated by activate.sh)
├── evaluator/               # GITIGNORED — private scoring code + history DB
│   ├── score.sh              # Runs scoring, extracts metrics as JSON
│   ├── evaluate.py           # Serial evaluation loop (poll, score, merge/discard)
│   ├── server.py             # Webhook-driven web evaluator (PR-based workflow)
│   └── history.db            # SQLite evaluation history (created on first run)
└── test_problems/           # All optimization problems
    ├── activate.sh           # Switch repo to a problem
    ├── rastrigin/            # 10-D function minimization (score: ~170 → 0)
    ├── tsp/                  # Traveling salesman, 20 cities (score: ~1914 → ~680)
    ├── packing/              # Rectangle packing, 12 rects (score: 13250 → ~6975)
    └── gpt/                  # GPT pretraining, val_bpb (~1.15 → ?, requires GPU)
```

## Commands

```bash
uv sync                                    # install dependencies

# Activate a problem (copies files into root)
bash test_problems/activate.sh rastrigin   # or: tsp, packing, gpt
bash evaluator/score.sh                    # verify scoring works

# Evaluator (run on the scoring machine, not by agents)
uv run evaluator/evaluate.py               # start the serial evaluation loop
uv run evaluator/evaluate.py --baseline-only  # just establish the baseline score
uv run evaluator/evaluate.py --push        # push leaderboard updates to origin
uv run evaluator/server.py                 # start the webhook-driven web evaluator
uv run evaluator/server.py --push          # web evaluator with auto-push

# GPT problem only (after activating gpt)
uv run context/prepare.py                  # one-time: download data + train tokenizer
uv run state/train.py                      # run a single training experiment (~5 min)

# Simulated test run (generates progress chart, doesn't touch working tree)
uv run test_problems/run_test.py rastrigin              # run with 15 submissions
uv run test_problems/run_test.py tsp -n 20              # more submissions
uv run test_problems/run_test.py packing --include-failures  # with crash submissions
uv run test_problems/plot_progress.py evaluator/history.db   # chart from real evaluator
```

## How Problems Work

Every problem follows the same structure — a directory under `test_problems/` with:

```
test_problems/<name>/
├── problem.yaml           # Problem definition (name, score direction, constraints)
├── agent_instructions.md  # Protocol for agents
├── state/*.py             # Mutable file(s) agents edit
├── context/*.py           # Read-only context
└── evaluator/score.sh     # Scoring script (outputs JSON on last line)
```

`activate.sh` copies these into the repo root. The evaluator (`evaluate.py`, `server.py`) is problem-agnostic — it reads the score metric name from `problem.yaml` and delegates scoring to `score.sh`.

## Agent Protocol

1. Pull latest master, create branch: `proposals/<name>/<description>`
2. Read `problem.yaml`, `context/`, and `leaderboard.md` for context
3. Modify ONLY the files listed under `mutable` in `problem.yaml`
4. Commit with a clear message explaining the approach
5. Push the branch or open a PR targeting master — the evaluator scores it and merges if improved

## Evaluator Design

- **Two modes**: polling (`evaluate.py` watches for branches) or webhook (`server.py` receives PR events)
- **Serial evaluation**: one proposal at a time, no race conditions
- **Blind scoring**: agents never see `evaluator/` (gitignored)
- **SQLite history**: all evaluations recorded in `evaluator/history.db`
- **Auto-leaderboard**: `leaderboard.md` updated after each evaluation

## Available Problems

| Problem | Description | Starting → Optimum | Requirements |
|---------|-------------|-------------------|-------------|
| `rastrigin` | Minimize 10-D Rastrigin function (many local minima) | ~169.7 → 0.0 | None |
| `tsp` | Shortest tour of 20 fixed cities | ~1914 → ~680 | None |
| `packing` | Pack 12 rectangles into smallest bounding box | 13250 → ~6975 | None |
| `gpt` | Optimize GPT training script (val_bpb) | ~1.15 → ? | NVIDIA GPU |

The first three score instantly (<1ms) and need no GPU — use them for framework development. The `gpt` problem is the real-world use case (~5 min scoring, GPU required).
