# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AutoAnything — a framework for autonomous optimization via AI agents. Agents propose changes, an evaluator scores them against a black-box metric, and only improvements are kept. Any optimization problem with a scoring function can be plugged in.

## Repository Structure

```
autoanything/
├── src/autoanything/        # Installable package (the framework)
│   ├── cli.py                # CLI entry point (click)
│   ├── evaluator.py          # Polling evaluation loop
│   ├── server.py             # Webhook server (FastAPI)
│   ├── scoring.py            # Run score.sh, parse JSON output
│   ├── problem.py            # Parse + validate problem.yaml (PyYAML)
│   ├── leaderboard.py        # Render leaderboard.md from history
│   ├── plotting.py           # Progress chart generation (matplotlib)
│   ├── history.py            # SQLite history management
│   └── git.py                # Git operations (subprocess wrappers)
├── problem.yaml             # Problem definition (populated by activate.sh)
├── agent_instructions.md    # Protocol for agents (populated by activate.sh)
├── leaderboard.md           # Auto-updated scoreboard
├── state/                   # MUTABLE — file(s) agents modify (populated by activate.sh)
├── context/                 # READ-ONLY — background for agents (populated by activate.sh)
├── evaluator/               # GITIGNORED — private scoring code + history DB
│   └── score.sh              # Runs scoring, extracts metrics as JSON
├── .autoanything/           # GITIGNORED — evaluator state (history.db)
├── examples/                # Example optimization problems
│   ├── activate.sh           # Switch repo to a problem
│   ├── rastrigin/            # 10-D function minimization (score: ~170 → 0)
│   ├── tsp/                  # Traveling salesman, 20 cities (score: ~1914 → ~680)
│   ├── packing/              # Rectangle packing, 12 rects (score: 13250 → ~6975)
│   └── gpt/                  # GPT pretraining, val_bpb (~1.15 → ?, requires GPU)
└── tests/                   # Test suite (106 tests)
```

## Commands

```bash
uv sync                                    # install dependencies

# Activate a problem (copies files into root)
bash examples/activate.sh rastrigin        # or: tsp, packing, gpt
bash evaluator/score.sh                    # verify scoring works

# Evaluator (run on the scoring machine, not by agents)
autoanything evaluate                      # start the serial evaluation loop
autoanything evaluate --baseline-only      # just establish the baseline score
autoanything evaluate --push               # push leaderboard updates to origin
autoanything serve                         # start the webhook-driven web evaluator
autoanything serve --push                  # web evaluator with auto-push

# GPT problem only (after activating gpt)
uv run context/prepare.py                  # one-time: download data + train tokenizer
uv run state/train.py                      # run a single training experiment (~5 min)

# Progress charts
autoanything plot                         # chart from .autoanything/history.db
autoanything plot --db path/to/history.db  # chart from a specific database
autoanything plot -o chart.png            # save to a specific path

# Simulated test run (generates progress chart, doesn't touch working tree)
uv run examples/run_test.py rastrigin              # run with 15 submissions
uv run examples/run_test.py tsp -n 20              # more submissions
uv run examples/run_test.py packing --include-failures  # with crash submissions
```

## How Problems Work

Every problem follows the same structure — a directory under `examples/` with:

```
examples/<name>/
├── problem.yaml           # Problem definition (name, score direction, constraints)
├── agent_instructions.md  # Protocol for agents
├── state/*.py             # Mutable file(s) agents edit
├── context/*.py           # Read-only context
└── evaluator/score.sh     # Scoring script (outputs JSON on last line)
```

`activate.sh` copies these into the repo root. The evaluator is problem-agnostic — it reads the score metric name from `problem.yaml` and delegates scoring to `score.sh`.

## Agent Protocol

1. Pull latest master, create branch: `proposals/<name>/<description>`
2. Read `problem.yaml`, `context/`, and `leaderboard.md` for context
3. Modify ONLY the files listed under `state:` (or `mutable:`) in `problem.yaml`
4. Commit with a clear message explaining the approach
5. Push the branch or open a PR targeting master — the evaluator scores it and merges if improved

## Evaluator Design

- **Two modes**: polling (`autoanything evaluate` watches for branches) or webhook (`autoanything serve` receives PR events)
- **Serial evaluation**: one proposal at a time, no race conditions
- **Blind scoring**: agents never see `evaluator/` or `scoring/` (gitignored)
- **SQLite history**: all evaluations recorded in `.autoanything/history.db`
- **Auto-leaderboard**: `leaderboard.md` updated after each evaluation

## Available Problems

| Problem | Description | Starting → Optimum | Requirements |
|---------|-------------|-------------------|-------------|
| `rastrigin` | Minimize 10-D Rastrigin function (many local minima) | ~169.7 → 0.0 | None |
| `tsp` | Shortest tour of 20 fixed cities | ~1914 → ~680 | None |
| `packing` | Pack 12 rectangles into smallest bounding box | 13250 → ~6975 | None |
| `gpt` | Optimize GPT training script (val_bpb) | ~1.15 → ? | NVIDIA GPU |

The first three score instantly (<1ms) and need no GPU — use them for framework development. The `gpt` problem is the real-world use case (~5 min scoring, GPU required).
