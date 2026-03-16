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
│   ├── runner.py             # Local optimization loop (maxx run)
│   ├── server.py             # Webhook server (FastAPI)
│   ├── scoring.py            # Run scoring/score.py, parse JSON output
│   ├── problem.py            # Parse + validate problem.yaml (PyYAML)
│   ├── leaderboard.py        # Render leaderboard.md and history.md from DB
│   ├── plotting.py           # Progress chart generation (matplotlib)
│   ├── history.py            # SQLite history management
│   └── git.py                # Git operations (subprocess wrappers)
├── examples/                # Reference problem structures (read-only)
│   ├── rastrigin/            # 10-D function minimization (score: ~170 → 0)
│   ├── tsp/                  # Traveling salesman, 20 cities (score: ~1914 → ~680)
│   ├── packing/              # Rectangle packing, 12 rects (score: 13250 → ~6975)
│   ├── fib/                  # Fibonacci optimization (score: ~1.0s → ~0.000001s)
│   └── gpt/                  # GPT pretraining, val_bpb (~1.15 → ?, requires GPU)
└── tests/                   # Test suite
```

## Commands

```bash
uv sync                                    # install dependencies

# Try an example problem (quick demo with built-in agent)
maxx try rastrigin                 # set up + run demo agent + plot
maxx try fib --claude              # use Claude as the agent
maxx try tsp -a "./my_agent.sh"    # use a custom agent

# Local optimization loop (run from a problem directory)
maxx run -a "./my_agent.sh"        # run agent in a loop, score locally
maxx run -a "python opt.py" -n 50  # limit to 50 iterations
maxx run -a "claude -p 'improve'" -n 10  # use any command as the agent

# Remote evaluator (run from a problem directory, not by agents)
maxx evaluate                      # start the serial evaluation loop
maxx evaluate --baseline-only      # just establish the baseline score
maxx evaluate --push               # push leaderboard updates to origin
maxx serve                         # start the webhook-driven web evaluator
maxx serve --push                  # web evaluator with auto-push

# GPT problem only (after activating gpt)
uv run context/prepare.py                  # one-time: download data + train tokenizer
uv run state/train.py                      # run a single training experiment (~5 min)

# Progress charts
maxx plot                         # chart from .autoanything/history.db
maxx plot --db path/to/history.db  # chart from a specific database
maxx plot -o chart.png            # save to a specific path

# Runnable examples with test harness: https://github.com/kousun12/derby-examples
```

## How Problems Work

Every problem is a self-contained directory (typically its own git repo):

```
my-problem/
├── problem.yaml           # Problem definition (name, score direction)
├── agent_instructions.md  # Protocol for agents
├── state/                 # Mutable files — agents can create, modify, or delete
├── context/               # Read-only context (optional)
├── scoring/
│   └── score.py           # GITIGNORED — implement score() → dict
└── .autoanything/         # GITIGNORED — evaluator state (history.db)
```

The evaluator is problem-agnostic — it reads the score metric name from `problem.yaml` and runs `scoring/score.py`. The `score()` function returns a dict with the metric key. Reference examples live in `examples/` in this repo; runnable problem repos live at [derby-examples](https://github.com/kousun12/derby-examples).

## Agent Protocol

1. Pull latest master, create branch: `proposals/<name>/<description>`
2. Read `problem.yaml`, `context/`, `leaderboard.md`, and `history.md` for context
3. You may create, modify, or delete files in `state/`
4. Commit with a clear message explaining the approach
5. Push the branch or open a PR targeting master — the evaluator scores it and merges if improved

## Evaluator Design

- **Three modes**: local loop (`maxx run`), polling (`maxx evaluate`), or webhook (`maxx serve`)
- **Local loop**: runs a user-provided agent command repeatedly — the framework handles branching, scoring, merging improvements, and leaderboard. Scoring directory is hidden from the agent during execution. Agent gets env vars: `AUTOANYTHING_ITERATION`, `AUTOANYTHING_SCORE`, `AUTOANYTHING_DIRECTION`, `AUTOANYTHING_METRIC`, `AUTOANYTHING_PROBLEM`.
- **Serial evaluation**: one proposal at a time, no race conditions
- **Blind scoring**: agents never see `scoring/` (gitignored; physically hidden during `run`)
- **SQLite history**: all evaluations recorded in `.autoanything/history.db`
- **Auto-leaderboard**: `leaderboard.md` (best scores) and `history.md` (recent attempts) updated after each evaluation

## Available Problems

| Problem | Description | Starting → Optimum | Requirements |
|---------|-------------|-------------------|-------------|
| `rastrigin` | Minimize 10-D Rastrigin function (many local minima) | ~169.7 → 0.0 | None |
| `tsp` | Shortest tour of 20 fixed cities | ~1914 → ~680 | None |
| `packing` | Pack 12 rectangles into smallest bounding box | 13250 → ~6975 | None |
| `fib` | Optimize Fibonacci implementation for speed | ~1.0s → ~0.000001s | None |
| `gpt` | Optimize GPT training script (val_bpb) | ~1.15 → ? | NVIDIA GPU |

The first four score instantly or near-instantly and need no GPU — use them for framework development. The `gpt` problem is the real-world use case (~5 min scoring, GPU required).
