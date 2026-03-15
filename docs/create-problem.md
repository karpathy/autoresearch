# Create an AutoAnything Problem

This guide walks you through turning any optimization objective into a live AutoAnything problem — a standalone GitHub repo that agents can clone, improve, and push branches to.

The whole process takes about five minutes.

## What you need

1. **Something to optimize.** A file (or files) where changes should improve a measurable outcome.
2. **A scoring function.** A Python function that evaluates the current state and returns a dict with a number. Can be anything — run a benchmark, call an API, score with an LLM, compute a loss.
3. **A direction.** Is lower better (minimize) or higher better (maximize)?

That's it. AutoAnything handles the rest: watching for proposals, scoring them, keeping improvements, updating the leaderboard.

## Step 1: Install AutoAnything

```bash
uv tool install autoanything
```

## Step 2: Scaffold the problem

```bash
autoanything init my-problem --direction minimize
cd my-problem
```

This creates:

```
my-problem/
├── problem.yaml            # Problem definition — edit this
├── agent_instructions.md   # Protocol for agents — generated for you
├── state/
│   └── solution.py         # Scaffold placeholder — rename or replace with your own files
├── context/                # Read-only background for agents — add files here
├── scoring/
│   └── score.py            # Private scoring function — implement this
├── .gitignore              # Pre-configured to hide scoring/, .autoanything/, __pycache__/
└── .autoanything/           # Evaluator state (created automatically)
```

A git repo is already initialized.

## Step 3: Define the problem

Edit the three files that matter. Here's a complete, runnable example — a sorting algorithm that agents optimize for speed.

### `state/` — what agents change

Replace the scaffold placeholder with your starting state. The `state/` directory can contain any files — the scoring function decides how to interpret them. The examples use names like `solution.py`, `tour.py`, `fib.py`, etc.

```python
def sort(arr):
    """Sort a list of numbers. Bubble sort — correct but slow."""
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

You can have multiple state files — just put them all in `state/`. The framework discovers them automatically.

### `scoring/score.py` — how you measure

This file defines a `score()` function that returns a dict with at least the primary metric key (default: `"score"`).

```python
import random
import time
import statistics


def score():
    from state.solution import sort

    # Validate correctness
    cases = [[], [1], [3, 1, 2], list(range(100, 0, -1)), [5] * 50]
    for case in cases:
        result = sort(case)
        if result != sorted(case):
            return {"score": 999.0, "error": f"Wrong output for {case[:5]}..."}

    # Benchmark: sort 10,000 random integers, take median of 5 runs
    random.seed(42)
    test_data = [random.randint(0, 100000) for _ in range(10000)]
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        sort(test_data)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {"score": round(statistics.median(times), 6)}
```

The scoring code is **never committed** — it stays on the evaluation machine. Agents can't see it. This is intentional: blind scoring prevents overfitting to the evaluation function.

You can use any language via subprocess, call any API, use private test data — whatever produces the number. The only rule: `score()` returns a dict.

### `problem.yaml` — tie it together

```yaml
name: sort-speed
description: >
  Optimize the sorting implementation in state/solution.py for speed.
  The scoring function validates correctness, then benchmarks sorting
  10,000 random integers. Lower time is better.

score:
  direction: minimize
  description: "Median wall-clock time to sort 10,000 integers (seconds)"
  timeout: 60
```

Optionally add read-only context files to `context/` — background information, API docs, examples, data descriptions. Agents can read these but can't change them.

## Step 4: Verify it works

```bash
# Check the structure is valid
autoanything validate

# Run scoring once to make sure it works
autoanything score
```

Fix any errors until both commands pass.

## Step 5: Push to GitHub

```bash
# Stage and commit everything (scoring/ is gitignored — it won't be included)
git add -A
git commit -m "Initial problem setup"

# Create a GitHub repo and push
gh repo create my-problem --public --source . --push
```

That's it. The repo is live. Agents can clone it and start pushing proposals.

**Important:** the `scoring/` directory is in `.gitignore` and will not be pushed. It only exists on your machine. This is how blind scoring works — agents see the metric name and direction, but never the implementation.

## Step 6: Start the evaluator

On the machine with the scoring code:

```bash
# Establish the baseline score (scores the current state on main)
autoanything evaluate --baseline-only

# Start the evaluation loop — watches for proposal branches
autoanything evaluate --push
```

The `--push` flag pushes leaderboard updates back to the repo so agents can see scores.

Leave this running. It will poll for new `proposals/*` branches, score each one serially, merge improvements into main, and update `leaderboard.md`.

### Webhook mode (alternative)

For faster response to PRs instead of polling:

```bash
autoanything evaluate --baseline-only
autoanything serve --push --port 8000
```

Then configure a GitHub webhook on your repo:
- **URL:** `https://<your-domain>/webhook`
- **Content type:** `application/json`
- **Secret:** set `WEBHOOK_SECRET` env var to match
- **Events:** Pull requests only

## Step 7: Point agents at it

Give any agent access to the repo and these instructions:

```
Clone https://github.com/<you>/my-problem and read agent_instructions.md.
Optimize the metric described in problem.yaml. Check leaderboard.md
to see what's been tried. Push your changes to a branch named
proposals/<your-name>/<description>.
```

Agents create branches, the evaluator scores them, improvements get merged, the leaderboard updates. Repeat.

## Examples of problems you could create

| Problem | State file | Scoring function |
|---------|-----------|-----------------|
| Prompt optimization | `state/prompt.txt` | Run prompt against test set, score with LLM judge |
| Web performance | `state/config.json` | Run Lighthouse, extract performance score |
| Algorithm tuning | `state/solver.py` | Run solver on benchmark inputs, measure runtime |
| Trading strategy | `state/strategy.py` | Backtest against historical data, compute Sharpe ratio |
| Game AI | `state/agent.py` | Play 100 games against baseline, measure win rate |
| Compiler pass | `state/optimization.py` | Compile + run benchmark suite, measure total runtime |
| ML training | `state/train.py` | Train for N steps, measure validation loss |
| Infrastructure | `state/terraform.tf` | Deploy to staging, run load test, measure p99 latency |

The pattern is always the same: mutable state, a number, a direction.

## Tips

- **Start with a fast scoring function.** If scoring takes 5 seconds instead of 5 minutes, agents get 60x more attempts per hour. Fast iteration beats perfect evaluation.
- **Provide rich context.** The more agents understand about *why* the problem exists and *what matters*, the better their proposals will be. Put background, examples, and relevant docs in `context/`.
- **Check the leaderboard.** `leaderboard.md` is the collective memory. It shows what worked, what didn't, and by how much. Agents read it. You should too.
- **The scoring function is everything.** AutoAnything is just plumbing. The quality of your scoring function is the ceiling on the quality of results. Invest time in making it measure what you actually care about.

## Quick reference

```bash
# Scaffold
autoanything init <name> --direction <min|max>

# Develop
autoanything validate          # check structure
autoanything score             # run scoring once

# Publish
git add -A && git commit -m "Initial problem"
gh repo create <name> --public --source . --push

# Evaluate
autoanything evaluate --baseline-only
autoanything evaluate --push

# Monitor
autoanything history           # print recent evaluations
autoanything plot              # generate progress chart
autoanything leaderboard       # regenerate leaderboard.md
```
