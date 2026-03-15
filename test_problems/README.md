# Problems

All optimization problems for the AutoAnything framework live here. Each follows the same structure, scores via the same evaluator, and is activated the same way.

## Quick Start

```bash
# Activate a problem (copies files into the repo root)
bash test_problems/activate.sh rastrigin

# Verify scoring works
bash evaluator/score.sh

# Establish baseline and start evaluator
uv run evaluator/evaluate.py --baseline-only
uv run evaluator/evaluate.py

# Switch to a different problem
bash test_problems/activate.sh tsp
```

## Available Problems

### 1. Rastrigin Function Minimization (`rastrigin`)

**What:** Find the minimum of the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function) in 10 dimensions.

**State:** `state/solution.py` — a list of 10 floats.

**Score:** `f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))` — lower is better.

| Property | Value |
|----------|-------|
| Starting score | ~169.7 |
| Global optimum | 0.0 (all zeros) |
| Difficulty | Many local minima in a regular lattice |
| Scoring time | <1ms |

**Why it's good for testing:** Absolute minimum complexity. State is a list of numbers, scoring is a single function call. Tests the full loop (branch, score, merge/reject) without any possible infrastructure issues.

---

### 2. Traveling Salesman Problem (`tsp`)

**What:** Find the shortest tour visiting 20 fixed cities on a 200x200 grid.

**State:** `state/tour.py` — a permutation of `[0..19]`.

**Score:** Total Euclidean distance of the closed tour — lower is better.

| Property | Value |
|----------|-------|
| Starting score | ~1914 |
| Approximate optimum | ~680 |
| Difficulty | 20! possible permutations, combinatorial |
| Scoring time | <1ms |

**Why it's good for testing:** Slightly more complex state (permutation, not just numbers). Has a hard validity constraint (must be a valid permutation). Good for testing that agents respect constraints and that the evaluator handles invalid submissions gracefully.

---

### 3. Rectangle Packing (`packing`)

**What:** Pack 12 fixed-size rectangles into the smallest bounding box with no overlaps.

**State:** `state/packing.py` — a list of `(x, y, rotated)` placements.

**Score:** Bounding box area + 10000 per overlapping pair — lower is better.

| Property | Value |
|----------|-------|
| Starting score | 13250 |
| Theoretical minimum | 6975 (total rectangle area, assuming perfect packing) |
| Difficulty | Geometric reasoning, rotation choices |
| Scoring time | <1ms |

**Why it's good for testing:** More structured state (tuples with multiple fields). Has soft constraints (overlaps penalized, not rejected). Tests that agents can work with composite state and multi-objective scoring.

---

### 4. GPT Pretraining (`gpt`)

**What:** Optimize a GPT training script for lowest validation bits-per-byte (val_bpb).

**State:** `state/train.py` — full model architecture, optimizer, hyperparameters, training loop.

**Score:** val_bpb — lower is better.

| Property | Value |
|----------|-------|
| Starting score | ~1.15 |
| Optimum | Unknown (unbounded) |
| Difficulty | Large search space, slow evaluation |
| Scoring time | ~5 minutes |

**Requirements:** NVIDIA GPU with CUDA (tested on H100), data download via `uv run context/prepare.py`.

**Why it's useful:** The original real-world use case. Tests the full framework under realistic conditions with expensive scoring and complex state.

## Creating Your Own Problem

Every problem is a directory with the same layout:

```
test_problems/<name>/
├── problem.yaml           # Problem definition
├── agent_instructions.md  # Protocol for agents
├── state/*.py             # Mutable file(s) agents edit
├── context/*.py           # Read-only context
└── evaluator/score.sh     # Scoring script
```

Your `score.sh` must output a JSON object on its last line with at least the metric key named in `problem.yaml`. The evaluator reads the score name from `problem.yaml` and extracts it from this JSON — everything else is automatic.

Example for a minimization problem:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
python3 -c "
import json, sys
sys.path.insert(0, 'context')
sys.path.insert(0, 'state')
from problem import evaluate
from solution import x
print(json.dumps({'score': evaluate(x)}))
"
```

## Simulated Test Runs

`run_test.py` simulates an end-to-end optimization run with fake agent submissions and generates a progress chart. Runs in a temp directory — does not touch the repo working tree.

```bash
# Run a test (generates test_progress_<problem>.png in current dir)
uv run test_problems/run_test.py rastrigin
uv run test_problems/run_test.py tsp --submissions 20
uv run test_problems/run_test.py packing --include-failures -o chart.png
```

Options:
- `-n`, `--submissions` — number of simulated submissions (default: 15)
- `--include-failures` — include intentionally crashing submissions
- `-o`, `--output` — output chart path (default: `test_progress_<problem>.png`)
- `--seed` — random seed for reproducibility (default: 42)

Requires `matplotlib` (add with `uv add matplotlib` if not already in deps).

### Standalone chart generation

`plot_progress.py` generates a progress chart from any evaluation `history.db` — works with both test runs and real evaluator runs.

```bash
uv run test_problems/plot_progress.py evaluator/history.db
uv run test_problems/plot_progress.py evaluator/history.db -o chart.png --title "My Run"
```
