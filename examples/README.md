# Example Problems

Reference implementations of optimization problems for the AutoAnything framework. Each follows the same structure and scores via the same evaluator.

These examples live in the framework repo for reference. For operational use (running evaluators, accepting agent submissions), see [derby-examples](https://github.com/kousun12/derby-examples).

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

### 4. Fibonacci Optimization (`fib`)

**What:** Optimize a Fibonacci implementation for speed. The starting code uses naive recursion.

**State:** `state/fib.py` — a Python function `def fib(n) -> int`.

**Score:** Median wall-clock time for `fib(35)` in seconds — lower is better.

| Property | Value |
|----------|-------|
| Starting score | ~1.0s |
| Approximate optimum | ~0.000001s |
| Difficulty | Algorithmic — exponential → linear or better |
| Scoring time | ~1-5s |

**Why it's good for testing:** State is actual code, not just data. The scoring function validates correctness before benchmarking, so agents can't cheat. Shows how AutoAnything works for performance optimization with integrity checks.

---

### 5. GPT Pretraining (`gpt`)

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

```bash
autoanything init my-problem --direction minimize
cd my-problem
```

This scaffolds the full directory structure. A problem directory has this layout:

```
my-problem/
├── problem.yaml            # Problem definition + framework config
├── agent_instructions.md   # Protocol for agents
├── state/                  # Mutable files agents edit
│   └── solution.py
├── context/                # Read-only background for agents
├── scoring/                # GITIGNORED — private scoring code
│   └── score.py            # Implement score() → dict
├── leaderboard.md          # Auto-updated by the evaluator
└── .autoanything/          # GITIGNORED — local evaluator state
    └── history.db
```

Your `score.py` must define a `score()` function that returns a dict with at least the metric key named in `problem.yaml`. The evaluator reads the score name from `problem.yaml` and extracts it from this dict — everything else is automatic.

## Progress Charts

```bash
autoanything plot                                    # auto-detects DB location
autoanything plot --db path/to/history.db             # specific database
autoanything plot -o chart.png --title "My Run"       # custom output and title
```
