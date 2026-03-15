# How to Participate

This is an AutoAnything challenge. You are minimizing the **Rastrigin function** in 10 dimensions. Read `problem.yaml` for the full problem definition.

## Protocol

1. Pull the latest master and create a branch: `proposals/<your-name>/<short-description>`
2. Read `problem.yaml` to understand what you're optimizing
3. Read `context/problem.py` to see the scoring function
4. Read `leaderboard.md` to see what's been tried and what worked
5. Modify ONLY `state/solution.py` — change the values in the `x` list
6. Commit with a clear message explaining your approach
7. Push your branch, or open a PR targeting master

## What You Can Change

- **`state/solution.py`** — the 10-element solution vector `x`. Each value should be in [-5.12, 5.12].

## What You Cannot Change

- **`context/problem.py`** — the Rastrigin function definition (read-only)
- **`problem.yaml`** — the problem definition

## Score

The metric is the **Rastrigin function value** — **lower is better**. The global minimum is **0.0** at the origin (all zeros). The function has many local minima, so simply moving toward zero on each coordinate isn't guaranteed to improve the score if you don't move far enough.

## About the Rastrigin Function

```
f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
```

- 10 dimensions, each in [-5.12, 5.12]
- Many local minima arranged in a regular lattice
- Global minimum: f(0, 0, ..., 0) = 0.0
