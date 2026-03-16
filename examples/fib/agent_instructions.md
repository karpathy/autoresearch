# Agent Instructions — Fibonacci Optimization

## Goal

Minimize the execution time of `fib(n)` in `state/fib.py`.

## Protocol

1. Pull latest, create branch: `proposals/<name>/<description>`
2. Read `problem.yaml` and `context/problem.py` for details
3. Modify only `state/fib.py` — the function must remain `def fib(n)` and return correct values
4. Commit with a clear message explaining your optimization approach
5. Push the branch — the evaluator will score it and merge if improved

## Current state

`state/fib.py` contains a naive recursive implementation. It's correct but exponentially slow.

## What's measured

The scoring function:
1. Validates correctness against known Fibonacci values (fib(0) through fib(20))
2. Benchmarks `fib(20)` five times and takes the median wall-clock time
3. Returns the median time as the score (lower is better)
4. If any correctness check fails, returns a penalty score of 999.0

## Optimization ideas

The naive recursive approach recomputes the same values exponentially many times. Consider:
- Memoization
- Iterative approaches
- Matrix exponentiation
- Closed-form solutions

## Constraints

- Function signature must remain `def fib(n) -> int`
- Must return correct values for all non-negative integers
- No hardcoded return values
- No reading from `scoring/`
