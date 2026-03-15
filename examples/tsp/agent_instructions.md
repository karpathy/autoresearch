# How to Participate

This is an AutoAnything challenge. You are solving a **Traveling Salesman Problem** with 20 cities. Read `problem.yaml` for the full problem definition.

## Protocol

1. Pull the latest master and create a branch: `proposals/<your-name>/<short-description>`
2. Read `problem.yaml` to understand what you're optimizing
3. Read `context/cities.py` to see city coordinates and the distance function
4. Read `leaderboard.md` to see what's been tried and what worked
5. Modify ONLY `state/tour.py` — change the order of city indices
6. Commit with a clear message explaining your approach
7. Push your branch, or open a PR targeting master

## What You Can Change

- **`state/tour.py`** — the tour order. Must be a valid permutation of `[0, 1, 2, ..., 19]` (each city exactly once).

## What You Cannot Change

- **`context/cities.py`** — city coordinates and distance function (read-only)
- **`problem.yaml`** — the problem definition

## Score

The metric is **total Euclidean tour distance** (closed loop) — **lower is better**. The tour must visit all 20 cities exactly once and return to the starting city.

## Strategy Tips

- Look at the city coordinates in `context/cities.py` to reason about geography
- Nearest-neighbor, 2-opt, and other TSP heuristics all apply
- The tour is a closed loop — the last city connects back to the first
