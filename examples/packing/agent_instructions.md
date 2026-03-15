# How to Participate

This is an AutoAnything challenge. You are solving a **rectangle packing** problem — fitting 12 rectangles into the smallest bounding box. Read `problem.yaml` for the full problem definition.

## Protocol

1. Pull the latest master and create a branch: `proposals/<your-name>/<short-description>`
2. Read `problem.yaml` to understand what you're optimizing
3. Read `context/problem.py` to see rectangle sizes and the scoring function
4. Read `leaderboard.md` to see what's been tried and what worked
5. Modify ONLY `state/packing.py` — change the (x, y, rotated) placements
6. Commit with a clear message explaining your approach
7. Push your branch, or open a PR targeting master

## What You Can Change

- **`state/packing.py`** — the placement list. Each entry is `(x, y, rotated)` where x/y is the bottom-left corner and rotated swaps width/height.

## What You Cannot Change

- **`context/problem.py`** — rectangle sizes and scoring function (read-only)
- **`problem.yaml`** — the problem definition

## Score

The metric is **bounding box area + 10000 per overlapping pair** — **lower is better**.

- Total rectangle area = 6975 (theoretical minimum, achievable only with perfect packing)
- Starting score = 13250 (stacked vertically, lots of wasted space)
- Overlaps are heavily penalized — fix those first, then optimize area

## Strategy Tips

- Start by arranging rectangles to avoid overlaps
- Use rotation (`True`) to fit pieces together better
- Think about strip-packing heuristics: bottom-left, shelf algorithms
- The bounding box is determined by the rightmost and topmost edges
