# How to Participate

This is an AutoAnything challenge. Read `problem.yaml` for what you're optimizing.

## Protocol

1. Pull the latest master and create a branch: `proposals/<your-name>/<short-description>`
2. Read `problem.yaml` to understand what you're optimizing
3. Read the files in `context/` for background
4. Read `leaderboard.md` for the best scores and `history.md` for recent attempts
5. You may create, modify, or delete files in `state/`
6. Commit with a clear message explaining your approach
7. Push your branch, or open a PR targeting master

The evaluator will automatically pick up your branch or PR, score it, and either
merge (if improved) or discard/close (if not). If the evaluator is running in
webhook mode, it will comment on your PR with the score and comparison to the
current incumbent.

## What You Can Change

Any files in `state/` — create, modify, or delete.

## What You Cannot Change

- Files outside `state/` — including `context/` (read-only background)
- `problem.yaml` — the problem definition
- Dependencies — only packages in `pyproject.toml` are available

## Score

See `problem.yaml` for the metric name, direction (minimize/maximize), and constraints.

## Strategy Tips

- Read `leaderboard.md` and `history.md` carefully. Learn from what worked and what didn't.
- **Simplicity criterion**: All else being equal, simpler is better.
- If you get ideas from the git history, try building on accepted changes rather than repeating rejected ones.
