# Agent Instructions

## Objective

Optimize the score ({{direction}}).

## Protocol

1. Pull the latest main branch.
2. Create a branch: `proposals/<your-name>/<description>`
3. Read `problem.yaml` for the full problem definition.
4. Read files in `context/` for background information.
5. Check `leaderboard.md` for the best scores and `history.md` for recent attempts.
6. You may create, modify, or delete files in `state/`.
7. Commit with a clear message explaining your approach.
8. Push your branch or open a PR targeting main.
9. The evaluator will score your submission and update the leaderboard.

## Files

- `problem.yaml` — problem definition
- `state/` — files you can modify (create, edit, or delete)
- `context/` — read-only background information
- `leaderboard.md` — best accepted scores
- `history.md` — recent attempts with scores and outcomes
