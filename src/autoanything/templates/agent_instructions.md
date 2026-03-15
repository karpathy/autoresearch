# Agent Instructions

## Objective

Optimize the metric `{{metric}}` ({{direction}}).

## Protocol

1. Pull the latest main branch.
2. Create a branch: `proposals/<your-name>/<description>`
3. Read `problem.yaml` for the full problem definition and constraints.
4. Read files in `context/` for background information.
5. Check `leaderboard.md` for current scores and what has been tried.
6. Modify only the files listed under `state:` in `problem.yaml`.
7. Commit with a clear message explaining your approach.
8. Push your branch or open a PR targeting main.
9. The evaluator will score your submission and update the leaderboard.

## Files

- `problem.yaml` — problem definition and constraints
- `state/` — files you can modify
- `context/` — read-only background information
- `leaderboard.md` — current scores and history
