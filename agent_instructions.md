# How to Participate

This is an AutoAnything challenge. You are optimizing a GPT training script for the lowest validation bits-per-byte (val_bpb). Read `problem.yaml` for the full problem definition.

## Protocol

1. Pull the latest master and create a branch: `proposals/<your-name>/<short-description>`
2. Read `problem.yaml` to understand what you're optimizing
3. Read the files in `context/` for background (constants, data loading, evaluation)
4. Read `leaderboard.md` to see what's been tried and what worked
5. Modify ONLY the files listed under `mutable` in `problem.yaml` (currently just `state/train.py`)
6. Commit with a clear message explaining your approach
7. Push your branch (or open a PR)

The evaluator will automatically pick up your branch, score it, and either
merge (if improved) or discard (if not).

## What You Can Change

- **`state/train.py`** — everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

## What You Cannot Change

- **`context/prepare.py`** — read-only. Contains fixed constants (`MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`), data loading, tokenizer, and the evaluation function (`evaluate_bpb`).
- **`problem.yaml`** — the problem definition.
- **Dependencies** — only packages in `pyproject.toml` are available.

## Running Locally (Optional)

You can test your changes locally before pushing:

```bash
uv run state/train.py > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

This requires an NVIDIA GPU. Each run takes ~5 minutes.

## Score

The metric is **val_bpb** (validation bits per byte) — **lower is better**. This is an unbounded score with no known theoretical minimum. The scoring function in `context/prepare.py` (`evaluate_bpb`) is the ground truth metric.

VRAM is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

## Strategy Tips

- Read `leaderboard.md` carefully. Learn from what worked and what didn't.
- **Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.
- The training budget is fixed at 5 minutes. You're optimizing what happens in that time.
- If you get ideas from the git history, try building on accepted changes rather than repeating rejected ones.
