# autoresearch — Classic ML

An autonomous experiment loop for classic machine learning research. Give an agent a dataset and a baseline model, let it experiment overnight — modifying feature engineering, model selection, and hyperparameters — and wake up to a log of experiments and (hopefully) a better model.

The training code here starts with Ames Housing (house price prediction) as a concrete example, but the dataset is swappable. The agent only touches `train.py`. You program the research org by editing `program.md`.

## How it works

The repo has three files that matter:

- **`prepare.py`** — fixed constants, one-time data download (fetches dataset from OpenML), and the evaluation harness (`evaluate()`). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains feature engineering, model selection, and hyperparameter tuning. Everything here is fair game.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go.

The metric is **val_rmse** (RMSE on the held-out test set, original price scale) — lower is better. The train/test split is fixed by a random seed so experiments are directly comparable.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download dataset and verify setup (one-time, ~1 min)
uv run prepare.py

# 4. Run a baseline experiment
uv run train.py
```

If the above commands all work, your setup is working and you can go into autonomous research mode.

## Running the agent

Spin up Claude Code (or any agent) in this repo, then prompt:

```
Have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is the lightweight "research org program" the agent follows.

## Project structure

```
prepare.py      — constants, data download + evaluation harness (do not modify)
train.py        — feature engineering, model, hyperparameters (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Diffs stay reviewable.
- **Fixed train/test split.** `RANDOM_STATE = 42` in `prepare.py` ensures all experiments are evaluated on the same held-out test set — results are directly comparable.
- **Modular datasets.** `prepare.py` has a `DATASET_CONFIGS` registry. To add a new dataset, add an entry with a `fetch_fn` and `load_fn`. Change `DATASET = "..."` at the top to switch.
- **Configurable metric.** `METRIC` in `prepare.py` controls the evaluation: `"rmse"`, `"mae"`, or `"r2"`.
- **Self-contained.** No GPU required. Runs on any machine with Python.

## Adding a new dataset

1. Add a `fetch_fn` (downloads and caches raw data) and `load_fn` (returns `X, y` as pandas objects) to `DATASET_CONFIGS` in `prepare.py`.
2. Update `DATASET`, `TARGET`, and `METRIC` constants at the top of `prepare.py`.
3. Run `uv run prepare.py` to cache the new dataset.
4. Update `train.py` baseline as needed.

## License

MIT
