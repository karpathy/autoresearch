# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**autoresearch** is an autonomous ML research framework. An AI agent iterates on `train.py` to minimize `val_rmse` on a car price prediction task using XGBoost. The agent modifies feature engineering, hyperparameters, and model choice, then checks whether the validation RMSE improved, keeping or discarding each change.

The dataset is `car_sales_data.csv` — 50,000 UK used car listings with columns: `Manufacturer`, `Model`, `Engine size`, `Fuel type`, `Year of manufacture`, `Mileage`, `Price`.

## Commands

```bash
# Install dependencies
uv sync

# Run a single training experiment
uv run train.py

# Run and capture output (required for autonomous loop)
uv run train.py > run.log 2>&1

# Extract key metric from a run log
grep "^val_rmse:" run.log
tail -n 50 run.log  # inspect crashes
```

## File roles

- **`prepare.py`** — **DO NOT MODIFY.** Fixed constants (`DATA_PATH`, `TARGET`, `VAL_SIZE=0.2`, `RANDOM_STATE=42`, `CAT_COLS`, `NUM_COLS`), data loading (`load_raw()`), fixed train/val split (`get_train_val_split()`), and the canonical evaluation function (`evaluate_model(y_true, y_pred) → val_rmse`).
- **`plot.py`** — **DO NOT MODIFY.** Reads `results.tsv` and writes `progress.png`: only `keep` experiments, sorted by `val_rmse` descending. Called automatically at the end of `train.py`.
- **`train.py`** — **The only file the agent edits.** Feature engineering, model definition, training, and calls to `evaluate_model`. Has a `# Chart (do not modify)` section at the bottom that calls `plot.py`.
- **`program.md`** — Instructions for the autonomous agent: experiment loop protocol, logging format, rules.

## Architecture

### Data flow
`prepare.py` provides `load_raw()` (raw DataFrame) and `get_train_val_split(df)` (fixed 80/20 split by row index with `random_state=42`). `train.py` receives these DataFrames, applies all feature engineering, trains a model, generates predictions, and calls `evaluate_model(y_true, y_pred)` from `prepare.py`.

### Evaluation
`evaluate_model(y_true, y_pred)` computes RMSE on raw price (£). **Both arrays must be in the original price scale.** If `train.py` trains on `log(price)`, it must `np.expm1(preds)` before calling `evaluate_model`. The val split is always the same 10,000 rows, so results across experiments are directly comparable.

### Experiment loop (from `program.md`)
1. Create branch `autoresearch/<tag>` from master.
2. Baseline run first (unmodified `train.py`).
3. Log all results to `results.tsv` (tab-separated, **not** committed).
4. If `val_rmse` improves → keep commit and advance. If not → `git reset` to prior commit.
5. Never stop — loop until manually interrupted.

## Key constraints for experiments

- Only `train.py` may be edited.
- No new packages — only what's in `pyproject.toml` (pandas, numpy, scikit-learn, xgboost, matplotlib).
- `evaluate_model` in `prepare.py` is the ground truth; never modify it.
- Simplicity criterion: prefer simpler code when val_rmse is equal or better.
