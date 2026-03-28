# autoresearch — Classic ML

This is an autonomous experiment loop for classic machine learning research.
The agent iterates on `train.py`, runs experiments, logs results, and repeats.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar28`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, evaluation harness. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model, hyperparameters.
4. **Verify data exists**: Check that `~/.cache/autoresearch/house_prices/raw.pkl` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment modifies `train.py`, trains on the Ames Housing dataset, and evaluates on the held-out test set using `evaluate()` from `prepare.py`. You launch it as:

```
uv run train.py
```

**What you CAN do (all within `train.py`):**
- Feature engineering: imputation strategies, encoding schemes, log/power transforms, polynomial features, interaction terms, new derived features, feature selection
- Model selection: any model from sklearn, xgboost, lightgbm — or ensembles/stacking
- Hyperparameter tuning: manual, GridSearchCV, RandomizedSearchCV, or Optuna
- Cross-validation strategy: folds, stratification, repeated CV, etc.
- Target transformation: log1p, Box-Cox, Yeo-Johnson, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed `evaluate()` function and data split.
- Change the `evaluate(y_test, y_pred)` call to use a different test set or cheat in any way.
- Install new packages or add dependencies. Use only what is already in `pyproject.toml`.

**The goal is simple: get the lowest `val_rmse`.** This is the RMSE on the held-out test set in the original price scale (not log scale). Lower is better.

**Simplicity criterion**: All else being equal, simpler is better. A 50-RMSE improvement that adds 100 lines of hacky code might not be worth it. Removing something and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline — run the training script as-is, without any changes.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_rmse:         18423.456789
cv_rmse_log:      0.123456
total_seconds:    12.3
num_features:     245
model:            Ridge
train_rows:       1168
test_rows:        292
```

Extract the key metric:

```
grep "^val_rmse:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_rmse	cv_rmse_log	status	description
```

1. git commit hash (short, 7 chars)
2. val_rmse achieved (e.g. 18423.456789) — use 0.000000 for crashes
3. cv_rmse_log from cross-validation on log-transformed target (e.g. 0.123456) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_rmse	cv_rmse_log	status	description
a1b2c3d	18423.456789	0.123456	keep	baseline Ridge alpha=10
b2c3d4e	17500.234567	0.118900	keep	switch to XGBoost default params
c3d4e5f	19800.000000	0.134000	discard	remove scaling — worse result
d4e5f6g	0.000000	0.000000	crash	Optuna 1000 trials (timeout)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar28`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_rmse:\|^cv_rmse_log:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_rmse improved (lower), you "advance" the branch, keeping the git commit
9. If val_rmse is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should finish within 5 minutes wall clock. If a run exceeds 5 minutes (e.g. an Optuna search that won't stop), kill it and treat it as a failure — discard and revert.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — try feature interactions, target encoding, stacking, Bayesian hyperparameter search, polynomial features, outlier removal, etc. The loop runs until the human interrupts you, period.

## Ideas to try (not exhaustive)

- **Feature engineering**: log-transform skewed numeric features, total square footage, house age, remodel age, bathroom ratios, neighborhood interaction terms
- **Models**: ElasticNet, Lasso, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost-style encoding
- **Ensembling**: simple average of Ridge + XGBoost predictions, stacking with a meta-learner
- **Target transform**: compare log1p vs Box-Cox vs Yeo-Johnson
- **Hyperparameter tuning**: Optuna with early stopping (n_trials=50–200, timeout=120s)
- **Feature selection**: drop low-variance or high-multicollinearity features, RFECV
- **Outlier handling**: remove extreme outliers from training set (but not test)
