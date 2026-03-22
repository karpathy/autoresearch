# autoresearch-sklearn

Autonomous sklearn experimentation loop on a hard imbalanced classification problem.

**Dataset**: Credit Card Fraud Detection (OpenML id=1597)
- 284,807 transactions; only 492 (~0.17%) are fraud
- 30 features: V1-V28 (PCA components), Amount, Time
- Binary target: 0 = legitimate, 1 = fraud

**Metric**: macro-averaged F1 score — **higher is better**.
Macro F1 averages the F1 of each class equally, so doing well on the rare fraud class matters
as much as doing well on the majority class. A dumb "predict all 0" classifier gets ~0.5.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`).
   The branch `autoresearch-sklearn/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-sklearn/<tag>` from current master.
3. **Read the in-scope files**:
   - `prepare.py` — fixed constants, data loading, evaluation. Do not modify.
   - `model.py` — the file you modify. Model, features, hyperparameters.
4. **Sync environment**: Run `uv sync` (once) to install packages.
5. **First run downloads data**: `uv run model.py` will download ~30 MB on first run and cache it.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Confirm and go**.

## Experimentation

Run experiments as: `uv run model.py`

**What you CAN do:**
- Modify `model.py` — fully fair game: switch algorithms, tune hyperparameters, engineer
  features, build ensembles, resampling strategies (SMOTE etc. — but only if in pyproject.toml),
  threshold tuning, calibration, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed stratified data split and the ground-truth `evaluate()`.
- Add new packages beyond `pyproject.toml` (sklearn, lightgbm, xgboost, numpy, pandas).
- Change the data split, stratification, or random seed used in evaluate().
- Tune the decision threshold after seeing val labels — threshold must be the default 0.5 unless
  set via model hyperparameters (e.g. `scale_pos_weight`, `class_weight`).

**The goal: highest val_f1 (macro).** Higher is better.

**Simplicity criterion**: same as always — marginal improvement + complex code = not worth it.
Simplification that ties or beats = always keep.

**The first run**: run `model.py` as-is to establish the baseline.

## Output format

The script prints:
```
---
val_f1:          0.912345
train_seconds:   4.56
positive_rate:   0.0017
```

Extract the key metric: `grep "^val_f1:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Do not commit this file.

```
commit	val_f1	status	description
```

1. git commit hash (7 chars)
2. val_f1 (6 decimal places)
3. status: `keep`, `discard`, or `crash`
4. short description

Example:
```
commit	val_f1	status	description
a1b2c3d	0.912345	keep	baseline RandomForest class_weight=balanced
b2c3d4e	0.921000	keep	XGBoost scale_pos_weight=578
c3d4e5f	0.905000	discard	LogisticRegression underfits
d4e5f6g	0.000000	crash	SMOTE import error
```

## Experiment loop

LOOP FOREVER:

1. Check current git state (branch/commit).
2. Pick an idea, modify `model.py`.
3. `git commit`
4. `uv run model.py > run.log 2>&1`
5. `grep "^val_f1:" run.log`
6. If empty → crash. Run `tail -n 50 run.log`, attempt fix or skip.
7. Log to `results.tsv`.
8. If improved (higher val_f1): keep commit, advance.
9. If not improved: `git reset --hard HEAD~1`, discard.

**Run time**: most runs finish in 5-60 seconds. Keep individual runs under 5 minutes.
If a run exceeds 5 minutes, kill it and treat as failure.

**Idea space** (not exhaustive, roughly ordered by expected impact):
- Switch algorithm to LightGBM or XGBoost with `scale_pos_weight` = count(0)/count(1) ≈ 578
- Tune decision threshold on val (predict_proba, find threshold maximizing macro F1)
- HistGradientBoostingClassifier with `class_weight='balanced'`
- Feature engineering: log(Amount+1), time-of-day cyclical features, interaction terms
- Stacking: blend RF + XGBoost predictions as meta-features for a logistic regression
- ExtraTreesClassifier — often faster, sometimes better on anomaly-like problems
- Tune n_estimators, max_depth, min_child_weight, subsample, colsample_bytree
- Feature selection: drop low-importance features to reduce noise
- Calibration: CalibratedClassifierCV to improve probability estimates before thresholding

**Key insight**: with ~0.17% fraud, the naive "all-zeros" classifier gets macro F1 ≈ 0.50.
Getting above ~0.90 requires the model to actually find the fraud cases. The hardest part is
recall on the minority class — watch the per-class F1 breakdown printed by sklearn's
classification_report if you want to debug.

**NEVER STOP**: Once the loop begins, do NOT ask if you should continue. Run indefinitely
until manually interrupted.
