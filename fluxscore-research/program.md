# FluxScore autoresearch

Autonomous ML experiment loop for FluxScore — Innflux's credit scoring model.
This is the research companion to the karpathy/autoresearch pattern, adapted for
scikit-learn on synthetic borrower data (no GPU required, no PII involved).

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `fluxscore/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b fluxscore/<tag>` from current `feature/fluxscore-research`.
3. **Read the in-scope files**:
   - `fluxscore-research/README.md` — context and differences from original autoresearch.
   - `fluxscore-research/prepare.py` — fixed constants, synthetic data schema, evaluation harness. **Do not modify.**
   - `fluxscore-research/experiment.py` — the file you modify. Everything below `# AGENT ZONE` is yours.
4. **Verify data exists**: Check that `fluxscore-research/train.parquet` and `holdout.parquet` exist. If not, run `python fluxscore-research/prepare.py`.
5. **Initialize results.tsv**: Create `fluxscore-research/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**.

Once confirmed, kick off the experimentation loop.

## Experimentation

Each experiment runs with a **fixed time budget of 2 minutes** (wall clock, enforced by `signal.alarm(120)` in the fixed harness — you cannot override this).

**What you CAN do:**
- Modify `experiment.py` — but only the code below the `# AGENT ZONE` marker.
- Feature engineering on `_X_train` / `_y_train`.
- Try any model available in `pyproject.toml`: logistic regression, LightGBM, XGBoost, random forest, ensembles, shallow MLP.
- Add cross-validation folds, hyperparameter search (keep it fast — you only have 2 minutes).
- Interaction terms, polynomial features, target encoding, PCA — all fine.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It defines the feature schema and generates the data.
- Modify the fixed harness section of `experiment.py` (above `# AGENT ZONE`).
- Modify `_evaluate()` or `_print_summary()`.
- Add imports above the AGENT ZONE marker.
- Access any data other than `_X_train`, `_y_train`, `_X_holdout`, `_y_holdout`.

**The goal is simple: maximize `auc` on the holdout set.**

**CRITICAL — AUC higher is better.** This is the OPPOSITE of val_bpb in karpathy's autoresearch (where lower is better). AUC = 0.5 is random. AUC = 1.0 is perfect. Target AUC > 0.85. The baseline logistic regression typically achieves ~0.78.

**Promotion gate**: When a model beats baseline by > 0.02 AUC on synthetic holdout, flag it for manual validation against real data in the TEE. You cannot access real data — flag and move on.

**Simplicity criterion**: Same as karpathy — all else equal, simpler is better. A 0.001 AUC improvement that adds 50 lines of complexity is not worth it. An improvement of ~0 from simpler code? Keep.

**Class imbalance**: The dataset uses `class_weight="balanced"` in the baseline. This is intentional — defaults are rare (~8% overall). Keep this or handle imbalance explicitly (SMOTE, cost-sensitive learning, etc.).

**The first run**: Your very first run should establish the baseline — run `experiment.py` as-is without modification.

## Output format

The script prints a summary like this:

```
---
auc:     0.782341
gini:    0.564682
brier:   0.089123
elapsed: 4.2s
status:  ok
```

Extract the key metric:

```bash
grep "^auc:" run.log
```

Crashes output:

```
auc: 0.000000
gini: 0.000000
brier: 1.000000
status: crash (ModelError: ...)
```

Timeouts output the same format with `status: timeout`.

## Logging results

Log to `fluxscore-research/results.tsv` (tab-separated, NOT comma-separated).

Header and columns:

```
commit	auc	status	description
```

1. git commit hash (short, 7 chars)
2. auc achieved (6 decimal places, e.g. `0.782341`) — use `0.000000` for crashes
3. status: `keep`, `discard`, or `crash`
4. short description of what this experiment tried

Example:

```
commit	auc	status	description
a1b2c3d	0.782341	keep	baseline logistic regression
b2c3d4e	0.791200	keep	add interaction: velocity × counterparty_diversity
c3d4e5f	0.779000	discard	switch to raw features (no scaling)
d4e5f6g	0.000000	crash	LightGBM bad hyperparams (num_leaves=99999)
```

Do NOT commit `results.tsv` — leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `fluxscore/mar31`).

LOOP FOREVER:

1. Check current git state.
2. Modify `experiment.py` below the `# AGENT ZONE` marker with an experimental idea.
3. `git commit`
4. Run: `python fluxscore-research/experiment.py > run.log 2>&1`
5. Extract metric: `grep "^auc:" run.log`
6. If empty, grep failed — run `tail -n 30 run.log` to inspect. Attempt a fix if trivial. Otherwise log as crash and move on.
7. Record in `results.tsv`.
8. If AUC improved (even by 0.0001), **keep** — advance the branch. Same threshold as karpathy.
9. If AUC equal or worse, `git reset --hard HEAD~1`.

**Promotion**: If any experiment beats baseline by > 0.02 AUC, add a note to `results.tsv` description: `[PROMOTE: validate in TEE]`. Flag it to the human at next check-in.

**Timeout**: If a run exceeds 2 minutes (130s wall clock), the harness kills it automatically. Treat as crash, discard, move on.

**NEVER STOP**: Once the loop has begun, do not pause to ask the human if you should continue. The human may be asleep. You are autonomous. If you run out of ideas, re-read this program.md, look at which features have the highest coefficient magnitude in the baseline logistic regression, think about interaction terms between the top predictors, try tree-based models, try ensembles. The loop runs until manually interrupted.

## Research objectives (ordered by priority)

1. **Feature importance ranking**: Which of the 18 features carry the most predictive signal? Fit the baseline, read `model.named_steps['clf'].coef_[0]` — log the top 5 by absolute magnitude.

2. **Non-linear interactions**: The most promising: `txn_volatility × on_time_repayment_rate`, `avg_monthly_txn_count × counterparty_diversity`, `balance_to_request_ratio × loan_type`. Try these as explicit features.

3. **Alternative models**: LightGBM is likely the biggest single jump. Try `lgbm.LGBMClassifier(n_estimators=100, class_weight='balanced')`. XGBoost second. Stacking third.

4. **Minimum feature set**: Can we hit 95% of peak AUC with ≤ 10 features? Start from the full model, drop lowest-importance features one by one. A compact model is easier to audit in TEE.

5. **Class imbalance**: Try SMOTE oversampling (`imbalanced-learn`) vs cost-sensitive weighting vs threshold calibration. Does it move AUC or just calibration?

6. **Calibration**: After finding the best model, apply `CalibratedClassifierCV(cv=3)`. Check if Brier score improves without hurting AUC — good calibration matters for the PD→score conversion.

## Constraints (permanent, non-negotiable)

- **No real borrower data.** Synthetic data only. Real data never leaves the TEE. If you somehow get access to real data, stop immediately.
- **No on-chain deployment.** That happens in the TEE, manually, after human review.
- **No GPU required.** scikit-learn on CPU is intentional.
- **No PyTorch.** Out of scope for this research loop.

## Living document

Update this file when:
- A new research objective is added.
- The baseline model is promoted to production (update the "Baseline" reference AUC).
- A new version of the 18-feature schema is deployed.

Current baseline AUC reference: ~0.72 (logistic regression, all 18 features, balanced class weight).
Synthetic data default rate: ~20%. Promotion gate: +0.02 AUC over baseline (i.e., ≥ 0.74).
