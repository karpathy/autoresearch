# autotrader

This is an experiment to have the LLM do its own trading research by training ML models that predict forward returns.

## Philosophy

This project is about **training ML models that learn from data and generalize to unseen market conditions**. The agent in `train.py` defines a model, engineers features, trains on historical data, and produces predictions. The evaluation infrastructure in `prepare.py` tests those predictions across multiple time periods — including periods the model has never seen — and returns a single composite score.

Your job is to build a predictor that works not just on training data, but on genuinely out-of-sample data. Overfitting to the training period is the primary failure mode. A model with a moderate train-period Sharpe that generalizes is worth far more than a model with an amazing train-period Sharpe that fails on new data.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, backtesting engine, metric computation. Do not modify.
   - `train.py` — the file you modify. Model architecture, feature engineering, training loop.
4. **Verify data exists**: Check that `~/.cache/autotrader/` contains the cached parquet file. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. This file is gitignored — it lives only on disk and is never committed. Git operations (commit, reset) will not affect it.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine. The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup and evaluation overhead). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, feature engineering, loss function, optimizer, hyperparameters, training loop, regularization, data preprocessing, normalization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` function in `prepare.py` is the ground truth.
- Access validation or holdout data. Only `load_train_data()` is available. The evaluation function handles all other splits internally.

## How evaluation works

The evaluation is a **black box**. You call `evaluate_model(predict_on_data, n_params)` and it:

1. Passes the full dataset through your `predict_on_data` function
2. Tests predictions across **multiple time periods** (including periods your model has never seen)
3. Returns a single composite score and aggregate metrics

**You do NOT know:**
- How many evaluation periods there are
- Which period is your weakest
- Whether you're failing on training data or out-of-sample data

**What you DO see:**
- `score` — the single number to optimize (higher is better)
- `sharpe_min` — the minimum Sharpe ratio across evaluation periods
- `max_drawdown` — the worst drawdown across any period
- `total_trades` — trades across all periods combined
- `consistency` — how many subperiods are profitable (e.g. "5/7")

The score is based on the **minimum Sharpe ratio across all evaluation periods**. This means your score is capped by your worst period. A model that scores Sharpe 3.0 on training data but Sharpe -0.5 on unseen data will get a negative score. The only way to achieve a high score is to build a model that genuinely generalizes.

## Antipatterns — hard rules

These approaches are explicitly prohibited:

1. **No brute-force strategy search.** Do not replace the trainable model with hardcoded signal functions and search over parameter combinations.

2. **No evaluation loops.** Each run of `train.py` should call `evaluate_model()` exactly **once**. Do not call it multiple times to test different configurations within a single run.

3. **No placeholder models.** The model class must contain the actual prediction logic with learned parameters.

4. **No signal function factories.** Do not loop over thousands of parameter combinations. The experiment loop across git commits provides exploration.

5. **No random seed optimization.** The random seed must remain fixed at `42`. Changing the seed and keeping the best-scoring result is cherry-picking a lucky random draw — it doesn't improve the model and won't generalize. If you want stochasticity for regularization, use `subsample`, `max_features`, or dropout.

6. **No fine-grained parameter sweeps.** Do not run 3+ consecutive experiments that only vary a single numeric parameter by small increments. Bracketing a range with 1-2 values is fine; systematic grid search is not. Each experiment should test a meaningfully different idea.

The spirit of these rules: the 2-minute training budget should be spent **training a model** (fitting weights to data), not evaluating pre-built strategies or exploiting evaluation variance.

**Enforcement:** Score improvements are audited by the `experiment-auditor` subagent. If it detects gaming, the experiment is discarded regardless of score.

## Permitted output processing

These are standard architectural choices, not domain-knowledge injection:

- **Output smoothing.** A rolling average or EMA on predictions is permitted. This reduces noise from discontinuous model outputs (e.g. tree-based models). A 24-48h window is reasonable.
- **Target normalization and denormalization.** Training on transformed targets (e.g. vol-normalized returns) and inverse-transforming at prediction time is permitted.
- **Learned intercepts and biases.** Model parameters learned during training are fine.

These are **still prohibited:**

- **Hardcoded directional biases.** No `preds += constant`. The model should learn any directional bias through asymmetric loss functions or target engineering.
- **Hardcoded regime filters.** No "go flat when X" or "never go short" rules. The model should learn risk management from features and training signal.

## Key insight from previous experiments

**Train score is inversely correlated with out-of-sample performance for this problem.** In previous rounds, every increase in training-period score made generalization worse. Models that scored 4.0+ on train lost 40%+ on holdout. The models that actually made money on unseen data had train scores around 0.1-0.2.

This happens because BTC market regimes shift significantly across years. A model that memorizes patterns from 2018-2022 actively hurts itself on 2023-2025.

**Implications for your research:**
- Don't chase train-period metrics. A lower train score with better generalization is a better model.
- Prefer techniques that regularize and simplify: fewer features, shallower trees, more regularization.
- Time-decay sample weighting (weight recent training data more heavily) has shown promise for generalization.
- Output smoothing significantly reduces trade churn and improves out-of-sample performance.

## Research directions

Explore these roughly in order of priority:

**Generalization techniques** (highest priority):
- Output smoothing (rolling mean or EMA on predictions, 24-48h window)
- Time-decay sample weighting (weight recent training data more heavily)
- Aggressive regularization (high min_samples_leaf, low tree depth)
- Cross-validation within the training period to detect overfitting

**Loss functions and target engineering:**
- **Asymmetric loss**: penalize wrong-direction predictions more heavily (2-3x weight on missed upside). This lets the model learn directional bias from data.
- **Vol-normalized targets**: train on `forward_return / vol_168h`, denormalize at prediction time. Makes the model naturally cautious in high-vol regimes.
- **Excess returns**: train on `forward_return - rolling_mean_return`. The rolling mean captures drift; the model learns deviations.
- Huber loss for robustness to outliers

**Feature engineering:**
- Vol-normalized returns (returns divided by rolling volatility)
- Volatility measures and vol regime indicators
- Volume profiles
- Keep features minimal — more features = more overfitting

**Architectures:**
- Gradient boosted trees with heavy regularization (current baseline)
- **LSTM / GRU** — produces temporally coherent predictions naturally, solving the trade-churn problem architecturally. Start small (32-64 units, single layer, aggressive dropout).
- Small feedforward networks with dropout
- Ensembles of diverse simple models

## Simplicity criterion

All else being equal, simpler is better. The parameter count penalty is built into the score.

**Simplicity hierarchy** (prefer, in order):
- Fewer features over more features
- A shallower model over a deeper one
- Fewer hyperparameters over more
- Standard techniques over exotic ones

A small improvement that adds complexity is not worth it. Removing something and getting equal or better results is a great outcome.

## Stuck protocol

**If you've run 5 consecutive experiments without improving the best score, you MUST invoke the `experiment-coach` subagent.** Tell it your current best score and recent experiment history. It will diagnose why you're stuck and prescribe a specific next direction. Follow its prescription for your next experiment.

The coach will identify your binding constraint (drawdown, trade count, Sharpe, consistency) and give you a concrete change to implement — not vague advice. Trust its diagnosis; it has full context on the scoring system and experiment history.

After following the coach's prescription, if you're still stuck after another 5 experiments, invoke it again. Each invocation should result in a fundamentally different approach:

- Change the model architecture category (e.g. GBR → LSTM → feedforward → ensemble)
- Change the feature representation (e.g. raw returns → vol-normalized → rolling z-scores)
- Change the loss function or target engineering
- Change the training procedure (e.g. add time-decay weighting, change regularization approach)

Tweaking hyperparameters on the same approach does NOT count. Those are fine-tuning for when something is working.

## Output format

Once the script finishes it prints a summary like this:

```
---
score:            0.4200
sharpe_min:       0.5800
max_drawdown:     -15.2%
total_trades:     145
consistency:      5/7
n_params:         3206
training_seconds: 32.1
total_seconds:    45.3
```

You can extract the key metric from the log file:

```
grep "^score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 8 columns:

```
commit	score	sharpe_min	max_dd	total_trades	n_params	status	description
```

1. git commit hash (short, 7 chars)
2. score (e.g. 0.4200) — use 0.0000 for crashes
3. sharpe_min (e.g. 0.5800) — use 0.0000 for crashes
4. max drawdown as percentage (e.g. -15.2) — use 0.0 for crashes
5. total trades (e.g. 145) — use 0 for crashes
6. number of model parameters (e.g. 3206) — use 0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	score	sharpe_min	max_dd	total_trades	n_params	status	description
a1b2c3d	0.4200	0.5800	-15.2	145	3206	keep	baseline GBR
b2c3d4e	0.5100	0.6200	-12.8	178	3206	keep	add output smoothing
c3d4e5f	0.0000	0.0000	0.0	0	0	crash	LSTM (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Decide on an experimental idea. Consult the research directions and stuck protocol.
3. Modify `train.py` with the experiment. The model must be trained during the 2-minute budget.
4. `git add train.py && git commit -m "exp: <short description>"`
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^score:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace. If fixable, fix and re-run. Otherwise give up on this idea.
8. Append the results to `results.tsv` (gitignored — no need to commit it)
9. **If score improved**, invoke the `experiment-auditor` subagent to check for gaming. Tell it the experiment description and results. If the auditor returns FAIL, treat the experiment as discarded and `git reset --hard HEAD~1`. If PASS, keep the commit.
10. If score is equal or worse, `git reset --hard HEAD~1` to discard the experiment
11. Check the stuck protocol: if this is the 5th consecutive non-improvement, invoke the `experiment-coach` subagent for diagnosis and direction.

**Timeout**: Each experiment should take ~3 minutes total (2 minutes training + evaluation overhead). If a run exceeds 6 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use judgment: fix simple bugs and re-run, or skip and log "crash."

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human expects you to work indefinitely until manually stopped. If you run out of ideas, consult the research directions — there are dozens of combinations to try.

**The first run**: Always establish the baseline first by running the training script as is. Log it and move on to improvements.