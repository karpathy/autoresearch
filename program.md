# autotrader

This is an experiment to have the LLM do its own trading research by training ML models that predict forward returns.

## Philosophy

This project is about **training ML models that learn from data**. The agent in `train.py` defines a model, engineers features, trains on historical data, and produces predictions. The backtesting infrastructure in `prepare.py` converts those predictions into trading signals and scores them.

Your job is to build a better predictor — not to hand-craft trading rules or brute-force search over parameter grids. If your `train.py` doesn't contain a model whose weights are updated during training, you've drifted off course. The model must learn patterns from data through gradient descent (or equivalent fitting procedure for tree-based models). Predictions must flow through learned parameters.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, backtesting engine, metric computation. Do not modify.
   - `train.py` — the file you modify. Model architecture, feature engineering, training loop.
4. **Verify data exists**: Check that `~/.cache/autotrader/` contains the cached parquet file. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine. The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup and evaluation overhead). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, feature engineering, loss function, optimizer, hyperparameters, training loop, regularization, data preprocessing, normalization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, temporal splits, backtesting engine, and metric computation.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` function in `prepare.py` is the ground truth metric.
- Access or evaluate on holdout data. The holdout period (2024-07 to 2025-12) is reserved for human evaluation only.

## Antipatterns — hard rules

These approaches are explicitly prohibited. If you find yourself doing any of these, stop and change direction:

1. **No brute-force strategy search.** Do not replace the trainable model with hardcoded signal functions (e.g. numpy-computed mean-reversion or momentum indicators) and search over parameter combinations. That is curve-fitting, not learning.

2. **No evaluation loops.** Each run of `train.py` should call `evaluate_model()` at most **twice** — once on train split, once on val split. If you want to do hyperparameter selection or model selection within a run, use your own loss metric (e.g. MSE on a held-aside portion of training data), NOT the backtest evaluator. The official evaluator is for final scoring only.

3. **No placeholder models.** The model class must contain the actual prediction logic. If your predictions are computed entirely outside the model (e.g. raw numpy operations with hardcoded coefficients), and the model exists only for API compatibility, something is wrong.

4. **No signal function factories.** Do not write functions like `make_momentum_fn(scale)` or `make_mean_reversion_fn(lookback, threshold)` and loop over thousands of parameter combinations. The experiment loop across git commits is what provides exploration — not a search loop within a single run.

The spirit of these rules: the 2-minute training budget should be spent **training a model** (fitting weights to data), not evaluating thousands of pre-built strategies.

## The goal

**Get the highest score.** The score is a composite metric: Sharpe ratio penalized for model complexity, insufficient trades, excessive drawdown, and inconsistent performance across time periods. Higher is better.

**Score components:**
- Base: annualized Sharpe ratio of the backtested strategy
- Hard penalties (score → 0): fewer than 30 trades, max drawdown > 25%, unprofitable in too many subperiods
- Soft penalty: parameter count (simpler models score higher, all else equal)

**Validation pass/fail:** After training, the model is also evaluated on a held-out validation period. You see only `val_pass: true` or `val_pass: false`. You do NOT see the actual validation metrics. An experiment should only be kept if BOTH score improved AND val_pass is true.

**Baseline context**: The naive baseline scores 0.0 due to negative Sharpe and excessive drawdown (hard penalties zero out the score). This is expected and normal. Your first objective is achieving a positive score — this means getting positive Sharpe AND keeping max drawdown under 25% AND having at least 30 trades AND being profitable in at least 2 of 3 training subperiods. Improving prediction quality through better features and model architecture is the path forward.

## Simplicity criterion

All else being equal, simpler is better. The parameter count penalty is built into the score, but beyond that:

**Simplicity hierarchy** (prefer, in order):
- Fewer features over more features
- A shallower model over a deeper one
- Fewer hyperparameters over more
- Standard techniques over exotic ones

A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

A linear model with 5 good features that scores 0.4 is more valuable than a 10-layer network with 50 features that scores 0.45.

## Research directions

Explore these roughly in order of complexity. Don't skip ahead to exotic approaches before exhausting simpler ones:

**Feature engineering** (start here):
- Different return lookback windows (1h, 4h, 12h, 24h, 72h, 168h)
- Volatility measures (rolling std, ATR, high-low range)
- Volume profiles and volume-weighted features
- Rolling z-scores of returns (mean-reversion signals)
- Price relative to moving averages (trend signals)

**Loss functions:**
- MSE on forward returns (baseline)
- Huber loss for robustness to outliers
- Asymmetric loss that penalizes wrong-direction predictions more heavily
- Directional accuracy loss (predict sign correctly)
- Custom Sharpe-aware or downside-risk-aware objectives

**Architectures** (try in order):
- Linear regression / ridge regression (surprisingly strong baseline)
- Small feedforward networks (2-3 layers, 16-64 units)
- Gradient boosted trees (scikit-learn GradientBoostingRegressor)
- LSTM / GRU for temporal patterns
- 1D-CNN for local pattern detection
- Ensembles combining diverse model types

**Training techniques:**
- Learning rate scheduling (cosine, step decay, warmup)
- Early stopping on a held-aside portion of training data
- Data augmentation (noise injection)
- Feature normalization (z-score, robust scaling, winsorization)
- Walk-forward training within the train window

**Regularization:**
- Dropout
- Weight decay / L2 regularization
- Batch normalization
- Feature selection (train with subsets, keep what helps)

## Stuck protocol

**If you've run 5 consecutive experiments without improving the best score, you MUST change your fundamental approach.** "Fundamental" means one of:

- Change the model architecture category (e.g. feedforward → LSTM → gradient boosted trees → ensemble)
- Change the feature representation (e.g. raw returns → rolling z-scores → learned embeddings)
- Change the loss function (e.g. MSE → Huber → directional loss)
- Change the training procedure (e.g. full-batch → mini-batch, add early stopping, add curriculum learning)

Varying scale coefficients, threshold values, layer widths, or learning rates on the same fundamental approach does NOT count as changing direction. Those are fine-tuning — do them when something is working, not when you're stuck.

When changing direction, commit a clean version that represents the new approach, even if the first attempt scores poorly. The point is to explore the search space broadly before optimizing locally.

## Output format

Once the script finishes it prints a summary like this:

```
---
score:            0.4200
sharpe:           0.5800
max_drawdown:     -15.2%
n_trades:         127
total_return:     23.4%
n_params:         1249
val_pass:         true
training_seconds: 118.3
total_seconds:    134.7
```

You can extract the key metrics from the log file:

```
grep "^score:\|^val_pass:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 9 columns:

```
commit	score	sharpe	max_dd	n_trades	n_params	val_pass	status	description
```

1. git commit hash (short, 7 chars)
2. score achieved (e.g. 0.4200) — use 0.0000 for crashes
3. sharpe ratio (e.g. 0.5800) — use 0.0000 for crashes
4. max drawdown as percentage (e.g. -15.2) — use 0.0 for crashes
5. number of trades (e.g. 127) — use 0 for crashes
6. number of model parameters (e.g. 1249) — use 0 for crashes
7. val_pass: true, false, or N/A for crashes
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	score	sharpe	max_dd	n_trades	n_params	val_pass	status	description
a1b2c3d	0.4200	0.5800	-15.2	127	1249	true	keep	baseline
b2c3d4e	0.5100	0.6200	-12.8	145	1249	true	keep	add momentum features
c3d4e5f	0.3800	0.5500	-18.1	98	1249	false	discard	switch to Huber loss
d4e5f6g	0.0000	0.0000	0.0	0	0	N/A	crash	LSTM model (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Decide on an experimental idea. Consult the research directions above and the stuck protocol.
3. Modify `train.py` with the experiment. The model must be trained (weights updated from data) during the 2-minute budget.
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^score:\|^val_pass:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
8. Record the results in the tsv
9. If score improved (higher) AND val_pass is true, you "advance" the branch, keeping the git commit
10. If score is equal or worse, or val_pass is false, you git reset back to where you started
11. Check the stuck protocol: if this is the 5th consecutive non-improvement, change your fundamental approach before the next experiment.

**Timeout**: Each experiment should take ~3 minutes total (2 minutes training + evaluation overhead). If a run exceeds 6 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, consult the research directions section above — there are dozens of combinations to try. The loop runs until the human interrupts you, period.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is. The baseline will likely score 0.0 — that's expected and fine. Log it and move on to improvements.