# autotrader

This is an experiment to have the LLM do its own trading research.

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

**The goal is simple: get the highest score.** The score is a composite metric: Sharpe ratio penalized for model complexity, insufficient trades, excessive drawdown, and inconsistent performance across time periods. Higher is better.

**Score components:**
- Base: annualized Sharpe ratio of the backtested strategy
- Hard penalties (score → 0): fewer than 30 trades, max drawdown > 25%, unprofitable in too many subperiods
- Soft penalty: parameter count (simpler models score higher, all else equal)

**Validation pass/fail:** After training, the model is also evaluated on a held-out validation period. You see only `val_pass: true` or `val_pass: false`. You do NOT see the actual validation metrics. An experiment should only be kept if BOTH score improved AND val_pass is true.

**Simplicity criterion**: All else being equal, simpler is better. The parameter count penalty is built into the score, but beyond that: a small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

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
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^score:\|^val_pass:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv ~~(NOTE: do not commit the results.tsv file, leave it untracked by git)~~
8. If score improved (higher) AND val_pass is true, you "advance" the branch, keeping the git commit
9. If score is equal or worse, or val_pass is false, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~3 minutes total (2 minutes training + evaluation overhead). If a run exceeds 6 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try different architectures (LSTM, GBT, ensemble), different features (more lookback windows, price patterns, volume analysis), different loss functions, different regularization strategies. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~3 minutes then you can run approx 20/hour, for a total of about 160 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
