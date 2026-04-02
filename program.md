# autoresearch — Time Series Forecasting & Anomaly Detection

This is an autonomous research experiment for time series forecasting and anomaly detection. The agent iterates on model architecture, hyperparameters, and training strategies to minimize forecasting error and maximize anomaly detection performance.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr2-forecast`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, column discovery, scaling, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-ts/` contains `splits.pt`, `metadata.json`, `target_scaler.pkl`. If not, tell the human to run `python prepare.py --data path/to/data.csv`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Problem Context

**Domain**: Energy/utility consumption forecasting with anomaly detection.
**Data**: Hourly time series with weather features, temporal features, and a consumption target. Columns may vary between datasets — `prepare.py` auto-discovers them.
**Goal**: Minimize `val_scaled_mae` (primary) and maximize `anomaly_f1` (secondary). The `combined_score` = `val_scaled_mae - 0.1 * anomaly_f1` (lower is better) is the single metric to optimize.

## Experimentation

Each experiment runs on whatever compute is available (GPU, MPS, or CPU). The training script runs for a **fixed time budget of 5 minutes**. You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Model architecture (LSTM, GRU, Transformer, CNN-LSTM, TCN, hybrid, etc.)
  - Optimizer (Adam, AdamW, SGD, Muon, custom schedulers)
  - Hyperparameters (hidden size, layers, dropout, learning rate, batch size)
  - Loss functions (MSE, Huber, quantile loss, custom combinations)
  - Feature engineering within the model (lag features, rolling stats, embeddings)
  - Anomaly detection approach (reconstruction error, classification head, autoencoders)
  - Attention mechanisms, skip connections, normalization strategies
  - Ensemble methods within the single file

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and constants.
- Install new packages or add dependencies beyond what's already available (torch, numpy, pandas, sklearn).
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.
- Change `TIME_BUDGET` or evaluation constants.

**The goal is simple: get the lowest `combined_score`.**

The combined score balances forecasting accuracy (val_scaled_mae, lower=better) with anomaly detection (anomaly_f1, higher=better):
```
combined_score = val_scaled_mae - 0.1 * anomaly_f1
```
Lower is better. A model that forecasts well AND detects anomalies will score best.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but don't blow it up.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

**The first run**: Your very first run should always establish the baseline — run `train.py` as-is.

## Research Directions to Explore

Here are productive research directions, roughly ordered by expected impact:

### High Impact
1. **Architecture**: Try GRU (simpler, often competitive), Temporal Convolutional Networks (TCN), or Transformer encoder with positional encoding
2. **Bidirectional encoding**: Enable `BIDIRECTIONAL = True` or try bidirectional with attention
3. **Attention mechanisms**: Self-attention over LSTM outputs, multi-head attention
4. **Learning rate**: Try 3e-4, 5e-4, 2e-3 — this is often the single most impactful hyperparameter
5. **Hidden size scaling**: Try 128, 256 — larger capacity may help

### Medium Impact
6. **Loss function**: Switch from MSE to Huber loss (robust to outliers in consumption data)
7. **Feature interaction layers**: Add a small MLP or 1D-Conv before LSTM to capture feature interactions
8. **Residual connections**: Add skip connections around LSTM layers
9. **Layer normalization**: Add LayerNorm between LSTM layers
10. **Anomaly strategy**: Try reconstruction-based anomaly detection (autoencoder) instead of classification head

### Exploratory
11. **Multi-scale**: Parallel LSTMs at different sequence granularities
12. **Dilated convolutions**: TCN with exponentially increasing dilation
13. **Mixture of experts**: Route different time patterns to different sub-networks
14. **Contrastive anomaly**: Learn normal patterns, flag deviations
15. **Ensemble**: Train multiple models, average predictions

## Output format

Once the script finishes it prints a summary like this:

```
---
val_mae:            0.009002
val_scaled_mae:     0.045123
val_rmse:           0.024328
val_r2:             0.3957
anomaly_f1:         0.1200
anomaly_precision:  0.1500
anomaly_recall:     0.1000
combined_score:     0.033123
test_mae:           0.009500
test_r2:            0.3800
training_seconds:   300.1
total_seconds:      310.5
peak_vram_mb:       1024.0
num_steps:          4500
num_params:         45060
hidden_size:        64
num_layers:         2
batch_size:         64
learning_rate:      0.001
```

You can extract the key metric from the log:
```
grep "^combined_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	combined_score	val_scaled_mae	anomaly_f1	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. combined_score (e.g. 0.033123) — use 9.999999 for crashes
3. val_scaled_mae (e.g. 0.045123) — use 9.999999 for crashes
4. anomaly_f1 (e.g. 0.1200) — use 0.0000 for crashes
5. peak memory in MB (e.g. 1024.0) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	combined_score	val_scaled_mae	anomaly_f1	memory_mb	status	description
a1b2c3d	0.033123	0.045123	0.1200	1024.0	keep	baseline LSTM
b2c3d4e	0.028500	0.040500	0.1200	1100.0	keep	increase hidden to 128
c3d4e5f	0.035000	0.047000	0.1200	1024.0	discard	switch to GRU (worse)
d4e5f6g	9.999999	9.999999	0.0000	0.0	crash	transformer OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr2-forecast`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^combined_score:\|^val_scaled_mae:\|^anomaly_f1:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If `combined_score` improved (lower), you "advance" the branch, keeping the git commit
9. If `combined_score` is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: fix typos and re-run, or skip and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the research directions above, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## Key constraints to remember

- The data columns can change between datasets. Your model receives `n_features` from metadata — do not hardcode feature count.
- The model's `forward()` must return `{"forecast": tensor(B,), "anomaly": tensor(B,)}`. This interface is fixed.
- Anomaly labels are generated by z-score thresholding. They may be imbalanced (rare anomalies). Consider class weighting or focal loss.
- The time budget is wall clock — compilation time counts after warmup steps, so avoid overly complex models that take too long to compile.
