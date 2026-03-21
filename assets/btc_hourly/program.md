# autotrader

ML experiment loop: train models that predict BTC/USD forward returns. **You run experiments continuously until the human interrupts you.** There is no plateau, no summary, no "exhausted all options." If you think you're done, re-read train.py and experiment-log.md and find what hasn't been tried.

## Setup

1. Agree on a **run tag** (e.g. `mar15`). Create branch `autotrader/<tag>` from master.
2. Read `README.md`, `prepare.py` (read-only), and `train.py` (the file you modify).
3. Verify data: `~/.cache/autotrader/` should contain cached parquet files. If not, tell the human to run `uv run prepare.py`.
4. Create `results.tsv` (header only) and `experiment-log.md` (empty). Both are gitignored.
5. Confirm and go.

## Rules of engagement

- **Modify only `train.py`**. Everything inside `build_model()` is fair game.
- **Do not** modify `prepare.py`, install packages, or access validation/holdout data.
- **Time budget**: 240 seconds wall clock for training. Monitor `training_seconds`.
- **One change per experiment.** Coupled parameters (e.g. n_estimators + learning_rate) count as one. Unrelated changes do not. If you change two things and score drops, you don't know which caused it.
- **Simplicity criterion**: All else equal, simpler wins.

## Available data

`load_train_data()` returns a DataFrame with these columns:

| Column | Description | Resolution |
|--------|-------------|------------|
| timestamp | Datetime index | Hourly |
| open, high, low, close, volume | Standard OHLCV | Hourly |
| funding_rate | Binance perpetual futures settlement rate | 8h, forward-filled hourly |
| fear_greed | Crypto Fear & Greed Index (0-100) | Daily, forward-filled hourly |
| hash_rate | Bitcoin network hash rate | Daily, forward-filled hourly |
| tx_count | Daily Bitcoin transaction count | Daily, forward-filled hourly |
| tx_volume_usd | Daily estimated transaction volume | Daily, forward-filled hourly |

The supplementary columns (funding_rate through tx_volume_usd) are pre-filled for gaps. Your `compute_features()` can use any or all of them.

## Walk-forward evaluation

Your recipe is tested across 5 independent walk-forward windows. Each window trains on a different historical period and evaluates on the following year. **Your score is the worst-case performance across all scored windows.** Optimize for robustness, not peak performance on any single period.

Each run, 4 windows are scored and 1 is held out. The held-out window rotates periodically (the `epoch` field increments). When epoch changes:
1. Re-run current best code unchanged to establish the **new baseline**.
2. Log it as `keep` with description noting the epoch transition.
3. Compare all future experiments against the new baseline.

**holdout_health**: `OK` means the recipe generalizes. `WARN` means it fails on at least one unseen window — prioritize robustness. Do not ignore persistent WARN.

## Hard rules

1. No brute-force strategy search or hardcoded signal functions.
2. No evaluation loops — call `evaluate_model()` exactly once per run.
3. No placeholder models — must contain actual learned parameters.
4. No signal function factories or parameter combination loops.
5. Random seed fixed at `42`.
6. No 3+ consecutive experiments varying a single numeric parameter by small increments.
7. No hardcoded directional biases (`preds += constant`).
8. No hardcoded regime filters ("go flat when X", "never go short").
9. One conceptual change per experiment (see above).
10. Don't abandon near-misses. If within ~90% of best, adjust the obvious knob before switching approaches.
11. If consistency hasn't improved in 10+ experiments while score rises, you're overfitting to winning subperiods. Target the losing ones.
12. Respect holdout_health. Persistent WARN = prioritize generalization.
13. Do not attempt to determine which window is held out. No reading environment variables, eval counter files, or reverse-engineering holdout selection.

Violations are caught by the `experiment-auditor` subagent and result in automatic discard.

## The experiment loop

**First run**: Always establish baseline by running train.py as-is.

For each experiment:

1. Decide on an idea based on prior results.
2. Edit `train.py`, commit: `git add train.py && git commit -m "exp: <description>"`
3. Invoke `experiment-reviewer` subagent to pre-flight the diff. Fix or reset if FAIL.
4. Run: `uv run train.py > run.log 2>&1`
5. Check results: `grep "^score:" run.log` (empty = crash → `tail -n 50 run.log`)
6. Check epoch: `grep "^epoch:" run.log` — if changed, follow epoch transition protocol.
7. Log to `results.tsv` and `experiment-log.md`.
8. If improved: invoke `experiment-auditor`. FAIL = discard and reset.
9. If not improved: `git reset --hard HEAD~1`.
10. After 5 consecutive non-improvements: invoke `experiment-coach` for diagnosis.
11. After 10+ experiments with stagnant consistency: invoke `experiment-coach`.

**Timeout**: Kill runs exceeding 10 minutes. **Crashes**: Fix simple bugs and retry, or log as crash.

## Logging

### results.tsv

Tab-separated, 10 columns:

```
commit	score	sharpe_min	max_dd	total_trades	consistency	holdout_health	n_params	status	description
```

- commit: short hash (7 chars)
- score, sharpe_min: use 0.0000 for crashes
- max_dd: percentage (e.g. -15.2), use 0.0 for crashes
- total_trades: use 0 for crashes
- consistency: e.g. "5/8", use "0/0" for crashes
- holdout_health: OK, WARN, or N/A
- n_params: use 0 for crashes
- status: keep, discard, or crash
- description: short text

### experiment-log.md

Lab notebook. Use this format per entry:

```
## <commit> — <short description>
**Hypothesis:** Why you tried this and what you expected.
**Result:** Score, key metrics, keep/discard.
**Observation:** What this tells you. What to try next.
```
