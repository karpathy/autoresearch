# autoresearch — BTC 15-minute prediction (Kalshi KXBTC15M)

This is an autonomous research loop for discovering profitable BTC price prediction strategies for Kalshi KXBTC15M binary up/down contracts.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — data loading, feature engineering, Kalshi-accurate backtesting engine, evaluation. Do not modify.
   - `strategy.py` — the file you modify. Probability estimation, edge thresholds, entry timing logic.
   - `backtest.py` — entry point that runs strategy against validation data. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains `btc_1m.csv` (real data) or run `uv run prepare.py` to generate synthetic data.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## How Kalshi KXBTC15M works

The contract settles on BTC's price at expiration, calculated as the **1-minute VWAP** (average of ~60 per-second BRTI observations) in the final minute before expiry. Markets open at 15-minute intervals (:00, :15, :30, :45). You can enter at any minute 0-13 within a window (no trades at minute 14+).

The backtest engine models this accurately:
- **Window alignment**: Trades align to real clock boundaries (:00/:15/:30/:45)
- **Settlement**: Mean of close prices in the final minute (not a point-in-time snapshot)
- **Fair market price**: Estimated via binary option pricing (normal CDF based on displacement from window open, volatility, and time remaining)
- **Entry**: Strategy is called at each minute 0-13; multiple trades per window allowed
- **P&L**: Buy at fair market price. Correct = `+(1 - buy_price)`, wrong = `-buy_price`, minus fees

## The prediction task

You are estimating the **probability** that BTC's settlement price will be above the window opening price. The strategy sees the last 60 one-minute OHLCV candles (with pre-computed indicators) plus a `context` dict with window-specific information.

### Strategy API: `on_bar(window, context) → (probability, edge_threshold)`

**`window`** — DataFrame of the last 60 bars with columns:
- Base: `open`, `high`, `low`, `close`, `volume`
- Returns: `returns`
- Volatility: `volatility_20`
- Moving averages: `sma_20`, `sma_50`, `ema_12`, `ema_26`
- RSI: `rsi_14`
- MACD: `macd`, `macd_signal`, `macd_hist`
- Bollinger Bands: `bbands_lower`, `bbands_mid`, `bbands_upper`, `bbands_bandwidth`
- Other: `atr_14`, `volume_sma_20`

**`context`** — dict with Kalshi window info:
- `window_minute`: int (0-13), which minute within the 15-min window you're at
- `window_open_price`: float, BTC price at minute 0 of this window
- `minutes_remaining`: int, minutes until settlement
- `fair_price`: float (0-1), binary option fair value P(up) given current displacement and volatility

**Returns**: `(probability, edge_threshold)`
- `probability`: float 0-1, your model's estimate of P(BTC up at settlement)
- `edge_threshold`: float 0-1, minimum `|probability - fair_price|` required to trigger a trade

The backtest engine trades when your `probability` diverges from `fair_price` by more than `edge_threshold`. If `probability > fair_price + edge_threshold`, it buys YES (bet up). If `probability < fair_price - edge_threshold`, it buys NO (bet down).

**Scoring**: `score = sharpe × accuracy × trade_factor` where:
- `sharpe` = annualized Sharpe ratio of daily P&L
- `accuracy` = fraction of correct direction predictions
- `trade_factor` = `√(min(num_trades / 20, 1.0))` — penalty for too few trades

**Higher score is better.** A score of -999 means the strategy is invalid (too few trades or errors).

## Experimentation

Each experiment runs on CPU. The backtest runs for a **maximum of 120 seconds**. You launch it as: `uv run backtest.py`.

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: probability model, indicator combinations, edge thresholds, entry timing logic, ensemble methods, regime detection, adaptive parameters.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed evaluation, data loading, feature engineering, and backtesting engine.
- Modify `backtest.py`. It is the fixed entry point.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (pandas, numpy, pandas-ta, scipy).
- Modify the evaluation harness or scoring formula.

**The goal is simple: get the highest score.** Since the time budget is fixed, you don't need to worry about execution time. Everything in `strategy.py` is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the backtest as is.

## Ideas to explore

Here are productive directions for improving the strategy:

- **RSI period tuning**: Try RSI(8), RSI(21), or adaptive RSI periods
- **EMA crossover signals**: Short/long EMA crosses, with different period combinations
- **MACD signals**: Zero-line crosses, signal-line crosses, histogram divergence
- **Bollinger Band signals**: Squeeze breakouts, mean reversion from bands
- **Multi-signal ensembles**: Combine 3-5 signals with majority voting
- **Momentum**: Rate of change, price velocity, acceleration
- **Volume confirmation**: Volume spikes, volume-price divergence
- **Adaptive thresholds**: Adjust edge_threshold based on volatility regime
- **Regime detection**: Use volatility or trend strength to switch between strategies
- **Probability calibration**: Use signal strength to produce better-calibrated probabilities
- **Entry timing**: Trade early in window (more uncertainty, bigger edge) vs late (more certainty, smaller edge)
- **Displacement-aware**: Factor in how far price has moved from window open when estimating probability
- **Fair price disagreement**: Look for situations where your probability model systematically disagrees with the fair price model
- **Time-of-day patterns**: Different behavior during different market hours

## Output format

Once the backtest finishes it prints a summary like this:

```
---
score:      0.123456
sharpe:     1.560000
accuracy:   0.580000
num_trades: 142
max_dd:     0.050000
total_pnl:  12.3400
total_seconds: 8.5
```

You can extract the key metric from the log file:

```
grep "^score:\|^sharpe:\|^accuracy:\|^num_trades:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	score	sharpe	accuracy	num_trades	status	description
```

1. git commit hash (short, 7 chars)
2. score achieved (e.g. 0.123456) — use -999.000000 for crashes
3. sharpe ratio (e.g. 1.560000) — use 0.000000 for crashes
4. accuracy (e.g. 0.580000) — use 0.000000 for crashes
5. num_trades (e.g. 142) — use 0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	score	sharpe	accuracy	num_trades	status	description
a1b2c3d	0.123456	1.560000	0.580000	142	keep	baseline RSI probability strategy
b2c3d4e	0.156789	2.100000	0.610000	98	keep	add displacement-aware probability
c3d4e5f	0.098000	1.200000	0.550000	156	discard	switch to MACD only
d4e5f6g	-999.000000	0.000000	0.000000	0	crash	syntax error in ensemble logic
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `strategy.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run backtest.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^score:\|^sharpe:\|^accuracy:\|^num_trades:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If score improved (higher), you "advance" the branch, keeping the git commit
9. If score is equal or worse, you git reset back to where you started

The branch accumulates only winning commits, creating monotonic improvement. All experiments (keeps, discards, crashes) are logged to results.tsv for full audit.

**Timeout**: Each experiment should take <2 minutes total. If a run exceeds 4 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical signal combinations. The loop runs until the human interrupts you, period.
