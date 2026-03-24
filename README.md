# autotrader

Autonomous ML prediction research platform for BTC/USD. An AI agent
runs experiments continuously — modifying the model architecture,
running walk-forward evaluation across 5 expanding windows, and
keeping or discarding based on a composite score (Sharpe, drawdown,
consistency).

The system includes a paper trading pipeline that runs the trained
model in real time on a Raspberry Pi, logging predictions, positions,
and simulated P&L for live validation.

## How it works

The agent modifies only `train.py`. Everything inside `build_model()`
is fair game: features, model hyperparameters, post-processing
pipeline, position scaling. The evaluation infrastructure is fixed.

```
assets/btc_hourly/
  prepare.py    — data download, walk-forward eval wrapper (do not modify)
  train.py      — model recipe: features, training, inference (agent modifies)
  program.md    — agent instructions

core/
  backtesting.py  — position sizing + P&L engine
  config.py       — BacktestConfig, WalkForwardConfig
  evaluation.py   — walk-forward evaluation + diagnostics
  epoch.py        — epoch/holdout rotation

btc-paper-trader/
  src/            — hourly cron pipeline (inference, portfolio, logging)
  scripts/        — export artifacts, seed data, replay, install services
  tests/          — feature parity, inference parity, portfolio math
```

## Quick start

```bash
# Install dependencies
uv sync

# Download BTC/USD data (~2 min)
cd assets/btc_hourly
uv run python prepare.py

# Run a single training experiment
uv run python train.py

# Run walk-forward diagnostic (human only)
uv run python prepare.py --diagnose
```

## Running the agent

Point Claude Code at `assets/btc_hourly/program.md`:

```
Read program.md and kick off a new experiment.
```

The agent creates a branch, runs experiments, logs results to
`results.tsv` and `experiment-log.md`, and commits improvements.

## Evaluation

5 expanding walk-forward windows (2021-2025), each with 2 half-year
subperiods. Score = min Sharpe across scored windows, penalized for
drawdown and low trade count. Consistency = all 8 subperiods positive.

A rotating holdout window (changes every 30 evals) prevents
overfitting to any single test period.

## Paper trading

The `btc-paper-trader/` directory contains a headless paper trading
system that runs the frozen model in real time:

```bash
cd btc-paper-trader

# Export model artifacts (run after training)
cd .. && uv run btc-paper-trader/scripts/export_artifacts.py --train-end 2025-12-31

# Seed historical data
cd btc-paper-trader && python scripts/seed_data.py

# Run tests
uv run python -m pytest tests/ -v

# Manual test run
.venv/bin/python -m src.main

# Historical replay (infrastructure validation)
.venv/bin/python scripts/replay.py --start 2026-01-01
```

OHLCV from Binance US, funding rates from Kraken Futures.
Hourly cron at :05, daily report at 23:15 UTC.

## License

MIT
