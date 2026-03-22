# autoresearch — BTC prediction

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **BTC price prediction** targeting Kalshi 15-minute binary up/down contracts.

The idea: give an AI agent a backtesting framework and let it experiment with trading strategies autonomously overnight. It modifies the strategy code, runs a backtest, checks if the score improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better strategy.

## How it works

Three files that matter:

- **`prepare.py`** — fixed data pipeline, feature engineering, backtesting engine, and evaluation function. Not modified by the agent.
- **`strategy.py`** — the single file the agent edits. Contains the `Strategy` class with an `on_bar()` method that receives 60 minutes of OHLCV data with technical indicators and returns a direction signal + confidence. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the autonomous agent. **This file is edited and iterated on by the human.**

The metric is a composite **score = sharpe × accuracy × trade_factor** — higher is better. The backtest simulates Kalshi binary contract P&L with realistic fees.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU needed.

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Prepare data (generates synthetic data for testing)
uv run prepare.py

# 4. Run baseline backtest
uv run backtest.py
```

## Using real data

Place your BTC 1-minute OHLCV CSV at `~/.cache/autoresearch/btc_1m.csv` with columns:
```
timestamp,open,high,low,close,volume
```

Then run `uv run backtest.py` — it will automatically use the real data.

## Running the agent

Spin up Claude Code (or any coding agent) in this repo and prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent runs autonomously, ~30+ experiments/hour (backtests are fast).

## Project structure

```
prepare.py      — data pipeline, features, backtest engine, evaluation (do not modify)
strategy.py     — strategy logic (agent modifies this)
backtest.py     — entry point that runs evaluation (do not modify)
program.md      — agent instructions
analysis.ipynb  — notebook for analyzing results.tsv
pyproject.toml  — dependencies
```

## Configurable temporal split

Edit constants in `prepare.py` to control train/validation periods:

```python
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-12-31"
VAL_PERIODS = [("2023-01-01", "2023-12-31"), ("2025-01-01", "2025-03-22")]
```

## Based on

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original autonomous LLM research framework
- [Nunchi-trade/auto-researchtrading](https://github.com/Nunchi-trade/auto-researchtrading) — trading-specific fork for crypto futures

## License

MIT
