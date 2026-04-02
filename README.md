# autoresearch — Time Series Forecasting & Anomaly Detection

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for **time series forecasting and anomaly detection** instead of LLM training.

The idea: give an AI agent a time series model and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better forecasting + anomaly detection model.

## How it works

Three files that matter:

- **`prepare.py`** — fixed constants, data loading (auto-discovers CSV columns), scaling, train/val/test splits, and evaluation metrics. **Not modified by the agent.**
- **`train.py`** — the single file the agent edits. Contains the model (LSTM baseline), optimizer, loss functions, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the autonomous agent. **This file is edited by the human.**

By design, training runs for a **fixed 5-minute time budget** (wall clock). The primary metric is **combined_score** = `val_scaled_mae - 0.1 * anomaly_f1` — lower is better.

## Quick start

**Requirements:** Python 3.10+, PyTorch, pandas, scikit-learn, numpy.

```bash
# 1. Install dependencies
pip install torch pandas scikit-learn numpy

# 2. Prepare your data (one-time)
python prepare.py --data /path/to/your/data.csv

# 3. Run a single training experiment (~5 min)
python train.py
```

## Data format

Any CSV with:
- A **datetime column** (auto-detected)
- A **numeric target column** (auto-detected by name: consumption, demand, load, price, etc.)
- **Numeric feature columns** (weather, temporal, etc.)

The system auto-discovers columns, so you can swap datasets without changing code.

## Running the agent

Spin up Claude Code in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| `val_scaled_mae` | Forecasting MAE (0-1 scale) | Lower is better |
| `anomaly_f1` | Anomaly detection F1 score | Higher is better |
| `combined_score` | `val_scaled_mae - 0.1 * anomaly_f1` | **Lower is better** (primary) |
| `val_mae` | MAE in original units | Lower is better |
| `val_r2` | R-squared | Higher is better |

## Project structure

```
prepare.py      — constants, data prep, evaluation (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
README.md       — this file
```

## License

MIT
