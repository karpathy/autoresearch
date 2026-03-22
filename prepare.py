"""
Fixed data pipeline, backtesting engine, and evaluation for BTC autoresearch.

DO NOT MODIFY — this file is read-only. The agent only edits strategy.py.

Usage:
    python prepare.py              # generate synthetic data (for testing)
    python prepare.py --check      # verify data exists and print stats

Data is stored in ~/.cache/autoresearch/.
"""

import os
import math
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

# Configurable temporal split — change these dates to control train/val periods
VAL_PERIODS = [
    ("2023-01-01", "2023-12-31"),
    ("2025-01-01", "2025-03-22"),
]

# Prediction task
HORIZON_MINUTES = 15    # predict price direction this far ahead
LOOKBACK_MINUTES = 60   # number of 1-min bars the strategy sees per call

# Backtest parameters
TIME_BUDGET = 120       # max seconds per experiment run
FEE_BPS = 5             # round-trip fee in basis points (applied per trade)
MIN_TRADES = 20         # minimum trades for full score (penalty below this)
CONFIDENCE_THRESHOLD = 0.0  # minimum confidence to place a trade
MIN_CONFIDENCE = 0.01       # floor for confidence clamp (risk at least $0.01)
MAX_CONFIDENCE = 0.99       # ceiling for confidence clamp (risk at most $0.99)
INVALID_SCORE = -999.0      # sentinel for invalid/crashed strategies

# Data paths
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_FILE = os.path.join(CACHE_DIR, "btc_1m.csv")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Load BTC 1-minute OHLCV data from CSV.
    Expected columns: timestamp, open, high, low, close, volume
    Falls back to synthetic data if the CSV doesn't exist.
    """
    if os.path.exists(DATA_FILE):
        print(f"Loading data from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    else:
        print(f"Data file not found at {DATA_FILE}")
        print("Generating synthetic data for testing...")
        df = generate_synthetic_data()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Validate required columns
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    print(f"Data loaded: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def generate_synthetic_data(
    start="2022-06-01",
    end="2025-03-22",
    seed=42,
):
    """
    Generate realistic synthetic 1-minute BTC OHLCV data using geometric
    Brownian motion. Used for testing when real data is unavailable.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start, end=end, freq="1min")
    n = len(timestamps)

    # GBM parameters calibrated to BTC (~60% annual vol)
    dt = 1 / (365.25 * 24 * 60)  # 1 minute in years
    mu = 0.0       # no drift (neutral)
    sigma = 0.60   # 60% annualized volatility

    # Generate log returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n)
    log_returns[0] = 0.0

    # Price series starting at ~30,000
    close = 30000.0 * np.exp(np.cumsum(log_returns))

    # Generate OHLV from close
    noise = rng.uniform(0.0001, 0.002, size=n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Volume: log-normal with some autocorrelation
    base_vol = rng.lognormal(mean=10, sigma=1.5, size=n)
    volume = pd.Series(base_vol).rolling(10, min_periods=1).mean().values

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    # Save for reproducibility
    os.makedirs(CACHE_DIR, exist_ok=True)
    synth_path = os.path.join(CACHE_DIR, "btc_1m_synthetic.csv")
    df.to_csv(synth_path, index=False)
    print(f"Synthetic data saved to {synth_path} ({len(df)} rows)")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(df):
    """
    Add technical indicator columns to the OHLCV DataFrame.
    Uses pandas-ta for indicator computation.
    The strategy can use any of these columns.
    """
    import pandas_ta as ta

    df = df.copy()

    # Basic returns
    df["returns"] = df["close"].pct_change()

    # Volatility (20-bar rolling std of returns)
    df["volatility_20"] = df["returns"].rolling(20, min_periods=1).std()

    # Moving averages
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # MACD (derived from already-computed EMAs to avoid redundant passes)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = ta.ema(df["macd"], length=9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_df = ta.bbands(df["close"], length=20, std=2)
    df["bbands_lower"] = bb_df.iloc[:, 0]
    df["bbands_mid"] = bb_df.iloc[:, 1]
    df["bbands_upper"] = bb_df.iloc[:, 2]
    df["bbands_bandwidth"] = bb_df.iloc[:, 3] if bb_df.shape[1] > 3 else 0.0

    # ATR
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume SMA
    df["volume_sma_20"] = ta.sma(df["volume"], length=20)

    # Fill NaN from indicator warm-up periods
    df = df.ffill().bfill()

    return df


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def get_val_data(df):
    """
    Extract validation data based on configured date periods.
    Returns val_df (reset index for positional access in backtest).
    """
    ts = df["timestamp"]

    val_mask = pd.Series(False, index=df.index)
    period_counts = []
    for vstart, vend in VAL_PERIODS:
        period_mask = (ts >= vstart) & (ts < pd.Timestamp(vend) + pd.Timedelta(days=1))
        val_mask = val_mask | period_mask
        period_counts.append((vstart, vend, period_mask.sum()))
    val_df = df[val_mask].reset_index(drop=True)

    for vstart, vend, count in period_counts:
        print(f"Val:   {count} bars ({vstart} to {vend})")
    print(f"Val total: {len(val_df)} bars")

    return val_df


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

def run_backtest(strategy_class, df):
    """
    Run a backtest of the strategy against the given DataFrame.

    For each bar (after the lookback warm-up), calls strategy.on_bar(window)
    where window is the last LOOKBACK_MINUTES bars with all features.

    The strategy returns (signal, confidence):
      signal:     1 = predict up, -1 = predict down, 0 = no trade
      confidence: 0.0 to 1.0

    Simulates Kalshi binary contract P&L:
      - Buy at price = confidence (e.g., 0.6 means paying $0.60 for a $1 contract)
      - If correct: profit = 1.0 - confidence
      - If wrong:   loss = -confidence
      - Fee deducted per trade

    Returns a list of trade dicts.
    """
    strategy = strategy_class()
    trades = []
    error_count = 0

    fee_per_trade = FEE_BPS / 10000.0
    close_arr = df["close"].values
    ts_arr = df["timestamp"].values

    # Precompute future close prices (shifted by HORIZON_MINUTES)
    future_close_arr = np.empty(len(df), dtype=np.float64)
    future_close_arr[:len(df) - HORIZON_MINUTES] = close_arr[HORIZON_MINUTES:]
    future_close_arr[len(df) - HORIZON_MINUTES:] = np.nan

    max_idx = len(df) - HORIZON_MINUTES
    start_idx = LOOKBACK_MINUTES

    for i in range(start_idx, max_idx):
        window = df.iloc[i - LOOKBACK_MINUTES:i]

        try:
            result = strategy.on_bar(window)
            signal, confidence = int(result[0]), float(result[1])
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"WARNING: strategy.on_bar() raised {type(e).__name__}: {e}")
            continue

        if signal == 0 or confidence <= CONFIDENCE_THRESHOLD:
            continue

        confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

        current_close = close_arr[i]
        future_close = future_close_arr[i]
        if np.isnan(future_close):
            continue
        actual_direction = 1 if future_close > current_close else -1

        correct = (signal == actual_direction)
        if correct:
            pnl = (1.0 - confidence) - fee_per_trade
        else:
            pnl = -confidence - fee_per_trade

        trades.append({
            "timestamp": ts_arr[i],
            "signal": signal,
            "confidence": confidence,
            "actual_direction": actual_direction,
            "correct": correct,
            "pnl": pnl,
        })

    if error_count > 0:
        print(f"WARNING: strategy.on_bar() raised {error_count} total errors")

    return trades


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def compute_metrics(trades):
    """
    Compute performance metrics from a list of trades.
    Returns a dict of metrics.
    """
    if not trades or len(trades) < 2:
        return {
            "score": INVALID_SCORE,
            "sharpe": 0.0,
            "accuracy": 0.0,
            "num_trades": len(trades) if trades else 0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
        }

    trades_df = pd.DataFrame(trades)
    num_trades = len(trades_df)
    accuracy = trades_df["correct"].mean()
    total_pnl = trades_df["pnl"].sum()

    # Sharpe ratio: annualized from per-trade P&L
    # Group by day for daily P&L
    trades_df["date"] = trades_df["timestamp"].dt.date
    daily_pnl = trades_df.groupby("date")["pnl"].sum()

    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        sharpe = 0.0
    else:
        sharpe = daily_pnl.mean() / daily_pnl.std() * math.sqrt(365)

    # Max drawdown from cumulative P&L
    cum_pnl = trades_df["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0

    # Trade count penalty: sqrt(min(num_trades / MIN_TRADES, 1.0))
    trade_factor = math.sqrt(min(num_trades / MIN_TRADES, 1.0))

    # Composite score
    if num_trades < 5 or math.isnan(accuracy):
        score = INVALID_SCORE
    else:
        score = sharpe * accuracy * trade_factor

    return {
        "score": score,
        "sharpe": sharpe,
        "accuracy": accuracy,
        "num_trades": num_trades,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
    }


def evaluate(strategy_class):
    """
    Full evaluation pipeline:
    1. Load data
    2. Compute features
    3. Extract validation data
    4. Run backtest on validation data
    5. Compute and print metrics

    Returns metrics dict.
    """
    # Load and prepare data
    df = load_data()
    df = compute_features(df)
    val_df = get_val_data(df)

    if len(val_df) < LOOKBACK_MINUTES + HORIZON_MINUTES + 1:
        print("ERROR: not enough validation data")
        return {"score": INVALID_SCORE}

    # Run backtest on validation set
    print(f"\nRunning backtest on {len(val_df)} validation bars...")
    trades = run_backtest(strategy_class, val_df)

    # Compute metrics
    metrics = compute_metrics(trades)

    # Print in grep-able format
    print("\n---")
    print(f"score:      {metrics['score']:.6f}")
    print(f"sharpe:     {metrics['sharpe']:.6f}")
    print(f"accuracy:   {metrics['accuracy']:.6f}")
    print(f"num_trades: {metrics['num_trades']}")
    print(f"max_dd:     {metrics['max_drawdown']:.6f}")
    print(f"total_pnl:  {metrics['total_pnl']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Main — data preparation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for BTC autoresearch")
    parser.add_argument("--check", action="store_true", help="Check if data exists and print stats")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")
    args = parser.parse_args()

    if args.check:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
            print(f"Data file: {DATA_FILE}")
            print(f"Rows: {len(df)}")
            print(f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Columns: {list(df.columns)}")
        else:
            print(f"No data file found at {DATA_FILE}")
            print(f"Place your BTC 1-minute OHLCV CSV there, or run: python prepare.py --synthetic")
    elif args.synthetic:
        generate_synthetic_data()
        print("\nDone! Synthetic data ready for testing.")
    else:
        # Default: generate synthetic if no real data exists
        if not os.path.exists(DATA_FILE):
            print("No real data found. Generating synthetic data...")
            generate_synthetic_data()
        else:
            print(f"Data already exists at {DATA_FILE}")
        print("\nDone! Ready to run: uv run backtest.py")
