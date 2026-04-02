"""
Autoresearch: Time Series Forecasting & Anomaly Detection
Data preparation, evaluation, and runtime utilities.

Usage:
    python prepare.py                          # uses default sample data
    python prepare.py --data path/to/data.csv  # uses custom CSV

Data is stored in ~/.cache/autoresearch-ts/

THIS FILE IS READ-ONLY. The agent must NOT modify it.
"""

import os
import sys
import math
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
SEQ_LENGTH = 24            # default lookback window (hours)
FORECAST_HORIZON = 1       # predict 1 step ahead
VAL_RATIO = 0.15           # validation set ratio
TEST_RATIO = 0.05          # test set ratio (held out, never used during experiments)
ANOMALY_ZSCORE = 3.0       # z-score threshold for anomaly ground truth
EVAL_BATCHES = 50          # number of batches for evaluation
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-ts")

# ---------------------------------------------------------------------------
# Data Discovery & Loading
# ---------------------------------------------------------------------------

def discover_columns(df):
    """
    Auto-discover datetime, target, and feature columns from any CSV.
    Returns dict with keys: datetime_col, target_col, feature_cols, categorical_cols.

    Strategy:
    - datetime: first column parseable as datetime
    - target: first numeric column with 'consumption', 'value', 'target', 'demand',
              'load', 'price', 'flow', 'usage' in name, else first numeric column
    - features: all other numeric columns (excluding IDs and keys)
    - categorical: low-cardinality int columns (weekday, month, etc.)
    """
    result = {
        "datetime_col": None,
        "target_col": None,
        "feature_cols": [],
        "categorical_cols": [],
    }

    # Find datetime column
    for col in df.columns:
        if df[col].dtype == "object" or "datetime" in str(df[col].dtype):
            try:
                pd.to_datetime(df[col].head(100))
                result["datetime_col"] = col
                break
            except (ValueError, TypeError):
                continue

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID-like columns
    id_patterns = ["id", "key", "index", "locationid", "propkey", "custkey"]
    non_id_numeric = [c for c in numeric_cols
                      if not any(p in c.lower() for p in id_patterns)]

    # Find target column
    target_hints = ["consumption", "value", "target", "demand", "load",
                    "price", "flow", "usage", "power", "energy", "volume"]
    for hint in target_hints:
        for col in non_id_numeric:
            if hint in col.lower():
                result["target_col"] = col
                break
        if result["target_col"]:
            break
    if not result["target_col"] and non_id_numeric:
        result["target_col"] = non_id_numeric[0]

    # Categorize remaining columns
    time_derived = ["hour", "day", "month", "weekday", "dayofweek",
                    "is_weekend", "is_holiday", "daypartkey", "interval"]
    for col in non_id_numeric:
        if col == result["target_col"]:
            continue
        if col.lower() in time_derived or df[col].nunique() < 30:
            result["categorical_cols"].append(col)
        else:
            result["feature_cols"].append(col)

    return result


def load_and_prepare_data(csv_path, seq_length=SEQ_LENGTH):
    """
    Load CSV, discover columns, create scaled sequences.
    Returns a dict with everything needed for training.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    # Discover columns
    col_info = discover_columns(df)
    print(f"  Datetime column: {col_info['datetime_col']}")
    print(f"  Target column:   {col_info['target_col']}")
    print(f"  Feature columns ({len(col_info['feature_cols'])}): {col_info['feature_cols']}")
    print(f"  Categorical columns ({len(col_info['categorical_cols'])}): {col_info['categorical_cols']}")

    # Parse datetime and sort
    if col_info["datetime_col"]:
        df[col_info["datetime_col"]] = pd.to_datetime(df[col_info["datetime_col"]])
        df = df.sort_values(col_info["datetime_col"]).reset_index(drop=True)

    target_col = col_info["target_col"]
    if target_col is None:
        print("ERROR: No suitable target column found.")
        sys.exit(1)

    # Build feature matrix: target first, then features, then categoricals
    use_cols = [target_col] + col_info["feature_cols"] + col_info["categorical_cols"]
    # Drop columns with too many NaNs (>50%)
    use_cols = [c for c in use_cols if df[c].isna().mean() < 0.5]
    data = df[use_cols].copy()
    data = data.fillna(method="ffill").fillna(method="bfill").fillna(0)
    feature_names = list(data.columns)
    n_features = len(feature_names)

    print(f"  Final features ({n_features}): {feature_names}")

    # Compute anomaly labels BEFORE scaling (z-score on target)
    target_values = data[target_col].values
    mean_t = np.mean(target_values)
    std_t = np.std(target_values) + 1e-8
    z_scores = np.abs((target_values - mean_t) / std_t)
    anomaly_labels = (z_scores > ANOMALY_ZSCORE).astype(np.float32)
    anomaly_rate = anomaly_labels.mean()
    print(f"  Anomaly rate (z>{ANOMALY_ZSCORE}): {anomaly_rate:.4f} ({anomaly_labels.sum():.0f}/{len(anomaly_labels)} points)")

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values)

    # Target scaler (for inverse transform)
    target_scaler = MinMaxScaler()
    target_scaler.fit(data[[target_col]].values)

    # Create sequences
    X, y_forecast, y_anomaly = [], [], []
    for i in range(len(scaled_data) - seq_length - FORECAST_HORIZON + 1):
        X.append(scaled_data[i:i + seq_length])
        y_forecast.append(scaled_data[i + seq_length, 0])  # target is col 0
        y_anomaly.append(anomaly_labels[i + seq_length])

    X = np.array(X, dtype=np.float32)
    y_forecast = np.array(y_forecast, dtype=np.float32)
    y_anomaly = np.array(y_anomaly, dtype=np.float32)

    # Split: train / val / test (chronological, no shuffle)
    n = len(X)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    splits = {
        "train": {
            "X": torch.tensor(X[:n_train]),
            "y_forecast": torch.tensor(y_forecast[:n_train]),
            "y_anomaly": torch.tensor(y_anomaly[:n_train]),
        },
        "val": {
            "X": torch.tensor(X[n_train:n_train + n_val]),
            "y_forecast": torch.tensor(y_forecast[n_train:n_train + n_val]),
            "y_anomaly": torch.tensor(y_anomaly[n_train:n_train + n_val]),
        },
        "test": {
            "X": torch.tensor(X[n_train + n_val:]),
            "y_forecast": torch.tensor(y_forecast[n_train + n_val:]),
            "y_anomaly": torch.tensor(y_anomaly[n_train + n_val:]),
        },
    }

    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    metadata = {
        "n_features": n_features,
        "seq_length": seq_length,
        "feature_names": feature_names,
        "target_col": target_col,
        "col_info": col_info,
        "anomaly_rate": float(anomaly_rate),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }

    return splits, metadata, target_scaler


def save_prepared_data(splits, metadata, target_scaler):
    """Save prepared data to cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(splits, os.path.join(CACHE_DIR, "splits.pt"))
    with open(os.path.join(CACHE_DIR, "metadata.json"), "w") as f:
        # col_info may have non-serializable types
        meta_save = {k: v for k, v in metadata.items() if k != "col_info"}
        json.dump(meta_save, f, indent=2)
    with open(os.path.join(CACHE_DIR, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)
    print(f"  Saved to {CACHE_DIR}")


def load_prepared_data(device="cpu"):
    """Load prepared data from cache."""
    splits = torch.load(os.path.join(CACHE_DIR, "splits.pt"), map_location=device)
    with open(os.path.join(CACHE_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    with open(os.path.join(CACHE_DIR, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)
    return splits, metadata, target_scaler


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

def make_dataloader(split_data, batch_size, shuffle=False):
    """Simple batched data loader. Yields (X_batch, y_forecast_batch, y_anomaly_batch)."""
    X = split_data["X"]
    y_f = split_data["y_forecast"]
    y_a = split_data["y_anomaly"]
    n = len(X)
    indices = torch.randperm(n) if shuffle else torch.arange(n)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y_f[idx], y_a[idx]


def make_infinite_dataloader(split_data, batch_size, shuffle=True):
    """Infinite batched data loader for training."""
    epoch = 0
    while True:
        epoch += 1
        for batch in make_dataloader(split_data, batch_size, shuffle=shuffle):
            yield batch, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE - this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, split_data, target_scaler, batch_size, device,
             max_batches=EVAL_BATCHES):
    """
    Evaluate model on forecasting (MAE, RMSE, R2) and anomaly detection (F1).

    The model must return a dict with:
      - "forecast": tensor of shape (B,) - predicted next-step target (scaled 0-1)
      - "anomaly":  tensor of shape (B,) - anomaly score/probability (0-1)

    Primary metric: val_mae (lower is better)
    Secondary metrics: val_rmse, val_r2, anomaly_f1, anomaly_precision, anomaly_recall

    Returns dict of metrics.
    """
    model.eval()

    all_pred_forecast = []
    all_true_forecast = []
    all_pred_anomaly = []
    all_true_anomaly = []

    loader = make_dataloader(split_data, batch_size, shuffle=False)
    for i, (X_b, y_f_b, y_a_b) in enumerate(loader):
        if i >= max_batches:
            break
        X_b = X_b.to(device)
        y_f_b = y_f_b.to(device)
        y_a_b = y_a_b.to(device)

        output = model(X_b)
        all_pred_forecast.append(output["forecast"].cpu())
        all_true_forecast.append(y_f_b.cpu())
        all_pred_anomaly.append(output["anomaly"].cpu())
        all_true_anomaly.append(y_a_b.cpu())

    pred_f = torch.cat(all_pred_forecast)
    true_f = torch.cat(all_true_forecast)
    pred_a = torch.cat(all_pred_anomaly)
    true_a = torch.cat(all_true_anomaly)

    # Inverse transform for interpretable MAE/RMSE
    pred_f_inv = target_scaler.inverse_transform(
        pred_f.numpy().reshape(-1, 1)).flatten()
    true_f_inv = target_scaler.inverse_transform(
        true_f.numpy().reshape(-1, 1)).flatten()

    # Forecasting metrics
    mae = float(np.mean(np.abs(pred_f_inv - true_f_inv)))
    rmse = float(np.sqrt(np.mean((pred_f_inv - true_f_inv) ** 2)))
    ss_res = np.sum((pred_f_inv - true_f_inv) ** 2)
    ss_tot = np.sum((true_f_inv - np.mean(true_f_inv)) ** 2) + 1e-8
    r2 = float(1 - ss_res / ss_tot)

    # Scaled MAE (on 0-1 scale, for agent comparison - THIS is the primary metric)
    scaled_mae = float(torch.mean(torch.abs(pred_f - true_f)).item())

    # Anomaly metrics (threshold at 0.5)
    pred_a_binary = (pred_a > 0.5).float()
    tp = float(((pred_a_binary == 1) & (true_a == 1)).sum())
    fp = float(((pred_a_binary == 1) & (true_a == 0)).sum())
    fn = float(((pred_a_binary == 0) & (true_a == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Combined score: primary is scaled_mae, anomaly_f1 is secondary
    # Lower is better: score = scaled_mae - 0.1 * anomaly_f1
    combined_score = scaled_mae - 0.1 * f1

    metrics = {
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2,
        "val_scaled_mae": scaled_mae,
        "anomaly_f1": f1,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "combined_score": combined_score,
    }

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Main (one-time data prep)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare time series data for autoresearch experiments")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV file with time series data")
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH,
                        help=f"Lookback window length (default: {SEQ_LENGTH})")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: File not found: {args.data}")
        sys.exit(1)

    print(f"Cache directory: {CACHE_DIR}")
    print()

    splits, metadata, target_scaler = load_and_prepare_data(
        args.data, seq_length=args.seq_length)
    save_prepared_data(splits, metadata, target_scaler)

    print()
    print("Done! Ready to train with: python train.py")
