"""
Data preparation for autoresearch classic ML experiments.
Downloads and caches the dataset, provides evaluation harness.

Usage:
    uv run prepare.py                         # full prep (download + cache)
    uv run prepare.py --dataset house_prices  # explicit dataset choice

Data is cached in ~/.cache/autoresearch/<dataset>/.

DO NOT MODIFY — this is the fixed evaluation harness.
"""

import os
import sys
import argparse
import math
import pickle

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (edit these to switch dataset/metric)
# ---------------------------------------------------------------------------

DATASET      = "house_prices"  # dataset key — see DATASET_CONFIGS below
TARGET       = "SalePrice"     # target column name
METRIC       = "rmse"          # "rmse" | "mae" | "r2"
TEST_SIZE    = 0.2             # fraction held out for final test evaluation
RANDOM_STATE = 42              # fixed seed — do not change between experiments

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

CACHE_DIR   = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")

def _dataset_cache_dir(dataset=None):
    return os.path.join(CACHE_DIR, dataset or DATASET)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _fetch_house_prices(cache_dir):
    """Download Ames Housing dataset via sklearn OpenML, save as parquet."""
    from sklearn.datasets import fetch_openml
    raw_path = os.path.join(cache_dir, "raw.pkl")
    if os.path.exists(raw_path):
        return
    print("Downloading Ames Housing dataset from OpenML...")
    bunch = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    with open(raw_path, "wb") as f:
        pickle.dump(bunch, f)
    print(f"  Saved to {raw_path}")


def _load_house_prices(cache_dir):
    """Load raw Ames Housing bunch from cache."""
    raw_path = os.path.join(cache_dir, "raw.pkl")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Run `uv run prepare.py` first."
        )
    with open(raw_path, "rb") as f:
        bunch = pickle.load(f)
    X = bunch.data.copy()
    y = bunch.target.astype(float).copy()
    y.name = "SalePrice"
    return X, y


DATASET_CONFIGS = {
    "house_prices": {
        "target":   "SalePrice",
        "metric":   "rmse",
        "fetch_fn": _fetch_house_prices,
        "load_fn":  _load_house_prices,
        "description": "Ames Housing — predict residential home sale prices",
    },
    # To add a new dataset, add an entry here with fetch_fn and load_fn.
}

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    """
    Compute the configured metric between ground-truth and predictions.
    Both inputs should be array-like of the same length.
    Returns a scalar. For RMSE/MAE, lower is better. For R2, higher is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if METRIC == "rmse":
        return math.sqrt(np.mean((y_true - y_pred) ** 2))
    elif METRIC == "mae":
        return np.mean(np.abs(y_true - y_pred))
    elif METRIC == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        raise ValueError(f"Unknown METRIC: {METRIC!r}. Use 'rmse', 'mae', or 'r2'.")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset=None):
    """
    Load cached dataset and return a fixed train/test split.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Raw feature matrices — columns unchanged, no imputation, no scaling.
        The agent is responsible for all preprocessing in train.py.
    y_train, y_test : pd.Series
        Target values (original scale, not log-transformed).
    """
    from sklearn.model_selection import train_test_split

    dataset = dataset or DATASET
    cfg = DATASET_CONFIGS[dataset]
    cache_dir = _dataset_cache_dir(dataset)

    X, y = cfg["load_fn"](cache_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Reset indices so they're clean 0-based integers
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------
# Main — one-time data preparation
# ---------------------------------------------------------------------------

def _print_data_summary(X, y):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    missing  = X.isnull().sum()
    missing  = missing[missing > 0].sort_values(ascending=False)

    print(f"  Shape:              {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"  Numeric features:   {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    print(f"  Columns with nulls: {len(missing)}")
    if len(missing):
        top = missing.head(5)
        for col, cnt in top.items():
            print(f"    {col}: {cnt} ({100*cnt/len(X):.1f}%)")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
    print(f"  Target ({y.name}):")
    print(f"    min={y.min():,.0f}  median={y.median():,.0f}  max={y.max():,.0f}  std={y.std():,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch classic ML")
    parser.add_argument("--dataset", default=DATASET, choices=list(DATASET_CONFIGS),
                        help="Which dataset to prepare")
    args = parser.parse_args()

    dataset = args.dataset
    cfg     = DATASET_CONFIGS[dataset]
    cache_dir = _dataset_cache_dir(dataset)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Dataset:    {dataset}")
    print(f"            {cfg['description']}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Metric:     {METRIC} (lower is better)" if METRIC != "r2" else f"Metric: {METRIC} (higher is better)")
    print()

    # Download if needed
    cfg["fetch_fn"](cache_dir)

    # Load and summarise
    print("Loading data...")
    X, y = cfg["load_fn"](cache_dir)
    print("Data summary:")
    _print_data_summary(X, y)

    print()
    print("Done! Ready to run experiments with: uv run train.py")
