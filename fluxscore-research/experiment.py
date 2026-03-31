"""
FluxScore autoresearch — experiment harness.
The agent modifies everything BELOW the "AGENT ZONE" marker.
The fixed harness above it is read-only.

Usage: python experiment.py
Output: prints summary ending with "auc: X.XXXXXX" on its own line (grep ^auc:)
"""

# ===========================================================================
# FIXED HARNESS — DO NOT MODIFY (agent cannot touch this section)
# ===========================================================================
import signal
import sys
import time
import os
from pathlib import Path

# 2-minute hard ceiling — same budget regardless of model complexity
TIME_LIMIT_SECONDS = 120

def _timeout_handler(signum, frame):
    print("auc: 0.000000")
    print("gini: 0.000000")
    print("brier: 1.000000")
    print("status: timeout")
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(TIME_LIMIT_SECONDS)

# Startup check — must run before any model code
_DATA_DIR = Path(__file__).parent
_TRAIN_PATH = _DATA_DIR / "train.parquet"
_HOLDOUT_PATH = _DATA_DIR / "holdout.parquet"

if not _TRAIN_PATH.exists() or not _HOLDOUT_PATH.exists():
    print("ERROR: run prepare.py first")
    print("  cd fluxscore-research && python prepare.py")
    sys.exit(1)

import numpy as np
import pandas as pd

# Load data — fixed, agent does not touch this
_train = pd.read_parquet(_TRAIN_PATH)
_holdout = pd.read_parquet(_HOLDOUT_PATH)

FEATURE_COLS = [c for c in _train.columns if c != "default"]
_X_train = _train[FEATURE_COLS].values.astype(np.float32)
_y_train = _train["default"].values
_X_holdout = _holdout[FEATURE_COLS].values.astype(np.float32)
_y_holdout = _holdout["default"].values

_t0 = time.perf_counter()


def _evaluate(model):
    """
    Fixed evaluation harness. Agent calls this — cannot modify it.
    Returns (auc, gini, brier_score).
    AUC higher = better (opposite of val_bpb — this is important).
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    try:
        proba = model.predict_proba(_X_holdout)[:, 1]
    except AttributeError:
        proba = model.predict(_X_holdout)

    auc = roc_auc_score(_y_holdout, proba)
    gini = 2 * auc - 1
    brier = brier_score_loss(_y_holdout, proba)
    return auc, gini, brier


def _print_summary(auc, gini, brier, elapsed):
    """Machine-parseable summary. grep '^auc:' to extract the metric."""
    print("---")
    print(f"auc:     {auc:.6f}")
    print(f"gini:    {gini:.6f}")
    print(f"brier:   {brier:.6f}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"status:  ok")


# ===========================================================================
# AGENT ZONE — modify freely below this line
# ===========================================================================
# Goal: maximize AUC on the holdout set.
# AUC higher = better. Target: > 0.85. Promotion gate: +0.02 over baseline.
#
# Rules:
# - Call _evaluate(model) to get (auc, gini, brier) — do not modify _evaluate
# - Call _print_summary(auc, gini, brier, elapsed) at the end — do not modify
# - signal.alarm(120) is already set — your run will be killed at 120s regardless
# - You may use any package in pyproject.toml
# - Cross-validation is fine; _X_train / _y_train are yours to slice however
# - Try feature engineering, different models, ensembles, class weight tuning
# - Do NOT touch anything above the AGENT ZONE marker

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Baseline: logistic regression with standard scaling
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(_X_train, _y_train)

    _elapsed = time.perf_counter() - _t0
    _auc, _gini, _brier = _evaluate(model)
    _print_summary(_auc, _gini, _brier, _elapsed)

except Exception as e:
    # Crash handler — agent wrote bad code; emit parseable output and exit 1
    print(f"auc: 0.000000")
    print(f"gini: 0.000000")
    print(f"brier: 1.000000")
    print(f"status: crash ({type(e).__name__}: {e})")
    sys.exit(1)
