"""
Autoresearch sklearn experiment — edit this file to improve val_f1.
Dataset: Credit Card Fraud (~0.17% positive). Metric: macro F1 (higher is better).
Usage: uv run model.py
"""

import time
import numpy as np
from prepare import load_data, evaluate

# ---------------------------------------------------------------------------
# Load data (fixed, provided by prepare.py)
# ---------------------------------------------------------------------------

X_train, X_val, y_train, y_val, feature_names = load_data()

# ---------------------------------------------------------------------------
# Model definition — experiment here
# ---------------------------------------------------------------------------

from xgboost import XGBClassifier

# scale_pos_weight = count(negatives) / count(positives) ≈ 578
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=spw,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Fit and evaluate
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - t0

val_f1 = evaluate(model, X_val, y_val)

positive_rate = y_train.mean()
print("---")
print(f"val_f1:          {val_f1:.6f}")
print(f"train_seconds:   {train_seconds:.2f}")
print(f"positive_rate:   {positive_rate:.4f}")
