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

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",  # compensate for extreme imbalance
    n_jobs=-1,
    random_state=42,
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
