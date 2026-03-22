"""
Autoresearch sklearn experiment — edit this file to improve val_f1.
Dataset: Credit Card Fraud (~0.17% positive). Metric: macro F1 (higher is better).
Usage: uv run model.py
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from prepare import load_data, evaluate

# ---------------------------------------------------------------------------
# Load data (fixed, provided by prepare.py)
# ---------------------------------------------------------------------------

X_train, X_val, y_train, y_val, feature_names = load_data()

# ---------------------------------------------------------------------------
# Model definition — experiment here
# ---------------------------------------------------------------------------

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

# Carve a dev split to tune threshold (no val labels used).
X_tr, X_dev, y_tr, y_dev = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

t0 = time.perf_counter()

# --- Level 0: train two base models on X_tr ---
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", random_state=42, n_jobs=-1,
)
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    n_jobs=-1, random_state=0,
)
xgb.fit(X_tr, y_tr)
rf.fit(X_tr, y_tr)

# --- Level 1: stack probabilities, fit meta-model on dev ---
def stack_proba(xgb_m, rf_m, X):
    return np.column_stack([
        xgb_m.predict_proba(X)[:, 1],
        rf_m.predict_proba(X)[:, 1],
    ])

meta_X_dev = stack_proba(xgb, rf, X_dev)
meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
meta.fit(meta_X_dev, y_dev)

# Tune threshold on dev using meta-model probabilities
proba_dev = meta.predict_proba(meta_X_dev)[:, 1]
best_thresh, best_f1_dev = 0.5, 0.0
for thresh in np.linspace(0.01, 0.99, 199):
    preds = (proba_dev >= thresh).astype(int)
    score = f1_score(y_dev, preds, average="macro")
    if score > best_f1_dev:
        best_f1_dev, best_thresh = score, thresh

# --- Refit base models on full train ---
xgb_full = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", random_state=42, n_jobs=-1,
)
rf_full = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    n_jobs=-1, random_state=0,
)
xgb_full.fit(X_train, y_train)
rf_full.fit(X_train, y_train)

# Meta model was already fit on dev; use as-is (small and fast to refit if needed)
train_seconds = time.perf_counter() - t0

# --- Predict on val ---
meta_X_val = stack_proba(xgb_full, rf_full, X_val)
proba_val = meta.predict_proba(meta_X_val)[:, 1]
preds_val = (proba_val >= best_thresh).astype(int)
val_f1 = f1_score(y_val, preds_val, average="macro")

positive_rate = y_train.mean()
print("---")
print(f"val_f1:          {val_f1:.6f}")
print(f"best_thresh:     {best_thresh:.4f}")
print(f"train_seconds:   {train_seconds:.2f}")
print(f"positive_rate:   {positive_rate:.4f}")
