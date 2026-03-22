"""
Autoresearch sklearn experiment — edit this file to improve val_f1.
Dataset: Credit Card Fraud (~0.17% positive). Metric: macro F1 (higher is better).
Usage: uv run model.py
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from prepare import load_data, evaluate

# ---------------------------------------------------------------------------
# Load data (fixed, provided by prepare.py)
# ---------------------------------------------------------------------------

X_train, X_val, y_train, y_val, feature_names = load_data()

# ---------------------------------------------------------------------------
# Model definition — experiment here
# ---------------------------------------------------------------------------

from xgboost import XGBClassifier

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = neg / pos

# Carve a dev split from training data to tune the decision threshold.
# Val labels are never used here.
X_tr, X_dev, y_tr, y_dev = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

base_model = XGBClassifier(
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

t0 = time.perf_counter()
# Fit on 85% of train to find threshold on dev
base_model.fit(X_tr, y_tr)

# Scan thresholds on dev split, pick best macro F1
proba_dev = base_model.predict_proba(X_dev)[:, 1]
best_thresh, best_f1_dev = 0.5, 0.0
for thresh in np.linspace(0.01, 0.99, 199):
    preds = (proba_dev >= thresh).astype(int)
    score = f1_score(y_dev, preds, average="macro")
    if score > best_f1_dev:
        best_f1_dev, best_thresh = score, thresh

# Refit on full training set with same hyperparams
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
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - t0

# Predict on val using threshold found on dev
proba_val = model.predict_proba(X_val)[:, 1]
preds_val = (proba_val >= best_thresh).astype(int)
val_f1 = f1_score(y_val, preds_val, average="macro")

positive_rate = y_train.mean()
print("---")
print(f"val_f1:          {val_f1:.6f}")
print(f"best_thresh:     {best_thresh:.4f}")
print(f"train_seconds:   {train_seconds:.2f}")
print(f"positive_rate:   {positive_rate:.4f}")
