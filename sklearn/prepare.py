"""
Fixed evaluation harness for sklearn autoresearch.

Dataset: Credit Card Fraud Detection (OpenML id=1597).
  - 284,807 transactions, 492 fraud cases (~0.17% positive).
  - 30 features (V1-V28 are PCA components, plus Amount and Time).
  - Binary target: 0 = legit, 1 = fraud.

This is a deliberately hard, heavily imbalanced dataset.
The metric is macro-averaged F1 (higher is better).

DO NOT MODIFY THIS FILE.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Fixed constants — do not change
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
VAL_SIZE = 0.2          # 20 % stratified held-out validation set
OPENML_ID = 1597        # creditcard dataset

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Returns (X_train, X_val, y_train, y_val, feature_names).

    Downloads on first call (~30 MB), cached in ~/scikit_learn_data/ afterward.
    """
    dataset = fetch_openml(data_id=OPENML_ID, as_frame=False, parser="auto")
    X = dataset.data.astype(np.float32)
    # Target is strings '0'/'1' from OpenML — convert to int
    y = dataset.target.astype(int)
    feature_names = list(dataset.feature_names)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # preserve class ratio in both splits
    )
    return X_train, X_val, y_train, y_val, feature_names

# ---------------------------------------------------------------------------
# Evaluation — this is the ground-truth metric
# ---------------------------------------------------------------------------

def evaluate(model, X_val, y_val) -> float:
    """Returns macro F1 score (higher is better).

    model must implement a predict(X) method.
    """
    preds = model.predict(X_val)
    return float(f1_score(y_val, preds, average="macro"))
