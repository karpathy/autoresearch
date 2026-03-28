"""
Autoresearch classic ML experiment script.
This is the ONLY file the agent edits.

Usage: uv run train.py

Everything is fair game:
  - Feature engineering (imputation, encoding, transforms, new features)
  - Model selection (Ridge, RandomForest, XGBoost, LightGBM, stacking, ...)
  - Hyperparameter tuning (manual, GridSearch, Optuna, ...)
  - Cross-validation strategy

The only constraint: the script must run without crashing and call
evaluate(y_test, y_pred) using the fixed harness from prepare.py.
"""

import time
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

from prepare import load_data, evaluate, METRIC

t_start = time.time()

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# Log-transform target to reduce skew (common for price prediction)
y_train_log = np.log1p(y_train)

# Identify column types
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

# Numeric: median imputation + standard scaling
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# Categorical: constant imputation + one-hot encoding
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    Ridge(alpha=10.0)),
])

# ---------------------------------------------------------------------------
# Cross-validation (on log-transformed target)
# ---------------------------------------------------------------------------

cv_scores = cross_val_score(
    model, X_train, y_train_log,
    cv=5, scoring="neg_root_mean_squared_error",
)
# RMSE in log space; convert back to original scale for interpretability
cv_rmse_log = -cv_scores.mean()

# ---------------------------------------------------------------------------
# Final fit + evaluation
# ---------------------------------------------------------------------------

model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)
y_pred     = np.expm1(y_pred_log)   # invert log1p

val_rmse = evaluate(y_test, y_pred)

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary (format preserved for grep and results.tsv)
# ---------------------------------------------------------------------------

num_features_out = model.named_steps["preprocessor"].transform(X_train[:1]).shape[1]

print("---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"cv_rmse_log:      {cv_rmse_log:.6f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_features:     {num_features_out}")
print(f"model:            {model.named_steps['regressor'].__class__.__name__}")
print(f"train_rows:       {len(X_train)}")
print(f"test_rows:        {len(X_test)}")
