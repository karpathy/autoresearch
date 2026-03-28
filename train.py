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
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

from prepare import load_data, evaluate, METRIC

t_start = time.time()

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_features(df):
    df = df.copy()
    df["house_age"]   = df["YrSold"] - df["YearBuilt"]
    df["remodel_age"] = df["YrSold"] - df["YearRemodAdd"]
    df["total_sf"]    = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]
    df["total_baths"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                         + df.get("BsmtFullBath", pd.Series(0, index=df.index)).fillna(0)
                         + 0.5 * df.get("BsmtHalfBath", pd.Series(0, index=df.index)).fillna(0))
    df["qual_sf"]     = df["OverallQual"] * df["GrLivArea"]
    return df

X_train = add_features(X_train)
X_test  = add_features(X_test)

# Log-transform target
y_train_log = np.log1p(y_train)

# Identify column types
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# ---------------------------------------------------------------------------
# Model — LightGBM
# ---------------------------------------------------------------------------

model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

cv_scores = cross_val_score(
    model, X_train_proc, y_train_log,
    cv=5, scoring="neg_root_mean_squared_error",
)
cv_rmse_log = -cv_scores.mean()

# ---------------------------------------------------------------------------
# Final fit + evaluation
# ---------------------------------------------------------------------------

model.fit(X_train_proc, y_train_log)

y_pred_log = model.predict(X_test_proc)
y_pred     = np.expm1(y_pred_log)

val_rmse = evaluate(y_test, y_pred)

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"cv_rmse_log:      {cv_rmse_log:.6f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_features:     {X_train_proc.shape[1]}")
print(f"model:            LGBMRegressor")
print(f"train_rows:       {len(X_train)}")
print(f"test_rows:        {len(X_test)}")
