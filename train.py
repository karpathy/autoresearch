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
import xgboost as xgb

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

# Numeric: median imputation (XGBoost handles missing natively but pipeline needs it)
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

# Categorical: ordinal encode (XGBoost handles categories as ints)
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# ---------------------------------------------------------------------------
# Model — XGBoost
# ---------------------------------------------------------------------------

regressor = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="rmse",
)

# ---------------------------------------------------------------------------
# Cross-validation (on log-transformed target)
# ---------------------------------------------------------------------------

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

cv_scores = cross_val_score(
    xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    ),
    X_train_proc, y_train_log,
    cv=5, scoring="neg_root_mean_squared_error",
)
cv_rmse_log = -cv_scores.mean()

# ---------------------------------------------------------------------------
# Final fit + evaluation (with early stopping on a validation split)
# ---------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_proc, y_train_log, test_size=0.15, random_state=42
)

regressor.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

y_pred_log = regressor.predict(X_test_proc)
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
print(f"model:            XGBRegressor")
print(f"train_rows:       {len(X_train)}")
print(f"test_rows:        {len(X_test)}")
