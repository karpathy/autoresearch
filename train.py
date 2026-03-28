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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

from prepare import load_data, evaluate, METRIC

t_start = time.time()

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Outlier removal (training only)
# ---------------------------------------------------------------------------

outlier_mask = ~((X_train["GrLivArea"] > 4000) & (y_train < 200000))
X_train = X_train[outlier_mask].reset_index(drop=True)
y_train = y_train[outlier_mask].reset_index(drop=True)

y_train_log = np.log1p(y_train)

# ---------------------------------------------------------------------------
# Target encoding for Neighborhood
# ---------------------------------------------------------------------------

def target_encode_oof(train_col, train_target, test_col, n_splits=5, smoothing=10):
    global_mean = train_target.mean()
    train_enc = np.zeros(len(train_col))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for _, (tr_idx, val_idx) in enumerate(kf.split(train_col)):
        fold_map = train_target.iloc[tr_idx].groupby(train_col.iloc[tr_idx]).agg(["mean", "count"])
        fold_map["smooth"] = (
            (fold_map["mean"] * fold_map["count"] + global_mean * smoothing)
            / (fold_map["count"] + smoothing)
        )
        train_enc[val_idx] = train_col.iloc[val_idx].map(fold_map["smooth"]).fillna(global_mean).values
    full_map = train_target.groupby(train_col).agg(["mean", "count"])
    full_map["smooth"] = (
        (full_map["mean"] * full_map["count"] + global_mean * smoothing)
        / (full_map["count"] + smoothing)
    )
    test_enc = test_col.map(full_map["smooth"]).fillna(global_mean).values
    return train_enc, test_enc

X_train = X_train.copy()
X_test  = X_test.copy()
X_train["Neighborhood_enc"], X_test["Neighborhood_enc"] = target_encode_oof(
    X_train["Neighborhood"], y_train_log, X_test["Neighborhood"]
)

# ---------------------------------------------------------------------------
# Feature Engineering — extended interactions + more derived features
# ---------------------------------------------------------------------------

def add_features(df):
    df = df.copy()
    # Age features
    df["house_age"]      = df["YrSold"] - df["YearBuilt"]
    df["remodel_age"]    = df["YrSold"] - df["YearRemodAdd"]
    df["is_remodeled"]   = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # Area
    bsmt_sf = df["TotalBsmtSF"].fillna(0)
    df["total_sf"]       = bsmt_sf + df["1stFlrSF"] + df["2ndFlrSF"]
    df["bsmt_sf"]        = bsmt_sf
    df["has_bsmt"]       = (bsmt_sf > 0).astype(int)

    # Bathrooms
    bsmt_full = df.get("BsmtFullBath", pd.Series(0, index=df.index)).fillna(0)
    bsmt_half = df.get("BsmtHalfBath", pd.Series(0, index=df.index)).fillna(0)
    df["total_baths"]    = df["FullBath"] + 0.5 * df["HalfBath"] + bsmt_full + 0.5 * bsmt_half

    # Garage
    garage_cars = df.get("GarageCars", pd.Series(0, index=df.index)).fillna(0)
    garage_area = df.get("GarageArea", pd.Series(0, index=df.index)).fillna(0)
    df["garage_cars"]    = garage_cars
    df["garage_area"]    = garage_area
    df["has_garage"]     = (garage_area > 0).astype(int)

    # Quality interactions
    df["qual_sf"]        = df["OverallQual"] * df["GrLivArea"]
    df["qual_total_sf"]  = df["OverallQual"] * df["total_sf"]
    df["qual2"]          = df["OverallQual"] ** 2
    df["cond2"]          = df["OverallCond"] ** 2
    df["qual_cond"]      = df["OverallQual"] * df["OverallCond"]

    # Ratio features
    df["liv_lot_ratio"]  = df["GrLivArea"] / (df["LotArea"].clip(lower=1))
    df["sf_per_room"]    = df["GrLivArea"] / (df["TotRmsAbvGrd"].clip(lower=1))
    df["bsmt_living_ratio"] = bsmt_sf / (df["GrLivArea"].clip(lower=1))

    # Age x quality
    df["age_qual"]       = df["house_age"] * df["OverallQual"]
    df["remod_qual"]     = df["remodel_age"] * df["OverallQual"]

    # Neighborhood price x quality interaction (using target-enc)
    df["neigh_qual"]     = df["Neighborhood_enc"] * df["OverallQual"]
    df["neigh_sf"]       = df["Neighborhood_enc"] * df["GrLivArea"]

    # Porch
    porch = sum(df.get(c, pd.Series(0, index=df.index)) for c in
                ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"])
    df["total_porch"]    = porch

    return df

X_train = add_features(X_train)
X_test  = add_features(X_test)

num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# ---------------------------------------------------------------------------
# Model — Ridge with tuned alpha
# ---------------------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    Ridge()),
])

param_grid = {"regressor__alpha": [5.0, 10.0, 20.0, 50.0, 100.0]}
grid_search = GridSearchCV(
    model, param_grid, cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train_log)
best_model  = grid_search.best_estimator_
best_alpha  = grid_search.best_params_["regressor__alpha"]
cv_rmse_log = -grid_search.best_score_

y_pred_log = best_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)

val_rmse = evaluate(y_test, y_pred)

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

num_features_out = best_model.named_steps["preprocessor"].transform(X_train[:1]).shape[1]

print("---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"cv_rmse_log:      {cv_rmse_log:.6f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_features:     {num_features_out}")
print(f"model:            Ridge(alpha={best_alpha}) + extended interactions + neigh_qual neigh_sf")
print(f"train_rows:       {len(X_train)}")
print(f"test_rows:        {len(X_test)}")
