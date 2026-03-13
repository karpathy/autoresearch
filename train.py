"""
Car price prediction with XGBoost. Agent modifies this file.
Usage: uv run train.py
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from prepare import load_raw, get_train_val_split, evaluate_model, CAT_COLS, NUM_COLS, TARGET

t_start = time.time()

# ---------------------------------------------------------------------------
# Feature engineering (modify this)
# ---------------------------------------------------------------------------

df = load_raw()
train_df, val_df = get_train_val_split(df)

# Native XGBoost categorical support
for col in CAT_COLS:
    train_df[col] = train_df[col].astype("category")
    val_df[col] = val_df[col].astype("category")

FEATURE_COLS = CAT_COLS + NUM_COLS

X_train = train_df[FEATURE_COLS]
y_train = np.log1p(train_df[TARGET])
X_val = val_df[FEATURE_COLS]
y_val = val_df[TARGET]

# ---------------------------------------------------------------------------
# Model (modify this)
# ---------------------------------------------------------------------------

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

y_pred = np.expm1(model.predict(X_val))
val_rmse = evaluate_model(y_val, y_pred)

t_end = time.time()

print("---")
print(f"val_rmse:         {val_rmse:.2f}")
print(f"training_seconds: {t_end - t_start:.1f}")
print(f"num_samples:      {len(df)}")
print(f"num_features:     {X_train.shape[1]}")
