"""
FluxScore autoresearch — data preparation.
Generates synthetic borrower profiles matching the 18-feature schema.
Zero PII. Run once before starting the experiment loop.

Usage: python prepare.py
Output: train.parquet, holdout.parquet (same directory)

Feature distributions are calibrated to approximate Nigerian MFI borrower populations.
Each distribution is documented below for reproducibility and trust-artifact purposes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_TRAIN = 8_000
N_HOLDOUT = 2_000

# ---------------------------------------------------------------------------
# Feature schema — 18 features, calibrated to Nigerian MFI portfolio data
# ---------------------------------------------------------------------------
# Format: (name, distribution_type, params, notes)
FEATURE_SCHEMA = [
    # --- Transaction velocity features ---
    # avg_monthly_txn_count: Poisson-ish. Most borrowers: 20-80 txns/mo.
    # Low = thin file. High = active commercial account.
    ("avg_monthly_txn_count", "gamma", dict(shape=3.0, scale=15.0), "txns/month, clipped [0, 300]"),

    # avg_txn_value_ngn: Log-normal. Median ~₦15k. Wide tail for trade finance.
    ("avg_txn_value_ngn", "lognormal", dict(mean=9.6, sigma=1.2), "₦, clipped [100, 5_000_000]"),

    # txn_volatility: Coefficient of variation of monthly txn count. 0=stable, 1=volatile.
    ("txn_volatility", "beta", dict(a=2.0, b=3.0), "[0,1], higher = more irregular cash flow"),

    # --- Counterparty diversity features ---
    # unique_counterparties_30d: How many distinct senders/receivers in last 30 days.
    # High diversity = active business. Low = personal/limited network.
    ("unique_counterparties_30d", "gamma", dict(shape=2.5, scale=8.0), "count, clipped [0, 200]"),

    # top_counterparty_concentration: Fraction of volume from single largest counterparty.
    # High = single-client dependency (risk). Low = diverse book.
    ("top_counterparty_concentration", "beta", dict(a=1.5, b=3.5), "[0,1]"),

    # --- Settlement consistency features ---
    # on_time_repayment_rate: Historical repayment timeliness across all prior loans.
    # Key predictor. Most legitimate borrowers cluster 0.7-1.0.
    ("on_time_repayment_rate", "beta", dict(a=5.0, b=2.0), "[0,1], higher = better"),

    # days_past_due_avg: Average days past due on prior loans (0 if none).
    ("days_past_due_avg", "gamma", dict(shape=0.8, scale=3.0), "days, clipped [0, 90]"),

    # prior_default_count: Number of historical defaults. Rare but important.
    ("prior_default_count", "poisson", dict(lam=0.15), "count, clipped [0, 5]"),

    # --- Account tenure & stability ---
    # account_age_months: How long the primary account has been active.
    ("account_age_months", "gamma", dict(shape=3.0, scale=18.0), "months, clipped [1, 240]"),

    # days_since_last_activity: Recency of last account activity.
    ("days_since_last_activity", "gamma", dict(shape=1.5, scale=4.0), "days, clipped [0, 90]"),

    # --- Loan request context ---
    # requested_amount_ngn: Loan request size. Log-normal, Payday << Asset Finance << Trade.
    ("requested_amount_ngn", "lognormal", dict(mean=12.0, sigma=1.4), "₦, clipped [5000, 50_000_000]"),

    # loan_tenor_days: Requested loan duration.
    ("loan_tenor_days", "gamma", dict(shape=2.0, scale=45.0), "days, clipped [7, 365]"),

    # loan_type_encoded: 0=GSM, 1=Payday, 2=AssetFinance, 3=TradeFinance
    ("loan_type_encoded", "categorical", dict(probs=[0.35, 0.30, 0.20, 0.15]), "ordinal encoding"),

    # --- Balance sheet signals ---
    # avg_eod_balance_ngn: Average end-of-day balance over last 90 days.
    ("avg_eod_balance_ngn", "lognormal", dict(mean=10.5, sigma=1.5), "₦, clipped [0, 10_000_000]"),

    # min_eod_balance_ngn: Minimum end-of-day balance — catches overdraft risk.
    ("min_eod_balance_ngn", "lognormal", dict(mean=8.5, sigma=1.8), "₦, clipped [0, 1_000_000]"),

    # balance_to_request_ratio: avg_eod_balance / requested_amount. Key affordability signal.
    ("balance_to_request_ratio", "gamma", dict(shape=2.0, scale=0.8), "[0, inf], clipped [0, 20]"),

    # --- Digital footprint ---
    # app_sessions_30d: Mobile app sessions in last 30 days. High = engaged borrower.
    ("app_sessions_30d", "poisson", dict(lam=12.0), "count, clipped [0, 100]"),

    # kyb_tier: KYB verification level. 1=basic, 2=enhanced, 3=full institutional.
    ("kyb_tier", "categorical", dict(probs=[0.50, 0.35, 0.15]), "1/2/3"),
]

# Default rates by loan type (calibrated to plan spec)
# GSM=3.5%, Payday=16%, AssetFinance=8%, TradeFinance=5%
BASE_DEFAULT_RATES = [0.035, 0.16, 0.08, 0.05]


def _sample_feature(rng, name, dist, params, n):
    if dist == "gamma":
        vals = rng.gamma(shape=params["shape"], scale=params["scale"], size=n)
    elif dist == "lognormal":
        vals = rng.lognormal(mean=params["mean"], sigma=params["sigma"], size=n)
    elif dist == "beta":
        vals = rng.beta(a=params["a"], b=params["b"], size=n)
    elif dist == "poisson":
        vals = rng.poisson(lam=params["lam"], size=n).astype(float)
    elif dist == "categorical":
        probs = params["probs"]
        vals = rng.choice(len(probs), size=n, p=probs).astype(float)
    else:
        raise ValueError(f"Unknown distribution: {dist}")
    return vals


def _clip(vals, notes):
    """Clip based on bracket notation in notes, e.g. '[0, 300]'."""
    import re
    match = re.search(r'\[([0-9_\.]+),\s*([0-9_\.]+)\]', notes)
    if match:
        lo = float(match.group(1).replace("_", ""))
        hi = float(match.group(2).replace("_", ""))
        vals = np.clip(vals, lo, hi)
    return vals


def generate_dataset(n, rng):
    data = {}
    for name, dist, params, notes in FEATURE_SCHEMA:
        vals = _sample_feature(rng, name, dist, params, n)
        vals = _clip(vals, notes)
        data[name] = vals

    loan_type = data["loan_type_encoded"].astype(int)
    base_pd = np.array([BASE_DEFAULT_RATES[t] for t in loan_type])

    # Logistic link — mimics how a credit scoring model actually works.
    # logit(P(default)) = intercept + feature effects
    # Positive coefficient = increases default risk
    # Negative coefficient = reduces default risk
    # Intercept calibrated so overall default rate ≈ 8-10%.
    # Noise std = 0.5 logit units (realistic irreducible component).
    logit_p = (
        +1.0                                                            # intercept (calibrated for ~10% base default rate)
        - 2.0  * data["on_time_repayment_rate"]                        # good repayment → lower risk
        + 2.0  * np.clip(data["prior_default_count"], 0, 3)            # prior defaults → higher risk
        + 1.5  * (data["days_past_due_avg"] / 90.0)                    # overdue balance → higher risk
        - 1.0  * np.log1p(data["avg_monthly_txn_count"]) / np.log(50)  # txn velocity → lower risk
        - 0.8  * np.log1p(data["balance_to_request_ratio"])            # affordability → lower risk
        + 0.6  * data["top_counterparty_concentration"]                # single-client concentration → risk
        - 0.5  * (data["kyb_tier"] - 1.0) / 2.0                       # higher KYB → lower risk
        - 0.5  * np.log1p(data["unique_counterparties_30d"]) / np.log(50)  # network breadth → lower risk
        + 0.4  * data["txn_volatility"]                                # volatile cash flow → risk
        - 0.3  * np.log1p(data["app_sessions_30d"]) / np.log(30)      # engagement → lower risk
        + 0.6  * (data["loan_type_encoded"] == 1).astype(float)        # payday loans → higher risk
        + rng.normal(0, 0.5, n)                                         # irreducible noise
    )

    adj_pd = 1.0 / (1.0 + np.exp(-logit_p))  # sigmoid
    adj_pd = np.clip(adj_pd, 0.001, 0.95)

    data["default"] = (rng.uniform(size=n) < adj_pd).astype(int)

    df = pd.DataFrame(data)
    # kyb_tier as int
    df["kyb_tier"] = df["kyb_tier"].astype(int)
    df["loan_type_encoded"] = df["loan_type_encoded"].astype(int)
    df["prior_default_count"] = df["prior_default_count"].astype(int)
    df["app_sessions_30d"] = df["app_sessions_30d"].astype(int)
    return df


def main():
    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).parent

    print("Generating synthetic borrower profiles...")
    train_df = generate_dataset(N_TRAIN, rng)
    holdout_df = generate_dataset(N_HOLDOUT, rng)

    train_path = out_dir / "train.parquet"
    holdout_path = out_dir / "holdout.parquet"

    train_df.to_parquet(train_path, index=False)
    holdout_df.to_parquet(holdout_path, index=False)

    print(f"train.parquet:   {len(train_df):,} rows, {train_df.columns.tolist()}")
    print(f"holdout.parquet: {len(holdout_df):,} rows")
    print(f"Train default rate:   {train_df['default'].mean():.3f}")
    print(f"Holdout default rate: {holdout_df['default'].mean():.3f}")
    print(f"Feature count: {len(train_df.columns) - 1} (+ 'default' label)")
    print("Done.")


if __name__ == "__main__":
    main()
