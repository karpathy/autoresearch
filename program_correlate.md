# RiskWise Correlation Analysis — Agent Instructions

This is an autonomous analysis pipeline. The agent connects to the RiskWise Postgres database, discovers all available risk indices, and systematically finds correlations with a target company's financial outcomes (stock price). The goal: produce a rigorous, PhD-level statistical analysis proving that RiskWise indices are predictive of the client's business outcomes.

## Context

RiskWise constructs risk indices across heterogeneous data channels — supply chain volatility, maritime disruptions, policy shifts, etc. These indices are composed of sub-indices that track specific risk drivers. Clients pay for access to monitor emerging trends. **Your job is to prove the value** by finding which indices are correlated with, predictive of, or cointegrated with the client's stock price.

## Setup

1. **Agree on the target**: The user specifies a company ticker and name (e.g., `PEP` / `PepsiCo`).
2. **Verify DB access**: Confirm the Postgres connection works by running discovery queries.
3. **Verify market data**: Fetch stock data via yfinance and confirm it looks reasonable.
4. **Run the pipeline**: `uv run python -m correlate.run --ticker <TICKER> --name "<NAME>"`

## What the Pipeline Does

The pipeline is fully automated and runs these steps:

### Step 1: Index Discovery
Connects to Postgres and discovers all available risk indices — names, date ranges, observation counts. This tells us what we have to work with.

### Step 2: Market Data Fetch
Pulls historical stock data (default 5 years) for the target company. Derives log returns, realized volatility, and cumulative returns.

### Step 3: Alignment
Aligns index time series with market data on business-day frequency. Requires minimum 60 days of overlap.

### Step 4: Statistical Analysis (per index)
For each risk index, computes:
- **Pearson, Spearman, Kendall correlations** — three complementary measures
- **Optimal lead/lag** — scans ±90 days to find the lag maximizing |correlation|. Positive lag = index LEADS stock (predictive!)
- **Granger causality** — F-test for whether past index values predict future stock movements. Tests both directions.
- **Cointegration** — Engle-Granger test for long-run equilibrium relationships
- **Stationarity** — ADF test (non-stationary series need careful interpretation)
- **Rolling correlation** — 60-day window to assess stability over time
- **Bootstrap 95% CI** — non-parametric confidence intervals
- **Benjamini-Hochberg FDR** — multiple testing correction across all indices

### Step 5: Classification
Each index is classified as:
- **Leading** — moves before the stock (the money shot for sales)
- **Contemporaneous** — moves in tandem
- **Lagging** — follows the stock
- **Cointegrated** — shares long-run equilibrium

### Step 6: ML Predictive Models (Stage 2)
Using the significant indices identified in Stage 1 as features, the pipeline builds ML models:
- **Feature engineering**: raw values, lagged values (1/2/3/5/10/20d), rolling mean/std (5/10/20/60d), momentum, rate of change
- **Models**: Ridge, Lasso, ElasticNet, Random Forest, XGBoost
- **Validation**: Walk-forward (expanding window) cross-validation — strictly out-of-sample
- **Horizons**: 1-day, 5-day, 10-day, 20-day ahead prediction
- **Metrics**: Directional accuracy (can we predict up/down?), binomial significance test, R², RMSE vs naive baseline, Information Coefficient
- **Feature importance**: Permutation importance + SHAP values to identify which indices drive predictions

### Step 7: SHAP Explainability Analysis (Stage 3)
Deep SHAP-driven signal discovery using the best model from Stage 2:
- **Global SHAP rankings**: Mean |SHAP| per feature with percentile statistics — which signals genuinely drive predictions vs noise
- **SHAP interaction values**: TreeSHAP exact interactions reveal synergistic pairs of indices (e.g., supply chain volatility × maritime disruptions amplify each other's predictive power)
- **Temporal SHAP dynamics**: How feature importance shifts across time windows and market regimes (bull/bear/volatile/calm) — detects regime-dependent signals
- **Partial dependence curves**: Maps each index value to predicted outcome — identifies monotonic vs non-linear relationships
- **SHAP-driven feature selection**: Iterative pruning to find the minimal set of indices that preserves predictive accuracy — answers "which specific indices should the client subscribe to?"
- **Base index aggregation**: Rolls up all derived features (lags, rolling stats, momentum) back to the underlying RiskWise index names

### Step 8: Report Generation
Two reports:
1. **Technical Report** — full methodology, all test results (statistical + ML + SHAP), interaction tables, temporal regime analysis, partial dependence, feature selection. PhD-level rigor.
2. **Executive Summary** — headline findings, top predictive indices, lead times, ML accuracy, SHAP-driven insights ("these 5 indices are all you need"), regime-aware intelligence. Sales-oriented.

## Agent Loop (Post-Pipeline)

After the pipeline runs, the agent should:

1. **Review the reports** — read the generated markdown files
2. **Investigate anomalies** — if results look suspicious (e.g., absurdly high correlations), dig into the data
3. **Run additional analyses** — the agent can:
   - Re-run with `--use-levels` for price-level correlations
   - Adjust `--max-lag` for longer/shorter horizons
   - Run for specific sectors/industries
   - Write custom SQL via `db.run_query()` to explore index composition
   - Run `--stats-only` or `--ml-only` to iterate on a specific stage
   - Try `--ml-use-all-indices` to see if non-significant indices add ML value
   - Adjust `--ml-horizons` for different prediction windows
4. **Iterate on the report** — refine the narrative based on findings
5. **Cross-validate** — run against related tickers (e.g., if PEP, also check KO, MDLZ)

## Database Schema Notes

The default assumes a long-format table (`rw_indices`) with columns:
- `date` — observation date
- `index_name` — string identifier for the index
- `value` — numeric index value
- Optionally: `category`, `description`

If the schema differs, the agent should:
1. Run `db.discover_tables()` to see what's available
2. Run `db.discover_columns(table)` to inspect structure
3. Override config parameters or write custom queries

## Goal

**Sell the product.** The analysis should demonstrate:
1. RiskWise indices contain **predictive information** about the client's outcomes
2. Specific indices provide **quantified lead times** (e.g., "moves 15 days ahead")
3. The relationships are **statistically rigorous** (FDR-corrected, bootstrap CIs)
4. The relationships are **stable over time** (rolling correlations)
5. Some indices share **long-run equilibrium** with the stock (cointegration)
6. ML models trained on RiskWise indices achieve **above-chance directional accuracy** on out-of-sample data
7. Feature importance analysis identifies **which specific indices drive predictions** — justifying the subscription
8. Walk-forward validation proves this isn't overfitting — it's **real predictive power**

The executive summary should make a C-suite reader think: "We need this data."
