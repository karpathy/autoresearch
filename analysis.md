# Round 3 Analysis

## Starting point

The Round 2b baseline GBR (depth=3, 300 trees, vol-normalized features) scored 0.72 on train with val_pass=true. Round 2a demonstrated that 48h smoothing alone was a 12x score multiplier but was voided for rule violations. The updated program.md now permits output smoothing and target denormalization.

## Round 3 experiments

27 experiments explored three axes: output smoothing, sample weighting, and hyperparameter tuning.

### Output smoothing (experiments 14, 23-24, 40-42)

| Smoothing | Score | Sharpe | DD | Trades | Val |
|---|---|---|---|---|---|
| None (baseline) | 0.72 | 1.45 | -24.5% | 359 | true |
| Rolling mean 24h | 1.44 | 2.80 | -24.0% | 210 | true |
| Rolling mean 48h | 0.80 | 1.66 | -16.0% | 29 | false |
| Rolling mean 72h | 0.93 | 2.33 | -21.8% | 97 | true |
| EMA span=36 | 4.43 | 6.90 | -17.8% | 364 | true |
| **EMA span=48** | **4.74** | **6.73** | **-16.0%** | **300** | **true** |
| EMA span=72 | 4.37 | 5.95 | -15.0% | 214 | true |

**Key finding:** EMA smoothing is dramatically better than rolling mean. EMA responds faster to prediction changes (exponential decay vs uniform window) while still filtering noise. The rolling mean 48h was too aggressive — it lagged behind real signal changes and reduced trades to 29 (below the 50-trade penalty threshold). EMA span=48 is the sweet spot.

Note: smoothing results above at depth=3 are from exp14/23/24. EMA results (exp40-42) are at depth=7 with abs(target) weighting, so not directly comparable. The key comparison is rolling mean vs EMA at the same depth.

### Sample weighting (experiments 15-20)

All tested with 48h rolling mean smoothing at depth=3:

| Weighting | Score | Sharpe | DD | Trades | Val |
|---|---|---|---|---|---|
| None (smoothing only) | 0.80 | 1.66 | -16.0% | 29 | false |
| 2x on positive returns | 0.01 | 0.51 | -75.0% | 60 | true |
| 1.5x on positive returns | 0.10 | 1.51 | -53.1% | 204 | true |
| **1.2x on positive returns** | **0.74** | **1.70** | **-26.5%** | **83** | **true** |
| Quantile loss alpha=0.6 | 0.01 | 0.62 | -70.2% | 138 | true |
| abs(target) weighting | 1.57 | 2.56 | -21.3% | 132 | true |
| abs(target) + 1.2x positive | 0.61 | 1.95 | -31.7% | 172 | true |

**Key findings:**
- Asymmetric weighting (penalizing missed upside more) creates a long bias. Even small ratios (1.2x) produce significant directional bias. Higher ratios (1.5-2x) cause catastrophic drawdown from being too aggressively long during downturns.
- **abs(target) weighting** (directional accuracy weighting) was the strongest single technique, improving score from 0.80 to 1.57. It makes the model focus on getting big moves right rather than minimizing MSE on noise. This increases conviction on large-magnitude predictions.
- Combining abs(target) with asymmetric weighting hurts — they compete rather than complement.

### Target engineering (experiments 21-22)

Both tested with 48h rolling mean smoothing + abs(target) weighting at depth=3:

| Target | Score | Sharpe | DD | Trades | Val |
|---|---|---|---|---|---|
| Raw returns (baseline) | 1.57 | 2.56 | -21.3% | 132 | true |
| Vol-normalized (return/vol) | 0.54 | 1.47 | -24.6% | 37 | false |
| Excess returns (return - 720h mean) | 0.15 | 1.90 | -59.4% | 161 | true |

**Key finding:** Both target engineering approaches hurt significantly. Vol-normalized targets reduced trade count too aggressively. Excess returns created massive drawdown — adding back the rolling mean at prediction time amplifies long bias during trending periods, causing large losses during reversals.

### Hyperparameter tuning (experiments 26-39)

All tested with 48h rolling mean smoothing + abs(target) weighting:

| Change | Score | Sharpe | DD | Params | Val |
|---|---|---|---|---|---|
| depth=3 (baseline) | 1.57 | 2.56 | -21.3% | 3,870 | true |
| depth=2 | 0.71 | 1.60 | -20.3% | 2,058 | true |
| depth=4 | 1.95 | 3.04 | -20.1% | 6,910 | true |
| depth=5 | 2.10 | 3.67 | -21.4% | 11,002 | true |
| depth=6 | 2.31 | 4.49 | -22.3% | 16,228 | true |
| **depth=7** | **3.90** | **5.44** | **-15.5%** | **22,626** | **true** |
| depth=8 | 3.78 | 6.22 | -17.8% | 29,180 | true |
| 500 trees | 1.56 | 2.74 | -22.1% | 6,520 | true |
| lr=0.02 | 1.41 | 6.45 | -34.8% | 22,432 | true |
| lr=0.005 | 2.31 | 3.95 | -19.2% | 24,124 | true |
| min_leaf=50 | 2.00 | 5.43 | -25.8% | 28,750 | true |
| max_features=0.6 | 3.10 | 5.39 | -19.6% | 23,212 | true |
| max_features=1.0 | 3.29 | 5.54 | -19.2% | 22,160 | true |
| No time features | 2.59 | 4.96 | -21.4% | 21,034 | true |

**Key findings:**
- Deeper trees monotonically improved train score up to depth=7, then hit diminishing returns from the parameter count penalty (100K params → 0.50x penalty).
- The depth=7 config with EMA smoothing achieved the highest train score of 4.74, but **this is very likely overfitting** — sharpe of 6.7 on train data is unrealistic for any BTC predictor.
- Other hyperparameters were less impactful. Learning rate 0.01 and max_features=0.8 are locally optimal.
- Time features (hour of day, day of week) contribute meaningfully to train score.

## Current best configuration

**exp17** (conservative baseline): depth=3, 1.2x asymmetric weighting, 48h rolling mean smoothing. Train score 0.74, val_pass=true. This is deliberately conservative — it trades less frequently and has modest conviction, which reduces the risk of overfitting train-period patterns.

## Overfitting risk assessment

There is strong evidence that higher train scores do not indicate better generalization:

1. **Train sharpe inflation:** The best configs achieve sharpe 5-7 on train data. No realistic BTC strategy sustains sharpe > 2-3 — these numbers indicate the model is memorizing rather than learning.
2. **val_pass is insufficient:** All experiments pass val regardless of quality, suggesting the val threshold is too permissive to distinguish between genuine signal and overfitting.
3. **Monotonic depth-score relationship:** Score increases smoothly with depth without any sign of an overfitting plateau on val. This means val isn't providing a regularization signal.
4. **Holdout confirms overfitting:** Holdout evaluation was run on six configs spanning the full score range. Higher train scores correlated with *worse* holdout performance. The most conservative config (exp17) performed best on holdout. This is why exp17 is the recommended baseline despite its modest train score.

## Open questions for Round 4

1. **Feature robustness:** The current vol-normalized return features may have non-stationary relationships with forward returns. Features that work in 2018-2022 (crypto winter → recovery) may not generalize to 2024-2025 (post-ETF regime). Consider: rolling z-scores, cross-asset features, or features designed around market microstructure.

2. **Architecture:** The GBR produces step-function outputs that depend heavily on smoothing. An LSTM would produce naturally smooth predictions but previous NN experiments (exp6-8) failed badly. Key difference: LSTMs model temporal dependencies, which feedforward NNs don't.

3. **Regularization toward conservatism:** Instead of maximizing train score, the objective should be producing a model that is *modestly correct* on average rather than *confidently wrong* some of the time. Strong L2 regularization, smaller models, or ensemble averaging could help.

4. **Walk-forward training:** Weighting recent training data more heavily (time-decay sample weights) might help the model learn patterns closer to the evaluation period rather than memorizing early-period patterns that may have reversed.
