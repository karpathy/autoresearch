# Experiment Log — autotrader/mar20b

**Goal**: Recalibrate pipeline for expanding windows with sample weight decay.
**Starting recipe**: 36 features, 9 monotonic constraints, 2-model HGB ensemble, power transform x^0.7, 0.3 dampening, EMA span 45, 0.7x partial demeaning. Prior best (sliding windows): 0.6031.
**Infrastructure change**: Expanding windows (train on all data from 2018 to eval year) + exponential sample weight decay on older samples.

---

## d10b30b — baseline (expanding windows)
**Hypothesis:** Establish baseline under new expanding window + sample weight decay infrastructure.
**Result:** Score -1.9313, sharpe_min -0.2522, max_dd -2.6%, 121 trades, 6/8 consistency, holdout CAUTION. Keep (baseline).
**Observation:** Massive regression from sliding-window best (0.6031). sharpe_min is negative — at least one window has a losing strategy. Trade count dropped from 230→121 (more training data may be shifting prediction magnitudes). The 0.7x partial demeaning was tuned for 3-year windows and is likely miscalibrated for expanding windows with decay weights. Next: sweep partial demeaning (0.5x, 0.8x).

## 3176c60 — partial demeaning 0.5x
**Hypothesis:** Lower demeaning fraction may be better calibrated for expanding windows with different bias profile.
**Result:** Score -8.3149, sharpe_min -0.7374, max_dd -2.8%, 137 trades, 5/8 consistency, holdout OK. Discard.
**Observation:** Much worse than baseline (-1.93). Less demeaning = more residual bias = worse predictions. The expanding window bias needs MORE removal, not less. Next: test 0.8x.

