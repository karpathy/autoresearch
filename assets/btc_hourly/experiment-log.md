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

## 9b9e5b4 — partial demeaning 0.8x
**Hypothesis:** Expanding windows have more bias needing removal. 0.8x should help.
**Result:** Score -0.6283, sharpe_min -0.1128, max_dd -2.6%, 121 trades, 7/8 consistency, holdout CAUTION. Keep (best so far).
**Observation:** Clear monotonic improvement: 0.5x→-8.31, 0.7x→-1.93, 0.8x→-0.63. Consistency improved to 7/8. Optimum is at or above 0.8x. Next: narrow with 0.9x.

## a4d4d25 — partial demeaning 0.9x
**Hypothesis:** Continuing upward sweep. 0.8x improved over 0.7x, so 0.9x may be even better.
**Result:** Score -0.4678, sharpe_min -0.0720, max_dd -2.6%, 117 trades, 6/8 consistency, holdout CAUTION. Keep (best score).
**Observation:** Better score than 0.8x (-0.47 vs -0.63) but consistency dropped (6/8 vs 7/8). Sharpe still negative but approaching zero. Trend: 0.5x→-8.31, 0.7x→-1.93, 0.8x→-0.63, 0.9x→-0.47. Still improving. Next: test 1.0x (full demeaning).

## 6988541 — full demeaning 1.0x
**Hypothesis:** Score still improving at 0.9x — full demeaning might be optimal for expanding windows.
**Result:** Score -2.3782, sharpe_min -0.3105, max_dd -2.7%, 115 trades, 6/8 consistency, holdout CAUTION. Discard.
**Observation:** Full demeaning overshoots. The curve peaks at ~0.9x:
- 0.5x→-8.31, 0.7x→-1.93, 0.8x→-0.63, **0.9x→-0.47**, 1.0x→-2.38.
Best demeaning fraction is 0.9x. Reverting to 0.9x for EMA span sweep. Next: EMA span 40 and 50.

## 31f23ba — EMA span 40 (with 0.9x demeaning)
**Hypothesis:** More training data may shift optimal smoothing window. Test EMA 40 (from 45).
**Result:** Score -3.4790, sharpe_min -0.4460, max_dd -2.8%, 133 trades, 5/8 consistency, holdout CAUTION. Discard.
**Observation:** Much worse than EMA 45 (-3.48 vs -0.47). Shorter EMA adds noise. Same pattern as mar20 (EMA 30→35→40 all worse than 45). Next: test EMA 50.

## ad00832 — EMA span 50 (with 0.9x demeaning)
**Hypothesis:** Longer EMA may help with more training data producing noisier predictions.
**Result:** Score -1.1587, sharpe_min -0.1513, max_dd -2.5%, 105 trades, 6/8 consistency, holdout CAUTION. Discard.
**Observation:** Worse than EMA 45 (-1.16 vs -0.47). Too much smoothing kills trade count (105 vs 117). EMA sweep: 40→-3.48, **45→-0.47**, 50→-1.16. EMA 45 confirmed optimal. Next: test dampening 0.35.

## f06f1ff — dampening 0.35 (with 0.9x demeaning, EMA 45)
**Hypothesis:** Bear market data in every window should make model more robust. May handle slightly higher exposure.
**Result:** Score -7.4658, sharpe_min -1.4627, max_dd -5.5%, 245 trades, 4/8 consistency, holdout OK. Discard.
**Observation:** Catastrophic. Same as mar20 — 0.35 kills the worst window (sharpe -1.46). Dampening permanently confirmed at 0.3. Reverting.

---

**Phase 1 recalibration complete.** Best config: 0.9x demeaning, EMA 45, dampening 0.3. Best score: -0.4678.

All scores remain negative. The pipeline parameters are recalibrated but the model needs structural improvement to work with expanding windows. Moving to Phase 2.

## 51a4784 — early_stopping=False
**Hypothesis:** With expanding windows (26k+ samples), early stopping may truncate training. Disable to use all 300/500 iterations.
**Result:** Score -4.4577, sharpe_min -0.3953, max_dd -2.4%, 117 trades, 5/8 consistency, holdout OK. Discard.
**Observation:** Much worse. Early stopping's validation split was helping prevent overfitting. The 10% held-out validation data is more valuable than extra iterations. Reverting. Next: try power transform exponent adjustment.

## 740034e — power transform exponent 0.5 (from 0.7)
**Hypothesis:** Lower exponent amplifies weak predictions, increasing trade count from 117.
**Result:** Score -6.2818, sharpe_min -0.9985, max_dd -3.3%, 213 trades, 4/8 consistency, holdout OK. Discard.
**Observation:** Trade count increased (213 vs 117) but sharpe collapsed. Amplifying noisy predictions destroys signal quality. Next: try no power transform (exponent 1.0 = linear).

