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

## ba417fb — power transform exponent 1.0 (linear)
**Hypothesis:** Remove power transform entirely. With expanding windows, raw predictions may not need reshaping.
**Result:** Score -2.7366, sharpe_min -0.4169, max_dd -2.4%, 103 trades, 7/8 consistency, holdout WARN. Discard.
**Observation:** Better consistency (7/8) but much worse score. Too few trades (103) without amplification. Power exponent confirmed at 0.7. 7 consecutive experiments without improvement since best (0.9x demeaning, -0.47). Invoking coach.

---
**Coach invoked (7 consecutive non-improvements)**

**DIAGNOSIS:** The agent has spent 11 experiments tuning pipeline parameters (demeaning, EMA, dampening, power transform, early stopping) on an unchanged model architecture. All scores remain negative because sharpe_min < 0 in every experiment. The fundamental problem is structural: this model was designed and tuned for sliding 3-year windows. With expanding windows (4-7 years of training data + weight decay), the model's rigid monotonic constraints prevent it from learning regime-dependent behavior. Nine features are forced monotonically increasing -- meaning "higher past return always predicts higher future return." This pure momentum assumption fails in at least one evaluation window (likely a mean-reversion or crash period), producing negative sharpe_min and dragging the entire score negative. No amount of pipeline parameter tuning can fix a model that is structurally forced to predict momentum in a regime where momentum fails.

**BOTTLENECK:** sharpe_min = -0.0720. Since sharpe_min is the base of the composite score, a negative base makes the entire score negative regardless of how good the other multipliers are. The dd_mult is fine (max_dd=-2.6% vs 25% hard penalty threshold). Trade count at 117 is adequate. Consistency at 6/8 means at least 2 subperiods are losing. The binding constraint is sharpe_min -- it must become positive (> 0) for any score improvement to matter. Currently 9 monotonic constraints lock the model into a momentum-only prediction mode across all regimes.

**PRESCRIPTION:** Remove all monotonic constraints and increase regularization to compensate. In `/Users/ipatterson/dev/autoresearch/assets/btc_hourly/train.py`, make these specific changes:

1. **Delete the monotonic constraint block** (lines 292-302). Replace the entire `mono_cst` section and the `monotonic_cst=mono_cst.tolist()` parameter in both models with nothing -- remove the parameter entirely.

2. **Increase min_samples_leaf from 600 to 1000** in both models. With 60k+ training samples in later windows, the model needs stronger regularization to avoid fitting noise.

3. **Increase l2_regularization from 3.0 to 5.0** in both models. Same reason.

Concrete code: Replace the block from `# --- Monotonic constraints` through the two model `.fit()` calls with:

```python
    # --- Train: two-model ensemble for diversity ---
    model_conservative = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=4,
        min_samples_leaf=1000,
        learning_rate=0.01,
        max_leaf_nodes=20,
        l2_regularization=5.0,
        random_state=42,
    )
    model_conservative.fit(features, targets, sample_weight=sample_weight)

    model_aggressive = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=4,
        min_samples_leaf=1000,
        learning_rate=0.01,
        max_leaf_nodes=20,
        max_features=0.8,
        l2_regularization=5.0,
        random_state=42,
    )
    model_aggressive.fit(features, targets, sample_weight=sample_weight)
```

This is a single conceptual change: "remove monotonic constraints and strengthen regularization to let the model learn regime-adaptive behavior." Test this as one experiment.

If sharpe_min becomes positive with this change but score is still modest, the next step is feature reduction (drop the weakest features to reduce noise). If sharpe_min stays negative, the next step is to try a fundamentally simpler model -- a single HistGradientBoostingRegressor with max_depth=3 and very few features (just the 6 return lookbacks + 2 volatility windows).

**RATIONALE:** Every experiment so far has kept the 9 monotonic constraints intact. These constraints were useful for sliding 3-year windows where each window saw a relatively homogeneous regime. But expanding windows mix bull markets, bear markets, and chop -- forcing "higher past return = higher predicted return" across all of these is the equivalent of forcing a momentum-only strategy. The model has regime features (vol ratio, directional efficiency, vol percentile, etc.) that could allow it to learn "momentum works in trending regimes, fade it in choppy/high-vol regimes" -- but the monotonic constraints prevent this. Removing them is the single highest-leverage structural change available. The increased regularization (min_samples_leaf 600->1000, l2 3.0->5.0) compensates for the additional model freedom to prevent overfitting on the larger training sets.

## e98d63c — remove monotonic constraints + increase regularization
**Hypothesis:** Coach: constraints force pure momentum which fails in mixed-regime expanding windows. Remove all 9 constraints, increase min_samples_leaf 600→1000, l2 3.0→5.0.
**Result:** Score -1.9492, sharpe_min -0.4746, max_dd -2.6%, 152 trades, 7/8 consistency, holdout CAUTION. Discard.
**Observation:** Worse than constrained (-1.95 vs -0.47). sharpe_min much worse (-0.47 vs -0.07). Constraints were PROTECTING signal quality, not limiting it. The model learns reversal patterns when unconstrained that hurt more than regime-adaptation helps. Trade count improved (152 vs 117) but at cost of signal quality. Constraints stay.

## 354cb9c — prediction clip ±1.0 (from ±2.0)
**Hypothesis:** sharpe_min at -0.07 is barely negative. Tighter clips may prevent extreme wrong-direction predictions.
**Result:** Score -2.4856, sharpe_min -0.2705, max_dd -2.0%, 125 trades, 5/8 consistency, holdout CAUTION. Discard.
**Observation:** Too aggressive — removes signal. Max_dd improved but sharpe and consistency both worse.

## 609a341 — target winsorization 3.0 sigma (from 5.0)
**Hypothesis:** Tighter winsorization reduces influence of crash/rally outliers. Helped in mar20.
**Result:** Score -349.6484, sharpe_min -1.3112, max_dd -0.9%, 36 trades, 3/8 consistency, holdout OK. Discard.
**Observation:** Catastrophic. Winsorization 3.0 compresses target distribution so much that model barely produces predictions above trading threshold. Only 36 trades.

## c6e162f — demeaning 0.95x
**Hypothesis:** Fine-tuning at the steep knee between 0.9x (-0.47) and 1.0x (-2.38).
**Result:** Score -1.1088, sharpe_min -0.1448, max_dd -2.6%, 123 trades, 6/8 consistency, holdout CAUTION. Discard.
**Observation:** Confirmed 0.9x is the peak: 0.8→-0.63, **0.9→-0.47**, 0.95→-1.11, 1.0→-2.38. 11 consecutive failures since best. Invoking coach.

---
**Coach invoked (11 consecutive non-improvements)**

**DIAGNOSIS:** The agent has run 15 experiments and every single one has kept the core model hyperparameters (max_depth=4, max_leaf_nodes=20, learning_rate=0.01, max_iter=300/500) completely untouched. All 15 experiments tuned only post-prediction pipeline parameters (demeaning, EMA, dampening, power, clip, winsor) or meta-settings (early stopping, monotonic constraints). The pipeline is now fully optimized at 0.9x demeaning, EMA 45, dampening 0.3, power 0.7 -- further tuning of these knobs is exhausted. The model itself, however, has never been adjusted for expanding windows. With expanding windows, later evaluation windows train on 50-60k+ samples spanning 4-7 years of mixed regimes. A model with max_depth=4 and max_leaf_nodes=20 has enough capacity to memorize regime-specific noise in this larger training set, which then hurts the worst evaluation window. The previous coach's prescription (remove constraints + increase regularization) conflated two changes and the constraint removal dominated -- it made things worse. The regularization increase alone was never tested with constraints intact.

**BOTTLENECK:** sharpe_min = -0.0720. This is the binding constraint. Since sharpe_min is negative, the entire composite score is negative regardless of other components. The score is tantalizingly close to flipping positive -- sharpe_min just needs to cross zero. max_drawdown at -2.6% is nowhere near the penalty threshold. Trade count at 117 is adequate. Consistency at 6/8 is decent. The ONLY thing preventing a positive score is the worst evaluation window having slightly negative Sharpe. A small reduction in model complexity could prevent overfitting in that worst window without degrading the other windows.

**NEAR-MISSES TO NOTE:**
- Exp3 (0.8x demean): score=-0.63, sharpe_min=-0.11, consistency=**7/8** -- better consistency than best
- Exp7 (EMA 50): score=-1.16, sharpe_min=-0.15 -- within range, different smoothing
- Exp15 (0.95x demean): score=-1.11, sharpe_min=-0.14 -- confirmed 0.9x peak
- None of these suggest a pipeline param adjustment would help. All pipeline params are at their optima.

**PRESCRIPTION:** Reduce max_depth from 4 to 3 in both models. This is a pure model complexity reduction that has never been tested. In `/Users/ipatterson/dev/autoresearch/assets/btc_hourly/train.py`, change both HistGradientBoostingRegressor instantiations:

```python
    # Conservative model: change max_depth=4 to max_depth=3
    model_conservative = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=3,          # was 4
        min_samples_leaf=600,
        learning_rate=0.01,
        max_leaf_nodes=20,
        l2_regularization=3.0,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )

    # Aggressive model: change max_depth=4 to max_depth=3
    model_aggressive = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=3,          # was 4
        min_samples_leaf=600,
        learning_rate=0.01,
        max_leaf_nodes=20,
        max_features=0.8,
        l2_regularization=3.0,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
```

Leave everything else unchanged -- same constraints, same regularization, same features, same pipeline. This is one change: "shallower trees."

Why max_depth=3 specifically: At depth 4 with max_leaf_nodes=20, trees can have up to 20 leaves with 4 levels of splits. At depth 3, trees can have at most 8 leaves (2^3). The max_leaf_nodes=20 parameter becomes non-binding but does no harm. This halves the effective tree complexity, reducing capacity to memorize regime-specific noise in the larger expanding-window training sets. Depth 3 still allows two-way feature interactions (e.g., "high vol AND positive momentum") which is sufficient for the constrained momentum signal.

**IF THIS FAILS:** The next step is to increase learning_rate from 0.01 to 0.02 (also untested). With expanding windows having more samples, each gradient step contributes less per-sample. A higher learning rate compensates, letting the model learn a stronger signal within the 300/500 iteration budget. This is especially relevant for the earliest evaluation window which has the least training data and may be undertrained at lr=0.01.

**IF BOTH FAIL:** Try replacing one of the two HGB models with an ExtraTreesRegressor to get genuine algorithmic diversity (bagging vs boosting). Use confidence-scaled predictions from the ETR via `_confidence_scaled_predict` (which was built for exactly this but never used). ETR naturally handles regime uncertainty through tree disagreement.

**RATIONALE:** The agent has been trapped in a pipeline-tuning loop for 15 experiments. All 6 pipeline parameters have been swept to their optima. The model's core hyperparameters (max_depth, learning_rate, max_iter, max_leaf_nodes) have NEVER been varied despite being the primary controls for model complexity. With the infrastructure change from sliding to expanding windows, the training set roughly doubled in size for later windows. A model calibrated for 25k samples (3-year sliding window) is likely overcapacity for 50k+ samples spanning diverse regimes. Reducing max_depth is the simplest, most direct way to address this. The fact that sharpe_min=-0.07 (barely negative) means even a small robustness improvement could flip the score positive.

## 8eb76b8 — max_depth 3 (from 4)
**Hypothesis:** Coach: shallower trees reduce overfitting on larger expanding-window training sets.
**Result:** Score -17.7069, sharpe_min -0.9599, max_dd -1.9%, 59 trades, 4/8 consistency, holdout WARN. Discard.
**Observation:** Catastrophic. max_depth=3 is far too shallow — model can't capture useful patterns. Only 59 trades. n_params dropped from 7820 to 4950. Trying coach fallback: learning_rate 0.02.

## a9b2ec5 — learning_rate 0.02 (from 0.01)
**Hypothesis:** Coach fallback: higher LR compensates for larger training sets where each gradient step contributes less.
**Result:** Score -4.7244, sharpe_min -1.9016, max_dd -6.1%, 345 trades, 5/8 consistency, holdout OK. Discard.
**Observation:** Overfits badly. Model makes too-confident wrong predictions (345 trades, sharpe -1.9). LR 0.01 is correct. Next: try milder complexity reduction — max_leaf_nodes 20→15. This keeps depth 4 but reduces leaf count by 25%.

## 5397cba — max_leaf_nodes 15 (from 20) ★ NEW BEST
**Hypothesis:** Mild complexity reduction (25% fewer leaves) while keeping depth 4.
**Result:** Score -0.4077, sharpe_min -0.0627, max_dd -2.6%, 121 trades, 6/8 consistency, holdout CAUTION. Keep (new best).
**Observation:** First improvement in 14 experiments! sharpe_min improved from -0.072 to -0.063 (less negative, approaching zero). Trades increased slightly (117→121). The max_leaf_nodes reduction helped the worst window without hurting others. Next: try max_leaf_nodes=12 for further reduction.

## d9e9a55 — max_leaf_nodes 12 (from 15)
**Hypothesis:** Further reduction may continue improvement.
**Result:** Score -2.6258, sharpe_min -0.2794, max_dd -2.5%, 109 trades, 6/8 consistency, holdout CAUTION. Discard.
**Observation:** Too few leaves — model loses useful patterns. Leaf count: 12→-2.63, **15→-0.41**, 20→-0.47. Optimum near 15. Next: try 18 (other side of 15).

