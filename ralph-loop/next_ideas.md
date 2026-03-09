# Next Experiment Ideas

Ranked by expected impact. Pick from the top.

## Priority 1: Simplification experiments (speed wins)

1. **Remove value embeddings** — VE adds embedding params and compute on alternating layers. At depth 5, that's layers 1,3 (or 0,2,4). Removing could give speed + simplicity. Requires removing has_ve, ve_gate, value_embeds from model + optimizer.

2. **Remove x0 residual connection** — The model uses x0_lambdas to mix in the original embedding at each layer. Removing simplifies forward pass. Set x0_lambdas init to 0 or remove entirely.

3. **Remove resid_lambdas** — Per-layer residual scaling. Try removing (fix at 1.0).

## Priority 2: LR re-tuning at new model size

4. **Matrix LR 0.12 at depth 5** — Previously too high at depth 8 (50M params). At depth 5 (24.6M params, 358 steps) the dynamics differ. Worth retrying.

5. **Embedding LR 1.8 at depth 5** — Same reasoning. Was too high at depth 8.

6. **Unembedding LR 0.012 at depth 5** — Was too high at depth 8.

## Priority 3: Architecture variants

7. **Softcap 10** — Tighter logit capping. Forces better calibration early.

8. **Depth 5, aspect 72** — 5*72=360, rounds to 384 (same dim). No effect. Skip.

9. **Weight decay 0.3** — Try increasing rather than decreasing. More regularization could help small model generalize.

10. **FINAL_LR_FRAC = 0.15** — Fine-tune between 0.1 and 0.2.

## Priority 4: Training dynamics

11. **Muon momentum warmup faster** — Current: ramps 0.85→0.95 over 300 steps. With 358 steps total, most training is in warmup. Try faster ramp (150 steps) or start higher (0.90→0.95).

12. **Different warmdown shape** — Current is linear. Try cosine warmdown.
