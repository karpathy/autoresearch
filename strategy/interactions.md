# Interactions

Known couplings between parameters/features.

## Confirmed
- **SwiGLU ↔ MATRIX_LR**: SwiGLU requires lower LR (0.03, not 0.04). LR=0.04 causes loss spikes with SwiGLU (exp55).
- **k-Winners ↔ developmental pruning**: complementary — removing k-Winners from pruned model hurt (exp66). Both structural and activation sparsity contribute.
- **DEPTH ↔ tokens/5min**: DEPTH=4 gets significantly fewer tokens processed in budget. DEPTH=3 is sweet spot.
- **DEPTH ↔ pruning**: DEPTH=2 model too small to benefit from pruning (nothing to prune). DEPTH=3 benefits strongly.

## Suspected
- **SPARSE_K ↔ MLP intermediate size**: k-Winners at 10% of 8/3x intermediate might be different than 10% of 4x intermediate. (exp72 tested 4x but found it worse — coupled effect unclear)
- **MQA ↔ n_head**: MQA with n_kv_head=1 and n_head=3 — only 3 query heads total, so the "specialization" effect may be weaker than with more heads.
- **Homeostatic plasticity ↔ SPARSE_K**: Both regulate activation density — interaction unknown. Don't run both simultaneously until each is understood alone.

## Unknown
- How MQA interacts with attention output sparsity (untested)
- Whether variance-based pruning would interact with developmental schedule
