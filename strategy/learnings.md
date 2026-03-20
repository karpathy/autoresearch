# Learnings

## Architecture
- **SwiGLU beats relu²** [HIGH]: Massive win across all configs. Gated activations dramatically improve expressiveness at same param count.
- **DEPTH=3 is the sweet spot** [HIGH]: Depth=2 too little capacity; Depth=4 too slow (fewer tokens in 5min budget).
- **MQA (n_kv_head=1) helps** [MEDIUM]: Forces query head specialization, shrinks KV params. Small but real gain.
- **Developmental pruning (55%) helps** [HIGH]: Starting bigger then pruning to 55% beats starting small. Newborn-to-adult pattern.

## Sparsity
- **k-Winners 10% is sweet spot** [HIGH]: 5% too sparse, 25% too sparse. 10% neutral-to-slight-win.
- **Structural + activation sparsity are complementary** [MEDIUM]: Removing k-Winners from pruned model made it worse (exp66).
- **Progressive per-layer sparsification is neutral** [MEDIUM]: Budget-neutral redistribution gives no gain.

## Training Signal
- **Non-uniform token weighting is poison for pretraining** [HIGH]: exp57, 68, 69 all failed badly. Every token matters equally for TinyStories pretraining.
- **Multi-token prediction is incompatible at this scale** [HIGH]: All 3 variants worse, 2x VRAM. Model too small for spare capacity.

## Optimizer / Schedule
- **MATRIX_LR=0.03 is tuned** [HIGH]: 0.04 causes loss spikes. 0.03 stable.
- **WARMDOWN_RATIO=0.8 is optimal** [MEDIUM]: 0.5 marginally worse (exp71).

## Dead Ends
- Block-level soft skip gates: adds params without useful routing
- Inhibitory interneurons in MLP: extra suppression doesn't improve k-Winners
- Per-layer myelination gradient decay: 3-layer model too shallow for this
- Wider MLP (4x vs 8/3x): larger neuron pool for k-Winners doesn't help
- Homeostatic plasticity EMA alpha=0.005: too slow for 5min budget (exp80)
