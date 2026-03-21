# Learnings

## Architecture
- **SwiGLU beats relu²** [HIGH]: Massive win across all configs.
- **Peri-LN (pre+post norm) helps** [HIGH]: +0.004 improvement. Adopted by Gemma/OLMo. Better gradient flow at near-zero compute cost.
- **SwiGLU 3x > 8/3x intermediate** [HIGH]: Slightly wider MLP (+0.5M params) gives +0.002 improvement without losing too many steps. 3.5x and 4x too slow.
- **RoPE base=50000 > 10000** [HIGH]: +0.003 improvement. Higher frequency base gives better positional encoding for 2048 context. 75K and 100K worse — 50K is the sweet spot.
- **HEAD_DIM=128 (n_head=2) > HEAD_DIM=64 (n_head=4)** [HIGH]: Fewer but more powerful heads is better at dim=256.
- **Value embeddings are essential** [HIGH]: Removing VE gave -0.041 regression. Alternating layers is optimal (every layer is a wash).
- **D8/256 is the sweet spot** [HIGH]: D9, D10 lose too many steps. D6/192 was previous optimum before BATCH scaling.

## Batch Size / VRAM
- **BATCH=2^15 + DEVICE_BATCH=16 is optimal** [HIGH]: Biggest single win (+0.008). Eliminates grad accum overhead.
- **DEVICE_BATCH=20 or 32 OOMs or is too slow** [HIGH]: 5.3-7.7GB VRAM, 36-316 steps.
- **D8/384 too wide** [HIGH]: 33M params, 902-1673 steps — capacity isn't the bottleneck, steps are.
- **Gradient checkpointing kills step speed** [HIGH]: Saves VRAM but recomputation costs too many steps.

## Optimizer / Schedule
- **MATRIX_LR=0.03 with Peri-LN** [HIGH]: Re-tuned from 0.04 after adding post-norm.
- **WARMDOWN_RATIO=0.9** [HIGH]: Better than 0.8 and 0.95.
- **FINAL_LR_FRAC=0.05** [HIGH]: Keep a small residual LR at end of warmdown for continued learning.
- **EMBEDDING_LR=0.4** [HIGH]: Lower than original 0.6.
- **WEIGHT_DECAY=0.1** [HIGH]: Less than original 0.2.
- **ADAM_BETAS=(0.9, 0.98)** [HIGH]: Higher momentum and beta2 than original (0.8, 0.95).
- **Muon ns_steps=5 is essential** [HIGH]: ns_steps=3 loses gradient quality despite extra steps.
- **Muon momentum 0.95 final is optimal** [MEDIUM]: 0.98 too high.

## Dead Ends
- Parallel attn+MLP (PaLM-style): more steps but worse quality per step
- Label smoothing: destroys loss signal, catastrophic regression
- Post-norm only (no pre-norm): much worse gradient flow
- Softcap removal with Peri-LN: softcap=15 still helps
- MQA at HEAD_DIM=128: causes OOM/torch.compile issues
- VE gate channels 64 (was 32): no improvement, fewer steps
- SCALAR_LR tuning (0.2 or 1.0): 0.5 is optimal
- x0_lambdas init 0.2: slightly worse than 0.1
- EMBEDDING_LR=0.8/0.2: 0.4 is optimal
- UNEMBEDDING_LR tuning: 0.004 is optimal
- Developmental pruning at current scale: removes useful weights
- wte init std=0.5: worse than 1.0
- WARMUP_RATIO=0.02: wastes precious steps at 1500 step budget
