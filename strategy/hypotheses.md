# Hypotheses

## Active Queue (prioritized)

### H0: Scale wider — n_embd=384 or 512
**Hypothesis:** n_embd=256 gave +0.058. We still have 3.3GB VRAM headroom. Width is the strongest lever found so far.
**Mechanism:** More features per layer = richer representations. DEPTH=3 keeps step count high.
**Expected:** Significant improvement if we can fit it. Risk: fewer steps if model is too large per step.
**Tier:** 4 (but highest expected value — confirmed scaling direction)

### H0b: Predictive coding + DEPTH=4 at n_embd=256
**Hypothesis:** DEPTH=4 failed at small width. At n_embd=256 the model may have enough capacity to justify 4 layers.
**Mechanism:** PC factorizes cheap vs. expensive computation per layer. Extra depth = more abstraction.
**Expected:** Unknown. Might work now that width is sufficient. Will lose ~30% steps.
**Tier:** 1

### H1: Homeostatic plasticity with faster alpha
**Hypothesis:** EMA alpha=0.005 was too slow for 5min budget. Try alpha=0.05 or batch-level instantaneous firing stats.
**Mechanism:** Keeps average activation near target, prevents dead neurons, better utilization of sparse coding.
**Expected:** Small improvement (~0.001) or neutral. More likely to activate meaningfully with faster stats.
**Tier:** 1 (activation sparsity / dynamic compute)

### H2: Attention output k-Winners
**Hypothesis:** Apply k-Winners sparsity to attention output projections, not just MLP. Sparse attention outputs.
**Mechanism:** Forces attention heads to be selective. Only top-K attended values propagate. Less redundancy.
**Expected:** Neutral to small win. Could hurt if attention heads need dense output.
**Tier:** 1 (sparse activation)

### H3: Smaller batch size (2^13) for more optimizer steps
**Hypothesis:** 2^14 batch = ~953 steps/5min. 2^13 = ~1906 steps. More gradient updates = better convergence.
**Mechanism:** More optimizer steps with smaller batches can sometimes win on small models/datasets.
**Expected:** Small improvement or neutral. Risk: higher gradient variance.
**Tier:** 4 (hyperparam)

### H4: Linear attention / SSM replacement
**Hypothesis:** Replace O(n²) attention with linear attention kernel or Mamba-style SSM.
**Mechanism:** Completely different sequence modeling. No quadratic cost. Different inductive bias.
**Expected:** Unknown. Could be much worse initially. This is a structural class change.
**Tier:** 2 (replace core operation)

### H5: Glial-driven variance-based pruning
**Hypothesis:** Use variance of activations (not magnitude of weights) for pruning decisions.
**Mechanism:** Prune neurons with low activation variance across tokens — they're not doing useful computation.
**Expected:** Neutral to small improvement. More principled than magnitude pruning.
**Tier:** 1 (pruning)

### H6: Information bottleneck auxiliary loss
**Hypothesis:** Add IB auxiliary loss at intermediate layers to minimize I(X;Z) while maximizing I(Z;Y).
**Mechanism:** Forces representations to be minimal-sufficient. Penalizes redundant information.
**Expected:** Unknown. Could help generalization. Tricky to calibrate weight.
**Tier:** 3 (different learning signal)

## Tested / Resolved

| Hypothesis | Result | Commit |
|---|---|---|
| SwiGLU activation | WIN +0.039 | 0ed1b8f |
| k-Winners 10% | neutral/keep | 283b728 |
| DEPTH=3 + pruning | WIN +0.003 | 84ffad6 |
| PRUNE_TARGET=55% | best | 7d55f8e |
| MQA n_kv_head=1 | tiny WIN | f5ac918 |
| MTP 3 variants | fail | 9f159d7/70ba6a4/75b0705 |
| Homeostatic EMA=0.005 | neutral | b1f96dd |
| Block skip gates | fail | 2999248 |
| Inhibitory interneurons | fail | 70aaee4 |
| Per-layer myelination | fail | 5800ea2 |
| Token weighting (focal/soft) | catastrophic | 3ec0dba/623a108 |
| Wider MLP 4x | fail | 46b9be3 |
| Progressive sparsification | neutral | 4dc96da |
