# PubMed Run 1 — RTX 3070 8GB

**Date:** 2026-03-13 to 2026-03-15 (~48 hours)
**GPU:** NVIDIA RTX 3070 (8GB VRAM, 3.3GB used)
**Model:** Claude Sonnet 4 (`claude-sonnet-4-20250514`)
**Dataset:** PubMed medical abstracts
**Config:** depth=8, batch_size=8, sequence_len=2048, vocab_size=8192

## Results Summary

| Metric | Value |
|--------|-------|
| Total experiments | 100 |
| Keeps | 7 (+ baseline) |
| Discards | 70 |
| Crashes | 23 |
| Keep rate | 7% |
| Starting val_bpb | 1.3984 |
| Final val_bpb | 1.0498 |
| **Total improvement** | **24.9%** |

## Discoveries (in order)

| # | val_bpb | Change | Technique |
|---|---------|--------|-----------|
| 0 | 1.3984 | — | Baseline |
| 1 | 1.2870 | -8.0% | Weight tying + halved batch size for 2x training steps |
| 2 | 1.2867 | -0.0% | Weight decay (0.001) on value embedding parameters |
| 3 | 1.1856 | -7.9% | Gradient clipping at 0.5 |
| 4 | 1.1672 | -1.6% | MLP skip connections for better gradient flow |
| 5 | 1.0501 | -10.0% | Sequence packing to reduce padding waste |
| 6 | 1.0498 | -0.0% | Multi-query attention (KV heads 6→1) |

## Sample Text Progression

**Baseline (1.398):** No sample recorded.

**Weight tying (1.287):**
> The aim of this study was to investigate the effects of the 2-oxynucleotide (SN2) on the effects of the SN2 on the effects of SN2 on the SN2 on SN2 and SN2 on SN2 and SN2 on SN2.

**Gradient clipping (1.186):**
> The aim of this study was to evaluate the effect of a 3-year-old man with a 3-year-old man with a 3-year history of a 3-year-old man with a 3-year history of a 3-year-old man.

**MLP skip connections (1.167):**
> The aim of this study was to investigate the effects of a single-molecule (SR) on the expression of the SR1 gene in the human brain. The expression of SR1 gene was evaluated in the human brain.

**Multi-query attention (1.050):**
> The aim of this study was to investigate the effect of the use of a 3D magnetic resonance imaging (MRI) on the performance of a 3D magnetic resonance imaging (MRI) scanner. The MRI scanner was used to assess the performance of the MRI scanner.

## Notable Failures

- **38-experiment losing streak** early on — broken by rewriting the prompt with categorized summaries, near-miss detection, and adaptive temperature
- **val_bpb=-37.8 bug** — attention entropy regularization broke the metric, kept as "improvement" since -37.8 < 1.05. Fixed with sanity check (reject val_bpb <= 0 or > 20)
- **SwiGLU, MoE, FIRE optimizer, cosine warm restarts** — all crashed or regressed
- **Mamba-style convolution** — crashed (architecture too different)

## Key Learnings

1. **Structured prompt > raw logs** — categorized summaries with near-miss detection dramatically outperform dumping full experiment history
2. **Adaptive temperature works** — 0 default → 0.4 at 5 fails → 0.6 at 10 → 0.8 at 15+. Broke a 38-run streak.
3. **Guard your metric** — need sanity checks for impossible values
4. **Known techniques win** — sequence packing, weight tying, gradient clipping are all well-established. The agent's value is finding the right hyperparameters and combinations, not inventing novel architectures.
5. **Crash recovery is essential** — 23% crash rate means the agent must handle failures gracefully
