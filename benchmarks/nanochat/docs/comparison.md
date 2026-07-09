# Agent Run Comparison: Baseline vs LoopGraph

## GPT-4o-mini via Venice

Run: `20260628T190725Z-loopgraph-retrieval-comparison`

| Metric | Baseline | LoopGraph |
|---|---:|---:|
| Model | openai-gpt-4o-mini-2024-07-18 | openai-gpt-4o-mini-2024-07-18 |
| Baseline BPB | 2.180112 | 2.180112 |
| Best BPB | 2.204652 | 2.180112 |
| BPB improvement | -0.024540 | 0.000000 |
| Total tokens | 24,127 | 20,986 |
| Wall time | 225.9 s | 186.9 s |

LoopGraph best BPB is **0.02454 better** than baseline.
LoopGraph used **13.0% fewer tokens** and was **17.3% faster**.

## zai-org-glm-5-2 via Venice

Run: `20260628T230233Z-loopgraph-retrieval-comparison`

| Metric | Baseline | LoopGraph |
|---|---:|---:|
| Model | zai-org-glm-5-2 | zai-org-glm-5-2 |
| Baseline BPB | 2.158530 | 2.178723 |
| Best BPB | 2.128834 | 2.153019 |
| BPB improvement | -0.029696 | -0.025704 |
| Total tokens | 31,755 | 28,431 |
| Wall time | 314.6 s | 284.6 s |

Baseline best BPB is **0.02419 better** than LoopGraph.
LoopGraph used **10.5% fewer tokens** and was **9.5% faster**.
Query reuse fired on experiment 3 (`Reusing prior query 'MLP SwiGLU'`).

## Notes

- With GPT-4o-mini, LoopGraph wins on BPB, tokens, and time.
- With GLM 5.2, baseline wins on BPB, but LoopGraph still uses fewer tokens and less time.
- GLM 5.2 is a stronger optimizer in absolute terms, but the retrieval context may constrain it to a narrow search direction (e.g. SwiGLU/MLP) while baseline freely explores LR tuning.

## Published NanoChat Baselines

Recursive's public Case Study 1 numbers are a different tier from the quick local LoopGraph comparisons above:

| Benchmark | Mean BPB | Seeds | Device | Notes |
|---|---:|---:|---|---|
| Recursive `optimized_from_karpathy` | 0.9109 | 10 | NVIDIA B200 | Best reported Recursive result, starting from Karpathy's autoresearch seed |
| autoresearch@home community best | 0.9372 | 10 | NVIDIA B200 | Reward-hack-removed previous public best |
| Recursive `optimized_from_vanilla` | 0.9344 | 10 | NVIDIA B200 | Recursive result starting from a vanilla Transformer |
| Karpathy seed re-run | 0.9751 ± 0.0112 | 10 | NVIDIA B200 | Recursive comparability re-run |
| Vanilla Transformer | 1.0587 ± 0.0101 | 10 | NVIDIA B200 | Plain baseline in Recursive artifact |

The local comparison runs in this file report best BPB around 2.13–2.20, so they are roughly 1.19–1.29 BPB worse than autoresearch@home and 1.22–1.29 BPB worse than Recursive's best. That gap reflects different budgets, hardware, search depth, and validation rigor; these local runs are agent-loop smoke tests, not SOTA attempts.
