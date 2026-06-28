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
