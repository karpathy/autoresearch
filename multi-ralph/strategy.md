# Search Strategy

## Current best: ~1.181 (x0_lambda 0.05, agent2 task 005, 177 concurrent steps)
## Solo baseline: 1.094873 (355 steps, no contention)
## Concurrent baseline: 1.258207 (141 steps, 3 agents sharing GPU)

## CRITICAL: Concurrent Comparison
All experiments run with ~120-177 steps due to 3 agents sharing GPU.
Compare to CONCURRENT baseline (1.258), NOT solo baseline (1.095).
Step count varies 120-177 depending on contention — adds noise.

## Phase: exploitation + exploration of untested dimensions

## Hardware constraints (DO NOT VIOLATE)
- A100 SXM4 40GB, 3 agents sharing 1 GPU concurrently
- DEVICE_BATCH_SIZE = 32 ALWAYS. Never change this.
- TOTAL_BATCH_SIZE = 2**19 ALWAYS. Never change this.
- Max depth: 10 (VRAM constraint)
- BF16 native, no dtype hacks

## Rankings (vs concurrent baseline 1.258) — 12 experiments done
1. x0_lambda 0.05: 1.181 (-0.077) *** BEST — now in best/train.py ***
2. x0_lambda 0.05 + matrix_lr 0.08: 1.201 (-0.057) — matrix_lr hurts on top of x0
3. warmdown 0.3 (old baseline): 1.208 (-0.050)
4. RoPE 50K (old baseline): 1.223 (-0.035)
5. weight_decay 0.05 + x0_lambda: 1.240 (-0.018)
6. embedding_lr 0.8 + unembed 0.008: 1.242 (-0.016)
7. x0_lambda 0.05 + RoPE 50K: 1.253 (-0.005 — RoPE hurts on top of x0)
8. Concurrent baseline: 1.258 (reference)
9. Depth 9/AR 57: 1.259 (neutral, larger model = fewer steps)
10. SSSSL window: 1.280 (+0.022, WORSE)
11. Warmdown 0.7: 1.333 (+0.075, MUCH WORSE)
12. Lower LRs (half all): 1.362 (+0.104, TERRIBLE)

## What works (confirmed)
- x0_lambda 0.05 (strong winner, in best/)

## What fails (avoid)
- RoPE 50K on top of x0_lambda (neutral/worse)
- Matrix LR 0.08 on top of x0_lambda (slight worse)
- Warmdown > 0.5 (catastrophic)
- Lower LRs (underconverge)
- SSSSL window (worse)

## Key observations
- Combinations of winners often DON'T stack
- Schedule changes (warmdown) have huge impact due to few concurrent steps
- Untested: softcap, FINAL_LR_FRAC, adam betas, head_dim, scalar_lr

## Queued (experiments 013-016)
- 013: x0_lambda + warmdown 0.3 (combine two winners)
- 014: x0_lambda + warmdown 0.2 (push warmdown lower)
- 015: x0_lambda + softcap 30 (untested dimension)
- 016: x0_lambda + FINAL_LR_FRAC 0.1 (untested dimension)

## Next strategy
Test warmdown variants and untested dimensions on x0_lambda base.
After this round: combine best warmdown + best untested dimension winner.
