# Search Strategy

## Current best: ~1.181 (x0_lambda 0.05, agent2 task 005, 177 concurrent steps)
## Solo baseline: 1.094873 (355 steps, no contention)
## Concurrent baseline: 1.258207 (141 steps, 3 agents sharing GPU)

## CRITICAL: Concurrent Comparison
All experiments run with ~120-177 steps due to 3 agents sharing GPU.
Compare to CONCURRENT baseline (1.258), NOT solo baseline (1.095).

## Phase: exploitation (combining winners)

## Hardware constraints (DO NOT VIOLATE)
- A100 SXM4 40GB, 3 agents sharing 1 GPU concurrently
- DEVICE_BATCH_SIZE = 32 ALWAYS. Never change this.
- Max depth: 10 (VRAM constraint)
- BF16 native, no dtype hacks

## Rankings (vs concurrent baseline 1.258)
1. x0_lambda 0.05: 1.181 (-0.077) *** BEST — now in best/train.py ***
2. Matrix LR 0.08: 1.207 (-0.051)
3. Warmdown 0.3: 1.208 (-0.050) — on old baseline, should combine with x0_lambda
4. RoPE 50K: 1.223 (-0.035)
5. Embedding LR 0.8 + Unembed 0.008: 1.242 (-0.016)
6. Concurrent baseline: 1.258 (reference)
7. Depth 9/AR 57: 1.259 (neutral, but fewer steps due to larger model)
8. SSSSL window: 1.280 (+0.022, WORSE)
9. Warmdown 0.7: 1.333 (+0.075, MUCH WORSE)

## What works (confirmed)
- x0_lambda 0.05 (strong improvement, now in best/)
- Matrix LR 0.08 (second best single change)
- Warmdown 0.3 (third best, should combine with x0_lambda)
- RoPE 50K (moderate improvement)
- Embedding LR 0.8 + Unembed 0.008 (small improvement)

## What fails (avoid)
- Warmdown 0.7 (catastrophic with few concurrent steps)
- SSSSL window pattern (worse than baseline)
- Changing DEVICE_BATCH_SIZE

## Next strategy
best/train.py now has x0_lambda=0.05. Combine with other winners:
1. x0_lambda 0.05 + Matrix LR 0.08 (top 2 combined)
2. x0_lambda 0.05 + RoPE 50K (top 1 + top 3)
3. x0_lambda 0.05 + Matrix LR 0.08 + RoPE 50K (top 3 combined)
4. Try lower warmdown (0.3) on new best
