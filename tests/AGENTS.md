# tests
Benchmark suite measuring harness quality via Pareto ratchet.

## Metrics (tiered)
- T1 gate | task_success_rate, quality_gate_pass_rate | must not regress
- T2 constraint | rework_rate | subject to T1 floors
- T3 optimize | avg_token_consumption, time_per_turn | subject to T1+T2

## Files
- `benchmark.py` | task runner | `--all`, `--quick`, `--json`
- `evaluator.py` | Pareto ratchet | `--extract-composite`, `--check-ratchet`, `--reflect`
- `tasks/` | synthetic task definitions | YAML per task

## Constraints
- Tier priority T1 > T2 > T3 -- lower-tier gains rejected if higher-tier floors drop
- Results logged to results.tsv (untracked) -- local artifacts only
