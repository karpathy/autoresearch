# tests/

Benchmark suite that measures harness quality. Synthetic tasks exercise the full agent loop -- instructions, skills, hooks, guardrails -- and produce scores across five metrics organized in three tiers.

## Metrics

| Tier | Metric | ID | What it measures |
|------|--------|----|------------------|
| T1 | Task success rate | task_success | Percentage of benchmark tasks the agent completes correctly |
| T1 | Quality-gate pass rate | gate_pass | Percentage of outputs that pass self-eval and judge gates on first attempt |
| T2 | Rework rate | rework | Percentage of tasks requiring retry or human correction |
| T3 | Avg token consumption | tokens | Mean input + output tokens per task |
| T3 | Time per turn | time | Mean wall-clock seconds per agent turn |

## Pareto Ratchet

Improvements to lower-tier metrics are only accepted if higher-tier floors hold. A change that reduces token consumption (T3) but drops task success rate (T1) below its floor is rejected. Floors are recorded in the benchmark manifest and ratchet upward as the harness improves.

Tier priority: T1 > T2 > T3. Within a tier, all metrics must hold.

## Conventions

- Benchmark tasks and evaluator will be added in a subsequent step
- Each benchmark task is a self-contained directory with inputs, expected outputs, and an evaluator script
- Results are logged to `results.tsv` (TSV format, untracked by git)
- The Pareto ratchet floors are stored in a manifest file alongside the benchmark tasks

## Version

- Last updated: 2026-07-15
