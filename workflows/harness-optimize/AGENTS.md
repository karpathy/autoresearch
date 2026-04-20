# Harness Optimization

## Overview

Meta-optimization workflow. Iterates on the repo's own harness files (AGENTS.md, copilot-instructions.md, agent definitions, skill definitions, harness.yaml) to maximize agent effectiveness as measured by the benchmark suite.

This workflow is self-referential -- the harness optimizes itself.

## Key Files

| File | Role | Who Edits |
|------|------|-----------|
| workflow.yaml | Targets, metric, run command | Human (once) |
| program.md | Research strategy and constraints | Human (iteratively) |
| ../../AGENTS.md | Root manifest/router | Agent (each experiment) |
| ../../.github/copilot-instructions.md | Copilot instructions | Agent (each experiment) |
| ../../agents/research-runner.md | Agent behavior contract | Agent (each experiment) |
| ../../skills/autonomous-iteration.md | Core skill definition | Agent (each experiment) |
| ../../harness.yaml | Reference config | Agent (each experiment) |

## Constraints

- Do NOT modify tests/benchmark.py, tests/evaluator.py, or tests/tasks/
- Do NOT modify workflow.yaml or program.md
- Changes must be platform-agnostic (no Copilot-only or Claude-only improvements)
- Changes must improve benchmark metrics without gaming specific test cases
- Keep AGENTS.md concise -- router function, not documentation dump

## Metrics (Pareto Ratchet)

| Metric | Tier | Direction | Role |
|--------|------|-----------|------|
| task_success_rate | 1 | higher | Gate -- must not regress |
| quality_gate_pass_rate | 1 | higher | Gate -- must not regress |
| rework_rate | 2 | lower | Constraint -- subject to T1 floors |
| avg_token_consumption | 3 | lower | Optimization -- subject to T1+T2 floors |
| time_per_turn | 3 | lower | Optimization -- subject to T1+T2 floors |

## Running

```bash
cd workflows/harness-optimize
uv run python ../../tests/benchmark.py --all > run.log 2>&1
grep "^harness_score:" run.log
```
