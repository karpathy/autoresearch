---
name: loopgraph-retrieval-comparison
description: "Compare LoopGraph retrieval modes (baseline vs ResearchFS loopgraph) on nanochat_bpb"
benchmark: loopgraph-nanochat
second_brain: hybrid-local
workers: 1
max_hours: 2
max_experiments: 4
metric:
  name: val_bpb
  goal: minimize
  baseline: null
experiment:
  command: >-
    {python_bin} asep_loopgraph_benchmark.py
    --mode {mode}
    --metric val_bpb
    --max-experiments 1
    --steps-budget 50
    --time-budget 60
    --api-provider openrouter
    --model z-ai/glm-4.5-air:free
    --device mps
  timeout_seconds: 600
strategy:
  type: planner
  parameters:
    mode: [baseline, loopgraph]
autonomy_guardrails:
  durable_state: required
  upstream_refresh: before_exhaustion
  paired_related_ideas: required
  pruning_rounds: after_stacked_changes
  premature_stop_blocker: required
---

## Goal
Measure whether LoopGraph's ResearchFS retrieval mode reduces the validation
bits-per-byte (val_bpb) compared to the baseline agent mode.

## Constraints
Each experiment is a full LoopGraph agent run with 2 proposals, ~50 training
steps per experiment, and a 60-second training time budget. Runs execute on
`mps` for fast local execution.

## Use Hybrid Memory
Use hybrid-local memory:
- Durable/shared/human-facing: local LLM-Wiki + GBrain
- Private/local learning engine: Hindsight
