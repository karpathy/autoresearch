---
name: autonomous-iteration
description: Edit-run-measure-keep/discard loop.
---

## When to use
Iterating a file against a measurable metric.

## Steps
1. Parse `workflow.yaml` for targets, metric, cmd
2. Parse `program.md` for strategy
3. Loop per [protocol.md](protocol.md): edit > run > measure > keep/discard

## Constraints
- Only modify workflow.yaml targets -- integrity
- Never pause -- human away
- Cycle Add/Reduce/Reform

## Examples
`workflows/examples/ml-training/`
