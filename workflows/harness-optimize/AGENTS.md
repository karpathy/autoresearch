# harness-optimize
Meta-optimization: the harness iterates on itself to maximize benchmark scores.

## Targets (agent edits)
- `../../AGENTS.md` | root manifest
- `../../.github/agents/research-runner.agent.md` | agent definition
- `../../.github/skills/autonomous-iteration/SKILL.md` | core skill
- `../../.github/copilot-instructions.md` | Copilot instructions

## Fixed (read-only)
- `workflow.yaml`, `program.md` | manifest and strategy
- `../../.github/tests/` | benchmark suite

## Constraints
- No modifying tests or evaluator -- evaluation integrity
- Platform-agnostic changes only -- no vendor lock-in
- No gaming specific test cases -- generalize, not overfit
