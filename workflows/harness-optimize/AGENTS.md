# harness-optimize
Meta-optimize: the harness iterates on itself to maximize benchmark scores.

## Targets
- `../../AGENTS.md` | `../../harness.yaml` | `../../.github/copilot-instructions.md`
- `../../.github/agents/research-runner.agent.md` | `../../.github/skills/autonomous-iteration/SKILL.md`

## Fixed
- `workflow.yaml`, `program.md` | `../../.github/tests/`

## Background mode
- `python scaffold.py harness-bg` | `--resume` | `--refresh`

## State (gitignored)
- `results/results.tsv` | `results/musings.md` | `results/ratchet_state.json` | `results/audit/`

## Constraints
- No modifying tests/evaluator -- evaluation integrity
- Platform-agnostic only -- no vendor lock-in
- No gaming specific tasks -- generalize, not overfit
- No rebase during session -- breaks commit provenance
