# agents
Agent behavioral contracts. One markdown file per agent. Platform-agnostic.

## Definitions
- `research-runner.md` | autonomous experiment loop | propose-run-measure-keep/discard

## Constraints
- Agents never modify their own definitions -- prevents self-corruption
- Each file includes: role, inputs, loop, constraints, outputs -- completeness contract
