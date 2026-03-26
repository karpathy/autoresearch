# Phase 4: Validation - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove the complete system works end-to-end: establish baseline metric, demonstrate one full autonomous cycle, verify crash recovery, then launch the first autonomous overnight run.

</domain>

<decisions>
## Implementation Decisions

### Baseline
- **D-01:** Claude's discretion on how many baseline runs to perform for stability verification. At least 1 required.

### Post-Validation
- **D-02:** After validation passes, automatically launch the first autonomous overnight run. No manual step needed — validation flows directly into production use.

### Claude's Discretion
- Number of baseline runs for stability (1 vs 3)
- How to intentionally trigger OOM for crash recovery testing (increase batch size? allocate extra tensors?)
- Whether to set a maximum experiment count for the overnight run or let it run indefinitely

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Outputs (all must exist before Phase 4)
- `prepare.py` — Phase 1 output
- `train.py` — Phase 1 output
- `metrics.json` — Phase 2 output format
- `results.tsv` — Phase 2 output format
- `program.md` — Phase 3 output

### Research
- `.planning/research/SUMMARY.md` — Overall system architecture and validation approach

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- All Phase 1-3 outputs form the complete system to validate

### Integration Points
- Validation tests the full chain: agent → edit train.py → run → evaluate → keep/discard → log
- Crash recovery tests the OOM → catch → log → reset → continue chain

</code_context>

<specifics>
## Specific Ideas

- After validation, the autonomous run should start immediately — the user wants to wake up to results in the morning.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-validation*
*Context gathered: 2026-03-25*
