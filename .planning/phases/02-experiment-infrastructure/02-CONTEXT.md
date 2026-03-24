# Phase 2: Experiment Infrastructure - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the experiment loop harness that wraps train.py execution: git management (commit/reset), results logging (results.tsv), crash recovery (OOM handling), and VRAM tracking. The agent drives this loop directly — no wrapper script.

</domain>

<decisions>
## Implementation Decisions

### Loop Ownership
- **D-01:** Agent (Claude Code) drives the loop directly via shell commands. No run.sh wrapper. Agent calls `python train.py`, reads output, manages git, appends to results.tsv. Follows original autoresearch pattern.

### Metric Output
- **D-02:** train.py writes metrics to a JSON file (e.g., `metrics.json`) after training completes. Agent reads this file for structured metric parsing instead of grepping stdout. More reliable than stdout parsing.

### Results Logging
- **D-03:** Claude's discretion on whether agent or train.py writes to results.tsv. The key requirement is that every experiment gets a row with: commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status, description.

### Claude's Discretion
- Whether results.tsv is written by train.py (automatic) or by agent (after reading metrics.json)
- Exact git workflow (commit before run vs after, branch strategy)
- How to implement 3-consecutive-crash skip logic (in program.md instructions vs code)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Autoresearch Pattern
- `program.md` — Original autoresearch agent instructions (shows how agent manages git + results)

### Phase 1 Output
- `prepare.py` — Will exist after Phase 1; provides evaluation functions
- `train.py` — Will exist after Phase 1; the file agent modifies and runs

### Research
- `.planning/research/PITFALLS.md` — OOM cascade and crash recovery patterns
- `.planning/research/FEATURES.md` — Infrastructure table stakes

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `torch.cuda.max_memory_allocated()` — already used in original for VRAM tracking
- `EpochStats` dataclass in monolith — has all sub-metrics, can be serialized to JSON

### Established Patterns
- Original autoresearch: agent commits before running, resets on failure
- results.tsv is NOT git-tracked (in .gitignore)
- metrics.json is ephemeral — overwritten each run

### Integration Points
- train.py must write metrics.json at end of training
- Agent reads metrics.json after each run
- Agent appends to results.tsv after each run
- Git operations: commit train.py changes, reset on failure

</code_context>

<specifics>
## Specific Ideas

No specific requirements — follow original autoresearch infrastructure patterns adapted for JSON metric output.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-experiment-infrastructure*
*Context gathered: 2026-03-25*
