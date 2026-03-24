# Phase 3: Agent Instructions - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Write `program.md` — the ReID-specific agent instructions that tell Claude Code how to autonomously run experiments. Covers: search space, constraints, experiment strategy, history reasoning, and never-stop behavior.

</domain>

<decisions>
## Implementation Decisions

### Agent Platform
- **D-01:** Claude Code CLI is the agent. program.md is written for Claude Code's capabilities (shell commands, file editing, git operations).

### Experiment Hints
- **D-02:** Claude's discretion on hint detail level — balance between detailed experiment checklist and directional guidance. The goal is effective ReID experiments, not a rigid script.

### Search Space (agent can modify these in train.py)
- **D-03:** Loss weights/composition — distillation, ArcFace, VAT, separation loss weights and combinations
- **D-04:** Model architecture — projection head design, activation functions. **CRITICAL CONSTRAINT: model must remain edge-deployable. Agent must consider parameter count, GFLOPs, and keep embedding dim close to 256.** No unbounded model scaling.
- **D-05:** Optimizer/LR — optimizer choice, learning rate, scheduler type, warmup fraction
- **D-06:** Augmentation — training augmentation pipeline (RandomQualityDegradation params, color jitter, random erasing, etc.)

### Hard Constraints (encoded in program.md)
- **D-07:** Never edit prepare.py
- **D-08:** Never add pip dependencies
- **D-09:** Never exceed 10 epochs per experiment
- **D-10:** Never stop the loop (run until manually interrupted)
- **D-11:** Model must remain edge-deployable: monitor params, GFLOPs, embedding dim ≈ 256

### Claude's Discretion
- Exact experiment prioritization order
- Whether to include specific loss function alternatives (circle loss, proxy-anchor, etc.) or let agent discover them
- How to structure the "what to try when stuck" section

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Autoresearch
- `program.md` — Original autoresearch agent instructions (the template to adapt)

### Research
- `.planning/research/FEATURES.md` — Differentiators section has ReID-specific experiment ideas
- `.planning/research/PITFALLS.md` — Metric gaming and augmentation gaming risks
- `.planning/research/ARCHITECTURE.md` — What agent can/cannot touch

### Phase Outputs
- `prepare.py` — What's immutable (agent must know not to touch)
- `train.py` — What's editable (agent must know what constants to modify)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing `program.md` in repo — original autoresearch agent instructions for GPT training, can be adapted

### Established Patterns
- Original program.md structure: system prompt, constraints, experiment strategy, results reading
- Agent reads results.tsv to reason about what to try next

### Integration Points
- program.md is read by Claude Code at session start
- References train.py constants that agent can modify
- References metrics.json output format for agent to parse

</code_context>

<specifics>
## Specific Ideas

- **Edge deployment constraint is critical** — this must be prominently encoded in program.md so the agent never creates models too large to deploy
- Agent should understand the ReID domain (embedding similarity, retrieval, knowledge distillation) to make informed experiment choices

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-agent-instructions*
*Context gathered: 2026-03-25*
