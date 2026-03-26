---
phase: 03-agent-instructions
plan: 01
subsystem: agent-instructions
tags: [program.md, autoresearch, reid, knowledge-distillation, experiment-loop]

# Dependency graph
requires:
  - phase: 01-code-split
    provides: prepare.py/train.py split boundary and import contract
  - phase: 02-experiment-infra
    provides: metrics.json output format, results.tsv column definition, combined_metric formula
provides:
  - "Complete program.md with ReID-specific agent instructions for autonomous experimentation"
  - "Search space reference table with 16 tunable constants"
  - "3-tier experiment playbook with directional hints"
  - "6 hard constraints encoded as NEVER rules"
  - "History reasoning instructions for results.tsv analysis"
affects: [04-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [autoresearch-loop, keep-discard-gate, never-stop-autonomy]

key-files:
  created: []
  modified: [program.md]

key-decisions:
  - "Preserved original autoresearch structure while replacing all domain content for ReID"
  - "Used directional experiment hints (ranges) not prescriptive values per D-02"
  - "Encoded 6 NEVER constraints as dedicated prominent section"
  - "Included VRAM budget rule (22GB threshold) to prevent OOM cascade"

patterns-established:
  - "program.md section ordering: Setup, Experimentation, Constraints, Output, Logging, Loop, Never-Stop, Domain, Search Space, Playbook, Stuck, History, Crash"

requirements-completed: [AGNT-01, AGNT-02, AGNT-03, AGNT-04, AGNT-05]

# Metrics
duration: 3min
completed: 2026-03-25
---

# Phase 03 Plan 01: Agent Instructions Summary

**Complete ReID autoresearch program.md with 14 sections: experiment loop, 6 NEVER constraints, 16-constant search space, 3-tier playbook, and history reasoning**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-24T20:30:32Z
- **Completed:** 2026-03-24T20:33:55Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Replaced Karpathy's GPT-focused program.md with complete ReID knowledge distillation agent instructions
- Encoded all 11 locked decisions (D-01 through D-11) from CONTEXT.md into the document
- Created 14-section document covering full autonomous experiment lifecycle from setup to crash handling
- All 5 AGNT requirements verified by automated grep checks (AGNT-01: 9, AGNT-02: 24, AGNT-03: 10, AGNT-04: 3, AGNT-05: 13)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write program.md core structure (Setup, Constraints, Output Format, Experiment Loop, Never-Stop)** - `0772723` (feat)
2. **Task 2: Add ReID Search Space Reference, Experiment Playbook, Domain Context, and History Reasoning** - `fc3faab` (feat)

## Files Created/Modified
- `program.md` - Complete ReID-specific agent instructions (278 lines, 14 sections)

## Decisions Made
- Preserved original autoresearch structure (Setup, Experimentation, Output, Logging, Loop, Never-Stop) while replacing all domain content per adaptation map
- Used directional experiment hints with ranges (e.g., "ArcFace weight 0.01-0.2") not exact values, per D-02 guidance
- Encoded 6 hard constraints as a dedicated, prominently placed NEVER section for maximum visibility
- Included VRAM budget rule (22GB threshold on 24GB card) to prevent OOM cascade loops per PITFALLS.md Pitfall 3
- Kept NEVER STOP section closely adapted from original -- proven effective, only changed experiment rate numbers

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - program.md is a complete self-contained document with no placeholder content.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- program.md is complete and ready for Phase 04 validation (VALD-01: dry run baseline)
- All search space constants reference train.py variables that were established in Phase 01-02
- The agent can read program.md at session start and immediately begin autonomous experimentation

## Self-Check: PASSED

- program.md: FOUND
- 03-01-SUMMARY.md: FOUND
- Commit 0772723: FOUND
- Commit fc3faab: FOUND

---
*Phase: 03-agent-instructions*
*Completed: 2026-03-25*
