---
phase: 03-agent-instructions
verified: 2026-03-25T06:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
gaps: []
human_verification: []
---

# Phase 3: Agent Instructions Verification Report

**Phase Goal:** program.md gives an LLM agent everything it needs to autonomously run effective ReID experiments -- domain knowledge, search strategy, constraints, and history-reading capability
**Verified:** 2026-03-25T06:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | program.md contains a complete ReID experiment strategy with search space, constraints, and domain context | VERIFIED | 278-line document with 14 named sections; domain context section lines 162-177; search space table lines 183-202 (16 constants) |
| 2 | program.md encodes all hard constraints (never edit prepare.py, never add deps, never exceed 10 epochs, never stop, edge-deployable) | VERIFIED | "Hard Constraints -- NEVER VIOLATE" section lines 51-69; 10 NEVER occurrences; all 6 required constraints present |
| 3 | program.md describes the full autonomous experiment loop with keep/discard logic and never-stop behavior | VERIFIED | "LOOP FOREVER" at line 129; keep/discard/crash logic lines 147-152; "NEVER STOP" section lines 156-160 |
| 4 | program.md includes prioritized experiment hints organized by impact tier | VERIFIED | "ReID Experiment Playbook" section lines 204-227; Tier 1 (lines 208-212), Tier 2 (lines 214-220), Tier 3 (lines 222-227); 24 experiment hint occurrences |
| 5 | program.md instructs the agent to read results.tsv history and reason about past outcomes before each experiment | VERIFIED | "Before EVERY experiment, read results.tsv" at line 241; `cat results.tsv` command at line 244; 13 total results.tsv occurrences |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `program.md` | Complete ReID-specific agent instructions for autonomous experimentation | VERIFIED | Exists, 278 lines (min_lines: 200 satisfied), contains "NEVER STOP" (2x), substantive content across all 14 sections |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `program.md` | `train.py` | Search space reference table listing all tunable constants | WIRED | Lines 183-202: table contains LR, BATCH_SIZE, ARCFACE_LOSS_WEIGHT, EMBEDDING_DIM and 12 other constants. Pattern matched: all 4 required constants present |
| `program.md` | `results.tsv` | Instructions for reading history and reasoning about next experiment | WIRED | Line 241: "Before EVERY experiment, read results.tsv"; line 244: `cat results.tsv`; 13 total occurrences. History Reasoning section lines 239-257 |
| `program.md` | `prepare.py` | Hard constraint: never edit prepare.py | WIRED | Line 55: "NEVER edit prepare.py -- it contains evaluation, data loading, and teacher inference." Pattern "NEVER edit prepare.py" confirmed present |

---

### Data-Flow Trace (Level 4)

Not applicable. program.md is a documentation artifact (agent instructions), not a code component that renders dynamic data. No data-flow trace required.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| program.md exists and is non-trivial | `wc -l program.md` | 278 | PASS |
| Commits documented in SUMMARY exist in git | `git log --oneline \| grep "0772723\|fc3faab"` | Both commits found | PASS |
| Task 1 commit modified program.md | `git show --stat 0772723` | program.md: 107 insertions | PASS |
| Task 2 commit modified program.md | `git show --stat fc3faab` | program.md: 118 insertions | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| AGNT-01 | 03-01-PLAN.md | program.md contains ReID-specific experiment strategy, constraints, and search space documentation | SATISFIED | Domain Context section (lines 162-177), Search Space Reference (lines 179-202), 6-constraint Hard Constraints section (lines 51-69). Grep: "search space\|Search Space" = 1, "constraint\|Constraint" (search-space\|constraint) = 9 |
| AGNT-02 | 03-01-PLAN.md | program.md includes prioritized experiment hints: loss weights, backbone unfreezing, augmentation, LR schedule, projection head design | SATISFIED | 3-tier Experiment Playbook (lines 204-227). All 5 specified categories present: loss weights (Tier 1), backbone unfreezing (Tier 1), augmentation (Tier 2), LR schedule (Tier 1), projection head design (Tier 2). Grep: 24 matches |
| AGNT-03 | 03-01-PLAN.md | program.md encodes hard constraints: never edit prepare.py, never add dependencies, never exceed 10 epochs, never stop | SATISFIED | "Hard Constraints -- NEVER VIOLATE" section with 6 numbered rules. Grep: NEVER = 10 occurrences (>= 6 required). All 4 specified constraints plus 2 additional (edge limits, quality degradation) |
| AGNT-04 | 03-01-PLAN.md | Agent runs autonomously in a never-stop loop -- modify train.py, run, evaluate, keep/discard, repeat | SATISFIED | "LOOP FOREVER" at line 129; 9-step experiment loop (lines 129-154); "NEVER STOP" section (lines 156-160). Grep: "NEVER STOP\|LOOP FOREVER" = 3 occurrences |
| AGNT-05 | 03-01-PLAN.md | Agent reads results.tsv history to reason about what to try next | SATISFIED | "Reading results.tsv for History Reasoning" section (lines 239-257). "Before EVERY experiment, read results.tsv" explicit instruction at line 241. Grep: results.tsv = 13 occurrences (>= 5 required) |

**All 5 phase-3 requirements SATISFIED.**

No orphaned requirements: AGNT-01 through AGNT-05 are all mapped to Phase 3 in REQUIREMENTS.md traceability table and all appear in the 03-01-PLAN.md frontmatter.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | - |

No TODO/FIXME/placeholder comments detected. No empty implementations. No deferred content. SUMMARY correctly states "Known Stubs: None."

---

### Human Verification Required

None. program.md is a self-contained markdown document whose content can be fully verified by text inspection. There is no runtime behavior, UI, or external service integration to validate. All checks were resolved programmatically.

---

### Gaps Summary

No gaps. All 5 observable truths verified. All 3 key links wired. All 5 AGNT requirements satisfied. The two commits (0772723, fc3faab) documented in the SUMMARY are confirmed present in git history and each modifies program.md as stated.

The phase goal is achieved: program.md (278 lines, 14 sections) gives a Claude Code agent all necessary components for autonomous ReID experimentation -- domain knowledge (knowledge distillation primer), search strategy (3-tier playbook, 16-constant search space), hard constraints (6 NEVER rules), and history-reading capability (explicit pre-experiment results.tsv read instruction).

---

_Verified: 2026-03-25T06:00:00Z_
_Verifier: Claude (gsd-verifier)_
