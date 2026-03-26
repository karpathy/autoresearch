# Phase 3: Agent Instructions - Research

**Researched:** 2026-03-25
**Domain:** LLM agent instructions for autonomous ReID experimentation
**Confidence:** HIGH

## Summary

Phase 3 is a documentation/prompt-engineering phase: write `program.md` that gives Claude Code everything it needs to autonomously run effective ReID experiments. The existing `program.md` in the repo is Karpathy's original template for GPT training (wall-clock budget, BPB metric, generic architecture search). It must be adapted to the ReID knowledge distillation domain with: epoch-based budget, dual metric (recall@1 + mean_cosine), edge-deployment constraints, and domain-specific experiment hints.

The original template is well-structured and proven -- the adaptation is primarily about replacing the domain content (GPT -> ReID) and adding hard constraints that the original did not need (edge deployability, VRAM awareness, augmentation-gaming prevention). The structure (Setup, Constraints, Output Format, Logging, Experiment Loop, NEVER STOP) should be preserved almost exactly.

**Primary recommendation:** Preserve the original program.md structure. Replace domain content. Add a prominent "Hard Constraints" section and a "ReID Experiment Playbook" section with prioritized experiment hints and pitfall warnings.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Claude Code CLI is the agent. program.md is written for Claude Code's capabilities (shell commands, file editing, git operations).
- **D-02:** Claude's discretion on hint detail level -- balance between detailed experiment checklist and directional guidance.
- **D-03:** Search space includes loss weights/composition -- distillation, ArcFace, VAT, separation loss weights and combinations.
- **D-04:** Search space includes model architecture -- projection head design, activation functions. CRITICAL: model must remain edge-deployable (params, GFLOPs, embedding dim close to 256). No unbounded model scaling.
- **D-05:** Search space includes optimizer/LR -- optimizer choice, learning rate, scheduler type, warmup fraction.
- **D-06:** Search space includes augmentation -- training augmentation pipeline (RandomQualityDegradation params, color jitter, random erasing, etc.).
- **D-07:** Never edit prepare.py.
- **D-08:** Never add pip dependencies.
- **D-09:** Never exceed 10 epochs per experiment.
- **D-10:** Never stop the loop (run until manually interrupted).
- **D-11:** Model must remain edge-deployable: monitor params, GFLOPs, embedding dim approximately 256.

### Claude's Discretion
- Exact experiment prioritization order
- Whether to include specific loss function alternatives (circle loss, proxy-anchor, etc.) or let agent discover them
- How to structure the "what to try when stuck" section

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AGNT-01 | program.md contains ReID-specific experiment strategy, constraints, and search space documentation | Original program.md template provides structure; FEATURES.md and ARCHITECTURE.md provide ReID domain content; monolith analysis provides exact tunable parameters |
| AGNT-02 | program.md includes prioritized experiment hints: loss weights, backbone unfreezing, augmentation, LR schedule, projection head design | FEATURES.md differentiators section lists all ReID-specific experiment ideas; monolith argparse reveals exact parameter names and defaults |
| AGNT-03 | program.md encodes hard constraints: never edit prepare.py, never add dependencies, never exceed 10 epochs, never stop | Original program.md constraint section is the template; CONTEXT.md D-07 through D-11 are the exact constraints |
| AGNT-04 | Agent runs autonomously in a never-stop loop -- modify train.py -> run -> evaluate -> keep/discard -> repeat | Original program.md experiment loop section is the template; ARCHITECTURE.md data flow diagram is the reference |
| AGNT-05 | Agent reads results.tsv history to reason about what to try next | Original program.md logging section is the template; results.tsv format from INFRA-02 defines the columns |
</phase_requirements>

## Standard Stack

This phase produces a single markdown file (`program.md`). No libraries or packages are involved.

| Asset | Purpose | Why Standard |
|-------|---------|--------------|
| `program.md` (existing) | Template -- original Karpathy autoresearch agent instructions | Proven structure, 100+ experiments overnight pattern |
| `.planning/research/FEATURES.md` | Source for ReID experiment hints | Differentiators section has all domain-specific ideas |
| `.planning/research/PITFALLS.md` | Source for agent warnings | Metric gaming and augmentation gaming risks |
| `.planning/research/ARCHITECTURE.md` | Source for prepare.py/train.py contract | Defines what agent can/cannot touch |

## Architecture Patterns

### program.md Section Structure (adapted from original)

The original program.md has this proven structure. The ReID version should mirror it:

```
program.md
  1. Setup (branch creation, file reading, baseline)
  2. Experimentation (what to edit, what not to edit, goal metric)
  3. Output Format (grep-able summary block)
  4. Logging Results (results.tsv format and columns)
  5. The Experiment Loop (step-by-step loop with keep/discard)
  6. NEVER STOP (autonomous behavior)
```

**Additions for ReID version:**
```
  7. Hard Constraints (edge deployment, VRAM, parameter budget)
  8. Search Space Reference (all tunable constants with current defaults)
  9. ReID Experiment Playbook (prioritized hints, "when stuck" guidance)
  10. Domain Context (what is ReID, what is knowledge distillation, why this matters)
```

### Pattern 1: Hard Constraints as a Dedicated Section

**What:** A clearly delimited, prominently placed section listing absolute prohibitions.
**When to use:** Always -- hard constraints must be impossible to miss.
**Why:** In the original program.md, constraints are scattered across sections. For ReID, the edge-deployment constraint and prepare.py immutability are safety-critical. They deserve a dedicated, prominent section.

**Structure:**
```markdown
## Hard Constraints -- NEVER VIOLATE

1. **NEVER edit prepare.py** -- evaluation must be immutable
2. **NEVER add pip dependencies** -- only what's in pyproject.toml
3. **NEVER exceed 10 epochs** -- fixed budget for fair comparison
4. **NEVER stop** -- run until manually interrupted
5. **NEVER exceed edge-deployment limits:**
   - Parameter count must stay under [threshold]
   - Embedding dim must remain 256
   - Monitor GFLOPs per forward pass
6. **NEVER remove quality degradation augmentation** -- model must handle real-world degraded images
```

### Pattern 2: Search Space as a Constants Reference Table

**What:** A table listing every tunable constant in train.py, its current default value, and a brief note on what it controls.
**When to use:** Always -- the agent needs to know what knobs exist without reading the entire source.
**Why:** The original program.md just says "everything in train.py is fair game." For ReID, the search space is more complex (multiple loss functions, augmentation params, backbone unfreezing). A reference table prevents the agent from wasting experiments on things that don't exist.

**Source data (from monolith argparse analysis):**

| Constant | Default | Domain |
|----------|---------|--------|
| `LR` | 1e-1 | Optimizer |
| `WEIGHT_DECAY` | 1e-5 | Optimizer |
| `BATCH_SIZE` | 256 | Training |
| `ARCFACE_BATCH_SIZE` | 128 | Training |
| `ARCFACE_LOSS_WEIGHT` | 0.05 | Loss |
| `ARCFACE_S` | 32.0 | Loss (ArcFace scale) |
| `ARCFACE_M` | 0.50 | Loss (ArcFace margin) |
| `VAT_WEIGHT` | 0 (disabled) | Loss |
| `VAT_EPSILON` | 8.0 | Loss |
| `SEP_WEIGHT` | 1.0 | Loss (separation) |
| `UNFREEZE_EPOCH` | 5 | Backbone |
| `QUALITY_DEGRADATION_PROB` | 0.5 | Augmentation |
| `DROP_HARD_RATIO` | 0.2 | ArcFace |
| `ARCFACE_PHASEOUT_EPOCH` | 0 (disabled) | Loss scheduling |
| `EMBEDDING_DIM` | 256 | Model |
| `MODEL_NAME` | `hf-hub:timm/lcnet_050.ra2_in1k` | Model |

Note: These are from the monolith's argparse. Phase 1 refactoring will convert these to module-level constants in train.py. The exact names may change, but the semantics are fixed.

### Pattern 3: Experiment Playbook with Priority Tiers

**What:** Prioritized experiment suggestions organized by expected impact.
**When to use:** Always -- guides agent's exploration strategy.
**Why:** Without guidance, the agent may waste experiments on low-impact changes or try dangerous changes early. Priority tiers encode human research judgment.

**Recommended tier structure:**

**Tier 1 -- High Impact, Low Risk (try first):**
- Loss weight ratios (distillation vs ArcFace vs separation)
- Learning rate and schedule (cosine with warmup vs step decay)
- Unfreeze epoch and number of stages to unfreeze

**Tier 2 -- Medium Impact, Medium Risk:**
- Projection head design (add hidden layers, activation functions, dropout)
- Augmentation parameters (quality degradation intensity, color jitter strength)
- ArcFace margin and scale tuning
- VAT regularization (currently disabled -- try enabling with small weight)
- Optimizer choice (SGD vs AdamW)

**Tier 3 -- High Impact, Higher Risk (try when Tier 1/2 plateau):**
- Different backbone from timm registry (must remain edge-deployable!)
- Batch size changes (watch VRAM)
- ArcFace phaseout scheduling
- Novel loss combinations

### Anti-Patterns to Avoid in program.md

- **Overly prescriptive scripts:** "First try LR=0.05, then try LR=0.01" -- this turns the agent into a grid search. Give directional guidance, not exact values.
- **Missing metric context:** The agent needs to know what a "good" combined metric looks like to calibrate expectations (e.g., baseline is around X, improvements of 0.01 are meaningful).
- **Omitting the results.tsv reasoning instruction:** The original says "LOOP FOREVER" but doesn't explicitly say "read results.tsv before each experiment to reason about history." AGNT-05 requires this.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Agent instructions | Custom agent framework | A well-written program.md | Claude Code reads markdown natively; no framework needed |
| Experiment scheduling | Priority queue or scheduler | Hints in program.md + agent reasoning | The LLM IS the scheduler; encoding priorities as instructions is sufficient |
| Search space definition | YAML config of search bounds | Constants reference table in program.md | Agent reads constants directly from code; the reference table is for orientation only |

## Common Pitfalls

### Pitfall 1: Agent Removes Quality Degradation Augmentation
**What goes wrong:** Agent discovers removing RandomQualityDegradation improves recall@1 on clean validation images. Keeps the change. Model fails on real-world degraded inputs.
**Why it happens:** Validation set has clean images. Augmentation removal always helps on clean-image benchmarks in short runs.
**How to avoid:** Encode in program.md as a hard constraint: "NEVER remove or disable RandomQualityDegradation. You may tune its parameters (prob, downsample_ratio, quality_range) but it must remain active."
**Warning signs:** recall@1 jumps significantly after augmentation change.

### Pitfall 2: Agent Scales Model Beyond Edge Deployment
**What goes wrong:** Agent tries a larger backbone or wider projection head. Metric improves. Model is now too large for edge deployment.
**Why it happens:** Agent optimizes for combined_metric without considering deployment constraints.
**How to avoid:** Encode edge deployment limits explicitly in program.md. Agent must check parameter count and GFLOPs after architecture changes. Include the check command in the instructions.
**Warning signs:** Peak VRAM increases significantly after architecture change.

### Pitfall 3: OOM Cascade Loop
**What goes wrong:** Agent tries increasingly large batch sizes or model configurations, hits OOM repeatedly.
**Why it happens:** results.tsv shows "crash" but agent doesn't understand the VRAM implications.
**How to avoid:** In program.md, include a VRAM budget rule: "If peak VRAM > 22GB on a successful run, do NOT increase batch size or model size further. If you hit OOM, your next experiment must REDUCE compute, not try the same thing with a minor tweak."
**Warning signs:** Multiple consecutive "crash" entries in results.tsv.

### Pitfall 4: Agent Edits prepare.py Imports
**What goes wrong:** Agent adds import from prepare.py that accesses internal state, or modifies how prepare.py functions are called in a way that changes evaluation semantics.
**Why it happens:** The "never edit prepare.py" rule is clear, but agent might reinterpret how imports are used.
**How to avoid:** program.md should list the exact public API available from prepare.py. "These are the ONLY things you import. Do not add new imports from prepare.py."
**Warning signs:** git diff shows changes to import statements.

### Pitfall 5: Stale Experiment Strategy
**What goes wrong:** Agent follows the same strategy for 50+ experiments without improvement, trying minor variations of the same idea.
**Why it happens:** Program.md doesn't give guidance on when to switch strategies.
**How to avoid:** Include a "when stuck" section: "If 5 consecutive experiments show no improvement, try a radically different approach from a different tier. If 10 show no improvement, try combining the best ideas from your history."

## Code Examples

### Example 1: Setup Section Adaptation

Original (GPT autoresearch):
```markdown
## Setup
1. Agree on a run tag
2. Create the branch: `git checkout -b autoresearch/<tag>`
3. Read the in-scope files: README.md, prepare.py, train.py
4. Verify data exists
5. Initialize results.tsv
6. Confirm and go
```

ReID adaptation:
```markdown
## Setup
1. Agree on a run tag based on today's date (e.g. `mar25`)
2. Create the branch: `git checkout -b autoresearch/<tag>`
3. Read the in-scope files:
   - `program.md` -- these instructions (you are reading it)
   - `prepare.py` -- IMMUTABLE: data loading, teacher inference, evaluation, caching
   - `train.py` -- YOUR FILE: student model, losses, optimizer, scheduler, augmentations
4. Verify teacher cache exists at workspace/output/trendyol_teacher_cache2/
   - If not, the first run will build it automatically (takes ~10-30 min, excluded from budget)
5. Initialize results.tsv with header:
   `commit\tcombined_metric\trecall_1\tmean_cosine\tpeak_vram_mb\tstatus\tdescription`
6. Run baseline: `python train.py > run.log 2>&1`
7. Record baseline in results.tsv, commit as baseline
8. Begin experiment loop
```

### Example 2: Output Format Section

```markdown
## Output Format

After each run, the script prints a summary block:
---
combined_metric:  0.654321
recall@1:         0.432100
mean_cosine:      0.876543
peak_vram_mb:     18432.1
total_seconds:    342.5
epochs:           10

Extract the key metrics:
grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram_mb:" run.log
```

### Example 3: Hard Constraints Section

```markdown
## Hard Constraints -- NEVER VIOLATE

These are absolute rules. Violating any one invalidates all experiments.

1. **NEVER edit prepare.py** -- it contains evaluation, data loading, and teacher inference.
   Modifying it breaks the trust boundary. All experiments become non-comparable.

2. **NEVER install new packages** -- only use what's in pyproject.toml.
   Available: torch, timm, onnxruntime, transformers, torchvision, numpy, PIL.

3. **NEVER exceed 10 epochs** -- this is the fixed experiment budget.
   You optimize WHAT happens in 10 epochs, not how many epochs to train.

4. **NEVER stop the loop** -- run until manually interrupted.
   The human may be asleep. Do NOT ask "should I continue?"

5. **NEVER exceed edge deployment limits:**
   - Embedding dimension MUST remain 256
   - After any architecture change, check: `python -c "import train; ..."`
   - Parameter count must not grow unbounded
   - If you change the backbone, verify it's still lightweight (LCNet-class)

6. **NEVER remove quality degradation augmentation** -- real-world images are degraded.
   You may tune its parameters but it must remain active.
```

### Example 4: Experiment Loop (ReID-specific)

```markdown
## The Experiment Loop

LOOP FOREVER:

1. **Read history**: Check results.tsv. What has been tried? What improved?
   What patterns emerge? (e.g., "lower LR helped", "VAT hurts", "ArcFace weight 0.1 > 0.05")

2. **Choose next experiment**: Based on history, pick an idea from the playbook
   or formulate your own hypothesis. Prefer unexplored dimensions over
   minor variations of explored ones.

3. **Edit train.py**: Make your changes. Keep diffs minimal and focused.
   One idea per experiment for clear attribution.

4. **git commit**: Commit the change with a descriptive message.

5. **Run**: `python train.py > run.log 2>&1`

6. **Read results**: `grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram_mb:" run.log`
   - If grep is empty: run crashed. `tail -n 50 run.log` for stack trace.

7. **Log to results.tsv**: Record commit hash, metrics, VRAM, status, description.
   NOTE: results.tsv is NOT git-tracked.

8. **Keep or discard**:
   - combined_metric improved? KEEP (advance branch)
   - Same or worse? DISCARD (`git reset --hard HEAD~1`)
   - Crash? Log as crash, `git reset --hard HEAD~1`
   - 3+ consecutive crashes on same idea? SKIP that direction entirely.

9. GOTO 1
```

## Adaptation Map: Original program.md to ReID program.md

Detailed mapping of what changes between the original and the ReID version:

| Section | Original (GPT) | ReID Adaptation | Change Type |
|---------|----------------|-----------------|-------------|
| Setup | `uv run prepare.py` for data | Teacher cache verification; `python train.py` | Replace |
| Metric | `val_bpb` (lower is better) | `combined_metric` = 0.5*recall@1 + 0.5*mean_cosine (higher is better) | Replace |
| Budget | 5 minutes wall-clock | 10 epochs fixed | Replace |
| What to edit | "Everything in train.py is fair game" | Same, but with edge-deployment constraints | Extend |
| Output format | val_bpb, peak_vram_mb, mfu_percent | combined_metric, recall@1, mean_cosine, peak_vram_mb | Replace |
| results.tsv columns | commit, val_bpb, memory_gb, status, description | commit, combined_metric, recall_1, mean_cosine, peak_vram_mb, status, description | Extend |
| VRAM guidance | "Soft constraint" | Hard constraint: 24GB RTX 4090, agent must be VRAM-aware | Strengthen |
| Simplicity criterion | Yes -- prefer simpler code | Adapt -- prefer simpler but maintain edge-deployability | Extend |
| NEVER STOP | Present | Keep verbatim (it's perfect as-is) | Keep |
| Hard constraints | Scattered | New dedicated section | Add |
| Search space reference | Not present (generic) | New section: all tunable constants listed | Add |
| Experiment playbook | Not present (generic) | New section: ReID-specific prioritized hints | Add |
| Domain context | Not present (assumed GPT knowledge) | Brief ReID/distillation primer for agent | Add |
| Keep/discard direction | Keep if val_bpb lower | Keep if combined_metric higher | Invert |
| Run command | `uv run train.py` | `python train.py` | Replace |
| Crash handling | Brief | Extended: OOM-specific guidance, VRAM budget rule | Extend |

## Key Decisions for Planner

### Decision 1: Experiment Hint Detail Level

**Recommendation:** Use directional guidance with specific examples, not exact parameter values.

Good: "Try reducing ArcFace loss weight. The default is 0.05 -- experiment in the range 0.01-0.2."
Bad: "Set ARCFACE_LOSS_WEIGHT to 0.03."
Bad: "Try different loss weights." (too vague)

The agent should understand the direction and have a starting range, but choose its own values based on results history.

### Decision 2: Whether to Include Specific Alternative Loss Functions

**Recommendation:** Include brief mentions of circle loss and subcenter ArcFace as "advanced" ideas in Tier 3, but do NOT include implementation code. The agent should implement them if it decides to explore that direction, since it has access to PyTorch and the math is standard.

Reason: Including implementation code makes program.md too long and prescriptive. The agent is a capable coder -- it just needs to know these options exist.

### Decision 3: "When Stuck" Section Structure

**Recommendation:** Three escalation levels:

1. **Minor plateau (3-5 no-improvement experiments):** Switch to a different parameter dimension (e.g., if tuning LR, switch to loss weights)
2. **Medium plateau (5-10 no-improvement experiments):** Jump to a different tier. Re-read results.tsv and identify the single best-performing configuration, then try combining it with ideas from unexplored dimensions.
3. **Major plateau (10+ no-improvement experiments):** Try radical changes: different backbone, novel loss function, extreme augmentation parameter changes. Read the current code carefully for overlooked opportunities.

### Decision 4: results.tsv Column Format

**Recommendation (extends INFRA-02):** Use these columns for the ReID version:
```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
```

The decomposed metrics (recall_1, mean_cosine) are critical for agent reasoning per AGNT-05. The agent should use them to understand WHY a change helped or hurt.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual review (this phase outputs a markdown file, not code) |
| Config file | N/A |
| Quick run command | `test -f program.md && wc -l program.md` |
| Full suite command | Manual review against AGNT-01 through AGNT-05 checklist |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AGNT-01 | program.md has ReID strategy, constraints, search space | manual | `grep -c "search space\|constraint\|Search Space\|Constraint" program.md` | N/A -- Wave 0 |
| AGNT-02 | program.md has prioritized experiment hints | manual | `grep -c "loss weight\|backbone\|augmentation\|LR\|projection" program.md` | N/A -- Wave 0 |
| AGNT-03 | program.md encodes hard constraints | manual | `grep -c "NEVER" program.md` (expect >= 4) | N/A -- Wave 0 |
| AGNT-04 | program.md describes never-stop loop | manual | `grep -c "NEVER STOP\|LOOP FOREVER\|loop forever" program.md` | N/A -- Wave 0 |
| AGNT-05 | program.md instructs reading results.tsv | manual | `grep -c "results.tsv" program.md` (expect >= 3) | N/A -- Wave 0 |

### Sampling Rate
- **Per task commit:** `grep -c "NEVER\|results.tsv\|combined_metric" program.md`
- **Per wave merge:** Manual review of program.md against all AGNT requirements
- **Phase gate:** All 5 AGNT requirements verifiable by reading program.md

### Wave 0 Gaps
None -- this phase creates program.md from scratch. No test infrastructure needed. Validation is by manual review and content verification greps.

## Open Questions

1. **Baseline metric value**
   - What we know: The combined metric is 0.5 * recall@1 + 0.5 * mean_cosine
   - What's unclear: What is the expected baseline value from unmodified train.py? This matters for program.md so the agent can calibrate "good improvement" vs "noise"
   - Recommendation: Phase 4 (VALD-01) establishes the baseline. program.md should say "record the baseline first, then treat improvements of >0.005 as meaningful"

2. **Edge deployment thresholds**
   - What we know: LCNet050 is the baseline backbone, embedding dim = 256
   - What's unclear: Exact parameter count and GFLOPs limits for "edge-deployable"
   - Recommendation: Include a check command in program.md that prints params/GFLOPs. Set a soft limit of 2x the baseline model's parameters as the maximum. The agent should log these in the experiment description.

3. **Teacher model choice**
   - What we know: Two teachers available (ONNX and DINOv2). STATE.md flags this as a research gap.
   - What's unclear: Which teacher will be used by the time Phase 3 executes (Phase 1 decides this).
   - Recommendation: program.md should reference "the teacher" generically. The specific teacher is configured in prepare.py and is not the agent's concern.

## Sources

### Primary (HIGH confidence)
- `program.md` in repo -- original Karpathy autoresearch agent instructions (direct analysis)
- `finetune_trendyol_arcface3.py` -- current monolith with all tunable parameters (direct analysis)
- `.planning/research/FEATURES.md` -- differentiators for ReID experiments
- `.planning/research/PITFALLS.md` -- metric gaming and augmentation gaming risks
- `.planning/research/ARCHITECTURE.md` -- prepare.py/train.py contract and data flow

### Secondary (MEDIUM confidence)
- Karpathy autoresearch GitHub -- canonical three-file pattern and loop structure
- ReID domain knowledge -- ArcFace, knowledge distillation, VAT regularization patterns

## Metadata

**Confidence breakdown:**
- Template structure: HIGH - direct analysis of proven original program.md
- ReID domain adaptation: HIGH - thorough analysis of existing codebase, FEATURES.md, PITFALLS.md
- Experiment hints: MEDIUM - based on domain research and code analysis, but untested in this specific setup
- Edge deployment thresholds: MEDIUM - exact limits depend on Phase 1 output

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable -- program.md structure is settled, domain doesn't change fast)
