# Roadmap: ReID Autoresearch

## Overview

Transform the monolithic `finetune_trendyol_arcface3.py` (~1400 lines) into the autoresearch three-file pattern (prepare.py, train.py, program.md), wire up experiment infrastructure (git loop, results logging, crash recovery), write ReID-specific agent instructions, and validate with a live autonomous run. The end state: an AI agent that runs experiments overnight on an RTX 4090, autonomously discovering better ReID model configurations.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Refactoring** - Split monolith into prepare.py (immutable) and train.py (agent-editable) with clear trust boundary
- [ ] **Phase 2: Experiment Infrastructure** - Git management, results logging, crash recovery, and VRAM tracking for the experiment loop
- [ ] **Phase 3: Agent Instructions** - ReID-specific program.md with experiment strategy, search space, and hard constraints
- [ ] **Phase 4: Validation** - Baseline run, full autonomous loop cycle, and crash recovery verification

## Phase Details

### Phase 1: Core Refactoring
**Goal**: The monolith is cleanly split so that prepare.py owns all immutable concerns (data, teacher, evaluation, caching) and train.py is a self-contained, agent-editable training script
**Depends on**: Nothing (first phase)
**Requirements**: REFAC-01, REFAC-02, REFAC-03, REFAC-04, REFAC-05, REFAC-06, REFAC-07
**Success Criteria** (what must be TRUE):
  1. Running `python prepare.py` loads all datasets, builds/loads teacher cache, and is ready to evaluate a trained model -- without touching train.py internals
  2. Running `python train.py` trains for 10 epochs and produces a model whose `.encode(images)` returns L2-normalized `Tensor[B, 256]`
  3. prepare.py computes and prints the combined metric (`0.5 * recall@1 + 0.5 * mean_cosine`) after a train.py run completes
  4. All tunable parameters in train.py are module-level constants (no argparse, no config files) -- an agent can modify them by editing Python source
  5. Evaluation logic (recall@1/k, mean_cosine) exists only in prepare.py -- grep confirms zero evaluation code in train.py
**Plans:** 3 plans
Plans:
- [x] 01-01-PLAN.md -- Create prepare.py with all immutable infrastructure (data, teacher, evaluation, metrics)
- [x] 01-02-PLAN.md -- Create train.py with all agent-editable components (model, losses, augmentations, training loop)
- [x] 01-03-PLAN.md -- Smoke tests and boundary verification for the split (pytest suite)

### Phase 2: Experiment Infrastructure
**Goal**: A complete experiment loop harness exists that can run train.py, log results, manage git state, and recover from crashes -- ready for an agent to drive
**Depends on**: Phase 1
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07
**Success Criteria** (what must be TRUE):
  1. After a successful experiment, the change is committed to git and results.tsv contains a row with commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status "kept", and a description
  2. After a regression, git reset is performed, results.tsv logs the experiment as "discarded", and train.py reverts to its previous state
  3. An OOM crash is caught, logged as "crash" in results.tsv, git reset is performed, and the loop continues to the next iteration without human intervention
  4. Run output includes all decomposed sub-metrics (recall@1, recall@5, mean_cosine, distill_loss, arc_loss, vat_loss, sep_loss) in a greppable format
  5. Teacher cache build time is excluded from the 10-epoch experiment budget
**Plans:** 2 plans
Plans:
- [ ] 02-01-PLAN.md -- Add metrics.json output, crash handling, VRAM tracking, and greppable stdout summary to train.py
- [ ] 02-02-PLAN.md -- Create infrastructure test suite verifying metrics.json contract, results.tsv format, and crash handling

### Phase 3: Agent Instructions
**Goal**: program.md gives an LLM agent everything it needs to autonomously run effective ReID experiments -- domain knowledge, search strategy, constraints, and history-reading capability
**Depends on**: Phase 2
**Requirements**: AGNT-01, AGNT-02, AGNT-03, AGNT-04, AGNT-05
**Success Criteria** (what must be TRUE):
  1. program.md documents the ReID search space (loss weights, backbone unfreezing schedule, augmentation strategies, LR schedules, projection head designs) with prioritized experiment hints
  2. program.md encodes all hard constraints: never edit prepare.py, never add dependencies, never exceed 10 epochs, never stop the loop
  3. program.md instructs the agent to read results.tsv history and reason about what to try next based on past experiment outcomes
  4. program.md describes the full autonomous loop (modify train.py, run, evaluate, keep/discard, repeat) with explicit instructions for never-stop behavior
**Plans:** 1 plan
Plans:
- [ ] 03-01-PLAN.md -- Write complete program.md with ReID domain content, search space, experiment playbook, and autonomous loop instructions

### Phase 4: Validation
**Goal**: The complete system is proven to work end-to-end -- baseline established, autonomous loop demonstrated, crash recovery verified
**Depends on**: Phase 3
**Requirements**: VALD-01, VALD-02, VALD-03
**Success Criteria** (what must be TRUE):
  1. Baseline run of unmodified train.py produces a valid combined metric logged as the first row in results.tsv
  2. At least one full autonomous cycle completes: agent reads history, modifies train.py, runs experiment, evaluates, and correctly keeps or discards the result
  3. Intentional OOM trigger is caught, logged as crash, git-reset performed, and the agent continues to the next experiment without human intervention
**Plans:** 2 plans
Plans:
- [ ] 04-01-PLAN.md -- Baseline run + one full autonomous cycle (VALD-01, VALD-02)
- [ ] 04-02-PLAN.md -- Crash recovery verification + launch overnight autonomous run (VALD-03, D-02)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Refactoring | 0/3 | Planning complete | - |
| 2. Experiment Infrastructure | 0/2 | Planning complete | - |
| 3. Agent Instructions | 0/1 | Planning complete | - |
| 4. Validation | 0/2 | Planning complete | - |
