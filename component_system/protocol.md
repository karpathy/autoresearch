# autoresearch — Component-System Protocol

This document is the operating manual for the component-system workflow.
The system runs a continuous Seed -> P -> DCA loop to discover, generate,
adapt, evaluate, and promote improvements to the training stack.

The main objective is simple: improve `val_bpb` against the current baseline
without breaking the canonical component-system entrypoint or introducing
unreasonable complexity.

VRAM is a first-class constraint. Higher memory use is acceptable only when the
quality gain is meaningful; avoid candidates that produce small or ambiguous
`val_bpb` gains while causing large memory growth.

## Top-Level Bootstrap Rule

If you are an interactive code agent that was merely told to "follow this
protocol", do not manually simulate the entire workflow inside one foreground
session.

The intended control flow is:
1. Read this file and the required context files.
2. Ensure the queue and state layout exist.
3. Create or refine a seed from a human prompt.
4. Queue the seed for P.
5. Start the daemon: `uv run component_system/run.py`.
6. Let daemon workers execute P and DCA via file-based handoff.
7. Monitor the daemon, queue, and logs; do not simulate stages in-session.

Manual execution of an individual stage is only for the agent process that was
invoked by the daemon for that specific task.

## Architecture

```text
component_system/
  protocol.md               <- overall workflow protocol
  entrypoint.py             <- canonical training entrypoint
  PDCA-PLAN.md              <- P stage rules
  PDCA-DO-CHECK-ACTION.md   <- DCA stage rules
  run.py                    <- resident daemon and worker dispatch
  task.py                   <- queue and JSON state helpers
  baseline_branches.json    <- per-branch baseline mapping (workflow-managed; read-only)
  baseline_metrics.json     <- baseline run metrics (workflow-managed; read-only)
  config.py                 <- promotion threshold and static binding
  history/                  <- runtime dir (auto-created)
    logs/                   <- agent stdout/stderr logs
    queue/{p,dca,done,error}/ <- stage handoff and archival
    state/{seeds,runs,events}/<- durable workflow state
    worktrees/              <- per-seed git worktrees
  components/
    model.py
    optimizer.py
    trainer.py
  training/
    mainline.py
```

## Core Goal and Decision Rule

Optimize for lower `val_bpb`. A candidate is worth promoting only when the gain
is real, the implementation is understandable, and the cost in memory or
complexity is justified.

Apply this bias consistently:
- Lower `val_bpb` is the primary success metric.
- VRAM is a soft but important constraint: some increase is acceptable, but
  dramatic growth needs correspondingly strong quality gains.
- Simpler changes are preferred when results are similar.
- A tiny gain that adds brittle complexity is usually not worth promotion.
- A tiny gain that materially increases VRAM is usually not worth promotion.
- A simplification that preserves or slightly improves quality is a strong outcome.
- If the signal is ambiguous, treat it as `neutral` and do not promote.

## Required Reading Before Any Work

Read in this order:
1. `component_system/protocol.md`
2. The stage-specific document (right after protocol): `component_system/PDCA-DO-CHECK-ACTION.md` for DCA, `component_system/PDCA-PLAN.md` for P
3. `prepare.py` for fixed data and evaluation behavior; never modify it
4. `component_system/entrypoint.py` for the canonical execution path
5. `component_system/config.py` for promotion threshold and static binding

Baseline files (workflow-managed; read-only): `baseline_branches.json`, `baseline_metrics.json`. For interactive bootstrap, inspect recent `queue/done/` and baseline state.

## Workspace and Path Rules

When the daemon invokes you for a P or DCA task, your current working directory
is the seed worktree. In that mode:

- Read and edit only within the seed worktree.
- Use only relative paths from the current working directory.
- Do not request or depend on absolute paths or files outside the worktree.

## Hard Constraints

1. Never modify `prepare.py`.
2. `uv run component_system/entrypoint.py` must remain the canonical,
   working component-system training command.
3. The root repo must stay compatible with the upstream implementation;
   do not require changes to root `train.py`.
4. Stage-to-stage handoff must happen through files under `queue/`, not
   merely in memory or only in agent conversation state.
5. Only the DCA promotion flow may update `baseline_metrics.json` and `baseline_branches.json`.
6. Do not bypass the baseline mechanism by manually merging branches or
   force-advancing the baseline outside workflow control.

## Baseline-First Rule

Establish baseline before evaluating seeds: if `baseline_metrics.json` has no `last_val_bpb`, run the baseline (no-changes) measurement first. Use that result as the reference for promotion.

```mermaid
flowchart TD
    A[Create seed] --> B{Baseline result exists?}
    B -- No --> C[Create or reuse __baseline__ seed]
    C --> D[Queue baseline DCA]
    D --> E[Run baseline measurement from project root]
    E --> F[Save baseline metrics in baseline_metrics.json]
    F --> G[Release waiting seeds]
    B -- Yes --> G
    G --> H[Seed stays in draft or queued with no worktree]
    H --> I[Queue P run]
    I --> J[Create seed worktree at P start]
    J --> K[P agent plans and commits on seed branch]
    K --> L[Queue DCA run]
    L --> M[DCA agent adapts, runs training, and reports metrics]
    M --> N{Promotion signal?}
    N -- Positive --> O[Merge seed branch into baseline]
    O --> P{Merge conflict?}
    P -- No --> Q[Update baseline metadata and finish seed]
    P -- Yes --> R[Queue conflict-resolution DCA]
    R --> M
    N -- Neutral or Negative --> S[Keep result in state only]
```

## Workflow Stages

The sections below describe what each daemon-dispatched stage worker does.
They are not instructions for a top-level interactive agent to perform the
entire lifecycle manually.

### P — Discovery / Plan / Initial Generation

Read `component_system/PDCA-PLAN.md`.

Responsibilities:
- Refine the seed prompt into a concrete plan.
- Create or refresh the seed worktree from the active baseline.
- Generate the first candidate implementation in the worktree.
- Keep the change focused enough that DCA can evaluate it cleanly.
- Commit the generated candidate on the seed branch so DCA receives a stable snapshot.

P is about producing a plausible, testable first version, not claiming success.

### DCA — Delivery / Check / Action

Read `component_system/PDCA-DO-CHECK-ACTION.md`.

Responsibilities:
- Adapt and fix the generated candidate inside the seed worktree.
- Run the canonical training/evaluation entrypoint.
- Read the structured metrics from the run output.
- Decide whether the result is positive, neutral, or negative relative to baseline.
- Promote the seed branch into baseline only when the signal is strong enough.

DCA is the stage that turns a raw idea into a measured outcome.

## Canonical Run and Output

The canonical component-system execution path is:

```bash
uv run component_system/entrypoint.py
```

Allow **at least 600 seconds** when DCA runs this (e.g. `timeout 600 uv run ...`).

DCA must report a structured JSON summary (including `metrics`). Runner uses it first; falls back to stdout/stderr parsing if missing. No metrics → recovery DCA inspects logs. Canonical metrics:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
startup_seconds:  25.8
```

Treat `val_bpb` as the primary metric. `peak_vram_mb`, total runtime, and code
complexity are secondary constraints that influence promotion decisions.

## VRAM Rule

Track `peak_vram_mb` on every serious evaluation run and treat it as required
decision input, not a cosmetic metric.

- Some VRAM growth is acceptable when it buys a clear `val_bpb` improvement.
- Large VRAM increases require a correspondingly strong quality gain.
- If two candidates are similar on `val_bpb`, prefer the lower-VRAM one.
- If a candidate regresses or barely improves `val_bpb` while increasing VRAM
  substantially, treat it as a bad trade and do not promote it.
- Avoid changes that risk blowing up memory usage unless the expected upside is
  compelling enough to justify the experiment.

## Promotion Rule

A run is promotable only if all of the following hold:
- The run completed successfully.
- `val_bpb` improved enough over the active baseline to count as a real win.
- VRAM growth is not unreasonable for the magnitude of the gain.
- The change is understandable, maintainable, and reversible.

If the candidate is equal, worse, noisy, or hard to justify, do not promote it.
Record the outcome and move on.

## Failure Handling

Use the same judgment standard as the original autoresearch loop:

- If a run crashes because of a simple bug, fix it, rerun, and update the same
  run record.
- If the idea is fundamentally flawed, archive it without promotion.
- If the task cannot be recovered quickly, move it into the error flow and
  persist the failure details.
- Crashes are negative evidence; they should not silently disappear.

## Bootstrap Procedure for Interactive Sessions

1. Read baseline files and recent queue/state.
2. Ensure queue/state/worktree layout exists.
3. Create a seed from the human prompt and queue it for P.
4. Start `uv run component_system/run.py` and monitor; do not run P/DCA manually.

## Operating Loop

1. Seed persisted in `state/seeds/`, queued to `queue/p/`.
2. P refreshes worktree, generates code, commits on seed branch.
3. Daemon queues DCA.
4. DCA adapts, runs, evaluates; promotes or archives.
5. State persisted under `state/`; daemon continues with next work.

## State and Logging

- `component_system/history/state/seeds/`, `component_system/history/state/runs/`, `component_system/history/state/events/` — seed and run state.
- `component_system/history/queue/done/`, `component_system/history/queue/error/` — completed and failed tasks.
- `component_system/history/logs/` — agent stdout/stderr.

Use filesystem state as source of truth, not chat context.

## Daemon

`run.py` runs two single-threaded workers polling `component_system/history/queue/p/` and `component_system/history/queue/dca/`. Workers dispatch to an external code agent; the agent reads files, edits the worktree, runs the entrypoint, and prints structured summaries.

Start:

```bash
# Default backend
uv run component_system/run.py

# Alternate backends
PDCA_AGENT=codex    uv run component_system/run.py
PDCA_AGENT=opencode uv run component_system/run.py
```

### Agent Backends

| `PDCA_AGENT` | CLI invoked | Prompt delivery |
|--------------|-------------|-----------------|
| `claude` (default) | `claude -p --verbose` | stdin |
| `codex` | `codex exec -a never --sandbox workspace-write` | positional arg |
| `opencode` | `opencode run` | positional arg |

### Timeouts

Each stage has a default timeout in seconds and can be overridden through the
environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PDCA_TIMEOUT_P` | 900 | Planning and initial code generation |
| `PDCA_TIMEOUT_DCA` | 3600 | Adaptation, training, evaluation, and promotion |

### Logs

Agent stdout/stderr → `component_system/history/logs/`.
