# DCA — Do, Check, Action

## Responsibility
Take the generated plan from P, adapt/fix it in the seed worktree,
run the canonical training entrypoint, evaluate results against baseline, and
promote only when the signal is positive. Do not propose new ideas or optimize for better metrics; only adapt/fix so the plan runs and report outcomes.

## Workspace and paths
**CWD = seed worktree.** Read and edit only inside it; use relative paths only. Treat `component_system/` in the worktree as canonical context.

## Input
- Runner prompt (task content).
- Baseline: `component_system/baseline_branches.json`, `component_system/baseline_metrics.json`.
- Worktree-local files only.

## Baseline measurement (seed_id __baseline__)
Retry until the run succeeds and you report real metrics. No empty metrics.

- **OOM:** Reduce `device_batch_size` in `component_system/components/trainer.py` (default 128); keep `total_batch_size % (device_batch_size * sequence_length) == 0`. Rerun until training completes.
- Only trivial fixes (e.g. batch size); no model/training logic changes.
- **Commit before reporting.** Uncommitted changes break the follow-up merge.

## Workflow
1. Work in the seed worktree (one branch per seed).
2. Adapt/fix until it runs (runtime only: bugs, OOM, imports, config; no model/hyperparameter/training-logic changes for better metrics).
3. Run canonical command (**≥600s**): `timeout 600 uv run --active component_system/entrypoint.py`. **Must set command/tool timeout ≥600s running this command** when invoking this run (so the process is not killed early).
4. On bug/OOM: fix and rerun; for baseline, retry until success.
5. Commit on seed branch before reporting.
6. Print DCA summary block with `commit_sha` in JSON.
7. Runner evaluates signal and handles promotion.

## Output Format
Print the summary block. Put metrics in JSON; runner falls back to stdout/stderr parsing if missing.

```text
AUTORESEARCH_DCA_SUMMARY_BEGIN
{"checks":["entrypoint"],"notes":"...","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"git sha","metrics":{"val_bpb":1.24,...}}
AUTORESEARCH_DCA_SUMMARY_END
```

If no final metrics, use `"metrics": {}`. Runner extracts from stdout/stderr: `val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth`. No metrics → recovery DCA inspects logs; only then treat as failed.

## Check: Signal Rules

| Condition | Signal |
|-----------|--------|
| `val_bpb` drops >= 0.001 vs baseline | `positive_signal` |
| `val_bpb` rises >= 0.001 vs baseline | `negative_signal` |
| difference < 0.001 | `neutral` |
| no historical baseline `last_val_bpb` | `positive_signal` (first recording) |
| metrics missing or training error | `error` |

The threshold is defined in `component_system/config.py` (`PROMOTION_THRESHOLD`).

## Action: Promotion Rules

Only DCA may trigger a merge into baseline; P must not. Runner records `commit_sha`; on positive signal the workflow merges seed → baseline. Merge conflict → system queues merge-resolution DCA.

### Promotion (`positive_signal`)
1. System merges seed into baseline (you do not run merge).
2. Workflow updates `baseline_metrics.json` / `baseline_branches.json`.
3. Metadata in seed/run state.

### Merge failure
- **Normal seed:** In seed worktree: `git merge __baseline__`, resolve conflicts, commit, print DCA summary for retry.
- **Baseline seed (__baseline__):** Merge __baseline__ into target (e.g. master). Run from worktree that has target checked out (`git worktree list`); do not run from __baseline__ worktree or `git merge master` there.

### Non-promotion
`neutral` / `negative_signal` / `error`: log only. Failure info in queue/state logs.

## Constraints
- No model/optimizer/training-logic changes for better metrics; only make the plan run (bugs, OOM, etc.).
- Use `run_mainline_training` (or equivalent); do not skip `val_bpb` evaluation.
- Do not edit baseline JSON files; only DCA promotion updates them.
- Canonical runner: `component_system/entrypoint.py`. Traceability: git + state files.
