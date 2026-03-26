# CA — Do, Check, Action

## Responsibility
Take the generated plan from PD, adapt/fix it in the seed worktree,
run the project's canonical command (script defined in protocol and below; e.g. train.py) using the **Python executable injected by the daemon**, evaluate results against baseline, and
promote only when the signal is positive. Do not propose new ideas or optimize for better metrics; only adapt/fix so the plan runs and report outcomes.

## Workspace and paths
**CWD = seed worktree.** Read and edit only inside it; use relative paths only. Treat `pdca_system/` in the worktree as canonical context.

## Input
- Runner prompt (task JSON), including optional **Plan-Do summary** (`pd_summary`): authoritative idea, description, and PD commit for what was implemented (the daemon may also repeat this as a labeled block above the task JSON).
- Baseline: `pdca_system/baseline_branches.json`, `pdca_system/baseline_metrics.json`.
- Worktree-local files only.

## Baseline measurement (seed_id __baseline__)
Retry until the run succeeds and you report real metrics. No empty metrics.

- **OOM:** Reduce `DEVICE_BATCH_SIZE` in `train.py` (default 128); keep `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0`. Rerun until training completes.
- Only trivial fixes (e.g. batch size); no model/training logic changes.
- **Commit before reporting.** Uncommitted changes break the follow-up merge.

## Workflow
1. Work in the seed worktree (one branch per seed).
2. Adapt/fix until it runs (runtime only: bugs, OOM, imports, config; no model/hyperparameter/training-logic changes for better metrics).
3. Prefer **running the canonical command early** (right after you know which script to run from protocol/this doc): avoid a long upfront code review or redesign; iterate from command failures.
4. Run the **canonical command** (**≥900s**): the daemon injects the **Python executable** (the one running the daemon) into your task prompt. Use that Python for every Python command in this stage, together with the canonical script defined in this doc and protocol (e.g. `train.py` or the script your project uses), for example `{python_exe} train.py > training.log 2>&1` (or `{python_exe} train.py 2>&1 | tee training.log`). **Must set command/tool timeout ≥900s**. After the run, inspect `training.log` to confirm completion and recover or verify metrics.
5. On bug/OOM: fix and rerun; for baseline, retry until success.
6. Commit on seed branch before reporting.
7. Output CA summary block with `commit_sha` in JSON.
8. Runner evaluates signal and handles promotion.

## Output Format
Write the summary JSON to the file named `autoresearch_summary.json` in your current working directory (cwd root). Do not print it to stdout or stderr. Put metrics in the JSON; the runner reads only this file.

```json
{"checks":["entrypoint"],"notes":"...","completed_at":"YYYY-MM-DD HH:MM:SS","commit_sha":"git sha","metrics":{"val_bpb":1.24,...}}
```

If no final metrics, use `"metrics": {}`. The **target metric** key (see `pdca_system/config.py`: `TARGET_METRIC_KEY`), plus `training_seconds`, `total_seconds`, `peak_vram_mb`, `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`, `depth` must be in the JSON `metrics` block. No metrics → recovery CA inspects logs; only then treat as failed.

## Check: Signal Rules

The **target metric** (key and direction) is configured in `pdca_system/config.py`: `TARGET_METRIC_KEY`, `TARGET_METRIC_LOWER_IS_BETTER`, `TARGET_METRIC_LABEL`. Default: `val_bpb`, lower is better.

| Condition | Signal |
|-----------|--------|
| target metric improves by >= threshold vs baseline (e.g. `val_bpb` drops >= 0.001) | `positive_signal` |
| target metric regresses by >= threshold vs baseline (e.g. `val_bpb` rises >= 0.001) | `negative_signal` |
| difference < threshold | `neutral` |
| no historical baseline (best target metric) | `positive_signal` (first recording) |
| metrics missing or training error | `error` |

The threshold is defined in `pdca_system/config.py` (`PROMOTION_THRESHOLD`).

## Action: Promotion Rules

Only CA may trigger a merge into baseline; PD must not. Runner records `commit_sha`; on positive signal the workflow merges seed into baseline first, then updates metrics/state. Merge conflict → system queues merge-resolution CA.

### Promotion (`positive_signal`)
1. System merges seed into baseline first (you do not run merge).
2. Workflow updates `baseline_metrics.json` / `baseline_branches.json`.
3. Metadata in seed/run state.

### Merge failure
- **Normal seed:** In seed worktree: `git merge __baseline__`, resolve conflicts, commit, output CA summary for retry.
- **Baseline seed (__baseline__):** Merge __baseline__ into target (e.g. master). Run from worktree that has target checked out (`git worktree list`); do not run from __baseline__ worktree or `git merge master` there.

### Non-promotion
`neutral` / `negative_signal` / `error`: log only. Failure info in queue/state logs.

## Constraints
- No model/optimizer/training-logic changes for better metrics; only make the plan run (bugs, OOM, etc.).
- Run the canonical command from the worktree using the **Python executable injected by the daemon** and the script/command defined in protocol and this doc. Do not skip target metric evaluation (output `{TARGET_METRIC_KEY}: {value}` in stdout and/or include in JSON).
- Do not edit baseline JSON files; only CA promotion updates them.
- Traceability: git + state files.
