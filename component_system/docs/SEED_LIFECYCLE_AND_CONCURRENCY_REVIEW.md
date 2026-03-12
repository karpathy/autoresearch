# Seed Lifecycle, State Transitions, and Concurrency Review

## 1. Seed lifecycle and state transitions

### 1.1 Seed status enum (`SeedStatus`)

| Status       | Meaning |
|-------------|---------|
| `draft`     | Newly created, not yet queued for Plan |
| `queued`    | Plan run created and task in queue (or waiting for baseline) |
| `planning`  | P run in progress |
| `generated` | P completed; code generated, DCA not yet queued |
| `dca_queued`| DCA run created and task in queue (includes sync/merge resolution) |
| `adapting`  | DCA run in progress |
| `running`   | **Never set in code** — see gap below |
| `passed`    | DCA completed, no promotion |
| `failed`    | Terminal failure (P failed, DCA failed, or reconciled from passed+error) |
| `promoted`  | DCA completed with positive signal; seed merged into baseline |

### 1.2 Documented transitions (from code)

```
draft → queued          queue_p / _enqueue_plan_run
queued → queued         queue_p (waiting for baseline; latest_run_id cleared)
queued → planning       mark_run_started (stage P)
planning → generated    finish_p_run
planning → failed       mark_run_failed (P failed)
generated → dca_queued   queue_dca
generated → queued      finish_sync_resolution (then _enqueue_plan_run)
dca_queued → adapting   mark_run_started (stage DCA)
adapting → passed       finish_dca_run (neutral/negative signal)
adapting → failed       finish_dca_run (error) or mark_run_failed
adapting → promoted     finish_dca_run (positive_signal)
adapting → dca_queued   finish_dca_run (merge/sync failed → queue_dca merge_resolution=True)
adapting → generated    finish_dca_run (ralph neutral_signal)
passed → failed         _reconcile_seed_status_signal (passed but latest_signal=="error")
```

Baseline seed: `draft` → `generated` (ensure_baseline_result) → `dca_queued` → … → `passed` / `failed`.

---

## 2. Gaps and issues in state transitions

### 2.1 `SeedStatus.running` is never set

- **Code:** `SeedStatus.running` appears only in:
  - `is_seed_eligible_for_stage` (P not eligible if `adapting`, `running`, or `dca_queued`)
  - `ensure_baseline_result` (early return if `dca_queued`, `adapting`, `running`)
  - Dashboard `status_column_map` → `activeDca`
- **Issue:** No assignment `seed.status = SeedStatus.running` anywhere. DCA-in-progress uses `adapting`.
- **Recommendation:** Either remove `running` from the enum and all checks, or document it as reserved and start setting it (e.g. for a future “running but not adapting” phase). Otherwise it’s dead code and the enum is misleading.

### 2.2 Sync failure in `mark_run_started` (P): run/seed consistency

- **Flow:** When a P run is started, `mark_run_started`:
  1. Sets `run.status = RunStatus.running` (in memory).
  2. Calls `ensure_seed_worktree_ready`, then `sync_seed_worktree_with_baseline`.
  3. On `GitCommandError`, calls `queue_sync_resolution(seed_id)` and raises `SyncResolutionQueued`.
  4. Only at the end (after sync and other logic) does it `run_repo.save(run)` and `seed_repo.save(seed)`.
- **Effect:** When sync fails:
  - The **run** is never saved as `running`; it remains `queued` in the run repo.
  - The **seed** is updated by `queue_sync_resolution`: `seed.status = dca_queued`, new DCA run and task written.
  - The **P task** is moved to error in `run.py` (`move_to_error(task_path)`).
- **Result:** The original P **run** is orphaned: it stays `queued` forever and is never completed or failed. `seed.latest_run_id` points to the new sync-resolution DCA run. A later Plan enqueue creates a new P run and task.
- **Recommendation:** When raising `SyncResolutionQueued`, either:
  - Mark the current P run as failed (e.g. “sync_failed”) and save it, or
  - Explicitly not create a run for that P task until after sync succeeds (e.g. move run creation to after sync). That would require a larger refactor.

### 2.3 Other transitions

- All other transitions are consistent with the intended flow: P → generated/failed, DCA → passed/failed/promoted/dca_queued/generated, and reconciliation of `passed` + `error` → `failed`.

---

## 3. Multiple seeds running at the same time — race conditions and conflicts

### 3.1 Task claiming is atomic per task

- **Mechanism:** `claim_pending` uses `path.rename(path, IN_PROGRESS_DIR / path.name)`. Only one process can rename a given file; others get `FileNotFoundError` or `OSError` and skip that task.
- **Effect:** Each task file is claimed by at most one worker; no double execution of the same task.

### 3.2 Per-seed eligibility prevents P vs DCA overlap

- **Mechanism:** Before running a task, the worker calls `claim_pending(..., eligible_fn=eligible)`. `eligible` uses `WORKFLOW.is_seed_eligible_for_stage(seed_id, stage)`:
  - **P:** eligible only if `seed.status not in (adapting, running, dca_queued)`.
  - **DCA:** eligible only if `seed.status is not SeedStatus.planning`.
- **Effect:** For a given seed, P and DCA are never both considered eligible. So the same seed cannot have a P task and a DCA task running at the same time, and a seed in `planning` or `adapting` will not get another stage started until the run finishes.

### 3.3 Read–modify–write on seed/run state

- **Risk:** Multiple workers can run concurrently (e.g. 2 P workers, 1 DCA-GPU, 1 DCA-AUX). Each worker loads seed/run, modifies, and saves. There is no locking or optimistic concurrency (e.g. version field).
- **Mitigation in practice:**
  - Each **task** is for a specific (seed_id, run_id). Different tasks imply different runs (and usually different seeds for P/DCA).
  - Eligibility ensures that for a given seed, only one “kind” of work (P or DCA) is allowed at a time.
- **Remaining risk:** If two tasks for the same seed could ever be in flight (e.g. due to a bug or a restored task), two workers could both read the same seed, update it, and save; the last write would win and one update could be lost. With the current design (one active run per seed per stage), this should not happen for normal execution.

### 3.4 Git worktrees

- **Design:** Each seed has its own worktree (path `worktrees/<seed_id>`). Different seeds use different directories.
- **Effect:** No filesystem conflict between seeds; multiple seeds can run P or DCA in parallel in separate worktrees. Baseline seed uses `worktrees/__baseline__`.

### 3.5 Shared JSON state (repos)

- **State:** Seeds, runs, metrics, branch map, and queue dirs are file-based (JSON under `history/state/`, `history/queue/`).
- **Risk:** Two workers writing different seeds at the same time can overwrite each other only if they wrote the same file (same seed or same run). Since each task is bound to one run and one seed, and eligibility prevents overlapping stages for the same seed, concurrent updates to the same seed/run are not expected for correct flows.
- **Recommendation:** For extra safety, consider short-lived file locking or atomic write (write to temp + rename) for seed/run saves if the daemon scales to many workers.

---

## 4. Edge case: automatic merge fails — can dependent tasks start prematurely?

### 4.1 Sync failure (merge baseline into seed) before P

- **When:** In `mark_run_started` (stage P), `sync_seed_worktree_with_baseline(seed)` raises `GitCommandError`.
- **What happens:**
  1. `queue_sync_resolution(seed_id)` runs: seed set to `dca_queued`, new DCA task with `sync_resolution: True` is written.
  2. `SyncResolutionQueued` is raised; in `run.py` the P task is moved to error (not re-queued).
  3. Seed remains `dca_queued`; only the sync-resolution DCA task is for that seed.
- **Eligibility:** For P, a seed in `dca_queued` is **not** eligible. So no other P task for this seed can start. No dependent “normal” P runs until the sync-resolution DCA completes and Plan is re-queued in `finish_sync_resolution`. So **dependent tasks do not start prematurely**.

### 4.2 DCA merge into baseline fails (normal or baseline seed)

- **When:** In `finish_dca_run`, `promote_seed_branch` raises `GitCommandError`.
- **What happens:**
  1. A new DCA run is queued with `merge_resolution=True` (and seed stays `dca_queued`).
  2. No new P run or normal DCA run is enqueued for that seed until the merge-resolution DCA finishes.
- **Eligibility:** While seed is `dca_queued` or `adapting`, P is not eligible. So **dependent tasks do not start prematurely**.

### 4.3 Baseline merge fails

- Same pattern: baseline seed gets a merge-resolution DCA task, stays `dca_queued`. `_release_seeds_waiting_for_baseline` is only called after a successful merge (or after the “loop avoided” path). Waiting seeds are not released until baseline is merged. So **dependent tasks do not start prematurely**.

**Conclusion:** When the workflow’s automatic merge (sync or promote) fails, the seed is put in `dca_queued` with a resolution DCA task. Eligibility and the fact that no new P/normal DCA is enqueued until resolution completes ensure that dependent tasks do **not** start before merge resolution.

---

## 5. Other edge cases

### 5.1 Restored in-progress tasks

- On daemon start, `restore_in_progress_tasks()` moves all tasks from `in_progress/` back to the stage queue. Those tasks are then eligible to be claimed again.
- **Risk:** If a task was in progress (worker had already called `mark_run_started` and set seed to `planning`/`adapting`) and the daemon died before the worker finished, the run and seed are already updated. After restore, the task is back in the queue; a worker can claim it and call `mark_run_started` again. That would re-use the same run_id and could lead to duplicate “started” events or inconsistent state (e.g. two workers both thinking they own the run). The code does not detect “this run was already started.”
- **Recommendation:** Before updating run/seed in `mark_run_started`, check that `run.status` is still `queued`; if it is already `running`, treat the task as a duplicate (e.g. move to error or skip and don’t run again).

### 5.2 Ralph loop and merge_resolution / metrics_recovery

- After a failed DCA, `mark_run_failed` can call `queue_p(seed_id)` for Ralph seeds, but only when the task is not `merge_resolution` and not `metrics_recovery`. So Ralph does not re-queue P on merge-resolution or metrics-recovery DCA failure, which is correct.

### 5.3 Baseline seed and sync

- Baseline seed does not call `sync_seed_worktree_with_baseline` (early return in that function). So sync failure path does not apply to __baseline__. `queue_sync_resolution` explicitly raises if seed is baseline. No issue.

---

## 6. Summary table

| Area                         | Status | Notes |
|-----------------------------|--------|--------|
| Seed status enum            | Gap    | `SeedStatus.running` never set; remove or use. |
| P/DCA transition consistency| OK     | Transitions match design. |
| Sync fail (before P)        | Bug    | P run left `queued`; orphaned run. |
| Task claiming                | OK     | Atomic rename prevents double run of same task. |
| P vs DCA same seed          | OK     | Eligibility prevents concurrent P and DCA for one seed. |
| Multiple seeds concurrent   | OK     | Different worktrees; eligibility per seed. |
| Merge/sync fail → dependents| OK     | Seed stays `dca_queued`; no premature P/DCA. |
| Restored in-progress tasks  | Risk   | Re-claiming can lead to duplicate start for same run. |

---

## 7. Implemented fixes

1. **Sync failure in `mark_run_started`:** Before raising `SyncResolutionQueued`, mark the current P run as failed (e.g. error “sync with baseline failed”) and save it, so the run is not orphaned.
2. **`SeedStatus.running`:** Either remove it from the enum and from all checks, or introduce a clear rule (e.g. “DCA in progress” = `adapting` only) and document that `running` is unused.
3. **Restored tasks:** In `mark_run_started`, if `run.status != RunStatus.queued`, do not update run/seed and do not run the agent; move the task to error or a “duplicate” bucket and return.

**Not changed:** `SeedStatus.running` is still never set; it could be removed from the enum in a follow-up or left as reserved.
