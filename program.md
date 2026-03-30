# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Initialize state files**: Create `search_state.json` and `search_tree.md` to track the DFS exploration state (see State Persistence below). Add both files plus `run.log` to `.gitignore`. These are your map of the search space — without them you will lose track of where you are.
7. **Set up CLAUDE.md**: Add the Autoresearch Recovery Protocol block to the project's `CLAUDE.md` (see State Persistence below). This ensures any new session automatically recovers state.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## DFS Exploration Strategy

You are not randomly guessing improvements. You are conducting a **Depth-First Search (DFS)** over the space of possible modifications to `train.py`. Think of the exploration as a tree:

```
baseline (root)
├── Branch A: Learning rate experiments
│   ├── A1: LR 0.002 → improved ✓ (advance)
│   │   ├── A1a: LR 0.003 → improved ✓ (advance)
│   │   │   └── A1a-i: LR 0.004 → worse ✗ (backtrack to A1a)
│   │   └── A1b: LR 0.003 + warmup tweak → worse ✗ (backtrack to A1a)
│   │       (A1a exhausted → backtrack to A1 → backtrack to baseline)
│   └── A2: LR schedule change → worse ✗ (backtrack)
├── Branch B: Architecture changes
│   ├── B1: Add RMSNorm → improved ✓ (advance)
│   │   └── B1a: RMSNorm + wider FFN → ...
│   ...
├── Branch C: Optimizer experiments
│   ...
```

### Core DFS rules

1. **Maintain an idea stack.** Before you begin experimenting, brainstorm a ranked list of high-level research directions (branches). Write these to `search_tree.md`. This is your **frontier** — the unexplored neighbor nodes. As you discover new ideas during experimentation, push them onto the stack.

2. **Go deep first.** When an experiment improves val_bpb, **stay on that branch** and try to push further in the same direction. Ask: "What is the natural next refinement of this change?" Only move to a sibling or parent branch when the current direction is exhausted.

3. **Backtrack on dead ends.** When an experiment fails to improve (or crashes), revert (`git reset`) and **pop the next idea from the stack** for the current depth level. If all siblings at this depth are exhausted, backtrack up one level (to the parent node) and try the next untried sibling there.

4. **Track visited nodes.** Every experiment is a node. Log it in both `results.tsv` (quantitative) and `search_state.json` / `search_tree.md` (structural). Never re-run the exact same experiment. If you find yourself about to try something you've already tried (or something trivially equivalent), skip it.

5. **Combine winning branches.** After exploring several independent branches, DFS allows you to **merge discoveries**. Once you've found improvements on two separate branches (e.g., "higher LR" and "RMSNorm"), create a new branch that combines them. This is like exploring a cross-edge in the search graph. Log these as combination experiments.

6. **Depth limit.** If you've gone more than **5 levels deep** on a single branch without meaningful improvement (cumulative < 0.005 val_bpb gain across the last 3 experiments), force a backtrack to the root and try a completely different high-level direction. You may be over-fitting to a local minimum.

7. **Breadth at the root.** At the root level (high-level directions), aim for **breadth** before going deep on any single one. Spend your first few experiments after baseline doing quick "probe" runs on 3–5 different directions (1 experiment each) to identify which branches are most promising. Then DFS into the most promising one.

### Deciding the next experiment

Before each run, follow this decision procedure:

```
1. Read search_state.json to recall current position and stack.
2. Did the last experiment IMPROVE val_bpb?
   YES → Push deeper: generate 2-3 child ideas, push onto stack, pop the top one, run it.
   NO  → Backtrack: revert git, pop next idea from stack at same or shallower depth, run it.
3. Is the stack empty?
   YES → You've exhausted known ideas. Re-read train.py and results.tsv.
         Brainstorm new root-level directions. Push them. Continue.
4. Have you found 2+ independent winning branches?
   YES → Consider a combination experiment before going deeper.
5. Update search_state.json and search_tree.md with your decision and reasoning.
```

## State Persistence (Claude Code Sessions)

Claude Code sessions have finite context windows and may restart mid-experiment. Your DFS state **must survive context resets**. You maintain three files that together allow you to fully reconstruct your position in the search tree from a cold start.

### The three persistence files

| File | Purpose | Format | Audience |
|------|---------|--------|----------|
| `search_state.json` | Machine-readable DFS state | JSON | Agent (you) |
| `search_tree.md` | Human-readable exploration log | Markdown | Human + Agent |
| `CLAUDE.md` | Boot instructions for Claude Code | Markdown | Agent (on startup) |

All three files are **untracked by git** (add them to `.gitignore`).

### 1. `search_state.json` — The canonical state file

This is your ground truth. After **every** experiment, overwrite this file with the current state. On session start (or context reset), **read this file first** to reconstruct where you are.

Schema:

```json
{
  "version": 1,
  "run_tag": "mar5",
  "branch": "autoresearch/mar5",
  "created_at": "2025-03-05T10:00:00Z",
  "updated_at": "2025-03-05T14:32:00Z",
  "total_experiments": 12,

  "current_position": {
    "node_id": "A1a",
    "depth": 2,
    "git_commit": "a1b2c3d",
    "val_bpb": 0.9850,
    "description": "LR 0.003"
  },

  "best_result": {
    "node_id": "B1+A1a",
    "git_commit": "g7h8i9j",
    "val_bpb": 0.9852,
    "description": "combine RMSNorm + LR 0.003"
  },

  "idea_stack": [
    {"id": "A1a-i", "depth": 3, "parent": "A1a", "idea": "Try LR 0.004"},
    {"id": "A1b",   "depth": 2, "parent": "A1",  "idea": "Cosine decay from A1"},
    {"id": "B1",    "depth": 1, "parent": "root", "idea": "Add RMSNorm"},
    {"id": "C",     "depth": 0, "parent": "root", "idea": "Try Muon optimizer"}
  ],

  "explored_nodes": [
    {"id": "root",  "depth": 0, "commit": "a1b2c3d", "val_bpb": 0.9979, "status": "keep",    "description": "baseline"},
    {"id": "A1",    "depth": 1, "commit": "b2c3d4e", "val_bpb": 0.9932, "status": "keep",    "description": "LR 0.002"},
    {"id": "A1a",   "depth": 2, "commit": "c3d4e5f", "val_bpb": 0.9850, "status": "keep",    "description": "LR 0.003"},
    {"id": "A2",    "depth": 1, "commit": "e5f6g7h", "val_bpb": 0.9950, "status": "discard", "description": "cosine schedule"}
  ],

  "combination_candidates": [
    {"branches": ["A1a", "B1"], "tried": false}
  ],

  "pending_experiment": null
}
```

**Key fields:**
- `current_position` — the node you are sitting on right now (the best commit on the active branch)
- `idea_stack` — LIFO stack; pop from index 0, push to index 0
- `explored_nodes` — every experiment you have run, in order
- `best_result` — the single best val_bpb achieved across all experiments
- `combination_candidates` — pairs of independent winning branches to try combining
- `pending_experiment` — if non-null, an experiment was started but not completed (crash recovery, see below)

### 2. `search_tree.md` — Human-readable log

This is the same file specified in the DFS Exploration Strategy section, but with one addition: it **must be regenerable from `search_state.json`**. If the markdown file is lost or corrupted, you can reconstruct it from the JSON. The markdown is the display layer; the JSON is the source of truth.

Format:

```markdown
# Search Tree

**Run tag:** mar5
**Branch:** autoresearch/mar5
**Total experiments:** 12
**Best val_bpb:** 0.9852 (node B1+A1a, commit g7h8i9j)
**Last updated:** 2025-03-05 14:32 UTC

## Current position
Node: A1a (LR experiments → 0.003)
Depth: 2
Commit: c3d4e5f
val_bpb: 0.9850

## Idea stack (LIFO — next to try is at top)
1. [DEPTH 3] A1a-i: Try LR 0.004 (child of A1a)
2. [DEPTH 2] A1b: Cosine decay from A1 (child of A1)
3. [DEPTH 1] B1: Add RMSNorm (root branch)
4. [DEPTH 0] C: Try Muon optimizer (root branch)

## Explored branches
- [root] baseline: val_bpb 0.9979 → KEEP ✓ (commit a1b2c3d)
  - [A1] LR 0.002: val_bpb 0.9932 → KEEP ✓ (commit b2c3d4e)
    - [A1a] LR 0.003: val_bpb 0.9850 → KEEP ✓ (commit c3d4e5f) ← CURRENT
  - [A2] cosine schedule: val_bpb 0.9950 → DISCARD ✗ (commit e5f6g7h)

## Combination candidates
- [ ] A1a + B1 (if B1 succeeds independently)
```

### 3. `CLAUDE.md` — Boot instructions

Add the following block to the project's `CLAUDE.md` file (or create it if it doesn't exist). Claude Code reads this file on every session start:

```markdown
# CLAUDE.md

## Autoresearch Recovery Protocol

If this is a fresh session or a context reset, **do the following before anything else**:

1. Read `search_state.json` to recover the full DFS state.
2. Read `search_tree.md` for the human-readable exploration log.
3. Read `results.tsv` for the quantitative experiment history.
4. Run `git log --oneline -20` to verify the current branch and commit match `search_state.json`.
5. If `search_state.json` has a non-null `pending_experiment`, a previous run was interrupted — see Crash Recovery below.
6. Resume the experiment loop from step 1 (Read the map).

**Do NOT ask the human if you should continue.** You are autonomous. Resume immediately.
```

### Save protocol (after every experiment)

After **every** experiment completes (success, failure, or crash), execute the following save sequence **before** moving to the next experiment. This is non-negotiable — skipping a save risks losing state on the next context reset.

```
SAVE SEQUENCE (mandatory after every experiment):
─────────────────────────────────────────────────
1. Update search_state.json:
   a. Add the experiment to explored_nodes
   b. Update current_position (if keeping) or leave unchanged (if discarding)
   c. Update best_result if this is a new best
   d. Update idea_stack (pop the completed idea, push any new child ideas)
   e. Set pending_experiment to null
   f. Increment total_experiments
   g. Update updated_at timestamp
   h. Write the file atomically:
      python3 -c "
      import json, tempfile, os
      state = ...  # your updated state dict
      # Write to temp file first, then atomic rename
      fd, tmp = tempfile.mkstemp(dir='.', suffix='.json')
      with os.fdopen(fd, 'w') as f:
          json.dump(state, f, indent=2)
      os.rename(tmp, 'search_state.json')
      "

2. Append to results.tsv (the experiment row)

3. Regenerate search_tree.md from search_state.json
   (or manually update it — but if in doubt, regenerate)

4. git add/commit (for the train.py changes, if keeping)
   OR git reset (if discarding)
```

The atomic write (write to temp file, then rename) prevents corruption if the process is killed mid-write.

### Load protocol (on session start / context reset)

When you start a new session or detect that you've lost context:

```
LOAD SEQUENCE (on session start):
──────────────────────────────────
1. Check if search_state.json exists:
   YES → Read it. This is your state. Proceed to step 2.
   NO  → This is a fresh run. Proceed with Setup.

2. Validate state against reality:
   a. git branch → does it match state.branch?
   b. git log --oneline -5 → is state.current_position.git_commit in recent history?
   c. cat results.tsv | wc -l → does row count match state.total_experiments + 1 (header)?

   If any check fails:
   - Trust git as ground truth (commits are immutable)
   - Reconstruct state from results.tsv + git log
   - Regenerate search_state.json and search_tree.md

3. Check pending_experiment:
   a. If non-null → a run was interrupted. Check if run.log exists and has results.
      - If run.log has val_bpb → the run finished but state wasn't saved. Process the result.
      - If run.log has no val_bpb → the run crashed. Log as crash. Backtrack.
      - If run.log doesn't exist → the run never started. Re-run or skip.
   b. If null → clean state. Proceed to experiment loop.

4. Resume the experiment loop from step 1 (Read the map).
```

### Crash recovery edge cases

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Session killed during experiment run | `pending_experiment` is non-null, `run.log` exists but incomplete | Check `run.log` for results. If found, process. If not, treat as crash. |
| Session killed during save | `search_state.json` is corrupt or missing | Fall back to `results.tsv` + `git log` to reconstruct state |
| Session killed during git reset | Branch is in unexpected state | Use `git reflog` to find the correct commit, reset to it |
| Context window exhausted mid-loop | New session starts, `CLAUDE.md` triggers recovery | Follow Load Sequence above |
| `search_tree.md` lost/corrupt | `search_state.json` exists | Regenerate markdown from JSON |
| `search_state.json` lost/corrupt | `search_tree.md` and `results.tsv` exist | Parse TSV + markdown to reconstruct JSON |
| All state files lost | `results.tsv` and git history exist | Reconstruct from TSV rows + `git log --all --oneline` |

### Pre-experiment checkpoint

Before starting each experiment, set the `pending_experiment` field:

```json
{
  "pending_experiment": {
    "node_id": "A1a-i",
    "idea": "Try LR 0.004",
    "parent_commit": "c3d4e5f",
    "started_at": "2025-03-05T14:35:00Z"
  }
}
```

Write this to `search_state.json` **before** running `uv run train.py`. This way, if the session dies during the run, the next session knows exactly what was in progress and can recover.

## Output format

Once the script finishes it prints a summary like this:

```
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
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_bpb	memory_gb	status	branch	depth	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. branch: the DFS branch label (e.g. `A1a`, `B2`, `C1b-ii`) — use `root` for baseline
6. depth: the DFS depth level (0 for root-level experiments, 1 for first refinement, etc.)
7. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	branch	depth	description
a1b2c3d	0.997900	44.0	keep	root	0	baseline
b2c3d4e	0.993200	44.2	keep	A1	1	increase LR to 0.002
c3d4e5f	0.989100	44.2	keep	A1a	2	increase LR to 0.003
d4e5f6g	0.991000	44.2	discard	A1a-i	3	increase LR to 0.004 (overshot)
e5f6g7h	0.995000	44.0	discard	A2	1	switch to cosine LR schedule
f6g7h8i	0.988500	45.1	keep	B1	1	add RMSNorm pre-normalization
g7h8i9j	0.985200	45.3	keep	B1+A1a	1	combine RMSNorm + LR 0.003
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. **Read the map**: Read `search_state.json` (or `search_tree.md` as fallback) for your current DFS position, idea stack, and explored branches. If this is the first run, the map is empty — your position is the root (baseline).
2. **Pick the next node**: Follow the DFS decision procedure (see above). Pop the top idea from the stack. If the stack is empty, brainstorm new root-level ideas and push them.
3. **Set checkpoint**: Write `pending_experiment` to `search_state.json`. This marks "I am about to try X" so crash recovery knows what happened.
4. **Tune `train.py`** with the selected experimental idea by directly hacking the code.
5. **git commit** — `git commit -m "<node_id>: <short description>"`
6. **Run the experiment**: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. **Read out the results**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, log it as a crash and backtrack.
9. **Save state (mandatory — do not skip)**:
   a. Append row to `results.tsv` with the branch label and depth.
   b. Update `search_state.json` (full save sequence — see State Persistence above).
   c. Update `search_tree.md`.
   d. Clear `pending_experiment`.
   (NOTE: do not commit `results.tsv`, `search_state.json`, or `search_tree.md` — leave them untracked by git)
10. **Advance or backtrack**:
    - If val_bpb **improved** (lower): keep the commit, update current position deeper in the tree. Generate child ideas and push onto stack.
    - If val_bpb is **equal or worse**: `git reset` back to the parent commit. Pop the next idea from the stack.
    - If you've found **multiple independent wins**: consider a combination experiment next.
11. **Go to step 1.**

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on to the next node in the DFS.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. Consult your `search_tree.md` — there may be unexplored branches you overlooked. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to a complete DFS exploration tree of experimental results, all completed by you while they slept!
