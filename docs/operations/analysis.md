# Analyzing results

After a run, two artifacts matter:

- `results.tsv` — every experiment the agent tried, with status and description.
- `git log autoresearch/<tag>` — one commit per kept experiment, in order.

`analysis.ipynb` (in the repo root) is a small notebook for turning `results.tsv` into a "running best" plot and a few summary tables. This page is a guide to using it and reading the output.

## Quick run

```bash
uv run jupyter lab analysis.ipynb     # or your notebook tool of choice
# Run all cells.
```

The notebook expects `results.tsv` in the current directory and produces `progress.png` next to it.

## What the notebook does

### 1. Load and clean

```python
df = pd.read_csv("results.tsv", sep="\t")
df["val_bpb"]   = pd.to_numeric(df["val_bpb"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"]    = df["status"].str.strip().str.upper()
```

Statuses are normalized to upper-case (`KEEP`, `DISCARD`, `CRASH`).

### 2. Outcome counts

```python
counts = df["status"].value_counts()
keep_rate = n_keep / (n_keep + n_discard)
```

A useful sanity check: the keep rate should be a small but non-zero fraction (a few percent is typical). Zero kept means the agent never found anything; close to 50% means the experiments are too easy and the metric is noisy.

### 3. Kept-experiment list

Plain text dump of every kept experiment with its `val_bpb` and one-line description, in chronological order. This is the human-readable changelog of the run.

### 4. Running-best plot

The headline visualization. Save target is `progress.png`.

- X axis: experiment index.
- Y axis: `val_bpb` (lower is better), zoomed to `[best - margin, baseline + margin]`.
- Faint grey dots: discarded experiments at-or-below baseline.
- Green dots with black edges: kept experiments.
- Green step line: running minimum over the kept points.
- Each kept dot is annotated with its description.

The Y-axis is intentionally zoomed: you only see runs that came close to or beat the baseline. Far-worse discards are off-screen by design.

### 5. Summary stats

```
Baseline val_bpb:  …
Best val_bpb:      …
Total improvement: …  (X.XX%)
Best experiment:   <description>
```

Followed by a chronological list of every kept experiment showing cumulative effort.

### 6. Top hits by delta

Sorts kept experiments by per-step improvement (`prev_bpb - val_bpb`):

```
Rank   Delta       BPB         Description
   1   +0.004700   0.993200    increase MATRIX_LR to 0.05
   2   +0.001500   0.991700    add a 9th transformer layer
   …
```

Cumulative delta should equal the total improvement.

## Reading the run

Three quick judgments to make:

1. **Was the agent productive?** Look at the kept count and the running-best line. A flat-then-down step is healthy. Long flats are signs the agent got stuck — re-read its commits for the same idea on repeat.
2. **Did it overfit complexity?** Read the kept descriptions. If they're all subtle hyperparameter nudges, the agent is in local-optimization mode. If they include architectural changes, it's exploring.
3. **Were there many crashes?** A crash rate above ~10% probably means the agent is over-reaching VRAM or pushing too hard on numerical stability. Worth a look at whether `program.md` should ask for smaller-step changes.

## Inspecting individual experiments

To look at a specific commit:

```bash
git show <commit>                           # the diff
git log -1 --format=%B <commit>             # the commit message
```

The agent's commit messages tend to be the same as the description — but if you want the *exact* code state of an experiment, `git checkout <commit>` works because every experiment is a real commit on the branch.

## Comparing runs across machines

`val_bpb` is comparable across runs *with the same `MAX_SEQ_LEN`, `EVAL_TOKENS`, `VOCAB_SIZE`, and the same val shard*. If you've forked for smaller hardware and changed any of those, you can compare *within* a tag but not across hardware.

`mfu_percent` is always relative to H100 peak, even on other hardware — it's only meaningful as a relative trend within a single tag.

## Re-running the analysis on archived runs

`results.tsv` is gitignored. To preserve a run:

```bash
cp results.tsv results-mar5.tsv
git add results-mar5.tsv && git commit -m "archive mar5 run"
```

Or stash it on a `results/` branch. Then in the notebook, change the `pd.read_csv` filename or copy the archived TSV back to `results.tsv` before running.
