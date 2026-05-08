# Reference: `results.tsv`

The agent's experiment log. Append-only during a run, untracked by git so it survives `git reset`.

## Schema

Five columns, **tab-separated** (not CSV — descriptions often contain commas):

| Column | Type | Meaning |
|---|---|---|
| `commit` | 7-char short SHA | The commit the experiment was run on. For crashes/discards this is the commit *before* the rollback. |
| `val_bpb` | float, 6 decimals | Validation bits-per-byte from `evaluate_bpb`. `0.000000` for crashes. |
| `memory_gb` | float, 1 decimal | Peak VRAM in GiB. `peak_vram_mb / 1024`. `0.0` for crashes. |
| `status` | enum | `keep`, `discard`, or `crash`. |
| `description` | free text | Short human-readable summary of the idea (no tabs). |

## Header

The very first line is the header. The agent writes it during setup before the first experiment:

```
commit	val_bpb	memory_gb	status	description
```

## Examples

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase MATRIX_LR to 0.05
c3d4e5f	1.005000	44.0	discard	switch MLP activation to GeLU
d4e5f6g	0.000000	0.0	crash	double n_embd (OOM)
e5f6g7h	0.992100	44.5	keep	add a 9th transformer layer
f6g7h8i	0.992050	44.5	discard	tweak softcap to 12 (improvement too small)
```

## Status semantics

- **`keep`** — `val_bpb` strictly improved over the previous kept value. The branch advances. The `commit` column matches `git rev-parse --short HEAD` after the keep.
- **`discard`** — run completed but didn't improve enough (or violated the simplicity criterion). The agent runs `git reset --hard HEAD~1` after logging. The recorded `commit` is the *now-deleted* commit's short SHA, preserved here as the only trail of what was tried.
- **`crash`** — Python exception, OOM, NaN/exploding loss, or run >10 minutes. `val_bpb=0.000000` and `memory_gb=0.0` are sentinels.

## Why TSV and not CSV

Descriptions like `"crank LR, drop momentum, swap activation"` contain commas. Tabs almost never appear in free-text descriptions, and they're easy to type and read. Pandas reads the file with `sep="\t"` (see `analysis.ipynb`).

## Why untracked

`.gitignore` includes `results.tsv` so:

- `git reset --hard HEAD~1` after a discarded experiment doesn't lose the log.
- `git checkout autoresearch/<other-tag>` doesn't blow away the in-progress run's history.
- Multiple agents on different branches don't fight over commits to the same file.

The flip side: if you want to share results across machines or back them up, copy the file by hand or commit it to a separate "results" branch.

## Reading the file

```python
import pandas as pd
df = pd.read_csv("results.tsv", sep="\t")
df["val_bpb"]   = pd.to_numeric(df["val_bpb"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"]    = df["status"].str.strip().str.upper()
```

`analysis.ipynb` does exactly this and produces:

- A summary count of `KEEP` / `DISCARD` / `CRASH` rows.
- A list of every kept experiment with its `val_bpb` and description.
- The "running best" plot saved as `progress.png`.
- A ranked table of kept hits by `delta` improvement.

See [operations/analysis.md](../operations/analysis.md) for the analysis workflow.

## Common pitfalls

- **Mixing tabs and spaces.** If the agent ever writes spaces, `pandas` will parse the row as a single column. The agent's prompt explicitly says "tab-separated, NOT comma-separated"; if you customize `program.md`, keep this constraint.
- **Newlines in descriptions.** Don't. One row per experiment.
- **Recording the post-reset commit instead of the experiment's own commit.** Discards should record the experiment commit's SHA *before* `git reset`. Otherwise the description is orphaned from any git object.
- **Forgetting to log crashes.** If an experiment crashed and you skip logging it, the next analysis run misattributes effort. Always append a `crash` row.
