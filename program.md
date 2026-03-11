# autoresearch program for deception2

This repo is an autonomous research loop for:

1. mining sentence-level features from the deception2 7B datasets
2. training OOD models across datasets
3. maximizing:
   - classification mean OOD AUROC
   - regression mean OOD Pearson

## Setup

When starting a fresh run:

1. Pick a run tag based on today, for example `mar11`.
2. Create a new branch: `git checkout -b autoresearch/<tag>`.
3. Run `uv run prepare.py --run-tag <tag>`.
4. Read these files:
   - `README.md`
   - `runs/<tag>/run_config.json`
   - `sandbox/feature_extractor.py`
   - `sandbox/multidataset_ood_xgb.py`
5. Confirm the sandbox copies exist and the dataset validation passed.

Do not edit the source dataset tree.

## Editable scope

The main editable files are:

- `sandbox/feature_extractor.py`
- `sandbox/multidataset_ood_xgb.py`

The agent is free to replace the baseline XGBoost logic with any model family
that is available in the environment, as long as:

- the pipeline still uses only GPU 7
- the dataset tree stays read-only
- the final outputs and summary files are still produced

You may also edit the harness if necessary:

- `train.py`
- `prepare.py`
- `pyproject.toml`

But the default assumption is that most research changes belong in the two
sandboxed baseline scripts.

## Immutable inputs

These are fixed inputs and must not be modified:

- `/playpen-ssd/smerrill/deception2/Dataset/AdvisorAudit/DeepSeek-R1-Distill-Qwen-7B`
- `/playpen-ssd/smerrill/deception2/Dataset/BS/DeepSeek-R1-Distill-Qwen-7B`
- `/playpen-ssd/smerrill/deception2/Dataset/Gridworld/DeepSeek-R1-Distill-Qwen-7B`

The baseline source scripts under `/playpen-ssd/smerrill/deception2/src/` are
used only as seeds for the sandbox copies.

## Label definition

Regression target:

- `y_reg = deception_rate`

Classification target:

- `y_cls = 1 if deception_rate > threshold else 0`

The default threshold search is `0.3, 0.4, 0.5, 0.6`.

This thresholded label definition is mandatory unless you deliberately change
the experiment design and document why.

## GPU rule

All experiments must run with only GPU 7 visible.

Use `train.py` to launch experiments. It exports `CUDA_VISIBLE_DEVICES=7`
before calling the sandbox scripts. Do not override this.

## Required outputs

Every successful run should produce:

- feature caches under `runs/<tag>/feature_cache/`
- OOD modeling artifacts under `runs/<tag>/results/`
- logs under `runs/<tag>/logs/`
- `runs/<tag>/classification_leaderboard.csv`
- `runs/<tag>/regression_leaderboard.csv`
- `runs/<tag>/summary.json`
- `runs/<tag>/notebooks/paper_ood_figures.ipynb`

The generated notebook should be based on:

- `/playpen-ssd/smerrill/deception2/Notebooks/paper_ood_figures.ipynb`

## Run commands

Full pipeline:

```bash
uv run train.py --run-tag <tag>
```

Reuse the previous feature cache when only the modeling code changed:

```bash
uv run train.py --run-tag <tag> --skip-features
```

Dry-run the planned commands:

```bash
uv run train.py --run-tag <tag> --dry-run
```

Redirect logs during autonomous work:

```bash
uv run train.py --run-tag <tag> > runs/<tag>/logs/run.log 2>&1
```

## Scoring and keep/discard policy

After each run, inspect the summary printed by `train.py`.

Primary metrics:

- classification: `classification_mean_ood_auroc`
- regression: `regression_mean_ood_pearson`

Train on one datset, then the other two are OOD datasets we want to compute metrics on.  Keep a change when it materially improves one objective without causing a clear
collapse in the other. Prefer simpler changes when scores are effectively tied.

If a run crashes, fix obvious issues and rerun. If the idea is fundamentally
bad, log it as a crash and move on.

## results.tsv

`prepare.py` initializes `runs/<tag>/results.tsv` with:

```text
commit	classification_mean_ood_auroc	regression_mean_ood_pearson	status	description
```

Use:

- `status=keep`
- `status=discard`
- `status=crash`

For crashes, record `0.000000` for both metrics.

## Working style

- Prefer minimal, testable changes.
- If you only changed modeling, reuse features with `--skip-features`.
- If you changed feature extraction, rerun the whole pipeline.
- The dataset tree is read-only.
- The notebook artifact must continue to be produced at the end of good runs.
- Never use GPUs other than GPU 7.
