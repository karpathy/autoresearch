# autoresearch for deception2

This repo is a sandboxed autonomous research harness for feature mining and
cross-dataset OOD modeling on the 7B deception2 datasets.

The baseline code comes from:

- `/playpen-ssd/smerrill/deception2/src/feature_extractor.py`
- `/playpen-ssd/smerrill/deception2/src/multidataset_ood_xgb.py`

The immutable input data comes from:

- `/playpen-ssd/smerrill/deception2/Dataset/AdvisorAudit/DeepSeek-R1-Distill-Qwen-7B`
- `/playpen-ssd/smerrill/deception2/Dataset/BS/DeepSeek-R1-Distill-Qwen-7B`
- `/playpen-ssd/smerrill/deception2/Dataset/Gridworld/DeepSeek-R1-Distill-Qwen-7B`

## What this fork changes

- The editable code is copied into `sandbox/` so experiments stay inside this
  repo.
- The dataset tree is treated as read-only input.
- All generated features, model bundles, metrics, logs, and notebooks are
  written under `runs/<tag>/`.
- The runner is locked to `CUDA_VISIBLE_DEVICES=7`, so only GPU 7 is visible.
- The evaluation objective is no longer language-model BPB. It is:
  - classification: maximize mean OOD AUROC
  - regression: maximize mean OOD Pearson correlation

## Task definition

Each experiment may change:

- `sandbox/feature_extractor.py`
- `sandbox/multidataset_ood_xgb.py`

The dataset tree under `/playpen-ssd/smerrill/deception2/Dataset` must not be
edited.

The modeling side is not restricted to XGBoost. The agent may swap in any model
family available in the environment, provided the run still stays sandboxed,
uses only GPU 7, and emits the same metrics and notebook outputs.

### Classification target

The sentence-level regression target is `deception_rate`.

For classification, the binary label must be constructed as:

```python
y_binary = (deception_rate > threshold).astype(int)
```

The default threshold search space is `0.3, 0.4, 0.5, 0.6`.

### OOD objective

For each training dataset in `{AdvisorAudit, BS, Gridworld}`, train on one
dataset, validate on an in-domain split of that same dataset, and evaluate OOD
on the other datasets.

The harness summarizes:

- best classification candidate by mean OOD AUROC
- best regression candidate by mean OOD Pearson

## Files that matter

- `prepare.py`
  - validates the fixed 7B dataset layout
  - seeds the sandbox working copies from the deception2 baselines
  - creates `runs/<tag>/run_config.json`
- `train.py`
  - runs feature extraction into `runs/<tag>/feature_cache`
  - runs multidataset OOD modeling into `runs/<tag>/results`
  - builds leaderboards and a summary JSON
  - generates a notebook based on
    `/playpen-ssd/smerrill/deception2/Notebooks/paper_ood_figures.ipynb`
- `program.md`
  - instructions for the autonomous coding agent

## Quick start

Install dependencies if needed:

```bash
uv sync
```

Initialize a run:

```bash
uv run prepare.py --run-tag mar11
```

This creates:

- `sandbox/feature_extractor.py`
- `sandbox/multidataset_ood_xgb.py`
- `runs/mar11/run_config.json`

Run the full pipeline:

```bash
uv run train.py --run-tag mar11
```

Feature extraction assumes `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` can be
loaded by `transformers` from the local Hugging Face cache or from your normal
HF setup.

If you only changed modeling code and want to reuse the last feature cache:

```bash
uv run train.py --run-tag mar11 --skip-features
```

Dry-run the commands without executing them:

```bash
uv run train.py --run-tag mar11 --dry-run
```

## Outputs

Each run writes only inside `runs/<tag>/`:

- `feature_cache/`
- `results/`
- `logs/`
- `notebooks/paper_ood_figures.ipynb`
- `classification_leaderboard.csv`
- `regression_leaderboard.csv`
- `summary.json`
- `results.tsv`

At the end of `train.py`, the script prints a compact summary like:

```text
---
classification_mean_ood_auroc: 0.812345
classification_train_dataset: BS
classification_feature_set: full__before_at
classification_threshold: 0.40
regression_mean_ood_pearson: 0.456789
regression_train_dataset: AdvisorAudit
regression_feature_set: full__before
notebook_path: /playpen-ssd/smerrill/autoresearch/runs/mar11/notebooks/paper_ood_figures.ipynb
```

## Notebook generation

After modeling, `train.py` copies the existing paper notebook template from
`/playpen-ssd/smerrill/deception2/Notebooks/paper_ood_figures.ipynb` and patches
it to read the current run’s `results/` directory.

The notebook is not executed automatically. It is created so the best run has a
paper-style artifact ready to inspect or execute later.
