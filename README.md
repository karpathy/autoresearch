# autoresearch — HPC/SLURM fork

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted to run on HPC clusters managed by SLURM, using a Singularity container for a reproducible runtime.

## Why a container?

Flash Attention 3 is difficult to install on HPC systems due to conflicts with OS libraries (glibc, CUDA toolchains), and users typically lack sudo privileges. Packaging everything into a Singularity image sidesteps these issues and provides a portable runtime across cluster nodes.

## How it works

An AI agent reads `program.md` for instructions, then iteratively modifies `train.py`, runs 5-minute training experiments, evaluates `val_bpb` (validation bits per byte — lower is better), and keeps or discards changes based on results. This repeats indefinitely.

Only three files matter for the research loop:

| File | Role |
|------|------|
| `prepare.py` | Data prep, dataloader, evaluation. **Do not modify.** |
| `train.py` | Model, optimizer, training loop. **Agent modifies this.** |
| `program.md` | Instructions and constraints for the agent. **Human edits this.** |

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The `val_bpb` metric is vocab-size-independent, so architectural changes are fairly compared.

Each training run is executed as `./run.sh python train.py > run.log 2>&1`. The agent parses `val_bpb` and `peak_vram_mb` from `run.log` to decide whether to keep or revert the change. If the grep comes back empty, the run crashed — the agent reads the tail of `run.log` for the stack trace and attempts a fix.

## Setup

### 1. Clone and build the Singularity image (one-time)

```bash
git clone https://github.com/dmbala/autoresearch
cd autoresearch
singularity build autoresearch.sif autoresearch.def
```

This produces `autoresearch.sif` which bundles CUDA, Python, and all dependencies including Flash Attention 3.

The SIF path is hardcoded in `run.sh` — update it if you place the image elsewhere.

### 2. Prepare the data (one-time, ~2 min)

```bash
./run.sh python prepare.py
```

Data is cached at the `CACHE_DATA` path defined in `run.sh` and bind-mounted into the container automatically.

### 3. Test a single training run (~5 min)

```bash
./run.sh python train.py
```

If this completes and prints a `val_bpb` summary, your setup is working.

## Running the agent on SLURM

`train_run.slrm` submits a Claude agent as a SLURM job. The agent reads `program.md`, then runs the experiment loop autonomously — modifying `train.py`, training, evaluating, and repeating indefinitely.

```bash
sbatch train_run.slrm
```

The script **self-resubmits** on completion (`sbatch "$0"`), so the agent keeps running across job time limits without manual intervention. To stop it:

```bash
scancel <jobid>
```

**Note:** To fully stop the loop, rename or remove `train_run.slrm` so the self-resubmit cannot re-launch.

SLURM logs are written to `logs/` (created automatically). The working directory is hardcoded in `train_run.slrm` — update it if you move the repo.

## Running the container manually

`run.sh` is a thin wrapper around `singularity exec --nv`:

```bash
./run.sh python train.py
```

## Project structure

```
prepare.py        — constants, data prep, runtime utilities (do not modify)
train.py          — model, optimizer, training loop (agent modifies this)
program.md        — agent instructions (human edits this)
train_run.slrm    — SLURM job script (self-resubmitting)
run.sh           — Singularity wrapper script
autoresearch.def  — Singularity container definition
pyproject.toml    — Python dependencies (baked into Singularity image)
analysis.ipynb    — experiment analysis notebook
logs/             — SLURM stdout/stderr (created automatically)
run.log           — stdout/stderr from the latest training run (not tracked in git)
results.tsv       — experiment results log (not tracked in git)
```

## Upstream

This repo tracks [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The HPC-specific additions are `run.sh`, `autoresearch.def`, and `train_run.slrm`. The core research loop (`prepare.py`, `train.py`, `program.md`) follows upstream conventions.

