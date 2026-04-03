# autoresearch — HPC/SLURM fork

![teaser](progress.png)

This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted to run on HPC clusters managed by SLURM, using a Singularity container instead of a bare `uv` environment.

The motivation: Flash Attention 3 is difficult to install on HPC systems due to restricted internet access, module conflicts, and the need for specific CUDA toolchain versions. Packaging everything into a Singularity image sidesteps these issues and gives a reproducible, portable runtime across cluster nodes.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Setup

### 1. Build the Singularity image (one-time)

```bash
singularity build autoresearch.sif autoresearch.def
```

This produces `autoresearch.sif` which bundles CUDA, uv, and all Python dependencies including Flash Attention 3. Place it in the parent directory alongside the repo:

```
/path/to/workdir/
  autoresearch.sif
  autoresearch/        ← this repo
  srun.sh
```

### 2. Prepare the data (one-time, ~2 min)

```bash
./srun.sh uv run prepare.py
```

Data is cached at `~/.cache/autoresearch/` and bind-mounted into the container automatically.

### 3. Test a single training run (~5 min)

```bash
./srun.sh uv run train.py
```

If this completes and prints a `val_bpb` summary, your setup is working.

## Running the agent on SLURM

The `train_run.slrm` script submits a Claude agent as a SLURM job. The agent reads `program.md`, then runs the experiment loop autonomously — modifying `train.py`, training for 5 minutes, evaluating, keeping or discarding, and repeating indefinitely.

```bash
sbatch train_run.slrm
```

The script **self-resubmits** on completion (`sbatch "$0"` at the end), so the agent keeps running across job time limits without manual intervention. To stop it:

```bash
scancel <jobid>
```

SLURM output and error logs are written to `logs/` under the repo directory.

## Running the container manually

`srun.sh` is the thin wrapper around `singularity exec --nv`:

```bash
# Run from the parent directory (contains autoresearch.sif and autoresearch/)
./srun.sh uv run train.py

# Override the directory containing autoresearch.sif
./srun.sh /path/to/dir uv run train.py
```

## Project structure

```
prepare.py       — constants, data prep + runtime utilities (do not modify)
train.py         — model, optimizer, training loop (agent modifies this)
program.md       — agent instructions
train_run.slrm   — SLURM job script (self-resubmitting)
srun.sh          — Singularity wrapper script
autoresearch.def — Singularity container definition
pyproject.toml   — Python dependencies
logs/            — SLURM stdout/stderr (created automatically)
```

## Upstream

This repo tracks [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The only HPC-specific additions are `srun.sh`, `autoresearch.def`, and `train_run.slrm`. The core research loop (`prepare.py`, `train.py`, `program.md`) follows upstream conventions.

## License

MIT
