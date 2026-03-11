# autoresearch/fine_tuning

*The era of manual model fine-tuning ended with a whimper, then a bang. Research is no longer about humans staring at loss curves; it's about setting the right constraints and letting autonomous swarms of agents find the global minimum. We don't write the adapters; we write the programs that write the adapters. -@karpathy, March 2026*.

The idea: extend the autonomous research concept to **fine-tuning** pretrained language models. Instead of training from scratch, these agents take a strong base model and specialize it for a specific domain while strictly monitoring for catastrophic forgetting. The system uses LoRA (Low-Rank Adaptation) and experience replay to ensure the model learns new tricks without losing its mind.

The core principle remains the same: you don't edit the Python code. You program the `program.md` instructions, set your GPU cluster loose, and wake up to a model that is both a domain expert and as capable as the day it was born.

## How it works

The fine-tuning subdirectory contains everything needed for autonomous specialized research:

- **`prepare.py`** — Handles data preparation in `--mode finetune`. It splits domain trajectories (e.g., from `trajectories.jsonl`) and downloads general-domain replay data (e.g., from FineWeb-Edu) to prevent weight drift.
- **`train.py`** — The agent's playground. A multi-GPU Distributed Data Parallel (DDP) script that implements the LoRA fine-tuning loop. It treats the base model as frozen and optimizes only the adapters. **This file is edited and iterated on by the agent**.
- **`program.md`** — Instructions for the autonomous fine-tuning agent. It defines the "laws" of the research org (e.g., "never set REPLAY_RATIO to 0.0") and the loop of hypothesis -> code change -> multi-GPU run. **This file is edited and iterated on by the human**.

By design, experiments are multi-GPU by default (standard setup is 6x GPUs). The metrics are `val_bpb_domain` (improving specialization) and `val_bpb_general` (ensuring general performance remains within 15% of the baseline).

## Quick start

**Requirements:** One or multiple NVIDIA GPUs (tested on 6x RTX 3090), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
cd fine_tuning

# Install dependencies
uv sync

# Place your model (can also use HuggingFace download, but local is faster)
# If using a local model, set MODEL_NAME to the absolute path in train.py:
#   MODEL_NAME = "ADD_MODEL_NAME"
# Or let it auto-download from HuggingFace (needs internet + disk space):
#   MODEL_NAME = "nvidia/Nemotron-Mini-4B-Instruct"

# Place your domain dataset
mkdir -p domain_data
cp /path/to/dataset.jsonl ./domain_data/trajectories.jsonl

# Prepare fine-tuning data (domain + general replay)
uv run prepare.py --mode finetune

# Run a baseline training experiment across GPUs
# (Default uses 6 GPUs; adjust --nproc_per_node as needed)
torchrun --nproc_per_node=6 train.py
```

If the baseline runs successfully, your cluster is ready for the agent to take over.

## Running the agent

Point your coding agent (Claude, etc.) at the `program.md` in this directory:

```
I've set up the fine-tuning environment. Look at fine_tuning/program.md and start the first autonomous experiment on our domain data!
```

## Project structure

```
prepare.py      — data prep for fine-tuning + replay (human runs this)
train.py        — DDP + LoRA training loop (agent modifies this)
program.md      — autonomous agent instructions (human modifies this)
```

## Design choices

- **Anti-Forgetting Safeguards.** Fine-tuning often breaks general reasoning. This system mandates experience replay from a general-domain corpus and enforces a "hard floor" on general validation performance. If bit-per-byte (BPB) on general data degrades by more than 15%, the experiment is automatically rejected.
- **Multi-GPU Parallelism.** Fine-tuning large models requires serious compute. The setup is configured for `torchrun` and DDP out of the box to maximize throughput during autonomous runs.
- **LoRA-Only Adaptation.** By restricting the agent to parameter-efficient fine-tuning (PEFT), we keep the experimental surface area manageable and drastically reduce the risk of catastrophic weight collapse.

## License

MIT
