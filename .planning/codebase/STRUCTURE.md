# Codebase Structure

**Analysis Date:** 2026-03-25

## Directory Layout

```
autoresearch/
├── train.py                        # GPT model + optimizer + training loop (AGENT-EDITABLE)
├── prepare.py                      # Data prep + runtime utilities (READ-ONLY)
├── program.md                      # AI agent instructions / experiment protocol
├── finetune_trendyol_arcface3.py   # Separate CV fine-tuning script (not part of autoresearch pipeline)
├── analysis.ipynb                  # Jupyter notebook for experiment result analysis
├── progress.png                    # Generated chart of experiment progress
├── README.md                       # Project documentation
├── pyproject.toml                  # uv/pip dependencies
├── requirements.txt                # Alternate pip requirements (CV fine-tuning deps)
├── uv.lock                         # Locked dependency versions
├── .python-version                 # Python version pin
├── .gitignore                      # Git ignore rules
├── .venv/                          # Virtual environment (not committed)
├── .claude/                        # Claude agent configuration
├── .planning/                      # Planning/analysis documents
└── workspace/                      # Working directory for CV fine-tuning outputs
    └── output/
        ├── dinov3_teacher_cache/
        ├── distill_trendyol_lcnet050_retail/
        ├── marqo_teacher_cache/
        ├── siglip_teacher_cache/
        └── trendyol_teacher_cache2/
```

## Directory Purposes

**Project Root:**
- Purpose: All source code lives flat at the root; no `src/` directory
- Contains: Python scripts, config files, documentation
- Key files: `train.py`, `prepare.py`, `program.md`

**`~/.cache/autoresearch/` (external, not in repo):**
- Purpose: Cached training data and tokenizer artifacts
- Contains: `data/` (parquet shards), `tokenizer/` (tokenizer.pkl, token_bytes.pt)
- Key files: `~/.cache/autoresearch/tokenizer/tokenizer.pkl`, `~/.cache/autoresearch/data/shard_*.parquet`

**`workspace/output/`:**
- Purpose: Output artifacts from the CV fine-tuning script (`finetune_trendyol_arcface3.py`)
- Contains: Teacher model embedding caches, training outputs
- Generated: Yes
- Committed: Partially (directory structure committed, large files likely gitignored)

## Key File Locations

**Entry Points:**
- `train.py`: LLM pretraining entry point (module-level execution, no `__main__` guard)
- `prepare.py`: Data preparation entry point (has `__main__` guard with argparse)
- `finetune_trendyol_arcface3.py`: CV fine-tuning entry point (separate pipeline)

**Configuration:**
- `pyproject.toml`: Project metadata, Python version constraint (>=3.10), dependencies, PyTorch CUDA index
- `.python-version`: Pins Python 3.12
- `requirements.txt`: Alternate dependency list for the CV fine-tuning script (different deps: loguru, timm, onnxruntime-gpu, transformers)

**Core Logic:**
- `train.py` lines 30-291: GPT model architecture (`GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`)
- `train.py` lines 293-427: Optimizer (`MuonAdamW`, compiled step functions)
- `train.py` lines 429-631: Hyperparameters, setup, training loop, evaluation
- `prepare.py` lines 56-113: Data download from HuggingFace
- `prepare.py` lines 119-203: Tokenizer training
- `prepare.py` lines 209-365: Runtime utilities (Tokenizer class, dataloader, evaluation)

**Agent Protocol:**
- `program.md`: Full experiment protocol for the AI agent (setup, loop, logging, output format)

**Analysis:**
- `analysis.ipynb`: Jupyter notebook that reads `results.tsv` and generates `progress.png`

## Naming Conventions

**Files:**
- Python scripts: `snake_case.py` (e.g., `train.py`, `prepare.py`, `finetune_trendyol_arcface3.py`)
- Documentation: `lowercase.md` (e.g., `program.md`, `README.md`)
- Config: Standard names (`pyproject.toml`, `.gitignore`)

**Classes:**
- PascalCase: `GPT`, `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `MuonAdamW`, `Tokenizer`

**Functions:**
- snake_case: `apply_rotary_emb`, `make_dataloader`, `evaluate_bpb`, `get_lr_multiplier`
- Compiled kernels: `adamw_step_fused`, `muon_step_fused`

**Constants:**
- UPPER_SNAKE_CASE: `MAX_SEQ_LEN`, `TIME_BUDGET`, `TOTAL_BATCH_SIZE`, `DEPTH`, `ASPECT_RATIO`

**Variables:**
- snake_case for locals: `train_loss`, `grad_accum_steps`, `smooth_train_loss`
- Short math-style names in model code: `x`, `q`, `k`, `v`, `B`, `T`, `C`

## Where to Add New Code

**Modifying the LLM training pipeline:**
- Edit `train.py` only. This is the designated mutable file in the autoresearch experiment loop.
- All model architecture changes: `train.py` lines 30-291
- All optimizer changes: `train.py` lines 293-427
- All hyperparameter changes: `train.py` lines 429-451 (module-level constants)
- Do NOT modify `prepare.py` (contains fixed evaluation metric and data pipeline)

**Adding a new standalone script:**
- Place at project root as a new `.py` file (following the flat structure pattern)
- Example: `finetune_trendyol_arcface3.py` follows this pattern

**Adding new analysis:**
- Use `analysis.ipynb` or add new notebooks at project root

**Adding new dependencies:**
- Add to `pyproject.toml` `dependencies` list, then `uv sync`
- Note: the autoresearch agent protocol FORBIDS adding new dependencies during experiments

## Special Directories

**`~/.cache/autoresearch/`:**
- Purpose: External cache for data shards and trained tokenizer
- Generated: Yes, by `prepare.py`
- Committed: No (lives outside repo)

**`.venv/`:**
- Purpose: Python virtual environment managed by uv
- Generated: Yes
- Committed: No (gitignored)

**`workspace/output/`:**
- Purpose: CV fine-tuning artifacts (teacher caches, model outputs)
- Generated: Yes
- Committed: Partially

## Two Pipelines in One Repo

This repo contains two distinct pipelines that share no code:

1. **Autoresearch LLM pretraining** (`train.py` + `prepare.py`): Single-GPU GPT pretraining with autonomous AI agent experimentation. Dependencies in `pyproject.toml`.

2. **CV fine-tuning** (`finetune_trendyol_arcface3.py`): Qwen distillation + ArcFace on retail product images. Dependencies in `requirements.txt`. Uses `workspace/output/` for artifacts. Imports from a parent repo via `sys.path.insert`.

These are independent. The autoresearch pipeline is the primary purpose of this repo.

---

*Structure analysis: 2026-03-25*
