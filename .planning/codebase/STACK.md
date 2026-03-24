# Technology Stack

**Analysis Date:** 2026-03-25

## Languages

**Primary:**
- Python >=3.10 - All source code (`prepare.py`, `train.py`, `finetune_trendyol_arcface3.py`)

## Runtime

**Environment:**
- Python 3.10+ (specified in `pyproject.toml` `requires-python`)
- CUDA (NVIDIA GPU required; targets H100 but supports other NVIDIA GPUs)
- CUDA 12.8 toolkit (PyTorch installed from `cu128` index)

**Package Manager:**
- [uv](https://docs.astral.sh/uv/) - Astral's fast Python package manager
- Lockfile: `uv.lock` (present, committed)
- All commands run via `uv run` (e.g., `uv run train.py`, `uv run prepare.py`)

## Frameworks

**Core:**
- PyTorch 2.9.1 (`torch==2.9.1`) - Deep learning framework, single-GPU training
- `torch.compile` - Used for model compilation and fused optimizer kernels

**ML Libraries:**
- `kernels>=0.11.7` - Flash Attention 3 kernel loading (uses `varunneal/flash-attention-3` on Hopper GPUs, `kernels-community/flash-attn3` on others)
- `rustbpe>=0.1.0` - Rust-based BPE tokenizer training
- `tiktoken>=0.11.0` - OpenAI tokenizer library (used as encoding wrapper after rustbpe training)

**Data Processing:**
- `pyarrow>=21.0.0` - Parquet file reading for data shards
- `numpy>=2.2.6` - Numerical operations
- `pandas>=2.3.3` - Data analysis (used in `analysis.ipynb`)

**Visualization:**
- `matplotlib>=3.10.8` - Experiment progress plotting (`analysis.ipynb`)

**Networking:**
- `requests>=2.32.0` - HTTP downloads of data shards from Hugging Face

**Secondary (finetune script only, via `requirements.txt`):**
- `timm` - PyTorch Image Models (pretrained vision backbones)
- `transformers` - Hugging Face Transformers (AutoModel)
- `torchvision` - Image transforms and datasets
- `onnxruntime-gpu>=1.17,<1.20` - ONNX inference for teacher model embeddings
- `Pillow` - Image loading/manipulation
- `loguru` - Structured logging

## Key Dependencies

**Critical:**
- `torch==2.9.1` (pinned exact) - Core training framework. Sourced from PyTorch CUDA 12.8 wheel index
- `kernels>=0.11.7` - Provides Flash Attention 3 kernels at runtime. Dynamically selects kernel repo based on GPU capability (`cap == (9, 0)` for Hopper)
- `rustbpe>=0.1.0` - Trains BPE tokenizer from scratch on training data

**Infrastructure:**
- `pyarrow>=21.0.0` - Reads Parquet data shards (6542+ shards from Hugging Face dataset)
- `tiktoken>=0.11.0` - Wraps trained BPE tokenizer for fast encoding/decoding

## Configuration

**Environment:**
- No `.env` files used. Configuration is entirely in-code constants
- `PYTORCH_ALLOC_CONF=expandable_segments:True` set in `train.py` at import time
- `HF_HUB_DISABLE_PROGRESS_BARS=1` set in `train.py` at import time
- Cache directory: `~/.cache/autoresearch/` (data shards + trained tokenizer)

**Build:**
- `pyproject.toml` - Project metadata, dependencies, uv source configuration
- `uv.lock` - Locked dependency versions
- Custom PyTorch index: `https://download.pytorch.org/whl/cu128` (explicit, CUDA 12.8 wheels)

**Training Constants (in `prepare.py`, immutable):**
- `MAX_SEQ_LEN = 2048` - Context length
- `TIME_BUDGET = 300` - 5-minute fixed training time budget
- `EVAL_TOKENS = 40 * 524288` - Validation evaluation token count
- `VOCAB_SIZE = 8192` - BPE vocabulary size

**Training Hyperparameters (in `train.py`, agent-editable):**
- `DEPTH = 8` - Transformer layers
- `ASPECT_RATIO = 64` - Model dim = depth * aspect_ratio
- `TOTAL_BATCH_SIZE = 2**19` - ~524K tokens per step
- `DEVICE_BATCH_SIZE = 128` - Per-device batch size
- Optimizer: MuonAdamW (Muon for 2D matrix params, AdamW for embeddings/scalars)

## Platform Requirements

**Development:**
- Single NVIDIA GPU (tested on H100)
- Python 3.10+
- uv package manager
- ~2 minutes one-time data prep (`uv run prepare.py`)
- Disk: `~/.cache/autoresearch/` for data shards (multiple GB)

**Production:**
- Not a production service. Research experimentation tool
- Designed for autonomous AI agent operation (agent modifies `train.py`, runs 5-min experiments in a loop)
- Each experiment: ~5 min training + startup/compilation overhead

**GPU-Specific Kernel Selection:**
- Hopper (sm_90, e.g. H100): `varunneal/flash-attention-3`
- Other NVIDIA GPUs: `kernels-community/flash-attn3`
- BF16 mixed precision throughout (`torch.amp.autocast` with `torch.bfloat16`)

---

*Stack analysis: 2026-03-25*
