# External Integrations

**Analysis Date:** 2026-03-25

## APIs & External Services

**Hugging Face Hub (Data Download):**
- Used for: Downloading training data shards (Parquet files)
- Dataset: `karpathy/climbmix-400b-shuffle`
- Base URL: `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main`
- SDK/Client: `requests` library (direct HTTP GET, no HF SDK)
- Auth: None required (public dataset)
- Implementation: `prepare.py` `download_single_shard()` function
- Shards: Up to 6542 Parquet files (`shard_00000.parquet` through `shard_06542.parquet`)
- Retry logic: 5 attempts with exponential backoff (2^attempt seconds)

**Kernels Hub (Flash Attention):**
- Used for: Loading Flash Attention 3 CUDA kernels at runtime
- SDK/Client: `kernels` package (`from kernels import get_kernel`)
- Repos:
  - `varunneal/flash-attention-3` (Hopper / sm_90 GPUs)
  - `kernels-community/flash-attn3` (all other NVIDIA GPUs)
- Auth: None required
- Implementation: `train.py` lines 21-24

## Data Storage

**Databases:**
- None. No database used.

**File Storage:**
- Local filesystem only
- Cache directory: `~/.cache/autoresearch/`
  - `~/.cache/autoresearch/data/` - Parquet data shards
  - `~/.cache/autoresearch/tokenizer/` - Trained BPE tokenizer (`tokenizer.pkl`, `token_bytes.pt`)
- Workspace output: `workspace/output/` - Contains cached teacher embeddings for finetune script
  - `workspace/output/trendyol_teacher_cache2/`
  - `workspace/output/dinov3_teacher_cache/`
  - `workspace/output/marqo_teacher_cache/`
  - `workspace/output/siglip_teacher_cache/`

**Caching:**
- Data shards cached at `~/.cache/autoresearch/data/` after first download
- Tokenizer cached at `~/.cache/autoresearch/tokenizer/` after first training
- Teacher embedding caches in `workspace/output/` for finetune workflows

## Authentication & Identity

**Auth Provider:**
- None. No authentication system. This is a local research tool.

## Monitoring & Observability

**Error Tracking:**
- None. Stdout/stderr only.

**Logs:**
- Training metrics printed to stdout via `print()` with `\r` carriage return for live updates
- Experiment results logged to `results.tsv` (untracked, tab-separated) by the AI agent
- Finetune script uses `loguru` logger
- Agent workflow redirects output: `uv run train.py > run.log 2>&1`

## CI/CD & Deployment

**Hosting:**
- Not deployed. Local single-GPU research tool.

**CI Pipeline:**
- None detected. No GitHub Actions, no CI config files.

## Environment Configuration

**Required env vars:**
- None explicitly required. All configuration is in-code.

**Auto-set env vars (in `train.py`):**
- `PYTORCH_ALLOC_CONF=expandable_segments:True` - PyTorch memory allocator config
- `HF_HUB_DISABLE_PROGRESS_BARS=1` - Suppress HF download bars

**Finetune script env vars:**
- `LD_LIBRARY_PATH` - Auto-configured to include NVIDIA pip package lib dirs for ONNX runtime

**Secrets location:**
- No secrets. Public datasets, no API keys required.

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## External Model Dependencies

**Flash Attention 3 Kernels:**
- Downloaded at runtime via `kernels` package from GitHub-hosted repos
- Required for attention computation in `train.py`
- GPU capability auto-detected: `torch.cuda.get_device_capability()`

**ONNX Teacher Models (finetune script only):**
- Referenced at hardcoded path: `/data/mnt/mnt_ml_shared/joesu/reid/distill_qwen_lcnet050_retail_2.onnx`
- Used as teacher model for knowledge distillation in `finetune_trendyol_arcface3.py`
- Loaded via `onnxruntime.InferenceSession`

**Hugging Face Pretrained Models (finetune script only):**
- `timm` models loaded via `timm.create_model()`
- `transformers` models loaded via `AutoModel.from_pretrained()`

---

*Integration audit: 2026-03-25*
