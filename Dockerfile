# ─────────────────────────────────────────────────────────────────────
# autoresearch-mnemebrain — Linux CUDA Docker image
#
# Runs the full autoresearch loop with MnemeBrain sidecar inside a
# container. For Mac users who want CUDA training via Docker.
#
# Prerequisites (Mac host):
#   - OrbStack or Docker Desktop with NVIDIA GPU passthrough
#   - OR: run on a Linux cloud VM with NVIDIA drivers installed
#
# Build:
#   docker build -t autoresearch-mnemebrain .
#
# Run (Linux with GPU):
#   docker run --gpus all -it \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     -v $(pwd)/results:/app/results \
#     autoresearch-mnemebrain
#
# Run (Mac via OrbStack / Docker Desktop - GPU passthrough):
#   docker run --gpus all -it \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     autoresearch-mnemebrain
#
# ─────────────────────────────────────────────────────────────────────

# Base: PyTorch 2.9.1 + CUDA 12.8 on Ubuntu 22.04
# Matches autoresearch pyproject.toml torch==2.9.1 + pytorch-cu128
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Copy project files ────────────────────────────────────────────────
COPY pyproject.toml uv.lock* ./
COPY platform.py agent_runner.py ./
COPY mnemebrain_sidecar/ ./mnemebrain_sidecar/

# ── Copy autoresearch core files (from submodule or local copy) ───────
# These are the ORIGINAL karpathy files — not modified.
# Mount your own train.py / prepare.py / program.md via -v if you want
# to use a customised version.
COPY prepare.py train.py program.md ./

# ── Install Python dependencies ────────────────────────────────────────
# torch is already in the base image; skip the CUDA index for pip
RUN uv pip install --system --no-deps mnemebrain httpx fastapi uvicorn pydantic \
    anthropic openai matplotlib pandas pyarrow requests rustbpe tiktoken kernels \
    numpy 2>/dev/null || \
    pip install --no-cache-dir mnemebrain httpx fastapi uvicorn pydantic \
    anthropic openai matplotlib pandas pyarrow requests rustbpe tiktoken numpy

# ── Data directory ────────────────────────────────────────────────────
ENV AUTORESEARCH_CACHE=/app/.cache
RUN mkdir -p /app/.cache /app/results

# ── Ports ─────────────────────────────────────────────────────────────
# 7432 = MnemeBrain sidecar
# 8000 = MnemeBrain backend (if running inside container)
EXPOSE 7432 8000

# ── Entry point ───────────────────────────────────────────────────────
# Default: show help. Override CMD to run specific workflows.
#
# Run agent loop:
#   docker run ... autoresearch-mnemebrain python agent_runner.py --backend claude-api
#
# Run sidecar only:
#   docker run -p 7432:7432 ... autoresearch-mnemebrain python -m mnemebrain_sidecar
#
# Prepare data:
#   docker run ... autoresearch-mnemebrain uv run prepare.py
#
CMD ["python", "agent_runner.py", "--help"]
