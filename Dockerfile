# =============================================================================
# Dockerfile — autoresearch
# Source code lives on /working (network volume), NOT baked in.
# Image contains: CUDA runtime, system packages, Rust, uv, all Python deps.
# Rebuild trigger: push changes to Dockerfile, pyproject.toml, or uv.lock
# =============================================================================

FROM runpod/autoresearch:1.0.2-cuda1281-ubuntu2204

# System packages
RUN apt-get update && apt-get install -y \
    git curl wget htop tmux vim \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Rust (needed for rustbpe and other native extensions)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Pin venv to fixed path — uv run works from /working at runtime
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Install Python dependencies into /opt/venv (baked into image)
# Source code is NOT copied — it comes from /working at runtime
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-cache

# Entrypoint bootstrap (clones/pulls repo, sets up /data, calls startup.sh)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888 22

CMD ["/entrypoint.sh"]
