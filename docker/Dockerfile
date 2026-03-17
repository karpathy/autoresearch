# AutoResearch Docker Image
# Multi-stage build for efficiency

# Stage 1: Base Python environment
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as base

LABEL maintainer="AutoResearch Team"
LABEL description="Autonomous Research System with Multi-Agent Orchestration"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash -u 1000 autoresearch

WORKDIR /app

# Stage 2: Python dependencies
FROM base as builder

# Install uv (fast Python package manager)
RUN pip install uv

# Copy only pyproject.toml first (for better caching)
COPY pyproject.toml .

# Install dependencies
RUN uv pip install --system -r requirements.txt || pip install -e .

# Stage 3: Production image
FROM base as production

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/results/experiments \
    /app/results/knowledge-graph \
    /app/results/podcast-content \
    /app/results/cold-storage \
    /app/config/agents \
    /app/config/services \
    /app/agents/prompts \
    /app/agents/tools

# Create cache directory in user home
RUN mkdir -p /home/autoresearch/.cache/autoresearch && \
    chown -R autoresearch:autoresearch /app /home/autoresearch

# Switch to non-root user
USER autoresearch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Default entry point
ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]
CMD ["ar", "start", "--agents", "4"]

# Expose ports
EXPOSE 6006 8000 3004 5173

# Stage 4: Development image (with extra tools)
FROM production as development

USER root

# Install development tools
RUN pip install \
    ipython \
    jupyter \
    tensorboard \
    black \
    flake8 \
    pytest \
    pytest-cov

# Install debugger
RUN pip install debugpy

USER autoresearch

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]
