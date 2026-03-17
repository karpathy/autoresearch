# AutoResearch Installation Guide

Complete step-by-step instructions for installing AutoResearch on Linux, macOS, and Windows 11.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Linux/macOS Installation](#linuxmacos-installation)
3. [Windows 11 Installation](#windows-11-installation)
4. [Docker Installation](#docker-installation)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Prerequisites:**
- Python 3.10+ installed
- Git installed
- (Optional) NVIDIA GPU with CUDA 12.1+

### Linux/macOS:
```bash
cd autoresearch
bash install/install.sh --interactive
```

### Windows 11:
```powershell
cd autoresearch
.\install\install.ps1 -Mode interactive
```

### Docker (All platforms):
```bash
cd autoresearch
docker-compose -f docker/docker-compose.yml up -d
```

---

## Linux/macOS Installation

### Step 1: Install Python 3.10+

**macOS:**
```bash
brew install python3
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip git curl
```

**Verify:**
```bash
python3 --version  # Should be 3.10 or later
```

### Step 2: Clone Repository

```bash
git clone https://github.com/SuperInstance/autoclaw.git
cd autoclaw
```

### Step 3: Run Installation Script

```bash
bash install/install.sh --interactive
```

The installer will:
1. Validate Python and Git installation
2. Detect your GPU (if available)
3. Identify package manager (uv, pip, conda)
4. Install dependencies
5. Prepare data (one-time, ~5 minutes)
6. Run interactive configuration wizard
7. Execute baseline training test

### Step 4: Activate Configuration

After installation, edit configuration files:
```bash
# Add your API keys
nano config/services/openai.yaml
nano config/services/anthropic.yaml

# Review system config
nano config/system.yaml
```

### Step 5: Start Agents

```bash
# Start with 1 agent for testing
ar start --agents 1

# Monitor progress
ar status

# Chat with agent
ar chat default

# View results
ar metrics --last 1h
```

---

## Windows 11 Installation

### Step 1: Install Python

1. Download Python from https://www.python.org/downloads/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Verify:
   ```powershell
   python --version
   ```

**Alternative (Package Manager):**
```powershell
winget install Python.Python.3.11
```

### Step 2: Install Git

1. Download from https://git-scm.com/download/win
2. Run installer with defaults
3. Restart terminal to apply PATH changes
4. Verify:
   ```powershell
   git --version
   ```

### Step 3: Clone Repository

```powershell
git clone https://github.com/SuperInstance/autoclaw.git
cd autoclaw
```

### Step 4: Install with PowerShell

```powershell
# Allow script execution (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run installer
.\install\install.ps1 -Mode interactive
```

### Step 5-Step 5: Configure & Start

Same as Linux/macOS above.

---

## Docker Installation

### Prerequisites

1. **Docker Desktop** - Download from https://www.docker.com/products/docker-desktop
2. **Docker Compose** - Included with Docker Desktop

### Step 1: Verify Docker Installation

```bash
docker --version
docker-compose --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/SuperInstance/autoclaw.git
cd autoclaw
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp docker/.env.example docker/.env

# Edit with your API keys
nano docker/.env  # or use your favorite editor
```

**Key settings to update:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `POSTGRES_PASSWORD` - Change to secure password
- `AGENTS_COUNT` - Number of agents (1-4 recommended)

### Step 4: Enable GPU (Optional)

If you have an NVIDIA GPU:

1. Install NVIDIA Container Toolkit:
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

2. Verify:
   ```bash
   docker run --rm --runtime=nvidia nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

### Step 5: Start Container

```bash
# Start all services
docker-compose up -d

# Wait for startup (30-60 seconds)
docker-compose logs -f autoresearch

# Once running, you should see: "Starting AutoResearch..."
```

### Step 6: Verify Running

```bash
# Check status
docker-compose ps

# View logs
docker-compose logs autoresearch

# Access services:
# - AutoResearch CLI: docker-compose exec autoresearch ar status
# - Murmur Wiki: http://localhost:3004
# - Spreadsheet: http://localhost:5173
# - Prometheus: http://localhost:9090 (if enabled)
```

### Step 7: Stop Container

```bash
# Stop all services
docker-compose down

# Stop and remove data
docker-compose down -v

# Stop but keep data
docker-compose stop
```

---

## Configuration

### API Keys

Edit `config/services/*.yaml` to add your API keys:

```yaml
openai:
  enabled: true
  api_key: "sk-..."  # Replace with your key
```

Available services:
- **openai.yaml** - OpenAI (GPT-4, GPT-4-turbo)
- **anthropic.yaml** - Anthropic Claude
- **chinese-apis.yaml** - Deepseek, Qwen, Baichuan
- **local-services.yaml** - Ollama, vLLM (local)

### Agent Configuration

Edit `config/agents/*.yaml` to customize agents:

```yaml
hyperparameter-specialist:
  gpu_allocation: 0.35        # % of GPU
  api_rate_limit: 12000       # tokens/min
  focus: "Learning rate optimization"
```

Pre-configured agents:
- `hyperparameter-specialist.yaml` - Hyperparameter tuning
- `synthesis-agent.yaml` - Research synthesis & debate

### Data Retention

Edit `config/retention-policies.yaml`:

```yaml
retention:
  default:
    hot_storage_days: 1       # Immediate access
    warm_storage_days: 30     # Compressed cloud
    cold_storage_days: 180    # Archival
```

---

## Verification

### Test Installation

```bash
# Check system status
ar verify

# Expected output:
# ✓ Python 3.11.8
# ✓ Git 2.45.0
# ✓ GPU: NVIDIA H100 (80GB)
# ✓ All dependencies installed
```

### Run Baseline Experiment

```bash
# Manual training run (5 minutes)
uv run train.py

# Expected output:
# Training for 5 minutes...
# [████████████████████] 100%
# val_bpb: 1.042 (baseline)
```

### Start Single Agent

```bash
# Start 1 agent for testing
ar start --agents 1

# Monitor progress
ar status

# You should see:
# Agent: default
# Status: running
# Experiments: 3/12
# Best val_bpb: 1.038 (improvement!)
```

---

## Troubleshooting

### Python Not Found

**Error:** `python3: command not found`

**Solution:**
1. Verify Python is installed: `which python3`
2. If not in PATH, reinstall with "Add to PATH" option
3. Restart terminal after installation

### Git Not Found

**Error:** `git: command not found`

**Solution:**
1. Install Git from https://git-scm.com
2. Restart terminal
3. Verify: `git --version`

### Insufficient Disk Space

**Error:** `Error: Not enough space for data preparation`

**Solution:**
1. Free up disk space (need ~500GB for full setup)
2. Or reduce `MAX_SEQ_LEN` in `prepare.py`:
   ```python
   MAX_SEQ_LEN = 1024  # Down from 2048
   ```

### GPU Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
1. Reduce batch size in `train.py`:
   ```python
   DEVICE_BATCH_SIZE = 8  # Down from 16
   ```
2. Reduce model size (`DEPTH`, `EMB_SIZE`)
3. Use `cpu` mode for testing:
   ```bash
   ar start --agents 1 --mode cpu
   ```

### Docker Connection Refused

**Error:** `Connection refused: docker daemon not running`

**Solution:**
1. Start Docker Desktop (macOS/Windows)
2. Or start Docker daemon (Linux):
   ```bash
   sudo systemctl start docker
   ```

### Configuration Validation Failed

**Error:** `Configuration validation failed: API key invalid`

**Solution:**
1. Check API key format (should not include `sk-` prefix in template)
2. Verify key is from correct service (OpenAI key for OpenAI, etc.)
3. Uncomment the service in YAML before using

### Database Connection Failed

**Docker only**

**Error:** `PostgreSQL connection refused`

**Solution:**
1. Check PostgreSQL is running:
   ```bash
   docker-compose logs postgres
   ```
2. Verify password in `.env` matches config
3. Wait longer for startup (can take 30-60 seconds)

---

## Next Steps

1. **Review Configuration**: Edit `config/system.yaml`
2. **Add API Keys**: Update `config/services/*.yaml`
3. **Understand Agents**: Read agent profiles in `config/agents/`
4. **Start Research**: `ar start --agents 4`
5. **Monitor Progress**: `ar status` and `ar metrics`
6. **Read Documentation**: See [ONBOARDING.md](ONBOARDING.md)

---

## Performance Tips

### Optimize for Speed

1. Use `uv` instead of `pip`:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

2. Use vLLM instead of Ollama for local models:
   ```bash
   pip install vllm
   vllm serve mistral-7b
   ```

3. Allocate more GPU to agents:
   ```yaml
   hyperparameter-specialist:
     gpu_allocation: 0.5  # 50% of GPU
   ```

### Optimize for Cost

1. Use cheaper models (GPT-4 Mini, Claude Haiku)
2. Use local models (Ollama, vLLM) - no API costs
3. Reduce agent count: `ar start --agents 1`
4. Lower API rate limits in config

### Optimize for Reliability

1. Use multiple services (OpenAI + Anthropic)
2. Enable fallback models in config
3. Use cold storage archival for backups
4. Monitor with: `ar metrics --graph uptime`

---

## Support & Community

- **Issues**: https://github.com/SuperInstance/autoclaw/issues
- **Discussions**: https://github.com/SuperInstance/autoclaw/discussions
- **Discord**: [SuperInstance Community](https://discord.gg/superinstance)

Happy researching! 🚀
