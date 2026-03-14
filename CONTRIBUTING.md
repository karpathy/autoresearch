# Contributing to autoresearch

Thank you for your interest in contributing to autoresearch! This project explores autonomous AI-driven research using single-GPU training setups.

## Getting Started

### Prerequisites

- A single NVIDIA GPU (tested on H100, works on RTX 3090/4090 with reduced batch sizes)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Verify setup with a single training run (~5 min)
uv run train.py
```

## Project Structure

The deliberately minimal codebase has three core files:

- **`prepare.py`** — Fixed constants, data preparation, tokenizer training, dataloader, and evaluation. **Do not modify.**
- **`train.py`** — The main file to modify. Contains the GPT model, optimizer, and training loop.
- **`program.md`** — Agent instructions for autonomous research. Edit this to customize agent behavior.

## How to Contribute

### Types of Contributions

1. **Model architecture improvements** — New attention mechanisms, layer types, or structural changes to `train.py`
2. **Optimizer enhancements** — Modifications to Muon/AdamW or new optimization strategies
3. **Hyperparameter tuning** — Discovering better default configurations
4. **`program.md` improvements** — Better agent instructions for autonomous experimentation
5. **Documentation** — README improvements, code comments, usage examples
6. **Bug fixes** — Crash fixes, compatibility improvements

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-improvement`
3. Make your changes to `train.py` or `program.md`
4. Run a training experiment to validate: `uv run train.py`
5. Log your results to `results.tsv` with format: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`
6. Commit with clear messages describing the change and its impact
7. Submit a Pull Request

### Commit Message Format

```
<type>: <short description>

<optional body explaining the change and its motivation>
<optional footer with issue references>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Example:
```
feat: add grouped-query attention to reduce KV cache memory

Implements GQA with 4 query groups, reducing peak VRAM by ~15%
while maintaining similar val_bpb performance.

Tested on H100: val_bpb=0.982 (baseline 0.998), VRAM 38GB→32GB
```

### Evaluation Criteria

All contributions are judged by **val_bpb** (validation bits per byte) — lower is better. Since training runs on a fixed 5-minute budget, the metric is directly comparable across changes.

When evaluating a change, consider:
- **Magnitude**: How much does val_bpb improve?
- **Complexity**: Does the improvement justify added code complexity?
- **Simplicity**: Removing code with equal/better results is a big win

## Code Style

- Follow existing code patterns in `train.py`
- Use clear variable names over abbreviations
- Add comments for non-obvious mathematical operations
- Keep the single-file philosophy — avoid creating new modules

## GPU Compatibility

- **Primary target**: NVIDIA H100 (compute capability 9.0)
- **Compatible**: Any NVIDIA GPU with compute capability >= 8.0
- The code auto-detects GPU capability and selects appropriate flash attention implementation

## Questions?

Open an issue for discussion before starting large changes. We'd hate for your time to be wasted on a direction that doesn't fit the project.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
