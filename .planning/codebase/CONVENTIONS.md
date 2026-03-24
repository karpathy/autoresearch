# Coding Conventions

**Analysis Date:** 2026-03-25

## Naming Patterns

**Files:**
- Lowercase with underscores: `train.py`, `prepare.py`, `finetune_trendyol_arcface3.py`
- Single-purpose scripts: each file is a self-contained runnable script or module
- Versioning by suffix number: `arcface3` indicates iteration 3 of that approach

**Functions:**
- `snake_case` for all functions: `apply_rotary_emb()`, `get_lr_multiplier()`, `download_single_shard()`
- Private/internal functions prefixed with underscore: `_document_batches()`, `_precompute_rotary_embeddings()`, `_patch_transformers_compat()`
- Short utility functions use terse names: `norm()`, `has_ve()`

**Variables:**
- `snake_case` for local variables and function parameters
- `UPPER_SNAKE_CASE` for module-level constants: `MAX_SEQ_LEN`, `TIME_BUDGET`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`
- Single-letter variables used freely in math-heavy code: `B`, `T`, `C`, `x`, `q`, `k`, `v`, `d`
- Abbreviated variable names in ML contexts: `lrm` (learning rate multiplier), `dt` (delta time), `mfu` (model FLOPs utilization)

**Classes:**
- `PascalCase`: `GPT`, `GPTConfig`, `MuonAdamW`, `CausalSelfAttention`, `FrozenBackboneWithHead`
- Dataclasses for configuration: `GPTConfig`
- Transform classes as callable objects: `PadToSquare`, `RandomQualityDegradation`

**Types:**
- Type hints used extensively in `finetune_trendyol_arcface3.py`: full return type annotations, parameter types with `str | None` union syntax
- Minimal type hints in `train.py` and `prepare.py` (research/experiment code)
- Modern Python 3.10+ union syntax (`str | None`) preferred over `Optional[str]`

## Code Style

**Formatting:**
- No formatter configured (no `.prettierrc`, `pyproject.toml` has no `[tool.black]` or `[tool.ruff]` section)
- Consistent 4-space indentation throughout
- Line length varies: generally under 120 characters but long lines are not wrapped aggressively
- f-strings used universally for string formatting

**Linting:**
- No linter configured (no `.flake8`, no `[tool.ruff]`, no `[tool.pylint]`)
- Code relies on `assert` statements for runtime validation instead of explicit error handling
- Type checking via inline `# type: ignore` comments where needed

## Import Organization

**Order (observed in `train.py`):**
1. Standard library (`os`, `gc`, `math`, `time`, `dataclasses`)
2. Third-party packages (`torch`, `torch.nn`, `torch.nn.functional`)
3. Specialized/kernel imports (`kernels`)
4. Local imports (`from prepare import ...`)

**Order (observed in `finetune_trendyol_arcface3.py`):**
1. `__future__` imports
2. Standard library (`os`, `site`) -- sometimes with immediate side effects
3. `collections.abc`, `dataclasses`, `pathlib` etc.
4. Third-party (`torch`, `timm`, `onnxruntime`, `numpy`, `PIL`)
5. Local/project imports (`from transformers import ...`)

**Path Aliases:**
- None configured. All imports use relative or absolute paths.

**Side Effects at Import Time:**
- `train.py` sets `os.environ` values at module top level (lines 8-9)
- `train.py` executes GPU capability detection and kernel loading at import time (lines 21-24)
- `finetune_trendyol_arcface3.py` patches `LD_LIBRARY_PATH` at import time (lines 14-24)

## Error Handling

**Patterns:**
- `assert` statements for preconditions and invariants throughout: `assert self.n_embd % self.n_head == 0`, `assert T <= self.cos.size(1)`
- `try/except` with fallback for I/O operations: corrupted images fall back to random replacement (e.g., `finetune_trendyol_arcface3.py` line 520-524)
- Retry with exponential backoff for network operations: `download_single_shard()` in `prepare.py` (lines 66-88)
- Fast-fail on NaN/exploding loss: `if math.isnan(train_loss_f) or train_loss_f > 100: exit(1)` in `train.py` (lines 570-572)
- `sys.exit(1)` for unrecoverable errors in data preparation

**No structured error types:** The codebase does not define custom exception classes. Standard exceptions and `assert` are used throughout.

## Logging

**Framework:**
- `train.py` / `prepare.py`: Plain `print()` statements for all output. No logging framework.
- `finetune_trendyol_arcface3.py`: `loguru.logger` for structured logging with `logger.info()`, `logger.error()`, `logger.warning()`

**Patterns:**
- Training progress logged via `\r` carriage return (single overwriting line): `train.py` line 590
- Summary metrics printed as `key: value` pairs after training completes: `train.py` lines 622-631
- Use `print()` for user-facing output in script mode
- Use `loguru.logger` for library/component-level logging in `finetune_trendyol_arcface3.py`

## Comments

**When to Comment:**
- Section headers use ASCII divider lines: `# ---------------------------------------------------------------------------`
- Brief inline comments for non-obvious math or ML concepts: `# Value residual (ResFormer)`, `# Nesterov momentum`, `# Polar express orthogonalization`
- Docstrings on key classes and public functions (brief, 1-3 lines)
- Comments reference research paper names: `ResFormer`, `NorMuon`

**Docstring Style:**
- Triple-quoted docstrings, typically single-line or short paragraph
- Args documented inline in `finetune_trendyol_arcface3.py` using Google-style `Args:` sections (e.g., `save_batch_visualization`)
- No enforced docstring standard in `train.py`/`prepare.py`

## Function Design

**Size:**
- Functions range from 1-line utilities (`norm()`) to 50+ line training loops
- Complex logic kept in single functions rather than decomposed (e.g., `make_dataloader` is ~60 lines)
- Training loop is inline in module scope, not wrapped in a function (`train.py` lines 543-604)

**Parameters:**
- Hyperparameters as module-level constants, not function parameters: `TOTAL_BATCH_SIZE`, `DEPTH`, etc.
- Configuration via `@dataclass`: `GPTConfig`
- Default parameter values used liberally for configuration

**Return Values:**
- Functions return raw values (tensors, floats, dicts)
- No result wrapper types or error monads
- Generator functions for data iteration: `make_dataloader()` uses `yield`

## Module Design

**Exports:**
- `prepare.py` exports constants and utilities via direct import: `from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb`
- No `__all__` definitions
- No barrel files or `__init__.py` (flat structure)

**Script Pattern:**
- Both `train.py` and `prepare.py` are executable scripts
- `prepare.py` uses `if __name__ == "__main__":` guard with `argparse`
- `train.py` executes immediately at module level (no main guard) -- designed to be run as `uv run train.py`

## Torch-Specific Conventions

**Device Management:**
- Explicit `device = torch.device("cuda")` assignment
- Meta device for lazy initialization: `with torch.device("meta"): model = GPT(config)` then `model.to_empty(device=device)`
- `pin_memory=True` for CPU-to-GPU transfers

**Compilation:**
- `torch.compile` used on both model and optimizer step functions
- `@torch.compile(dynamic=False, fullgraph=True)` decorator for fused optimizer kernels
- `torch.set_float32_matmul_precision("high")` set globally

**Precision:**
- `torch.bfloat16` as default training precision via `torch.amp.autocast`
- Explicit `.float()` upcast for loss computation and certain math
- Rotary embeddings computed in `float32` then cast to `bfloat16`

**Memory Management:**
- Manual GC control: `gc.collect(); gc.freeze(); gc.disable()` after first step to avoid GC stalls
- Periodic GC every 5000 steps
- Pre-allocated pinned memory buffers for dataloader

---

*Convention analysis: 2026-03-25*
