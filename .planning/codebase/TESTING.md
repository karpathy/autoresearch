# Testing Patterns

**Analysis Date:** 2026-03-25

## Test Framework

**Runner:**
- No test framework configured
- No `pytest`, `unittest`, or any test runner in `pyproject.toml` dependencies
- No test configuration files (`pytest.ini`, `conftest.py`, `tox.ini`, `setup.cfg`) present

**Assertion Library:**
- Not applicable (no test framework)

**Run Commands:**
```bash
# No test commands available
# The project has no automated test suite
```

## Test File Organization

**Location:**
- No test files exist in the repository
- No `tests/` directory, no `test_*.py` files, no `*_test.py` files

**Naming:**
- Not applicable

## Test Structure

**This codebase has no automated tests.** Validation is performed entirely through runtime execution.

## Validation Strategy (In Lieu of Tests)

The project uses a runtime validation approach instead of traditional unit/integration tests:

**Inline Assertions (`train.py`, `prepare.py`):**
```python
# Precondition checks via assert
assert self.n_embd % self.n_head == 0                    # train.py line 68
assert self.n_kv_head <= self.n_head                      # train.py line 69
assert T <= self.cos.size(1)                              # train.py line 270
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0          # train.py line 496
assert len(parquet_paths) > 0                             # prepare.py line 257
```

**Tokenizer Sanity Check (`prepare.py` line 200-202):**
```python
# Roundtrip test after training tokenizer
test = "Hello world! Numbers: 123. Unicode: 你好"
encoded = enc.encode_ordinary(test)
decoded = enc.decode(encoded)
assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
```

**Fast-Fail on Training Divergence (`train.py` lines 570-572):**
```python
# Abort if loss is NaN or exploding
if math.isnan(train_loss_f) or train_loss_f > 100:
    print("FAIL")
    exit(1)
```

**Experiment-Based Validation (`program.md`):**
- The entire `autoresearch` workflow is a validation loop: run `train.py`, check `val_bpb`, keep or discard
- `evaluate_bpb()` in `prepare.py` serves as the acceptance test (fixed evaluation metric)
- Results logged to `results.tsv` with status `keep`, `discard`, or `crash`

## Mocking

**Framework:** Not applicable (no tests)

**Patterns:** Not applicable

## Fixtures and Factories

**Test Data:** Not applicable

## Coverage

**Requirements:** None enforced. No coverage tooling configured.

## Test Types

**Unit Tests:**
- None exist

**Integration Tests:**
- None exist

**E2E Tests:**
- The closest analog is running `uv run train.py` end-to-end, which:
  1. Loads tokenizer and data
  2. Builds and compiles model
  3. Trains for 5 minutes
  4. Evaluates on validation set
  5. Reports `val_bpb` metric
- This is invoked manually or by the autonomous agent loop described in `program.md`

## Recommendations for Adding Tests

If tests are to be added, follow these patterns based on the codebase structure:

**Unit test candidates:**
- `GPTConfig` dataclass validation and `_compute_window_sizes()` in `train.py`
- `Tokenizer.encode()` / `Tokenizer.decode()` roundtrip in `prepare.py`
- `get_lr_multiplier()`, `get_muon_momentum()`, `get_weight_decay()` schedule functions in `train.py`
- `norm()`, `apply_rotary_emb()`, `has_ve()` utility functions in `train.py`

**Suggested framework:** `pytest` (standard for Python ML projects)

**Suggested test location:** `tests/` directory at project root, with files like:
- `tests/test_model.py` - GPT model forward pass, config validation
- `tests/test_optimizer.py` - MuonAdamW step correctness
- `tests/test_prepare.py` - Tokenizer, dataloader
- `tests/test_schedules.py` - LR schedule, momentum schedule

**Suggested test pattern:**
```python
import pytest
import torch
from train import GPT, GPTConfig, norm, apply_rotary_emb, get_lr_multiplier

def test_norm_preserves_shape():
    x = torch.randn(2, 10, 768)
    y = norm(x)
    assert y.shape == x.shape

def test_lr_schedule_warmup():
    assert get_lr_multiplier(0.0) == 0.0 or WARMUP_RATIO == 0
    assert get_lr_multiplier(0.5) == 1.0

def test_gpt_forward():
    config = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2,
                       n_head=2, n_kv_head=2, n_embd=64)
    model = GPT(config)
    model.init_weights()
    idx = torch.randint(0, 256, (1, 64))
    targets = torch.randint(0, 256, (1, 64))
    loss = model(idx, targets)
    assert loss.ndim == 0 and not torch.isnan(loss)
```

---

*Testing analysis: 2026-03-25*
