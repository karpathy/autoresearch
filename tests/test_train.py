"""
tests/test_train.py — baseline tests for model components in models/nanochat.py.

These tests run on CPU without requiring Flash Attention 3 or a GPU.
"""
from __future__ import annotations
import importlib
import pytest

# Skip the entire module if torch is not installed (e.g. on macOS dev machines
# where the CUDA wheel can't be installed). These tests are designed for the
# Linux/CUDA target environment.
torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(
    torch_spec is None,
    reason="torch not installed in this environment",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEAD_DIM = 64  # canonical head dimension used in train.py


def _tiny_config():
    """Return a minimal GPTConfig suitable for CPU unit tests."""
    from models.nanochat import GPTConfig  # noqa: PLC0415
    return GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=HEAD_DIM * 2,   # 128 — smallest valid dim (HEAD_DIM multiple)
        window_pattern="SL",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.cpu
def test_build_model_config():
    """build_model_config() returns a GPTConfig with valid derived values.

    Specifically:
    - n_embd must be a multiple of HEAD_DIM
    - n_head must evenly divide n_embd
    - n_kv_head must evenly divide n_head
    - sequence_len must be positive
    """
    from models.nanochat import GPTConfig, build_model_config  # noqa: PLC0415
    cfg = build_model_config(vocab_size=256)
    assert isinstance(cfg, GPTConfig)
    assert cfg.n_embd % HEAD_DIM == 0, f"n_embd {cfg.n_embd} not a multiple of HEAD_DIM {HEAD_DIM}"
    assert cfg.n_embd % cfg.n_head == 0, "n_embd not divisible by n_head"
    assert cfg.n_head % cfg.n_kv_head == 0, "n_head not divisible by n_kv_head"
    assert cfg.sequence_len > 0
    assert cfg.vocab_size > 0


@pytest.mark.unit
@pytest.mark.cpu
def test_gpt_forward_shape():
    """A tiny GPT on CPU produces output tensors of the correct shape.

    Input shape:  (B, T) integer token ids
    Output shape: (B, T, vocab_size) logits when targets=None
    """
    import torch
    from models.nanochat import GPT  # noqa: PLC0415
    cfg = _tiny_config()
    model = GPT(cfg)
    model.train(False)  # inference mode, no dropout

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits = model(idx)

    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected logits shape {(B, T, cfg.vocab_size)}, got {logits.shape}"
    )


@pytest.mark.unit
@pytest.mark.cpu
def test_gpt_loss_finite():
    """Forward pass with targets returns a finite scalar cross-entropy loss."""
    import torch
    from models.nanochat import GPT  # noqa: PLC0415
    cfg = _tiny_config()
    model = GPT(cfg)
    model.train(False)  # inference mode

    B, T = 2, 16
    idx     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        loss = model(idx, targets=targets)

    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, "Loss should be positive"


@pytest.mark.unit
@pytest.mark.cpu
def test_optimizer_creation():
    """setup_optimizer() creates a MuonAdamW with the correct param group kinds.

    Expected groups:
    - 'adamw' kind for embeddings, unembedding, value embeddings, scalars
    - 'muon' kind for 2-D matrix parameters (one group per unique shape)
    """
    import torch
    from models.nanochat import GPT, MuonAdamW  # noqa: PLC0415
    cfg = _tiny_config()
    model = GPT(cfg)

    optimizer = model.setup_optimizer()

    assert isinstance(optimizer, MuonAdamW)
    kinds = {g["kind"] for g in optimizer.param_groups}
    assert "adamw" in kinds, "Expected at least one 'adamw' param group"
    assert "muon" in kinds, "Expected at least one 'muon' param group"

    # Every param group must have a positive learning rate
    for g in optimizer.param_groups:
        assert "lr" in g and g["lr"] > 0, f"Param group missing positive lr: {g}"


@pytest.mark.unit
@pytest.mark.cpu
def test_one_training_step():
    """A single forward + backward + optimizer.step() completes without error.

    Checks:
    - Loss is finite before and after the step
    - At least one parameter gradient is non-None after backward
    - optimizer.step() runs without raising
    """
    import torch
    from models.nanochat import GPT  # noqa: PLC0415
    cfg = _tiny_config()
    model = GPT(cfg)
    model.train(True)  # training mode

    optimizer = model.setup_optimizer()

    B, T = 2, 16
    idx     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    optimizer.zero_grad()
    loss = model(idx, targets=targets)
    assert torch.isfinite(loss), f"Pre-step loss not finite: {loss.item()}"

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward()"

    optimizer.step()

    # Second forward to confirm parameters moved
    with torch.no_grad():
        loss2 = model(idx, targets=targets)
    assert torch.isfinite(loss2), f"Post-step loss not finite: {loss2.item()}"
