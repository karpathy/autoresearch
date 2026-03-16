"""
tests/test_model_registry.py — tests for models/ package registry and factory.

All tests run on CPU without requiring Flash Attention 3 or a GPU.
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
# Registry tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.cpu
def test_registry_lists_models():
    """list_models() returns the expected model names."""
    from models import list_models  # noqa: PLC0415
    names = list_models()
    assert "nanochat" in names, f"'nanochat' not in {names}"
    assert "gpt2" in names, f"'gpt2' not in {names}"


@pytest.mark.unit
@pytest.mark.cpu
def test_create_nanochat():
    """create_model('nanochat') returns a (model, config) pair."""
    from models import create_model  # noqa: PLC0415
    from models.nanochat import GPT, GPTConfig  # noqa: PLC0415
    model, config = create_model("nanochat", vocab_size=256, n_layer=2, n_embd=128,
                                 n_head=2, n_kv_head=2, sequence_len=64)
    assert isinstance(model, GPT)
    assert isinstance(config, GPTConfig)
    assert config.vocab_size == 256
    assert config.n_layer == 2


@pytest.mark.unit
@pytest.mark.cpu
def test_create_gpt2():
    """create_model('gpt2') returns a (model, config) pair."""
    from models import create_model  # noqa: PLC0415
    from models.gpt2 import GPT2, GPT2Config  # noqa: PLC0415
    model, config = create_model("gpt2", vocab_size=256, n_layer=2, n_embd=64,
                                 n_head=2, sequence_len=64)
    assert isinstance(model, GPT2)
    assert isinstance(config, GPT2Config)
    assert config.vocab_size == 256
    assert config.n_layer == 2


@pytest.mark.unit
@pytest.mark.cpu
def test_unknown_model_raises():
    """create_model('nonexistent') raises KeyError with informative message."""
    from models import create_model  # noqa: PLC0415
    with pytest.raises(KeyError, match="nonexistent"):
        create_model("nonexistent")


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.cpu
def test_nanochat_forward_cpu():
    """Tiny nanochat model forward pass on CPU returns correct shapes."""
    import torch  # noqa: PLC0415
    from models.nanochat import GPT, GPTConfig  # noqa: PLC0415
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="SL",
    )
    model = GPT(cfg)
    model.train(False)

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits = model(idx)

    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected {(B, T, cfg.vocab_size)}, got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    # Test loss path
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        loss = model(idx, targets=targets)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


@pytest.mark.unit
@pytest.mark.cpu
def test_gpt2_forward_cpu():
    """Tiny GPT-2 model forward pass on CPU returns correct shapes."""
    import torch  # noqa: PLC0415
    from models.gpt2 import GPT2, GPT2Config  # noqa: PLC0415
    cfg = GPT2Config(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = GPT2(cfg)
    model.init_weights()
    model.train(False)

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits = model(idx)

    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected {(B, T, cfg.vocab_size)}, got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    # Test loss path
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        loss = model(idx, targets=targets)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
