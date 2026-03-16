# Adding a New Model

This guide walks through adding a new model architecture to autoresearch.

## Step 1: Create the Model File

Create `models/your_model.py`. Your model must implement the `TrainableModel` protocol defined in `models/base.py`:

```python
class TrainableModel(Protocol):
    def forward(self, idx, targets=None, reduction='mean') -> torch.Tensor: ...
    def setup_optimizer(self, **kwargs) -> torch.optim.Optimizer: ...
    def init_weights(self) -> None: ...
    def estimate_flops(self) -> int: ...
    def num_scaling_params(self) -> dict: ...
```

### Template

```python
"""
Your model description.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class YourModelConfig:
    sequence_len: int = 1024
    vocab_size: int = 32768
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class YourModel(nn.Module):
    def __init__(self, config: YourModelConfig):
        super().__init__()
        self.config = config
        # ... build layers ...

    def init_weights(self):
        """Initialize all parameters. Called after model is on device."""
        # ... weight initialization ...

    def forward(self, idx, targets=None, reduction='mean'):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices, or None for inference
            reduction: 'mean' or 'none' for per-token loss

        Returns:
            logits (B, T, vocab_size) if targets is None
            loss scalar (or per-token if reduction='none') if targets provided
        """
        # ... forward pass ...
        logits = ...

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits

    def setup_optimizer(self, **kwargs):
        """Create and return an optimizer for this model."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.1)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        return 6 * nparams  # rough approximation

    def num_scaling_params(self):
        """Return dict of parameter counts by category."""
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
```

### Key Requirements

1. **`forward()` must handle both logits and loss paths.** When `targets` is `None`, return logits `(B, T, V)`. When targets are provided, return a loss scalar (or per-token losses if `reduction='none'`).

2. **`setup_optimizer()` must set `initial_lr`.** The training loop uses `group["initial_lr"]` for LR scheduling. If using the Muon+AdamW optimizer (from nanochat), set `group['kind']` to `'muon'` or `'adamw'`. For standard optimizers, the training loop uses `group.get('kind')` which returns `None` safely.

3. **`init_weights()` is called after `.to(device)`.** Don't assume device in `__init__`.

4. **`estimate_flops()` is used for MFU calculation.** It should return total FLOPs per token for forward + backward. A rough approximation of `6 * nparams` works as a starting point.

## Step 2: Register the Model

In `models/__init__.py`, add your model to the registry:

```python
from models.your_model import YourModel, YourModelConfig

REGISTRY = {
    "nanochat": (NanochatGPT, NanochatConfig),
    "gpt2": (GPT2, GPT2Config),
    "your_model": (YourModel, YourModelConfig),  # add this
}
```

## Step 3: Add Tests

In `tests/test_model_registry.py`, add:

```python
@pytest.mark.unit
@pytest.mark.cpu
def test_your_model_forward_cpu():
    """Tiny YourModel forward pass on CPU."""
    import torch
    from models.your_model import YourModel, YourModelConfig

    cfg = YourModelConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=2, n_embd=64)
    model = YourModel(cfg)
    model.init_weights()

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    # Test logits path
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    # Test loss path
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        loss = model(idx, targets=targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

Also add a `test_create_your_model()` test for the registry.

## Step 4: Run Tests

```bash
uv sync --extra dev
uv run pytest tests/ -m unit -v
```

## Step 5: Train

```bash
AUTORESEARCH_MODEL=your_model uv run train.py
```

## Tips

- **Keep it simple.** The agent will modify `train.py` hyperparameters, not your model code. Make your model work well with a range of depths and embedding sizes.
- **Use SDPA for attention.** `F.scaled_dot_product_attention` works on all platforms. Only use FA3 if you specifically need sliding window support on CUDA.
- **Match the interface exactly.** `train.py` expects `forward(x, y)` to return a loss, `forward(x)` to return logits, and `forward(x, y, reduction='none')` to return per-token losses.
- **Test on CPU first.** All models should work on CPU for unit tests, even if they're intended for CUDA training.
