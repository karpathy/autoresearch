# Model Selection

Autoresearch supports multiple model architectures via a registry in `models/`. The default is `nanochat` (the original architecture from this repo). Select a model via environment variable:

```bash
AUTORESEARCH_MODEL=gpt2 uv run train.py
```

## Available Models

### nanochat (default)

The original architecture, cherry-picked from [nanochat](https://github.com/karpathy/nanochat). A modern GPT with several advanced features.

| Feature | Details |
|---|---|
| Normalization | RMSNorm |
| Positional encoding | RoPE (Rotary Position Embeddings) |
| Attention | Grouped-Query Attention (GQA), Flash Attention 3 on CUDA |
| Activation | ReluSquared (`relu(x)^2`) in MLP |
| Value embeddings | Per-layer value residual connections (ResFormer) |
| Residual scaling | Learned per-layer `resid_lambdas` + `x0_lambdas` |
| Logit processing | Softcapping (`15 * tanh(logits/15)`) |
| Sliding window | Alternating short/long windows (`SSSL` pattern) |
| Optimizer | MuonAdamW (Muon for 2D matrices, AdamW for embeddings/scalars) |

**Config (`GPTConfig`):**
```python
GPTConfig(
    sequence_len=2048,
    vocab_size=32768,
    n_layer=12,
    n_head=6,
    n_kv_head=6,      # GQA: can be < n_head
    n_embd=768,
    window_pattern="SSSL",
)
```

**When to use:** Default choice. Best results with 5-minute training budget on CUDA. The Muon optimizer and value embeddings give it an edge at small scale.

### gpt2

Standard GPT-2 architecture. Simpler, fewer parameters at the same depth, good baseline.

| Feature | Details |
|---|---|
| Normalization | LayerNorm |
| Positional encoding | Learned absolute position embeddings |
| Attention | Standard multi-head (no GQA) |
| Activation | GELU in MLP |
| Value embeddings | None |
| Residual scaling | None |
| Logit processing | None (raw logits) |
| Sliding window | No (full context always) |
| Optimizer | AdamW (standard, with weight decay separation) |

**Config (`GPT2Config`):**
```python
GPT2Config(
    sequence_len=1024,
    vocab_size=32768,
    n_layer=6,
    n_head=6,
    n_embd=384,
)
```

**When to use:** Good baseline for comparison. Useful when you want to isolate the effect of nanochat's architectural innovations. Also a simpler starting point for agent modifications.

## Using the Model Registry

From Python:

```python
from models import create_model, list_models

# List available models
print(list_models())  # ['nanochat', 'gpt2']

# Create a model with custom config
model, config = create_model("gpt2", vocab_size=8192, n_layer=4, n_embd=256)
```

From `train.py`, the model is selected via:
```python
MODEL_NAME = os.environ.get("AUTORESEARCH_MODEL", "nanochat")
```

For `nanochat`, `train.py` uses `build_model_config()` with its hyperparameters (`DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, etc.). For other models, it uses `create_model()` with `vocab_size`, `n_layer=DEPTH`, and `sequence_len=MAX_SEQ_LEN`.

## Adding a New Model

See [docs/adding-models.md](adding-models.md) for a step-by-step guide.

Quick summary:

1. Create `models/your_model.py` with a model class and config dataclass
2. Implement the `TrainableModel` protocol (see `models/base.py`):
   - `forward(idx, targets=None, reduction='mean')` — returns logits or loss
   - `setup_optimizer(**kwargs)` — returns an optimizer
   - `init_weights()` — initialize parameters
   - `estimate_flops()` — estimated FLOPs per token
   - `num_scaling_params()` — dict of parameter counts
3. Add to the registry in `models/__init__.py`
4. Add a test in `tests/test_model_registry.py`
