# Method Pack: Autoresearch Optimization and Architecture Catalog

This document is a reference pack of valid optimization techniques, architectural modifications, and hyperparameter tuning configurations for the autoresearch `train.py` model. 

---

## 1. MLP Activation and Gating Patterns

The baseline transformer in `train.py` uses a squared ReLU activation in the MLP block:
$$MLP_{ReLU^2}(x) = (ReLU(x W_{fc}))^2 W_{proj}$$
We can replace this with modern gated activation functions (GLUs) such as SwiGLU or GeGLU.

### 1.1. SwiGLU MLP
SwiGLU uses the Swish (SiLU) activation function as a gate:
$$MLP_{SwiGLU}(x) = (SiLU(x W_{gate}) \cdot (x W_{val})) W_{proj}$$
To maintain a similar parameter count to the baseline ReLU^2 (which has intermediate dimension $4 \times d_{model}$), the intermediate dimension of the GLU projections should be scaled by $\approx 2/3$. The typical hidden dimension is:
$$d_{ff} = \text{int}\left(\frac{8}{3} \times d_{model}\right)$$
We round this to the nearest multiple of 128 or 64 to ensure optimal matrix multiplication layout alignment.

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Parameter-matched hidden dimension (scaled by 2/3 and aligned to 64)
        hidden_dim = int(8 / 3 * config.n_embd)
        hidden_dim = 64 * ((hidden_dim + 32) // 64)
        
        # Combined projection for gate and value to allow fused operations and Muon optimization
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # Chunk into gate (x1) and value (x2)
        x1, x2 = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.c_proj(x)
        return x
```

### 1.2. GeGLU MLP
GeGLU replaces SiLU with GELU in the gating path:
$$MLP_{GeGLU}(x) = (GELU(x W_{gate}) \cdot (x W_{val})) W_{proj}$$

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Parameter-matched hidden dimension
        hidden_dim = int(8 / 3 * config.n_embd)
        hidden_dim = 64 * ((hidden_dim + 32) // 64)
        
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x1, x2 = self.c_fc(x).chunk(2, dim=-1)
        x = F.gelu(x1) * x2
        x = self.c_proj(x)
        return x
```

### 1.3. MLP Bounds and Invariants
* **Hidden Dimension ($d_{ff}$)**: $2 \times d_{model} \le d_{ff} \le 6 \times d_{model}$. If using GLUs, the input projection creates $2 \times d_{ff}$ channels.
* **Initialization**: The input linear layer `c_fc` should be initialized with uniform bounds $s = \sqrt{3 / d_{model}}$, and `c_proj` should be initialized to zero to allow residual identity startup.

---

## 2. Attention Mechanisms

The baseline uses Grouped-Query/Multi-Query Attention via configurable head counts:
* **Multi-Head Attention (MHA)**: `n_kv_head` = `n_head`
* **Grouped-Query Attention (GQA)**: $1 <$ `n_kv_head` $<$ `n_head` (where `n_head` is a multiple of `n_kv_head`)
* **Multi-Query Attention (MQA)**: `n_kv_head` = $1$

The fallback PyTorch `scaled_dot_product_attention` and `FlashAttention` implementations in `train.py` support GQA and MQA natively.

### 2.1. Configuration Templates
Adjusting head configuration is achieved directly in `GPTConfig` or `train.py` config section:

```python
# Grouped-Query Attention (GQA) Configuration
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 2  # 3x query heads per KV head (6 % 2 == 0)
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

```python
# Multi-Query Attention (MQA) Configuration
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 1  # 1 KV head shared across all query heads
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

### 2.2. Attention Invariants
* **Divisibility**: `n_head` must be strictly divisible by `n_kv_head`.
* **Head Dimension**: `head_dim` = `n_embd // n_head` (must be an integer, typically 64 or 128).
* **Rotary Embeddings**: `apply_rotary_emb` uses broadcasting and is compatible with arbitrary `n_kv_head` counts without modifications.

---

## 3. Sliding Window Attention Patterns

The sliding window pattern determines which layers use full context and which use local (half) context. 
* **`S` (Short)**: `sequence_len // 2` window size.
* **`L` (Long)**: `sequence_len` window size.

```python
# Examples of Sliding Window Configuration
WINDOW_PATTERN = "SSSL" # Alternating: 3 short, 1 long (baseline)
WINDOW_PATTERN = "SLSL" # Alternating: 1 short, 1 long
WINDOW_PATTERN = "LLLL" # All layers use full context
WINDOW_PATTERN = "SSSS" # Mostly short context (last layer is always L)
```

---

## 4. Normalization and Regularization

### 4.1. Functional LayerNorm vs RMSNorm
The baseline uses functional parameter-free RMSNorm:
```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```
This can be swapped with functional parameter-free LayerNorm:
```python
def norm(x):
    return F.layer_norm(x, (x.size(-1),))
```

### 4.2. Learnable RMSNorm (with scale parameter)
If learnable scale parameters are desired, they must be registered as modules in the architecture (to ensure correct parameter grouping and state dict checkpointing):

```python
class LearnableRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight
```

### 4.3. Dropout Regularization
For smaller architectures or longer runs, Dropout can prevent overfitting:

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.attn_dropout = nn.Dropout(0.1)
        self.mlp_dropout = nn.Dropout(0.1)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn_dropout(self.attn(norm(x), ve, cos_sin, window_size))
        x = x + self.mlp_dropout(self.mlp(norm(x)))
        return x
```

---

## 5. Optimizer & Hyperparameter Ranges

The model uses the `MuonAdamW` optimizer, which applies Muon (orthogonal update) for 2D weight matrices (transformer weights) and AdamW for 1D parameters (scalars, embeddings, biases).

| Hyperparameter | Description | Baseline | Lower Bound | Upper Bound |
| :--- | :--- | :--- | :--- | :--- |
| `MATRIX_LR` | Muon LR for 2D matrices | 0.04 | 0.005 | 0.15 |
| `EMBEDDING_LR` | Adam LR for Token Embeddings | 0.6 | 0.05 | 1.5 |
| `UNEMBEDDING_LR`| Adam LR for LM Output Head | 0.004 | 0.0005 | 0.02 |
| `SCALAR_LR` | Adam LR for Res / x0 scalars | 0.5 | 0.05 | 2.0 |
| `WEIGHT_DECAY` | Muon weight decay | 0.2 | 0.0 | 0.5 |
| `WARMUP_RATIO` | Warmup steps ratio | 0.0 | 0.0 | 0.3 |
| `WARMDOWN_RATIO`| Warmdown steps ratio | 0.5 | 0.1 | 0.8 |
| `FINAL_LR_FRAC` | LR floor at the end of cooldown | 0.0 | 0.0 | 0.3 |

### 5.1. Learning Rate Schedules
The learning rates scale automatically with $d_{model}$ to ensure stable convergence:
$$\text{LR}_{scaled} = \text{LR} \times \left(\frac{d_{model}}{768}\right)^{-0.5}$$

---

## 6. Verification and Safety Invariants

1. **Memory / OOM Safety**:
   * If running on macOS (MPS), lower the `DEVICE_BATCH_SIZE` to `16` or `32` (from the default `128`) to prevent `MPS backend out of memory`.
   * Ensure `TOTAL_BATCH_SIZE` is a multiple of `DEVICE_BATCH_SIZE * MAX_SEQ_LEN` so that `grad_accum_steps` is an integer.
2. **Newton-Schulz iteration steps (`ns_steps`)**:
   * Muon step uses Newton-Schulz iteration (default `ns_steps = 5`). Lowering to `3` increases speed slightly but may degrade convergence; raising to `7` increases numerical stability at high dimensionality.
3. **Compilation Safety**:
   * The model uses `@torch.compile(dynamic=False, fullgraph=True)` on optimizer step functions. Modifying optimizer logic must preserve compile-friendly operations (e.g. no control flow on tensor shapes).
