"""
Autoresearch pretraining script — Apple Silicon / MLX edition.
Single-file, native MLX implementation. No CUDA required.

Architecture mirrors train.py exactly:
  • GQA causal self-attention with RoPE + QK-norm
  • Value residual embeddings on alternating layers  (ResFormer)
  • Sliding-window attention pattern               (SSSL)
  • Squared ReLU MLP
  • Muon (2-D transformer matrices) + AdamW (embeddings / scalars)

Usage:
    uv run train_mlx.py
"""

import gc
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu        # tree_flatten / tree_unflatten / tree_map
import numpy as np

from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    make_dataloader_mlx,
    get_token_bytes_np,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array) -> mx.array:
    """RMS normalisation — no learned scale."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE to x of shape (B, T, H, head_dim)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return mx.concatenate([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], axis=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len:   int = 512
    vocab_size:     int = 32768
    n_layer:        int = 6
    n_head:         int = 4
    n_kv_head:      int = 4
    n_embd:         int = 256
    window_pattern: str = "SSSL"


def _has_ve(layer_idx: int, n_layer: int) -> bool:
    """True if layer should carry a Value Embedding (alternating, last always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def _compute_window_sizes(config: GPTConfig) -> list[int]:
    pattern  = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern)
    long_w   = config.sequence_len
    short_w  = long_w // 2
    char_map = {"L": long_w, "S": short_w}
    sizes    = [char_map[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    sizes[-1] = long_w   # last layer is always full context
    return sizes


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        hd = config.n_embd // config.n_head
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = hd
        self.n_rep     = config.n_head // config.n_kv_head   # GQA expansion

        self.c_q    = nn.Linear(config.n_embd, config.n_head    * hd, bias=False)
        self.c_k    = nn.Linear(config.n_embd, config.n_kv_head * hd, bias=False)
        self.c_v    = nn.Linear(config.n_embd, config.n_kv_head * hd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,         bias=False)

        self.ve_gate_channels = 32
        if _has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(self.ve_gate_channels, config.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def __call__(
        self,
        x:           mx.array,          # (B, T, C)
        ve:          Optional[mx.array], # (B, T, kv_dim) or None
        cos:         mx.array,           # (1, T, 1, head_dim/2)
        sin:         mx.array,           # (1, T, 1, head_dim/2)
        window_size: int,
    ) -> mx.array:
        B, T, _ = x.shape

        q = self.c_q(x).reshape(B, T, self.n_head,    self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): input-gated mixing with value embeddings
        if ve is not None and self.ve_gate is not None:
            ve_r = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2.0 * mx.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels])
            )   # (B, T, n_kv_head)
            v = v + gate[:, :, :, None] * ve_r

        # RoPE + QK-norm
        q = apply_rotary(q, cos[:, :T], sin[:, :T])
        k = apply_rotary(k, cos[:, :T], sin[:, :T])
        q = rms_norm(q)
        k = rms_norm(k)

        # GQA: repeat k/v to match query head count
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=2)
            v = mx.repeat(v, self.n_rep, axis=2)

        # (B, T, H, D) → (B, H, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5

        # Build causal + optional sliding-window additive mask
        i = mx.arange(T)[:, None]    # (T, 1)
        j = mx.arange(T)[None, :]    # (1, T)
        causal = j <= i
        if window_size < T:
            causal = causal & ((i - j) < window_size)
        attn_bias = mx.where(
            causal[None, None, :, :],
            mx.zeros((1, 1, T, T)),
            mx.full((1, 1, T, T), float("-inf")),
        )

        scores = (q @ k.transpose(0, 1, 3, 2)) * scale + attn_bias  # (B,H,T,T)
        attn   = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(mx.square(nn.relu(self.c_fc(x))))   # squared ReLU


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def __call__(self, x, ve, cos, sin, window_size):
        x = x + self.attn(rms_norm(x), ve, cos, sin, window_size)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config       = config
        self.window_sizes = _compute_window_sizes(config)

        hd   = config.n_embd // config.n_head
        kv_d = config.n_kv_head * hd

        self.wte     = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks  = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Per-layer learned residual scalars (trained via AdamW)
        self.resid_lambdas = mx.ones(config.n_layer)
        self.x0_lambdas    = mx.zeros(config.n_layer)

        # Value embeddings on alternating layers (trained via AdamW)
        self.value_embeds: dict[str, nn.Embedding] = {
            str(i): nn.Embedding(config.vocab_size, kv_d)
            for i in range(config.n_layer)
            if _has_ve(i, config.n_layer)
        }

        # RoPE tables: stored as NUMPY arrays so MLX does NOT include them in
        # trainable_parameters() (MLX only tracks mx.array module attributes).
        rotary_len = config.sequence_len * 2
        half       = hd // 2
        inv_freq   = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
        t          = np.arange(rotary_len, dtype=np.float32)
        freqs      = np.outer(t, inv_freq)
        # shape: (1, rotary_len, 1, half)
        self._cos_np = np.cos(freqs).astype(np.float16)[None, :, None, :]
        self._sin_np = np.sin(freqs).astype(np.float16)[None, :, None, :]

    def __call__(
        self,
        idx:     mx.array,           # (B, T)  int32
        targets: Optional[mx.array] = None,
    ) -> mx.array:
        B, T = idx.shape

        # Convert RoPE tables to MLX arrays for this forward pass
        cos = mx.array(self._cos_np[:, :T])
        sin = mx.array(self._sin_np[:, :T])

        x  = rms_norm(self.wte(idx))
        x0 = x

        for i, block in enumerate(self.blocks):
            x  = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x  = block(x, ve, cos, sin, self.window_sizes[i])

        x = rms_norm(x)

        # Logit soft-cap (tanh at ±15) to prevent blow-up
        softcap = 15.0
        logits  = softcap * mx.tanh(self.lm_head(x).astype(mx.float32) / softcap)

        if targets is not None:
            B_, T_, V = logits.shape
            return nn.losses.cross_entropy(
                logits.reshape(B_ * T_, V),
                targets.reshape(B_ * T_),
                reduction="mean",
            )
        return logits

    def estimate_flops(self) -> float:
        """Rough FLOPs/token estimate (6N approximation + attention term)."""
        flat    = mu.tree_flatten(self.trainable_parameters())
        nparams = sum(v.size for _, v in flat if isinstance(v, mx.array))
        wte_n   = self.wte.weight.size
        ve_n    = sum(e.weight.size for e in self.value_embeds.values())
        h, d    = self.config.n_head, self.config.n_embd // self.config.n_head
        t       = self.config.sequence_len
        attn_f  = sum(12 * h * d * min(w, t) for w in self.window_sizes)
        return 6 * (nparams - wte_n - ve_n) + attn_f


# ---------------------------------------------------------------------------
# Newton–Schulz orthogonalisation  (Polar Express, runs on CPU / numpy)
# ---------------------------------------------------------------------------

_POLAR_COEFFS = [
    ( 8.156554524902461,  -22.48329292557795,   15.878769915207462),
    ( 4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    ( 3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    ( 3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    ( 2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
]


def _ns_ortho(g: np.ndarray, steps: int = 5) -> np.ndarray:
    """Return the orthogonalised gradient (approximate polar factor U)."""
    X = g.astype(np.float32)
    X /= np.linalg.norm(X) * 1.02 + 1e-6
    for a, b, c in _POLAR_COEFFS[:steps]:
        if X.shape[0] > X.shape[1]:
            A  = X.T @ X
            B_ = b * A + c * (A @ A)
            X  = a * X + X @ B_
        else:
            A  = X @ X.T
            B_ = b * A + c * (A @ A)
            X  = a * X + B_ @ X
    return X


# ---------------------------------------------------------------------------
# Muon + AdamW hybrid optimizer
# ---------------------------------------------------------------------------
#
# Design:
#   • Parameters are identified by their dot-separated path in the model tree
#     (from mlx.utils.tree_flatten).
#   • Optimizer state is stored as numpy arrays on CPU, keyed by path string.
#   • After computing updates, we reconstruct the full parameter tree and call
#     model.update() — the correct MLX idiom for writing back parameters.
#   • Muon is applied to all 2-D weight matrices inside transformer blocks.
#     AdamW is applied to everything else (embeddings, scalars, lm_head).
# ---------------------------------------------------------------------------

# Path substrings that identify Muon parameters (2-D matrices in blocks)
_MUON_SUBSTRINGS = ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj",
                    "mlp.c_fc", "mlp.c_proj", "attn.ve_gate")


def _is_muon_param(path: str, arr: np.ndarray) -> bool:
    return arr.ndim == 2 and any(s in path for s in _MUON_SUBSTRINGS)


class MuonAdamW:
    """
    Hybrid optimizer:
      • Muon  — Newton–Schulz orthogonalisation + NorMuon variance reduction
                for all 2-D matrices inside transformer blocks.
      • AdamW — standard AdamW for embeddings, lm_head, per-layer scalars.

    All moment buffers live on CPU (numpy). Parameter writes use model.update().
    """

    def __init__(self, lr_config: dict):
        """
        lr_config keys:
            embedding_lr, unembedding_lr, matrix_lr, scalar_lr,
            adam_betas, eps, muon_wd
        """
        self._cfg   = lr_config
        self._state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # AdamW update for a single parameter
    # ------------------------------------------------------------------
    def _adamw(
        self,
        path: str,
        p:    np.ndarray,
        g:    np.ndarray,
        lr:   float,
        betas: tuple[float, float] = (0.8, 0.95),
    ) -> np.ndarray:
        b1, b2 = betas
        eps = self._cfg.get("eps", 1e-10)
        wd  = self._cfg.get("weight_decay_adamw", 0.0)

        s = self._state.setdefault(path, {
            "step": 0,
            "exp_avg":    np.zeros_like(p),
            "exp_avg_sq": np.zeros_like(p),
        })
        s["step"] += 1
        t_ = s["step"]

        s["exp_avg"]    = b1 * s["exp_avg"]    + (1 - b1) * g
        s["exp_avg_sq"] = b2 * s["exp_avg_sq"] + (1 - b2) * (g * g)

        bc1   = 1.0 - b1 ** t_
        bc2   = 1.0 - b2 ** t_
        denom = np.sqrt(s["exp_avg_sq"] / bc2) + eps
        step_ = lr / bc1

        return p * (1.0 - lr * wd) - step_ * s["exp_avg"] / denom

    # ------------------------------------------------------------------
    # Muon update for a single 2-D matrix parameter
    # ------------------------------------------------------------------
    def _muon(
        self,
        path:     str,
        p:        np.ndarray,
        g:        np.ndarray,
        lr:       float,
        momentum: float = 0.95,
        wd:       float = 0.2,
        beta2:    float = 0.95,
        ns_steps: int   = 5,
    ) -> np.ndarray:
        rows, cols = p.shape
        red_ax = -1 if rows >= cols else -2

        # Initialise buffers on first call
        if path not in self._state:
            v_shape = (rows, 1) if rows >= cols else (1, cols)
            self._state[path] = {
                "mom_buf": np.zeros_like(p),
                "v_buf":   np.zeros(v_shape, dtype=np.float32),
            }
        s = self._state[path]

        # Nesterov momentum
        s["mom_buf"] = (1 - momentum) * g + momentum * s["mom_buf"]
        g_ = (1 - momentum) * g + momentum * s["mom_buf"]   # Nesterov blend

        # Polar-Express orthogonalisation
        g_ = _ns_ortho(g_, ns_steps)

        # NorMuon variance reduction
        v_mean = np.mean(g_ ** 2, axis=red_ax, keepdims=True)
        s["v_buf"] = (1 - beta2) * v_mean + beta2 * s["v_buf"]
        step_size = 1.0 / np.sqrt(np.maximum(s["v_buf"], 1e-10))

        raw_nrm = np.sqrt(np.sum(v_mean) * g_.shape[red_ax])
        new_sq  = np.sum(v_mean * step_size ** 2 * g_.shape[red_ax])
        new_nrm = np.sqrt(max(float(new_sq), 1e-10))
        g_ = g_ * step_size * (raw_nrm / new_nrm)

        # Fan-out LR correction (√(max/min dimension))
        eff_lr = lr * max(1.0, (rows / cols) ** 0.5)

        # Cautious weight decay + parameter update
        mask = ((g_ * p) >= 0).astype(np.float32)
        return p - eff_lr * g_ - eff_lr * wd * p * mask

    # ------------------------------------------------------------------
    # Public step: takes flat param/grad lists, writes back via model.update
    # ------------------------------------------------------------------
    def step(
        self,
        model:        "GPT",
        grad_tree:    dict,
        lr_scale:     float = 1.0,
        muon_momentum: float = 0.95,
        muon_wd:      float = 0.2,
    ) -> None:
        flat_p = mu.tree_flatten(model.trainable_parameters())
        flat_g = mu.tree_flatten(grad_tree)
        assert len(flat_p) == len(flat_g), (
            f"Param/grad tree size mismatch: {len(flat_p)} vs {len(flat_g)}"
        )

        cfg = self._cfg

        updates = []
        for (path, p_mx), (gpath, g_mx) in zip(flat_p, flat_g):
            p_np = np.array(p_mx, copy=False).astype(np.float32)
            g_np = np.array(g_mx, copy=False).astype(np.float32)

            if _is_muon_param(path, p_np):
                new_p = self._muon(
                    path, p_np, g_np,
                    lr       = cfg["matrix_lr"] * lr_scale,
                    momentum = muon_momentum,
                    wd       = muon_wd,
                    beta2    = 0.95,
                    ns_steps = 5,
                )
            else:
                # Determine AdamW LR by path matching
                if "wte" in path or "value_embeds" in path:
                    lr = cfg["embedding_lr"] * lr_scale
                    betas = cfg["adam_betas"]
                elif "lm_head" in path:
                    lr = cfg["unembedding_lr"] * lr_scale
                    betas = cfg["adam_betas"]
                elif "x0_lambdas" in path:
                    lr = cfg["scalar_lr"] * lr_scale
                    betas = (0.96, 0.95)
                else:  # resid_lambdas and any other scalars
                    lr = cfg["scalar_lr"] * 0.01 * lr_scale
                    betas = cfg["adam_betas"]

                new_p = self._adamw(path, p_np, g_np, lr, betas=betas)

            updates.append((path, mx.array(new_p.reshape(p_np.shape))))

        # Write all updated parameters back to the model in one call
        model.update(mu.tree_unflatten(updates))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bpb(model: GPT, tokenizer: Tokenizer, batch_size: int) -> float:
    """
    Compute bits-per-byte (BPB) on the validation split.
    Lower is better; vocab-size-independent metric.
    """
    from prepare import EVAL_TOKENS

    token_bytes = get_token_bytes_np()                            # (vocab,) int32
    val_loader  = make_dataloader_mlx(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps       = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))

    total_nats  = 0.0
    total_bytes = 0

    for _ in range(steps):
        x_np, y_np, _ = next(val_loader)
        x = mx.array(x_np)

        logits = model(x)                      # (B, T, V) float32
        B_, T_, V = logits.shape

        loss_flat = nn.losses.cross_entropy(
            logits.reshape(B_ * T_, V),
            mx.array(y_np.reshape(B_ * T_)),
            reduction="none",
        )
        mx.eval(loss_flat)

        y_flat = y_np.reshape(-1)
        nb     = token_bytes[y_flat]
        mask   = nb > 0
        loss_np = np.array(loss_flat)

        total_nats  += float((loss_np * mask).sum())
        total_bytes += int(nb.sum())

    return total_nats / (math.log(2) * max(total_bytes, 1))


# ---------------------------------------------------------------------------
# Hyperparameters
# (Edit freely — same style as train.py; tuned for a Mac with ~16–96 GB RAM)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO   = 32          # model_dim = DEPTH * ASPECT_RATIO
HEAD_DIM       = 64          # target head dimension
WINDOW_PATTERN = "SSSL"      # L = full context, S = half-context sliding window

# Optimisation
TOTAL_BATCH_SIZE  = 2**14    # ~16K tokens/step  (Mac-friendly; H100 uses 2**19)
EMBEDDING_LR      = 0.2
UNEMBEDDING_LR    = 0.004
MATRIX_LR         = 0.02
SCALAR_LR         = 0.5
WEIGHT_DECAY      = 0.2
ADAM_BETAS        = (0.8, 0.95)
WARMUP_RATIO      = 0.05
WARMDOWN_RATIO    = 0.4
FINAL_LR_FRAC     = 0.0

# Model size (primary knob — lower = faster per-step on smaller Macs)
DEPTH             = 6
DEVICE_BATCH_SIZE = 8        # micro-batch size; reduce if you see memory pressure

# Apple Silicon fp16 peak TFLOPS reference (adjust to your chip)
# M1 Pro ~11 TFLOPS | M2 Max ~13 TFLOPS | M3 Ultra ~60 TFLOPS | M4 Max ~68 TFLOPS
APPLE_SILICON_PEAK_FLOPS = 13e12

# ---------------------------------------------------------------------------
# LR / momentum schedules
# ---------------------------------------------------------------------------

def lr_multiplier(progress: float) -> float:
    """Trapezoidal schedule: warmup → flat → warmdown."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        t = (1.0 - progress) / WARMDOWN_RATIO
        return t + (1 - t) * FINAL_LR_FRAC


def muon_momentum_schedule(step: int) -> float:
    """Linearly ramp Muon momentum from 0.85 → 0.95 over 300 steps."""
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def muon_weight_decay_schedule(progress: float) -> float:
    """Decay Muon weight decay to zero at end of training."""
    return WEIGHT_DECAY * (1.0 - progress)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    mx.random.seed(42)

    # ---- Tokenizer --------------------------------------------------------
    tokenizer  = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # ---- Model ------------------------------------------------------------
    def build_config(depth: int) -> GPTConfig:
        base_dim  = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        num_heads = max(1, model_dim // HEAD_DIM)
        return GPTConfig(
            sequence_len   = MAX_SEQ_LEN,
            vocab_size     = vocab_size,
            n_layer        = depth,
            n_head         = num_heads,
            n_kv_head      = num_heads,
            n_embd         = model_dim,
            window_pattern = WINDOW_PATTERN,
        )

    config = build_config(DEPTH)
    print(f"Model config: {asdict(config)}")

    model = GPT(config)

    # Trigger MLX lazy initialisation — a tiny dummy forward pass
    _ = model(mx.zeros((1, 2), dtype=mx.int32))
    mx.eval(model.parameters())

    flat_p  = mu.tree_flatten(model.trainable_parameters())
    nparams = sum(v.size for _, v in flat_p if isinstance(v, mx.array))
    print(f"Parameters: {nparams / 1e6:.2f}M")

    flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs/token: {flops_per_token:.2e}")

    # ---- Optimizer --------------------------------------------------------
    tokens_per_micro = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_micro == 0, (
        f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by "
        f"DEVICE_BATCH_SIZE * MAX_SEQ_LEN ({tokens_per_micro})"
    )
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_micro

    dmodel_scale = (config.n_embd / 768) ** -0.5   # μP-style LR scaling

    lr_config = dict(
        embedding_lr    = EMBEDDING_LR    * dmodel_scale,
        unembedding_lr  = UNEMBEDDING_LR  * dmodel_scale,
        matrix_lr       = MATRIX_LR,
        scalar_lr       = SCALAR_LR,
        adam_betas       = ADAM_BETAS,
        eps             = 1e-10,
        weight_decay_adamw = 0.0,
    )
    optimizer = MuonAdamW(lr_config)

    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Time budget: {TIME_BUDGET}s")

    # ---- Dataloader -------------------------------------------------------
    train_loader = make_dataloader_mlx(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

    # ---- Forward / backward function (compiled) ---------------------------
    # nn.value_and_grad returns (loss_scalar, grad_tree) where grad_tree
    # matches the structure of model.trainable_parameters().
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m(x, y))

    # ---- Training loop ----------------------------------------------------
    t_start_training   = time.time()
    total_training_time = 0.0
    smooth_loss        = 0.0
    step               = 0

    while True:
        t0 = time.time()

        # -- Gradient accumulation over micro-steps -------------------------
        acc_grads   = None
        accum_loss  = 0.0

        for _micro in range(grad_accum_steps):
            x_np, y_np, epoch = next(train_loader)
            x = mx.array(x_np)
            y = mx.array(y_np)

            loss, grads = loss_and_grad(model, x, y)
            # Materialise before accumulating to keep the compute graph small
            mx.eval(loss, grads)

            accum_loss += float(loss)

            if acc_grads is None:
                acc_grads = grads
            else:
                # Add gradient trees element-wise using mlx.utils.tree_map
                acc_grads = mu.tree_map(lambda a, b: a + b, acc_grads, grads)

        # Average over micro-steps
        if grad_accum_steps > 1:
            acc_grads = mu.tree_map(lambda g: g / grad_accum_steps, acc_grads)

        train_loss = accum_loss / grad_accum_steps

        # Fast-fail guard
        if train_loss > 100:
            print("\nFAIL — loss exploded, aborting.")
            return

        # -- Schedules ------------------------------------------------------
        progress   = min(total_training_time / TIME_BUDGET, 1.0)
        lrm        = lr_multiplier(progress)
        mom        = muon_momentum_schedule(step)
        wd         = muon_weight_decay_schedule(progress)

        # -- Optimizer step -------------------------------------------------
        optimizer.step(model, acc_grads,
                       lr_scale=lrm, muon_momentum=mom, muon_wd=wd)
        mx.eval(model.parameters())

        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # -- Logging --------------------------------------------------------
        ema          = 0.9
        smooth_loss  = ema * smooth_loss + (1 - ema) * train_loss
        debiased     = smooth_loss / (1 - ema ** (step + 1))
        pct_done     = 100.0 * progress
        tok_per_sec  = int(TOTAL_BATCH_SIZE / max(dt, 1e-9))
        mfu          = (100.0 * flops_per_token * TOTAL_BATCH_SIZE
                        / max(dt, 1e-9) / APPLE_SILICON_PEAK_FLOPS)
        remaining    = max(0.0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased:.6f} | lrm: {lrm:.2f} | "
            f"dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu:.1f}% | epoch: {epoch} | "
            f"remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

        # Python GC: collect once at start then freeze to avoid GC stalls
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()   # newline after \r log

    total_tokens = step * TOTAL_BATCH_SIZE

    # ---- Evaluation -------------------------------------------------------
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    # ---- Summary ----------------------------------------------------------
    t_end       = time.time()
    startup_sec = t_start_training - t_start
    steady_mfu  = (
        100.0 * flops_per_token * TOTAL_BATCH_SIZE * (step - 10)
        / total_training_time / APPLE_SILICON_PEAK_FLOPS
        if total_training_time > 0 else 0.0
    )

    # MLX unified memory report (available on Metal backend)
    try:
        peak_mem_mb = mx.metal.get_peak_memory() / 1024 / 1024
    except Exception:
        peak_mem_mb = float("nan")

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"startup_seconds:  {startup_sec:.1f}")
    print(f"peak_memory_mb:   {peak_mem_mb:.1f}")
    print(f"mfu_percent:      {steady_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {nparams / 1e6:.2f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()