"""
Autoresearch pretraining script — Apple Silicon MLX backend.
Self-contained single-file. Adds MLX support for Apple Silicon Macs.
Usage: uv run train_mlx.py
Prerequisites: uv run prepare.py (downloads data + trains tokenizer)
"""

import os
import gc
import math
import time
import pickle
import re
import subprocess
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants (must match prepare.py — do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Tokenizer (loads tiktoken encoder saved by prepare.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled by prepare.py."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


# ---------------------------------------------------------------------------
# Data Loading (MLX-native, replaces CUDA dataloader from prepare.py)
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing (MLX-native).
    Every row starts with BOS. Documents packed using best-fit to minimize
    cropping. 100% utilization (no padding). Yields (inputs, targets, epoch)
    as mx.array on each call to next().
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate numpy buffer
    row_buffer = np.empty((B, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)),
                                       key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = mx.array(row_buffer[:, :-1])
        targets = mx.array(row_buffer[:, 1:])
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (MLX-native, replaces CUDA evaluate_bpb from prepare.py)
# ---------------------------------------------------------------------------

def get_token_byte_lengths(tokenizer):
    """
    Compute byte length for each token directly from tiktoken encoder.
    Returns an mx.array of shape (vocab_size,) with byte lengths.
    Special tokens get length 0 (excluded from BPB calculation).
    """
    enc = tokenizer.enc
    vocab_size = enc.n_vocab
    lengths = np.zeros(vocab_size, dtype=np.int32)
    for i in range(vocab_size):
        try:
            token_bytes = enc.decode_single_token_bytes(i)
            lengths[i] = len(token_bytes)
        except (KeyError, ValueError):
            lengths[i] = 0  # special token
    return mx.array(lengths)


def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_byte_lengths(tokenizer)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        nats = (loss_flat * mask).sum()
        byte_count = nbytes.sum()
        mx.eval(nats, byte_count)
        total_nats += nats.item()
        total_bytes += byte_count.item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Apple Silicon Hardware Detection
# ---------------------------------------------------------------------------

def get_apple_silicon_info():
    """Detect Apple Silicon chip and GPU cores for MFU calculation."""
    chip_name = "Apple Silicon"
    gpu_cores = 8  # conservative default
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            chip_name = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            m = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
            if m:
                gpu_cores = int(m.group(1))
    except Exception:
        pass
    return chip_name, gpu_cores


def estimate_peak_flops(chip_name, gpu_cores):
    """Estimate peak bf16 FLOPS for Apple Silicon."""
    m = re.search(r"(m[1-5])", chip_name.lower())
    gen = m.group(1) if m else "m4"
    flops_per_core = {
        "m1": 0.5e12, "m2": 0.55e12, "m3": 0.65e12, "m4": 0.7e12,
    }.get(gen, 0.65e12)
    return gpu_cores * flops_per_core


# ---------------------------------------------------------------------------
# Muon Optimizer — Newton-Schulz (Polar Express) Orthogonalization
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def newton_schulz_orthogonalize(X, ns_steps=5):
    """
    Polar express: approximate the orthogonal polar factor of X
    using Newton-Schulz iterations with precomputed optimal coefficients.

    Note: Uses float32 throughout. The original CUDA version uses bf16 because
    tensor cores give 2x speedup. On Apple Silicon, float32 is nearly as fast
    and avoids norm precision loss that causes divergence with bf16 reductions.
    """
    X = X.astype(mx.float32)
    norms = mx.sqrt((X * X).sum(axis=(-2, -1), keepdims=True))
    X = X / (norms * 1.02 + 1e-6)

    M, N = X.shape[-2], X.shape[-1]
    if M > N:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            Xt = mx.swapaxes(X, -2, -1)
            A = Xt @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            Xt = mx.swapaxes(X, -2, -1)
            A = X @ Xt
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X.astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# Muon + AdamW Combined Optimizer (MLX)
# ---------------------------------------------------------------------------

def _navigate_part(obj, part):
    """Navigate one level into a nested MLX model structure."""
    if part.isdigit():
        if isinstance(obj, dict):
            return obj[part]
        return obj[int(part)]
    return getattr(obj, part)


def _set_param_by_path(model, path, value):
    """Set a parameter in an MLX model by its dot-separated path."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = _navigate_part(obj, part)
    last = parts[-1]
    if last.isdigit():
        if isinstance(obj, dict):
            obj[last] = value
        else:
            obj[int(last)] = value
    else:
        setattr(obj, last, value)


class MuonAdamWMLX:
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others.
    Full MLX port of the original PyTorch MuonAdamW.

    param_groups: list of dicts, each with:
        kind: 'adamw' or 'muon'
        paths: list of parameter paths (dot-separated)
        lr: learning rate
        For adamw: betas, eps, weight_decay
        For muon: momentum, ns_steps, beta2, weight_decay
    """

    def __init__(self, param_groups):
        self.param_groups = param_groups
        self.state = {}
        self._step_count = 0
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    def _step_adamw(self, path, grad, param, group):
        """Standard AdamW update for a single parameter."""
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        wd = group['weight_decay']

        state = self.state.setdefault(path, {
            'm': mx.zeros_like(grad_f32),
            'v': mx.zeros_like(grad_f32),
            't': 0,
        })
        state['t'] += 1
        t = state['t']

        state['m'] = beta1 * state['m'] + (1 - beta1) * grad_f32
        state['v'] = beta2 * state['v'] + (1 - beta2) * (grad_f32 * grad_f32)

        bias1 = 1 - beta1 ** t
        bias2 = 1 - beta2 ** t
        denom = mx.sqrt(state['v'] / bias2) + eps
        step_size = lr / bias1

        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (state['m'] / denom)
        return param_f32.astype(param.dtype)

    def _step_muon(self, stacked_grads, stacked_params, group):
        """
        Muon update for a group of same-shape 2D parameters (stacked).
        Implements: Nesterov momentum -> Newton-Schulz -> NorMuon -> cautious WD.
        """
        momentum_val = group['momentum']
        lr = group['lr']
        wd = group['weight_decay']
        beta2 = group.get('beta2', 0.95)
        ns_steps = group.get('ns_steps', 5)
        group_id = group['_group_id']

        shape = stacked_grads.shape
        M, N = shape[-2], shape[-1]
        red_dim = -1 if M >= N else -2

        state = self.state.setdefault(group_id, {})

        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = mx.zeros_like(stacked_grads)
        if 'second_momentum_buffer' not in state:
            if M >= N:
                smb_shape = list(shape[:-1]) + [1]
            else:
                smb_shape = list(shape[:-2]) + [1, N]
            state['second_momentum_buffer'] = mx.zeros(smb_shape, dtype=mx.float32)

        # Nesterov momentum
        momentum = mx.array(momentum_val, dtype=stacked_grads.dtype)
        state['momentum_buffer'] = (momentum * state['momentum_buffer']
                                    + (1 - momentum) * stacked_grads)
        g = (1 - momentum) * stacked_grads + momentum * state['momentum_buffer']

        # Polar express orthogonalization
        g = newton_schulz_orthogonalize(g, ns_steps)

        # NorMuon variance reduction — all in float32
        g_f32 = g.astype(mx.float32)
        beta2_f32 = mx.array(beta2, dtype=mx.float32)
        v_mean = (g_f32 ** 2).mean(axis=red_dim, keepdims=True)
        red_dim_size = g.shape[red_dim]
        v_norm_sq = v_mean.sum(axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = mx.sqrt(v_norm_sq)

        smb_f32 = state['second_momentum_buffer']
        smb_f32 = smb_f32 + (1 - beta2_f32) * (v_mean - smb_f32)
        state['second_momentum_buffer'] = smb_f32

        step_size = mx.rsqrt(mx.maximum(smb_f32, mx.array(1e-10)))
        scaled_sq_sum = (v_mean * red_dim_size) * (step_size ** 2)
        v_norm_new = mx.sqrt(scaled_sq_sum.sum(axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, mx.array(1e-10)))
        g = (g_f32 * final_scale).astype(g.dtype)

        # Cautious weight decay + parameter update
        lr_val = mx.array(lr, dtype=g.dtype)
        wd_val = mx.array(wd, dtype=g.dtype)
        mask = (g * stacked_params) >= 0
        updated = stacked_params - lr_val * g - lr_val * wd_val * stacked_params * mask

        return updated

    def update(self, model, grads):
        """Apply one optimization step."""
        self._step_count += 1
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))

        # AdamW groups
        for group in self.param_groups:
            if group['kind'] != 'adamw':
                continue
            for path in group['paths']:
                if path not in flat_grads:
                    continue
                new_param = self._step_adamw(
                    path, flat_grads[path], flat_params[path], group)
                _set_param_by_path(model, path, new_param)

        # Muon groups (stacked by shape for batched operations)
        for group in self.param_groups:
            if group['kind'] != 'muon':
                continue
            paths = group['paths']
            if not paths:
                continue

            grads_list = [flat_grads[p] for p in paths if p in flat_grads]
            params_list = [flat_params[p] for p in paths if p in flat_grads]
            if not grads_list:
                continue

            stacked_g = mx.stack(grads_list)
            stacked_p = mx.stack(params_list)

            # Scale LR by aspect ratio (same as original)
            shape = params_list[0].shape
            group['lr'] = group['lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5

            updated = self._step_muon(stacked_g, stacked_p, group)

            active_paths = [p for p in paths if p in flat_grads]
            for i, path in enumerate(active_paths):
                _set_param_by_path(model, path, updated[i])

    def set_lr_multiplier(self, multiplier):
        """Scale all learning rates by multiplier (for warmup/cooldown)."""
        for group in self.param_groups:
            group['lr'] = group['initial_lr'] * multiplier


def build_param_groups(model, config_dict):
    """
    Build param_groups for MuonAdamWMLX from a model and configuration.

    config_dict should have:
        matrix_lr, embedding_lr, unembedding_lr, scalar_lr,
        adam_betas, weight_decay, model_dim
    """
    flat_params = dict(tree_flatten(model.parameters()))
    model_dim = config_dict['model_dim']
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    embedding_paths = []
    value_embed_paths = []
    lm_head_paths = []
    resid_paths = []
    x0_paths = []
    matrix_paths_by_shape = {}

    for path, param in flat_params.items():
        if 'wte' in path and 'weight' in path:
            embedding_paths.append(path)
        elif 'value_embeds' in path:
            value_embed_paths.append(path)
        elif 'lm_head' in path:
            lm_head_paths.append(path)
        elif 'resid_lambdas' in path:
            resid_paths.append(path)
        elif 'x0_lambdas' in path:
            x0_paths.append(path)
        elif 'blocks' in path and param.ndim == 2:
            shape = param.shape
            if shape not in matrix_paths_by_shape:
                matrix_paths_by_shape[shape] = []
            matrix_paths_by_shape[shape].append(path)
        elif param.ndim == 2:
            shape = param.shape
            if shape not in matrix_paths_by_shape:
                matrix_paths_by_shape[shape] = []
            matrix_paths_by_shape[shape].append(path)

    adam_betas = config_dict.get('adam_betas', (0.8, 0.95))
    param_groups = []

    if lm_head_paths:
        param_groups.append(dict(
            kind='adamw', paths=lm_head_paths,
            lr=config_dict['unembedding_lr'] * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if embedding_paths:
        param_groups.append(dict(
            kind='adamw', paths=embedding_paths,
            lr=config_dict['embedding_lr'] * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if value_embed_paths:
        param_groups.append(dict(
            kind='adamw', paths=value_embed_paths,
            lr=config_dict['embedding_lr'] * dmodel_lr_scale,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if resid_paths:
        param_groups.append(dict(
            kind='adamw', paths=resid_paths,
            lr=config_dict['scalar_lr'] * 0.01,
            betas=adam_betas, eps=1e-10, weight_decay=0.0,
        ))
    if x0_paths:
        param_groups.append(dict(
            kind='adamw', paths=x0_paths,
            lr=config_dict['scalar_lr'],
            betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0,
        ))

    for i, (shape, paths) in enumerate(sorted(matrix_paths_by_shape.items())):
        param_groups.append(dict(
            kind='muon', paths=paths,
            lr=config_dict['matrix_lr'],
            momentum=0.95, ns_steps=5, beta2=0.95,
            weight_decay=config_dict.get('weight_decay', 0.0),
            _group_id=f'muon_shape_{i}_{shape[0]}x{shape[1]}',
        ))

    for i, group in enumerate(param_groups):
        if '_group_id' not in group:
            group['_group_id'] = f'adamw_{i}'

    return param_groups


# ---------------------------------------------------------------------------
# GPT Model (MLX)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    """RMS norm."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding."""
    return layer_idx % 2 == (n_layer - 1) % 2


def create_causal_mask(seq_len, dtype=mx.float32):
    """Create standard causal attention mask."""
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype),
                    mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    """Create causal attention mask with sliding window."""
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype),
                    mx.array(0.0, dtype=dtype))


class RotaryEmbedding:
    """Precomputed rotary embeddings."""
    def __init__(self, head_dim, max_seq_len, base=10000):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        self.cos = freqs.cos().astype(mx.bfloat16)
        self.sin = freqs.sin().astype(mx.bfloat16)

    def apply(self, x, offset=0):
        """Apply rotary embeddings. x: (B, T, n_heads, head_dim)"""
        T = x.shape[1]
        cos = self.cos[offset:offset + T][None, :, None, :]
        sin = self.sin[offset:offset + T][None, :, None, :]
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return mx.concatenate([y1, y2], axis=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.has_ve = has_ve(layer_idx, config.n_layer)
        if self.has_ve:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve, rotary, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.has_ve:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = rotary.apply(q)
        k = rotary.apply(k)
        q, k = norm(q), norm(k)

        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = mx.repeat(k, rep, axis=2)
            v = mx.repeat(v, rep, axis=2)

        q = mx.swapaxes(q, 1, 2)
        k = mx.swapaxes(k, 1, 2)
        v = mx.swapaxes(v, 1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ mx.swapaxes(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        y = weights @ v

        y = mx.swapaxes(y, 1, 2).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReluSquared
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, rotary, mask):
        x = x + self.attn(norm(x), ve, rotary, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones(config.n_layer)
        self.x0_lambdas = mx.zeros(config.n_layer)

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self.value_embeds[str(i)] = nn.Embedding(config.vocab_size, kv_dim)

        self.rotary = RotaryEmbedding(head_dim, config.sequence_len * 10)

        self._masks = {}
        for ws in set(tuple(w) if isinstance(w, (list, tuple)) else (w,)
                      for w in self.window_sizes):
            window = ws[0] if isinstance(ws, tuple) else ws
            T = config.sequence_len
            if window > 0 and window < T:
                self._masks[window] = create_sliding_window_mask(T, window, mx.float32)
            else:
                self._masks[window] = create_causal_mask(T, mx.float32)

    def init_weights(self):
        """Initialize weights matching the original."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        self.wte.weight = mx.random.normal(self.wte.weight.shape).astype(mx.bfloat16)
        self.lm_head.weight = mx.random.normal(self.lm_head.weight.shape) * 0.001

        for block in self.blocks:
            block.attn.c_q.weight = mx.random.uniform(-s, s, block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            if block.attn.has_ve:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

        self.resid_lambdas = mx.ones(self.config.n_layer)
        self.x0_lambdas = mx.full(self.config.n_layer, 0.1)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, ve.weight.shape).astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        flat = dict(tree_flatten(self.parameters()))
        nparams = sum(p.size for p in flat.values())
        value_embeds_numel = sum(self.value_embeds[k].weight.size
                                 for k in self.value_embeds)
        nparams_exclude = (self.wte.weight.size + value_embeds_numel +
                          self.resid_lambdas.size + self.x0_lambdas.size)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        flat = dict(tree_flatten(self.parameters()))
        wte = self.wte.weight.size
        value_embeds = sum(self.value_embeds[k].weight.size
                           for k in self.value_embeds)
        lm_head = self.lm_head.weight.size
        block_params = sum(p.size for name, p in flat.items()
                           if 'blocks' in name)
        scalars = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte + value_embeds + lm_head + block_params + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': block_params, 'scalars': scalars,
            'total': total,
        }

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape

        x = self.wte(idx)
        x = norm(x)
        x0 = x

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = str(i)
            ve = (self.value_embeds[ve_key](idx)
                  if ve_key in self.value_embeds else None)
            window = self.window_sizes[i][0]
            mask = self._masks.get(window)
            if mask is not None and mask.shape[0] > T:
                mask = mask[:T, :T]
            x = block(x, ve, self.rotary, mask)

        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            if reduction == 'none':
                loss = nn.losses.cross_entropy(logits_flat, targets_flat,
                                               reduction='none')
                return loss.reshape(B, T)
            else:
                loss = nn.losses.cross_entropy(logits_flat, targets_flat,
                                               reduction='mean')
                return loss
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step (tuned for Apple Silicon)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32   # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

chip_name, gpu_cores = get_apple_silicon_info()
PEAK_FLOPS = estimate_peak_flops(chip_name, gpu_cores)
print(f"Backend: MLX ({chip_name}, {gpu_cores} GPU cores)")
print(f"Peak bf16 FLOPS: {PEAK_FLOPS:.2e}")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, \
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by " \
    f"DEVICE_BATCH_SIZE * MAX_SEQ_LEN ({tokens_per_fwdbwd})"
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Build optimizer with Muon
model_dim = config.n_embd
param_groups = build_param_groups(model, {
    'model_dim': model_dim,
    'matrix_lr': MATRIX_LR,
    'embedding_lr': EMBEDDING_LR,
    'unembedding_lr': UNEMBEDDING_LR,
    'scalar_lr': SCALAR_LR,
    'adam_betas': ADAM_BETAS,
    'weight_decay': WEIGHT_DECAY,
})
optimizer = MuonAdamWMLX(param_groups)

# Loss + gradient function
def loss_fn(model, x, y):
    return model(x, y, reduction='mean')

loss_grad_fn = nn.value_and_grad(model, loss_fn)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    # Gradient accumulation
    accum_loss = mx.array(0.0)
    accum_grads = None

    for micro_step in range(grad_accum_steps):
        loss, grads = loss_grad_fn(model, x, y)
        accum_loss = accum_loss + loss

        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

        x, y, epoch = next(train_loader)

    # Average gradients
    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda g: g / grad_accum_steps, accum_grads)

    train_loss_val = (accum_loss / grad_accum_steps)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    # Update optimizer schedules
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * lrm
        if group['kind'] == 'muon':
            group['momentum'] = muon_momentum
            group['weight_decay'] = muon_weight_decay

    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), train_loss_val)

    train_loss_f = train_loss_val.item()

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
          f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
          f"mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
          end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
steady_state_mfu = (100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10)
                    / total_training_time / PEAK_FLOPS
                    if total_training_time > 0 else 0)

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     n/a (unified memory)")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
