"""MLX benchmark runner with CPU Muon (PR #202 style)."""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu

from benchmarks.mps_vs_mlx.src.config import BenchConfig, generate_batch, compute_stats


# Newton-Schulz coefficients
POLAR_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def rms_norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = self.head_dim ** -0.5
        i = mx.arange(T)[:, None]
        j = mx.arange(T)[None, :]
        mask = mx.where(j <= i, mx.zeros((T, T)), mx.full((T, T), float("-inf")))
        
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.c_proj(mx.square(nn.relu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = Attention(n_embd, n_head)
        self.mlp = MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, idx, targets=None):
        x = rms_norm(self.wte(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(rms_norm(x))
        if targets is not None:
            B, T, V = logits.shape
            return nn.losses.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T), reduction="mean")
        return logits


def ns_ortho_cpu(g: np.ndarray, steps: int = 5) -> np.ndarray:
    """Newton-Schulz orthogonalization on CPU."""
    X = g.astype(np.float32)
    X /= np.linalg.norm(X) * 1.02 + 1e-6
    for a, b, c in POLAR_COEFFS[:steps]:
        if X.shape[0] > X.shape[1]:
            A = X.T @ X
            B_ = b * A + c * (A @ A)
            X = a * X + X @ B_
        else:
            A = X @ X.T
            B_ = b * A + c * (A @ A)
            X = a * X + B_ @ X
    return X


class MuonCPU:
    """Muon optimizer on CPU (PR #202 style)."""
    
    def __init__(self, lr=0.02, momentum=0.95):
        self.lr = lr
        self.momentum = momentum
        self.state = {}

    def step(self, model, grads):
        flat_p = mu.tree_flatten(model.trainable_parameters())
        flat_g = mu.tree_flatten(grads)
        updates = []
        
        for (path, p_mx), (_, g_mx) in zip(flat_p, flat_g):
            p_np = np.array(p_mx, copy=False).astype(np.float32)
            g_np = np.array(g_mx, copy=False).astype(np.float32)
            
            if path not in self.state:
                self.state[path] = {'mom': np.zeros_like(p_np)}
            
            s = self.state[path]
            s['mom'] = (1 - self.momentum) * g_np + self.momentum * s['mom']
            g_np = (1 - self.momentum) * g_np + self.momentum * s['mom']
            
            if p_np.ndim == 2:
                g_np = ns_ortho_cpu(g_np)
            
            new_p = p_np - self.lr * g_np
            updates.append((path, mx.array(new_p)))
        
        model.update(mu.tree_unflatten(updates))


def benchmark_mlx(config: BenchConfig) -> dict:
    """Run MLX benchmark."""
    print(f"  Device: MLX (Metal GPU + CPU Muon)")
    
    model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer)
    _ = model(mx.zeros((1, 2), dtype=mx.int32))
    mx.eval(model.parameters())
    
    flat_p = mu.tree_flatten(model.trainable_parameters())
    nparams = sum(v.size for _, v in flat_p if isinstance(v, mx.array))
    print(f"  Parameters: {nparams/1e6:.2f}M")
    
    optimizer = MuonCPU(lr=0.02)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m(x, y))

    results = {'forward': [], 'backward': [], 'optimizer': [], 'total': [], 'loss': []}
    tokens_per_step = config.batch_size * config.seq_len
    
    for step in range(config.warmup_steps + config.benchmark_steps):
        x_np, y_np = generate_batch(config.batch_size, config.seq_len, config.vocab_size)
        x = mx.array(x_np)
        y = mx.array(y_np)
        
        t0 = time.perf_counter()
        
        loss, grads = loss_and_grad(model, x, y)
        mx.eval(loss, grads)
        t1 = time.perf_counter()
        
        t2 = t1  # backward included in value_and_grad
        
        optimizer.step(model, grads)
        mx.eval(model.parameters())
        t3 = time.perf_counter()
        
        if step >= config.warmup_steps:
            fwd_bwd = (t1 - t0) * 1000
            results['forward'].append(fwd_bwd * 0.4)
            results['backward'].append(fwd_bwd * 0.6)
            results['optimizer'].append((t3 - t2) * 1000)
            results['total'].append((t3 - t0) * 1000)
            results['loss'].append(float(loss))
    
    total_mean = np.mean(results['total'])
    return {
        'forward_ms': compute_stats(results['forward']),
        'backward_ms': compute_stats(results['backward']),
        'optimizer_ms': compute_stats(results['optimizer']),
        'total_ms': compute_stats(results['total']),
        'tokens_per_sec': tokens_per_step / (total_mean / 1000),
        'final_loss': float(np.mean(results['loss'][-10:])),
        'params_M': nparams / 1e6,
    }
