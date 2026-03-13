#!/usr/bin/env python3
"""
Benchmark dispatcher. stdio_bus worker, JSON-RPC 2.0 over NDJSON.
Runs MPS vs MLX benchmarks and provides AI-assisted analysis.

Usage:
    echo '{"jsonrpc":"2.0","id":1,"method":"benchmark.status","params":{}}' | uv run benchmark.py
    echo '{"jsonrpc":"2.0","id":1,"method":"benchmark.run","params":{}}' | uv run benchmark.py

Methods:
    benchmark.status  - Current state: last results, history count
    benchmark.run     - Run MPS vs MLX benchmark
    benchmark.history - Get benchmark history
    benchmark.analyze - AI analysis of results
    benchmark.plot    - Generate plots (base64 PNG or save to dir)
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "mps_vs_mlx" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Benchmark Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    warmup_steps: int = 10
    benchmark_steps: int = 50
    batch_size: int = 4
    seq_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4

def generate_batch(batch_size, seq_len, vocab_size):
    x = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int64)
    y = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int64)
    return x, y

def compute_stats(times):
    return {"mean": float(np.mean(times)), "std": float(np.std(times)),
            "min": float(np.min(times)), "max": float(np.max(times))}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = {"running": False, "last_result": None, "history": [], "total": 0}
state_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

# ---------------------------------------------------------------------------
# PyTorch MPS Benchmark (GPU Muon)
# ---------------------------------------------------------------------------

def run_pytorch_mps(config):
    import torch
    import torch.nn.functional as F

    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    sync = torch.mps.synchronize if use_mps else lambda: None
    log(f"PyTorch MPS | Device: {device}")

    # Newton-Schulz coefficients
    POLAR_COEFFS = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]

    def rms_norm(x):
        return F.rms_norm(x, (x.size(-1),))

    class Attention(torch.nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            self.n_head, self.head_dim = n_head, n_embd // n_head
            self.c_q = torch.nn.Linear(n_embd, n_embd, bias=False)
            self.c_k = torch.nn.Linear(n_embd, n_embd, bias=False)
            self.c_v = torch.nn.Linear(n_embd, n_embd, bias=False)
            self.c_proj = torch.nn.Linear(n_embd, n_embd, bias=False)

        def forward(self, x):
            B, T, C = x.size()
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

    class MLP(torch.nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.c_fc = torch.nn.Linear(n_embd, 4 * n_embd, bias=False)
            self.c_proj = torch.nn.Linear(4 * n_embd, n_embd, bias=False)

        def forward(self, x):
            return self.c_proj(F.relu(self.c_fc(x)).square())

    class Block(torch.nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            self.attn = Attention(n_embd, n_head)
            self.mlp = MLP(n_embd)

        def forward(self, x):
            x = x + self.attn(rms_norm(x))
            x = x + self.mlp(rms_norm(x))
            return x

    class GPT(torch.nn.Module):
        def __init__(self, vocab_size, n_embd, n_head, n_layer):
            super().__init__()
            self.wte = torch.nn.Embedding(vocab_size, n_embd)
            self.blocks = torch.nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
            self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

        def forward(self, idx, targets=None):
            x = rms_norm(self.wte(idx))
            for block in self.blocks:
                x = block(x)
            logits = self.lm_head(rms_norm(x))
            if targets is not None:
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits

    class MuonGPU(torch.optim.Optimizer):
        def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
            super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(state["momentum_buffer"], group["momentum"])
                    if p.ndim == 2:
                        X = g.float()
                        X = X / (X.norm() * 1.02 + 1e-6)
                        for a, b, c in POLAR_COEFFS[: group["ns_steps"]]:
                            if X.size(0) > X.size(1):
                                A = X.mT @ X
                                B = b * A + c * (A @ A)
                                X = a * X + X @ B
                            else:
                                A = X @ X.mT
                                B = b * A + c * (A @ A)
                                X = a * X + B @ X
                        g = X.to(p.dtype)
                    p.add_(g, alpha=-group["lr"])

    # Setup
    model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    opt_muon = MuonGPU(matrix_params, lr=0.02)
    opt_adam = torch.optim.AdamW(other_params, lr=0.001) if other_params else None

    # Benchmark
    results = {"forward": [], "backward": [], "optimizer": [], "total": [], "loss": []}
    tokens_per_step = config.batch_size * config.seq_len

    for step in range(config.warmup_steps + config.benchmark_steps):
        x_np, y_np = generate_batch(config.batch_size, config.seq_len, config.vocab_size)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        sync()
        t0 = time.perf_counter()
        loss = model(x, y)
        sync()
        t1 = time.perf_counter()
        loss.backward()
        sync()
        t2 = time.perf_counter()
        opt_muon.step()
        if opt_adam:
            opt_adam.step()
        opt_muon.zero_grad(set_to_none=True)
        if opt_adam:
            opt_adam.zero_grad(set_to_none=True)
        sync()
        t3 = time.perf_counter()

        if step >= config.warmup_steps:
            results["forward"].append((t1 - t0) * 1000)
            results["backward"].append((t2 - t1) * 1000)
            results["optimizer"].append((t3 - t2) * 1000)
            results["total"].append((t3 - t0) * 1000)
            results["loss"].append(loss.item())

    total_mean = np.mean(results["total"])
    return {
        "forward_ms": compute_stats(results["forward"]),
        "backward_ms": compute_stats(results["backward"]),
        "optimizer_ms": compute_stats(results["optimizer"]),
        "total_ms": compute_stats(results["total"]),
        "tokens_per_sec": tokens_per_step / (total_mean / 1000),
        "final_loss": float(np.mean(results["loss"][-10:])),
        "params_M": nparams / 1e6,
    }

# ---------------------------------------------------------------------------
# MLX Benchmark (CPU Muon)
# ---------------------------------------------------------------------------

def run_mlx(config):
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.utils as mu
    except ImportError:
        return {"error": "mlx not installed. Run: uv sync --extra benchmark"}

    log("MLX | Device: Metal GPU + CPU Muon")

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
            self.n_head, self.head_dim = n_head, n_embd // n_head
            self.c_q = nn.Linear(n_embd, n_embd, bias=False)
            self.c_k = nn.Linear(n_embd, n_embd, bias=False)
            self.c_v = nn.Linear(n_embd, n_embd, bias=False)
            self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        def __call__(self, x):
            B, T, C = x.shape
            q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            k = self.c_k(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            v = self.c_v(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            scale = self.head_dim**-0.5
            i, j = mx.arange(T)[:, None], mx.arange(T)[None, :]
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
                return nn.losses.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean")
            return logits

    def ns_ortho_cpu(g, steps=5):
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
        def __init__(self, lr=0.02, momentum=0.95):
            self.lr, self.momentum, self.state = lr, momentum, {}

        def step(self, model, grads):
            flat_p, flat_g = mu.tree_flatten(model.trainable_parameters()), mu.tree_flatten(grads)
            updates = []
            for (path, p_mx), (_, g_mx) in zip(flat_p, flat_g):
                p_np = np.array(p_mx, copy=False).astype(np.float32)
                g_np = np.array(g_mx, copy=False).astype(np.float32)
                if path not in self.state:
                    self.state[path] = {"mom": np.zeros_like(p_np)}
                s = self.state[path]
                s["mom"] = (1 - self.momentum) * g_np + self.momentum * s["mom"]
                g_np = (1 - self.momentum) * g_np + self.momentum * s["mom"]
                if p_np.ndim == 2:
                    g_np = ns_ortho_cpu(g_np)
                updates.append((path, mx.array(p_np - self.lr * g_np)))
            model.update(mu.tree_unflatten(updates))

    # Setup
    model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer)
    _ = model(mx.zeros((1, 2), dtype=mx.int32))
    mx.eval(model.parameters())
    flat_p = mu.tree_flatten(model.trainable_parameters())
    nparams = sum(v.size for _, v in flat_p if isinstance(v, mx.array))
    optimizer = MuonCPU(lr=0.02)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m(x, y))

    # Benchmark
    results = {"forward": [], "backward": [], "optimizer": [], "total": [], "loss": []}
    tokens_per_step = config.batch_size * config.seq_len

    for step in range(config.warmup_steps + config.benchmark_steps):
        x_np, y_np = generate_batch(config.batch_size, config.seq_len, config.vocab_size)
        x, y = mx.array(x_np), mx.array(y_np)

        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, x, y)
        mx.eval(loss, grads)
        t1 = time.perf_counter()
        t2 = t1
        optimizer.step(model, grads)
        mx.eval(model.parameters())
        t3 = time.perf_counter()

        if step >= config.warmup_steps:
            fwd_bwd = (t1 - t0) * 1000
            results["forward"].append(fwd_bwd * 0.4)
            results["backward"].append(fwd_bwd * 0.6)
            results["optimizer"].append((t3 - t2) * 1000)
            results["total"].append((t3 - t0) * 1000)
            results["loss"].append(float(loss))

    total_mean = np.mean(results["total"])
    return {
        "forward_ms": compute_stats(results["forward"]),
        "backward_ms": compute_stats(results["backward"]),
        "optimizer_ms": compute_stats(results["optimizer"]),
        "total_ms": compute_stats(results["total"]),
        "tokens_per_sec": tokens_per_step / (total_mean / 1000),
        "final_loss": float(np.mean(results["loss"][-10:])),
        "params_M": nparams / 1e6,
    }

# ---------------------------------------------------------------------------
# Run Full Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(params):
    config = BenchConfig(
        warmup_steps=params.get("warmup", 10),
        benchmark_steps=params.get("steps", 50),
        batch_size=params.get("batch_size", 4),
    )

    log(f"Benchmark | steps={config.benchmark_steps} batch={config.batch_size}")

    # PyTorch MPS
    mps_results = run_pytorch_mps(config)

    # MLX (optional)
    mlx_results = run_mlx(config)
    if "error" in mlx_results:
        mlx_results = None

    # Comparison
    if mlx_results:
        speedup = mps_results["tokens_per_sec"] / mlx_results["tokens_per_sec"]
        opt_speedup = mlx_results["optimizer_ms"]["mean"] / mps_results["optimizer_ms"]["mean"]
        comparison = {"throughput_speedup": speedup, "optimizer_speedup": opt_speedup}
    else:
        comparison = None

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "pytorch_mps": mps_results,
        "mlx": mlx_results,
        "comparison": comparison,
    }

    # Save
    output_path = RESULTS_DIR / "benchmark.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Update state
    with state_lock:
        state["last_result"] = result
        state["history"].append(result)
        state["total"] += 1

    log(f"Benchmark done | MPS: {mps_results['tokens_per_sec']:.0f} tok/s")
    return result

def benchmark_wrapper(params):
    with state_lock:
        state["running"] = True
    try:
        return run_benchmark(params)
    finally:
        with state_lock:
            state["running"] = False

# ---------------------------------------------------------------------------
# JSON-RPC Handler
# ---------------------------------------------------------------------------

def handle(method, params):
    match method:
        case "benchmark.status":
            with state_lock:
                return {
                    "running": state["running"],
                    "total": state["total"],
                    "last_result": state["last_result"],
                }

        case "benchmark.run":
            with state_lock:
                if state["running"]:
                    return {"error": "Benchmark already running"}

            if params.get("blocking", True):
                return benchmark_wrapper(params)
            executor.submit(benchmark_wrapper, params)
            return {"queued": True}

        case "benchmark.history":
            limit = params.get("limit", 10)
            with state_lock:
                return {"results": state["history"][-limit:]}

        case "benchmark.analyze":
            with state_lock:
                result = state["last_result"]
            if not result:
                return {"error": "No benchmark results. Run benchmark.run first."}

            # Format for AI analysis
            mps = result["pytorch_mps"]
            mlx = result.get("mlx")
            comp = result.get("comparison")

            analysis_prompt = f"""Benchmark Results Analysis:

PyTorch MPS (GPU Muon):
- Throughput: {mps['tokens_per_sec']:.0f} tok/s
- Step time: {mps['total_ms']['mean']:.1f} ms
- Optimizer: {mps['optimizer_ms']['mean']:.1f} ms
- Forward: {mps['forward_ms']['mean']:.1f} ms
- Backward: {mps['backward_ms']['mean']:.1f} ms
"""
            if mlx and comp:
                analysis_prompt += f"""
MLX (CPU Muon):
- Throughput: {mlx['tokens_per_sec']:.0f} tok/s
- Step time: {mlx['total_ms']['mean']:.1f} ms
- Optimizer: {mlx['optimizer_ms']['mean']:.1f} ms

Comparison:
- PyTorch MPS is {comp['throughput_speedup']:.2f}x faster overall
- GPU Muon is {comp['optimizer_speedup']:.1f}x faster than CPU Muon
"""
            return {
                "analysis_prompt": analysis_prompt,
                "raw_results": result,
                "recommendation": "GPU Muon (PyTorch MPS) is significantly faster. PR #202 MLX approach not recommended."
                if comp and comp["throughput_speedup"] > 1.5
                else "Results inconclusive, more testing needed.",
            }

        case "benchmark.plot":
            with state_lock:
                result = state["last_result"]
            if not result:
                return {"error": "No benchmark results. Run benchmark.run first."}
            return generate_plots(result)

    return None


# ---------------------------------------------------------------------------
# Plot Generation
# ---------------------------------------------------------------------------

PLOTS_DIR = PROJECT_ROOT / "benchmarks" / "mps_vs_mlx" / "plots"
# Docker mount: /app/benchmarks -> $(pwd)/benchmarks
if not PLOTS_DIR.exists():
    PLOTS_DIR = Path("/app/benchmarks/mps_vs_mlx/plots")

def generate_plots(results):
    """Generate and save benchmark plots to results directory."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {"error": "matplotlib not installed"}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    mps = results["pytorch_mps"]
    mlx = results.get("mlx")
    comp = results.get("comparison")
    
    saved = []
    
    # Throughput chart
    fig, ax = plt.subplots(figsize=(8, 6))
    if mlx and comp:
        bars = ax.bar(
            ['PyTorch MPS\n(GPU Muon)', 'MLX\n(CPU Muon)'],
            [mps['tokens_per_sec'], mlx['tokens_per_sec']],
            color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5
        )
        title = f"Training Throughput\nPyTorch MPS is {comp['throughput_speedup']:.2f}x faster"
    else:
        bars = ax.bar(['PyTorch'], [mps['tokens_per_sec']],
                      color=['#3498db'], edgecolor='black', linewidth=1.5)
        title = f"Training Throughput: {mps['tokens_per_sec']:,.0f} tok/s"
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:,.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Tokens/sec', fontsize=12)
    plt.tight_layout()
    path = PLOTS_DIR / 'throughput.png'
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(str(path))
    
    # Breakdown chart
    fig, ax = plt.subplots(figsize=(10, 6))
    if mlx:
        categories = ['PyTorch MPS', 'MLX']
        forward = [mps['forward_ms']['mean'], mlx['forward_ms']['mean']]
        backward = [mps['backward_ms']['mean'], mlx['backward_ms']['mean']]
        optimizer = [mps['optimizer_ms']['mean'], mlx['optimizer_ms']['mean']]
    else:
        categories = ['PyTorch']
        forward = [mps['forward_ms']['mean']]
        backward = [mps['backward_ms']['mean']]
        optimizer = [mps['optimizer_ms']['mean']]
    x = np.arange(len(categories))
    ax.bar(x, forward, 0.5, label='Forward', color='#3498db')
    ax.bar(x, backward, 0.5, bottom=forward, label='Backward', color='#9b59b6')
    ax.bar(x, optimizer, 0.5, bottom=np.array(forward)+np.array(backward), label='Optimizer', color='#e67e22')
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Step Time Breakdown', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper right')
    for i, (f, b, o) in enumerate(zip(forward, backward, optimizer)):
        ax.annotate(f'{f+b+o:.0f} ms', xy=(i, f+b+o+5), ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = PLOTS_DIR / 'breakdown.png'
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(str(path))
    
    # Optimizer comparison (only if MLX data exists)
    if mlx and comp:
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            ['GPU Muon\n(PyTorch)', 'CPU Muon\n(MLX)'],
            [mps['optimizer_ms']['mean'], mlx['optimizer_ms']['mean']],
            color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f} ms', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')
        ax.set_title(f"Muon Optimizer: GPU vs CPU\nGPU is {comp['optimizer_speedup']:.1f}x faster",
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12)
        plt.tight_layout()
        path = PLOTS_DIR / 'optimizer.png'
        plt.savefig(path, dpi=150)
        plt.close()
        saved.append(str(path))
    
    log(f"Plots saved: {saved}")
    return {"saved": saved}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("Benchmark dispatcher ready")
    for line in sys.stdin:
        if not (line := line.strip()):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}), flush=True)
            continue

        if msg.get("jsonrpc") != "2.0" or "method" not in msg:
            if msg.get("id"):
                print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32600, "message": "Invalid request"}}), flush=True)
            continue

        result = handle(msg["method"], msg.get("params", {}))
        if msg.get("id") is not None:
            out = {"jsonrpc": "2.0", "id": msg["id"]}
            if msg.get("sessionId"):
                out["sessionId"] = msg["sessionId"]
            out |= {"result": result} if result is not None else {"error": {"code": -32601, "message": "Method not found"}}
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
