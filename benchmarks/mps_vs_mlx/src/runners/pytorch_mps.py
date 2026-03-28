"""PyTorch MPS benchmark runner with GPU Muon."""

import time
import numpy as np
import torch
import torch.nn.functional as F

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
    return F.rms_norm(x, (x.size(-1),))


class Attention(torch.nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
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
    """Muon optimizer running on GPU."""
    
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                mom = group['momentum']
                state['momentum_buffer'].lerp_(g, 1 - mom)
                g = g.lerp_(state['momentum_buffer'], mom)
                if p.ndim == 2:
                    X = g.float()
                    X = X / (X.norm() * 1.02 + 1e-6)
                    for a, b, c in POLAR_COEFFS[:group['ns_steps']]:
                        if X.size(0) > X.size(1):
                            A = X.mT @ X
                            B = b * A + c * (A @ A)
                            X = a * X + X @ B
                        else:
                            A = X @ X.mT
                            B = b * A + c * (A @ A)
                            X = a * X + B @ X
                    g = X.to(p.dtype)
                p.add_(g, alpha=-group['lr'])


def benchmark_pytorch_mps(config: BenchConfig) -> dict:
    """Run PyTorch MPS benchmark."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = GPT(config.vocab_size, config.n_embd, config.n_head, config.n_layer).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {nparams/1e6:.2f}M")
    
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    opt_muon = MuonGPU(matrix_params, lr=0.02)
    opt_adam = torch.optim.AdamW(other_params, lr=0.001) if other_params else None

    results = {'forward': [], 'backward': [], 'optimizer': [], 'total': [], 'loss': []}
    tokens_per_step = config.batch_size * config.seq_len
    
    for step in range(config.warmup_steps + config.benchmark_steps):
        x_np, y_np = generate_batch(config.batch_size, config.seq_len, config.vocab_size)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        
        torch.mps.synchronize()
        t0 = time.perf_counter()
        
        loss = model(x, y)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        
        loss.backward()
        torch.mps.synchronize()
        t2 = time.perf_counter()
        
        opt_muon.step()
        if opt_adam:
            opt_adam.step()
        opt_muon.zero_grad(set_to_none=True)
        if opt_adam:
            opt_adam.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        t3 = time.perf_counter()
        
        if step >= config.warmup_steps:
            results['forward'].append((t1 - t0) * 1000)
            results['backward'].append((t2 - t1) * 1000)
            results['optimizer'].append((t3 - t2) * 1000)
            results['total'].append((t3 - t0) * 1000)
            results['loss'].append(loss.item())
    
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
