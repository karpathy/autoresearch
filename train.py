"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"      # persistent compile cache across runs
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"         # serialize kernel launches for better overlap

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config

# B200 (Blackwell / SM100) torch.compile optimizations
inductor_config.coordinate_descent_tuning = True       # fine-tune tile sizes per kernel
inductor_config.max_autotune_gemm_backends = "CUTLASS,Triton,ATen"  # include CUTLASS SM100 kernels
inductor_config.epilogue_fusion = True                 # fuse pointwise ops into matmul epilogues
inductor_config.aggressive_fusion = True               # fuse more ops into fewer kernels
inductor_config.shape_padding = True                   # pad matrices to tensor core tile boundaries

from kernels import get_kernel
cap = torch.cuda.get_device_capability()

USE_FLEX_ATTENTION = cap[0] >= 10  # Blackwell (SM100) and future architectures

if USE_FLEX_ATTENTION:
    from functools import partial
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _flex_attention_compiled = torch.compile(
        partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
        dynamic=False,
    )
    print("Using FlexAttention with FA4 backend (Blackwell)")
else:
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
    print(f"Using FA3 from {repo}")

if USE_FLEX_ATTENTION:
    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def _sliding_window_mask(window_size):
        def mask_fn(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)
        return mask_fn

    def _build_flex_block_mask(window_size_tuple, B, n_head, T):
        window = window_size_tuple[0]
        if window <= 0 or window >= T:
            mask_fn = _causal_mask
        else:
            mask_fn = _sliding_window_mask(window)
        return create_block_mask(mask_fn, B=B, H=n_head, Q_LEN=T, KV_LEN=T, device="cuda")

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
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
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Fused QKV: single GEMM (d → 3d) instead of three separate (d → d) GEMMs
        # torch.compile does NOT auto-fuse separate Linears with different weight tensors
        self.q_dim = self.n_head * self.head_dim
        self.k_dim = self.n_kv_head * self.head_dim
        self.v_dim = self.n_kv_head * self.head_dim
        self.c_qkv = nn.Linear(self.n_embd, self.q_dim + self.k_dim + self.v_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, block_mask=None):
        B, T, C = x.size()
        qkv = self.c_qkv(x)
        q, k, v = qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if USE_FLEX_ATTENTION:
            # FlexAttention expects (B, H, T, D) not (B, T, H, D)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = _flex_attention_compiled(q, k, v, block_mask=block_mask, enable_gqa=True)
            y = y.transpose(1, 2)
        else:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, block_mask=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, block_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self._flex_block_masks = None  # Lazily built on first forward (needs CUDA device)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings — stored as ModuleDict for parameter tracking,
        # but accessed via _ve_list (plain list) to avoid dict lookups that cause graph breaks
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self._ve_list = [self.value_embeds.get(str(i)) for i in range(config.n_layer)]
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
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
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
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
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        # Verify all params are covered (use >= to allow FP8 scale params in transformer.h)
        grouped_count = (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        total_count = len(list(self.parameters()))
        assert grouped_count >= total_count, f"Param group mismatch: {grouped_count} grouped vs {total_count} total"
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Block masks are built eagerly before torch.compile (no lazy init = no graph break)
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self._ve_list[i](idx) if self._ve_list[i] is not None else None
            block_mask = self._flex_block_masks[i] if USE_FLEX_ATTENTION else None
            x = block(x, ve, cos_sin, self.window_sizes[i], block_mask)
        x = norm(x)

        softcap = 15

        if targets is not None:
            # Chunked cross-entropy: avoid materializing full (B*T, V) logit tensor
            # Saves ~8 GB of FP32 memory traffic per micro-step at B=128, T=2048, V=8192
            x_flat = x.view(-1, x.size(-1))       # (B*T, d)
            targets_flat = targets.view(-1)         # (B*T,)
            chunk_size = 4096  # tokens per chunk — keeps peak logit buffer at ~134 MB

            if reduction == 'none':
                # Eval path: evaluate_bpb() needs per-token losses
                losses = []
                for i in range(0, x_flat.size(0), chunk_size):
                    logits_chunk = self.lm_head(x_flat[i:i+chunk_size]).float()
                    logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
                    losses.append(F.cross_entropy(logits_chunk, targets_flat[i:i+chunk_size],
                                                  ignore_index=-1, reduction='none'))
                return torch.cat(losses)
            else:
                # Training path: accumulate sum, divide at the end
                total_loss = torch.tensor(0.0, device=x.device)
                n_tokens = targets_flat.ne(-1).sum()
                for i in range(0, x_flat.size(0), chunk_size):
                    logits_chunk = self.lm_head(x_flat[i:i+chunk_size]).float()
                    logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
                    total_loss = total_loss + F.cross_entropy(
                        logits_chunk, targets_flat[i:i+chunk_size],
                        ignore_index=-1, reduction='sum')
                return total_loss / n_tokens

        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture — sized to cross B200 roofline (AI > 562 FLOP/byte)
ASPECT_RATIO = 96       # model_dim = depth * ASPECT_RATIO (was 64, gives dim=1152 at depth=12)
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size — scaled up for B200 (192 GB HBM3e)
DEPTH = 12              # number of transformer layers (was 8)
DEVICE_BATCH_SIZE = 64   # per-device batch size — est. ~83 GB at D=12/AR=96, headroom for 192 GB B200
                         # increase to 128 if peak_vram_mb < 140,000 (target: fill ~180 GB)
COMPILE_WARMUP_STEPS = 3 # extra warmup for max-autotune kernel benchmarking
TIMING_WARMUP_STEPS = 10 + COMPILE_WARMUP_STEPS  # total steps before timing starts

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
def get_gpu_peak_flops():
    """Detect GPU and return peak bf16 TFLOPS (without sparsity) for MFU calculation."""
    gpu_name = torch.cuda.get_device_name()
    # Peak bf16 tensor core FLOPS (without sparsity) for known GPUs.
    # More specific substrings must come before less specific ones.
    known_gpus = [
        # Blackwell data center
        ("B200",           4500.0e12),
        # Hopper data center
        ("H200",            989.5e12),
        ("H100 PCIe",       756.0e12),
        ("H100 NVL",        835.0e12),
        ("H100",            989.5e12),
        # Ampere data center
        ("A100",            312.0e12),
        ("A10G",            125.0e12),
        ("A10",             125.0e12),
        # Ada Lovelace data center
        ("L40S",            362.0e12),
        ("L40",             181.0e12),
        ("L4",              121.0e12),
        # Volta
        ("V100",            125.0e12),
        # RTX 50 series (Blackwell)
        ("RTX 5090",        419.0e12),
        ("RTX 5080",        225.0e12),
        ("5070 Ti",         176.0e12),
        ("RTX 5070",        124.0e12),
        # RTX 40 series (Ada Lovelace)
        ("RTX 4090",        330.3e12),
        ("4080 SUPER",      209.0e12),
        ("RTX 4080",        195.0e12),
        ("4070 Ti SUPER",   176.0e12),
        ("4070 Ti",         160.0e12),
        ("4070 SUPER",      142.0e12),
        ("RTX 4070",        117.0e12),
        # RTX 30 series (Ampere)
        ("3090 Ti",         160.0e12),
        ("RTX 3090",        142.0e12),
        ("3080 Ti",         136.0e12),
        ("RTX 3080",        119.0e12),
    ]
    for pattern, flops in known_gpus:
        if pattern in gpu_name:
            print(f"GPU: {gpu_name} ({flops/1e12:.1f} TFLOPS bf16 peak)")
            return flops
    # Fallback: rough estimate from compute capability and SM count
    props = torch.cuda.get_device_properties(0)
    num_sms = props.multi_processor_count
    cap = torch.cuda.get_device_capability()
    tflops_per_sm = {7: 1.6, 8: 2.2, 9: 7.5, 10: 17.6}.get(cap[0], 2.5)
    estimated_flops = num_sms * tflops_per_sm * 1e12
    print(f"WARNING: Unknown GPU '{gpu_name}' (CC {cap[0]}.{cap[1]}, {num_sms} SMs)")
    print(f"  Estimated {estimated_flops/1e12:.1f} TFLOPS bf16 peak — MFU% will be approximate.")
    print(f"  Add your GPU to get_gpu_peak_flops() for accurate MFU reporting.")
    return estimated_flops

GPU_PEAK_FLOPS = get_gpu_peak_flops()

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

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# FP8 training: 2.2x throughput on B200 (10,000 TFLOPS FP8 vs 4,500 BF16)
# Only for Blackwell (SM100+) where FP8 tensor cores are mature
USE_FP8 = cap[0] >= 10 and config.n_embd >= 1024  # skip for tiny models (FP8 scaling overhead)
if USE_FP8:
    # Try MXFP8 first (per-block-of-32 scaling, better precision than per-tensor FP8)
    _using_mxfp8 = False
    try:
        from torchao.prototype.mx_formats.mx_linear import MXLinearConverter
        mx_converter = MXLinearConverter("mxfp8_cublas")
        # Skip ve_gate layers (32-wide, too narrow for FP8 benefit)
        def _mx_filter(mod, fqn):
            return isinstance(mod, nn.Linear) and "ve_gate" not in fqn
        mx_converter.convert(model, module_filter_fn=_mx_filter)
        _using_mxfp8 = True
        GPU_PEAK_FLOPS = GPU_PEAK_FLOPS * 2.22  # 10000/4500 ≈ 2.22
        print(f"MXFP8 training enabled (per-block-of-32 scaling, peak {GPU_PEAK_FLOPS/1e12:.0f} TFLOPS)")
    except (ImportError, Exception) as e:
        # Fall back to standard per-tensor FP8
        print(f"MXFP8 unavailable ({e}), falling back to standard FP8")
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        fp8_config = Float8LinearConfig(pad_inner_dim=True)
        def _fp8_module_filter(mod, fqn):
            return isinstance(mod, nn.Linear) and "ve_gate" not in fqn
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=_fp8_module_filter)
        GPU_PEAK_FLOPS = GPU_PEAK_FLOPS * 2.22  # 10000/4500 ≈ 2.22
        print(f"FP8 training enabled (per-tensor scaling, peak {GPU_PEAK_FLOPS/1e12:.0f} TFLOPS)")
else:
    print(f"FP8 disabled (cap={cap}, n_embd={config.n_embd})")

# Optimizer setup AFTER FP8 conversion — FP8 replaces nn.Linear with Float8Linear,
# creating new parameter objects. Optimizer must reference the final parameters.
optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# Eagerly build FlexAttention block masks before compile (eliminates graph break in forward)
if USE_FLEX_ATTENTION:
    unique_windows = set(model.window_sizes)
    mask_cache = {ws: _build_flex_block_mask(ws, DEVICE_BATCH_SIZE, config.n_head, MAX_SEQ_LEN)
                  for ws in unique_windows}
    model._flex_block_masks = [mask_cache[ws] for ws in model.window_sizes]

model = torch.compile(model, dynamic=False, mode="max-autotune", fullgraph=True)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

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
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > TIMING_WARMUP_STEPS:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / GPU_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > TIMING_WARMUP_STEPS and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - TIMING_WARMUP_STEPS) / total_training_time / GPU_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
