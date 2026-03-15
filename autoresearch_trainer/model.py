from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .optimizer import MuonAdamW


def resolve_attention_backend():
    if os.name == "nt":
        try:
            from flash_attn import flash_attn_func

            return "flash_attn", flash_attn_func
        except ImportError:
            pass

    from kernels import get_kernel

    cap = torch.cuda.get_device_capability()
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    flash_attn_func = get_kernel(repo).flash_attn_interface.flash_attn_func
    return repo, flash_attn_func


def estimate_device_peak_flops(device_props):
    flops_per_cycle_by_capability = {
        (10, 0): 8192,  # Blackwell
        (9, 0): 4096,   # Hopper
        (8, 0): 2048,   # Ampere (A100)
        (8, 6): 1024,   # Ampere (Consumer)
        (8, 9): 1024,   # Ada Lovelace
        (7, 0): 1024,   # Volta
        (7, 5): 1024,   # Turing
    }
    capability = (device_props.major, device_props.minor)
    flops_per_cycle = flops_per_cycle_by_capability.get(capability)
    if flops_per_cycle is None:
        return None
    return device_props.multi_processor_count * flops_per_cycle * device_props.clock_rate * 1000


def compute_mfu(peak_flops, num_flops_per_token, tok_per_sec):
    if peak_flops is None or tok_per_sec is None or peak_flops <= 0 or tok_per_sec <= 0:
        return None
    return 100 * num_flops_per_token * tok_per_sec / peak_flops


def target_tok_per_sec_for_mfu(peak_flops, num_flops_per_token, target_mfu_percent):
    if peak_flops is None or peak_flops <= 0 or num_flops_per_token <= 0 or target_mfu_percent <= 0:
        return None
    return peak_flops * (target_mfu_percent / 100.0) / num_flops_per_token


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    activation_checkpoint: str = "none"
    ve_gate_channels: int = 32
    softcap: float = 15.0


def build_model_config(*, depth, max_seq_len, vocab_size, aspect_ratio, head_dim, window_pattern, activation_checkpoint, ve_gate_channels=32, softcap=15.0):
    # Width is derived from depth*aspect_ratio, then rounded to a whole number
    # of heads so the profile knobs stay simple while tensor shapes stay legal.
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
        activation_checkpoint=activation_checkpoint,
        ve_gate_channels=ve_gate_channels,
        softcap=softcap,
    )


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    # Value embeddings are only enabled on alternating layers, matching the
    # original small-model recipe this project evolved from.
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, attention_op):
        super().__init__()
        self.attention_op = attention_op
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = config.ve_gate_channels
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, cos_sin: tuple[torch.Tensor, torch.Tensor], window_size: tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(bsz, seq_len, self.n_kv_head, self.head_dim).to(v.dtype)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1).to(v.dtype) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q)
        k = norm(k)
        attn_input_dtype = q.dtype
        if q.is_cuda and q.dtype not in (torch.float16, torch.bfloat16):
            q = q.bfloat16()
            k = k.bfloat16()
            v = v.bfloat16()

        y = self.attention_op(q, k, v, causal=True, window_size=window_size)
        if y.dtype != attn_input_dtype:
            y = y.to(attn_input_dtype)
        y = y.contiguous().view(bsz, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx, attention_op):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, attention_op)
        self.mlp = MLP(config)
        self.activation_checkpoint = config.activation_checkpoint

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, cos_sin: tuple[torch.Tensor, torch.Tensor], window_size: tuple[int, int]) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        mlp_input = norm(x)
        if self.activation_checkpoint == "mlp_only" and self.training and torch.is_grad_enabled():
            # Checkpointing only the MLP keeps the larger activation branch cheap
            # enough to fit wider profiles without recomputing attention.
            x = x + checkpoint(self.mlp, mlp_input, use_reentrant=False)
        else:
            x = x + self.mlp(mlp_input)
        return x


class GPT(nn.Module):
    def __init__(self, config, attention_op):
        super().__init__()
        self.config = config
        if config.activation_checkpoint not in {"none", "mlp_only"}:
            raise ValueError(f"Unsupported activation checkpoint mode: {config.activation_checkpoint}")
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i, attention_op) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale, scale)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -scale, scale)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # These embeddings stay in bf16 for the fast training path; eager
        # forwards are promoted back to fp32 before they hit fp32 linear layers.
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
        cos = cos.bfloat16()[None, :, None, :]
        sin = sin.bfloat16()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        # Repeat the user-facing pattern string across layers so profiles can
        # switch attention layouts with a short token like `SSSL` or `LLLL`.
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[layer_idx % len(pattern)]] for layer_idx in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        # This intentionally uses the same approximate token FLOP accounting for
        # every benchmark so MFU comparisons stay consistent across experiments.
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        num_heads = self.config.n_head
        head_dim = self.config.n_embd // self.config.n_head
        seq_len = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = seq_len if window < 0 else min(window, seq_len)
            attn_flops += 12 * num_heads * head_dim * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        *,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        optimizer_compile_backend="off",
        compile_mode="default",
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
        )

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon operates on same-shaped 2D matrices as grouped stacks; everything
        # else stays on AdamW where scalar and embedding behavior is simpler.
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )
        optimizer = MuonAdamW(
            param_groups,
            optimizer_compile_backend=optimizer_compile_backend,
            compile_mode=compile_mode,
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward_trunk(self, x: torch.Tensor, idx: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos_sin = cos, sin
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        return norm(x)

    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
        softcap = self.config.softcap
        if targets is not None:
            flat_hidden = x.reshape(-1, x.size(-1))
            flat_targets = targets.reshape(-1)
            logits = F.linear(flat_hidden, self.lm_head.weight).float()
            logits = torch.tanh(logits / softcap) * softcap
            return F.cross_entropy(
                logits,
                flat_targets,
                ignore_index=-1,
                reduction=reduction,
            )
        logits = F.linear(x, self.lm_head.weight).float()
        logits = torch.tanh(logits / softcap) * softcap
        return logits

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
        _, seq_len = idx.size()
        assert seq_len <= self.cos.size(1)
        cos = self.cos[:, :seq_len]
        sin = self.sin[:, :seq_len]
        x = self.transformer.wte(idx)
        x = norm(x)
        if not torch.is_autocast_enabled() and x.dtype != self.lm_head.weight.dtype:
            x = x.to(self.lm_head.weight.dtype)
        x = self.forward_trunk(x, idx, cos, sin)
        return self.compute_loss(x, targets=targets, reduction=reduction)
