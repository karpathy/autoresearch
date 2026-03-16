"""
GPT-2 architecture variant.

Standard GPT-2: learned positional embeddings, standard multi-head attention
(no GQA, no value embeddings), GELU activation, LayerNorm, no logit
softcapping. Uses a simple AdamW optimizer (no Muon).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    sequence_len: int = 1024
    vocab_size: int = 32768
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # Fused QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, n_head, T, head_dim) for SDPA
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """Standard GPT-2 architecture."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.sequence_len, config.n_embd),
            "h": nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        # Standard GPT-2 initialization
        nn.init.normal_(self.transformer.wte.weight, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, std=0.01)
        for block in self.transformer.h:
            nn.init.normal_(block.attn.c_attn.weight, std=0.02)
            nn.init.zeros_(block.attn.c_attn.bias)
            nn.init.normal_(block.attn.c_proj.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))
            nn.init.zeros_(block.attn.c_proj.bias)
            nn.init.normal_(block.mlp.c_fc.weight, std=0.02)
            nn.init.zeros_(block.mlp.c_fc.bias)
            nn.init.normal_(block.mlp.c_proj.weight, std=0.02 / math.sqrt(2 * self.config.n_layer))
            nn.init.zeros_(block.mlp.c_proj.bias)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def estimate_flops(self) -> int:
        """Estimated FLOPs per token (forward + backward, 6x params heuristic)."""
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude positional embeddings (not in standard scaling law params)
        nparams_exclude = self.transformer.wpe.weight.numel()
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        # Full-context attention per layer
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self) -> dict:
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_blocks = sum(p.numel() for p in self.transformer.h.parameters())
        ln_f = sum(p.numel() for p in self.transformer.ln_f.parameters())
        total = wte + wpe + lm_head + transformer_blocks + ln_f
        return {
            'wte': wte, 'wpe': wpe, 'lm_head': lm_head,
            'transformer_blocks': transformer_blocks, 'ln_f': ln_f, 'total': total,
        }

    def setup_optimizer(self, lr: float = 3e-4, weight_decay: float = 0.1,
                        betas: tuple = (0.9, 0.95), **kwargs) -> torch.optim.Optimizer:
        """Simple AdamW optimizer (no Muon — GPT-2 does not use Muon)."""
        # Apply weight decay only to 2D weight tensors, not biases or norms
        decay_params = [p for p in self.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.dim() < 2]
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return optimizer

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                reduction: str = 'mean') -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.config.sequence_len, (
            f"Sequence length {T} exceeds model max {self.config.sequence_len}"
        )
        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).float()

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits
