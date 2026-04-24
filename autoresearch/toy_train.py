"""Tiny CPU-friendly training loop for smoke-tests and CI.

Trains the :class:`GPT` in ``autoresearch.model`` on synthetic copy-style data
for a handful of steps. The goal is NOT research-grade loss — it is to prove
forward+backward works and that loss actually decreases.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .config import Config
from .model import GPT, GPTConfig
from .metrics import write_metrics


def _make_fixed_corpus(cfg: Config, generator: torch.Generator) -> torch.Tensor:
    """Build a single fixed token sequence the model can memorize.

    We sample one sequence of length ``block_size + 1`` once and reuse it on
    every step. This guarantees there is signal to learn (the model just has
    to memorize it), so loss provably decreases across a handful of steps —
    which is what the smoke-test asserts.
    """
    return torch.randint(
        0, cfg.vocab_size, (cfg.block_size + 1,), generator=generator,
    )


def _synthetic_batch(corpus: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq = corpus.unsqueeze(0).expand(batch_size, -1)
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def toy_train(
    cfg: Optional[Config] = None,
    *,
    metrics_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Run a tiny training loop and return a metrics dict.

    Writes a ``run_metrics.json`` sidecar at ``cfg.metrics_path`` (or the
    override passed in). Returns the same dict.
    """
    if cfg is None:
        cfg = Config()
    cfg.validate()

    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg.device='cuda' but CUDA is not available")

    model = GPT(GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    corpus = _make_fixed_corpus(cfg, gen).to(device)
    losses: List[float] = []
    model.train()
    for step in range(cfg.max_steps):
        x, y = _synthetic_batch(corpus, cfg.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if verbose:
            print(f"step {step:03d} | loss {losses[-1]:.4f}")

    metrics: Dict[str, object] = {
        "version": "1.0.0",
        "num_params": model.num_params(),
        "num_steps": cfg.max_steps,
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "config": cfg.to_dict(),
        "device": cfg.device,
    }
    out_path = metrics_path or cfg.metrics_path
    try:
        write_metrics(out_path, metrics)
        metrics["metrics_path"] = str(out_path)
    except OSError:
        # Metrics sidecar is best-effort; don't fail training because we
        # couldn't write to disk (e.g. sandboxed CI).
        metrics["metrics_path"] = None
    return metrics
