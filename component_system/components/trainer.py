from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.utils as nn_utils

from prepare import MAX_SEQ_LEN, TIME_BUDGET, evaluate_bpb, make_dataloader


H100_BF16_PEAK_FLOPS = 989.5e12


@dataclass
class TrainingSettings:
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"
    total_batch_size: int = 2**19
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple[float, float] = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    max_grad_norm: float = 1.0
    final_lr_frac: float = 0.0
    depth: int = 8
    device_batch_size: int = 32  # 24GB vram
    seed: int = 42
    compile_model: bool = True


def default_training_settings() -> TrainingSettings:
    return TrainingSettings()


def get_lr_multiplier(progress: float, settings: TrainingSettings) -> float:
    if progress < settings.warmup_ratio:
        return progress / settings.warmup_ratio if settings.warmup_ratio > 0 else 1.0
    if progress < 1.0 - settings.warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / settings.warmdown_ratio
    return cooldown + (1 - cooldown) * settings.final_lr_frac


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float, settings: TrainingSettings) -> float:
    return settings.weight_decay * (1 - progress)


def run_training_session(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    settings: TrainingSettings,
    param_counts: dict[str, int],
    num_flops_per_token: float,
    baseline_binding: dict[str, Any],
) -> dict[str, Any]:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16)

    tokens_per_fwdbwd = settings.device_batch_size * MAX_SEQ_LEN
    assert settings.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = settings.total_batch_size // tokens_per_fwdbwd
    train_loader = make_dataloader(
        tokenizer, settings.device_batch_size, MAX_SEQ_LEN, "train"
    )
    x, y, epoch = next(train_loader)

    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print("Training session started")

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0

    while True:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        t0 = time.time()
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress, settings)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress, settings)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

        # Gradient clipping for training stability
        if settings.max_grad_norm > 0:
            nn_utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)

        optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        if train_loss_f > 100:
            raise RuntimeError(
                "Training aborted because loss exceeded the fast-fail threshold."
            )

        torch.cuda.synchronize(device=device)
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(settings.total_batch_size / dt)
        mfu = (
            100
            * num_flops_per_token
            * settings.total_batch_size
            / dt
            / H100_BF16_PEAK_FLOPS
        )
        remaining = max(0.0, TIME_BUDGET - total_training_time)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

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
    total_tokens = step * settings.total_batch_size
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, settings.device_batch_size)

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    steady_state_mfu = (
        100
        * num_flops_per_token
        * settings.total_batch_size
        * (step - 10)
        / total_training_time
        / H100_BF16_PEAK_FLOPS
        if total_training_time > 0
        else 0.0
    )
    num_params = param_counts["total"]
    metrics = {
        "val_bpb": float(val_bpb),
        "training_seconds": float(total_training_time),
        "total_seconds": float(t_end - t_start),
        "peak_vram_mb": float(peak_vram_mb),
        "mfu_percent": float(steady_state_mfu),
        "total_tokens_M": float(total_tokens / 1e6),
        "num_steps": int(step),
        "num_params_M": float(num_params / 1e6),
        "depth": int(settings.depth),
        "startup_seconds": float(t_start_training - t_start),
    }

    print("---")
    print(f"val_bpb:          {metrics['val_bpb']:.6f}")
    print(f"training_seconds: {metrics['training_seconds']:.1f}")
    print(f"total_seconds:    {metrics['total_seconds']:.1f}")
    print(f"peak_vram_mb:     {metrics['peak_vram_mb']:.1f}")
    print(f"mfu_percent:      {metrics['mfu_percent']:.2f}")
    print(f"total_tokens_M:   {metrics['total_tokens_M']:.1f}")
    print(f"num_steps:        {metrics['num_steps']}")
    print(f"num_params_M:     {metrics['num_params_M']:.1f}")
    print(f"depth:            {metrics['depth']}")
    return metrics
