from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict

import torch

from prepare import Tokenizer, evaluate_bpb, make_dataloader

from .compile import (
    AVAILABLE_INDUCTOR_MODES,
    maybe_compile_function,
    prepare_compile_environment,
    resolve_compile_backend,
    resolve_optimizer_compile_backend,
    validate_compile_backend,
    validate_compile_mode,
)
from .config import build_runtime_config, parse_args
from .model import (
    GPT,
    build_model_config,
    compute_mfu,
    estimate_device_peak_flops,
    norm,
    resolve_attention_backend,
    target_tok_per_sec_for_mfu,
)


H100_BF16_PEAK_FLOPS = 989.5e12


def _compile_status(runtime):
    if runtime.compile.model_backend == "off":
        return "torch.compile: disabled"
    if runtime.compile.model_backend == "inductor":
        return (
            f"torch.compile: enabled ({runtime.compile.model_backend}, "
            f"mode={runtime.compile.mode}, scope={runtime.compile.scope})"
        )
    return f"torch.compile: enabled ({runtime.compile.model_backend}, scope={runtime.compile.scope})"


def _build_execution_functions(raw_model, runtime, autocast_ctx, grad_accum_steps):
    model_for_execution = raw_model
    if runtime.compile.use_compiled_model:
        model_for_execution = maybe_compile_function(
            raw_model,
            backend=runtime.compile.model_backend,
            compile_mode=runtime.compile.mode,
            dynamic=False,
        )

    def run_microstep(x, y):
        with autocast_ctx:
            loss = model_for_execution(x, y)
        (loss / grad_accum_steps).backward()
        return loss.detach()

    def run_trunk_forward(x, idx, cos, sin):
        return raw_model.forward_trunk(x, idx, cos, sin)

    microstep_fn = run_microstep
    trunk_fn = run_trunk_forward
    if runtime.compile.use_compiled_microstep:
        microstep_fn = maybe_compile_function(
            run_microstep,
            backend=runtime.compile.model_backend,
            compile_mode=runtime.compile.mode,
            dynamic=False,
        )
    elif runtime.compile.use_compiled_trunk:
        # `trunk` scope keeps embedding/loss eager while compiling the deep stack
        # of transformer blocks, which is useful for isolating compile issues.
        trunk_fn = maybe_compile_function(
            run_trunk_forward,
            backend=runtime.compile.model_backend,
            compile_mode=runtime.compile.mode,
            dynamic=False,
        )

    model_for_eval = model_for_execution if runtime.compile.use_compiled_model else raw_model
    return model_for_execution, model_for_eval, microstep_fn, trunk_fn


def main() -> int:
    args = parse_args(AVAILABLE_INDUCTOR_MODES)
    model_compile_backend = resolve_compile_backend(args.compile_backend)
    validate_compile_backend(model_compile_backend)
    validate_compile_mode(model_compile_backend, args.compile_mode)
    optimizer_compile_backend = resolve_optimizer_compile_backend(args.optimizer_compile_backend, model_compile_backend)
    runtime = build_runtime_config(
        args,
        model_compile_backend=model_compile_backend,
        optimizer_compile_backend=optimizer_compile_backend,
    )
    compile_prep = prepare_compile_environment(
        model_backend=runtime.compile.model_backend,
        optimizer_backend=runtime.compile.optimizer_backend,
    )

    attention_backend, attention_op = resolve_attention_backend()

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_peak_flops = estimate_device_peak_flops(device_props)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = build_model_config(
        depth=runtime.model.depth,
        max_seq_len=runtime.model.max_seq_len,
        vocab_size=vocab_size,
        aspect_ratio=runtime.model.aspect_ratio,
        head_dim=runtime.model.head_dim,
        window_pattern=runtime.model.window_pattern,
        activation_checkpoint=runtime.model.activation_checkpoint,
        ve_gate_channels=runtime.model.ve_gate_channels,
        softcap=runtime.model.softcap,
    )

    print(f"Vocab size: {vocab_size:,}")
    print(f"Attention backend: {attention_backend}")
    print(f"Experiment profile: {runtime.experiment_profile}")
    print(f"Device batch size: {runtime.model.device_batch_size}")
    print(f"Mode: {'benchmark' if runtime.benchmark.enabled else 'train'}")
    print(f"Compile backend: {runtime.compile.model_backend}")
    if runtime.compile.model_backend == "inductor":
        print(f"Compile mode: {runtime.compile.mode}")
    print(f"Compile scope: {runtime.compile.scope}")
    print(f"Optimizer compile backend: {runtime.compile.optimizer_backend}")
    if compile_prep.msvc_cl_path is not None:
        print(f"MSVC cl.exe: {compile_prep.msvc_cl_path}")
    if compile_prep.msvc_help_encoding is not None:
        print(f"MSVC help decode override: {compile_prep.msvc_help_encoding}")
    if runtime.benchmark.enabled:
        print(f"Benchmark steps: {runtime.benchmark.steps}")
        print(f"Benchmark warmup steps: {runtime.benchmark.warmup_steps}")
    print(f"Log interval: {runtime.benchmark.log_interval}")
    if device_peak_flops is not None:
        print(f"Device peak BF16/FP16 TFLOPS est: {device_peak_flops / 1e12:.2f}")
    print(f"Model config: {asdict(config)}")

    with torch.device("meta"):
        raw_model = GPT(config, attention_op)
    raw_model.to_empty(device=device)
    raw_model.init_weights()

    param_counts = raw_model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = raw_model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = runtime.model.device_batch_size * runtime.model.max_seq_len
    total_batch_size = runtime.optimization.total_batch_size
    if runtime.grad_accum_steps_override > 0:
        grad_accum_steps = runtime.grad_accum_steps_override
        total_batch_size = tokens_per_fwdbwd * grad_accum_steps
    else:
        assert total_batch_size % tokens_per_fwdbwd == 0
        grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    optimizer = raw_model.setup_optimizer(
        unembedding_lr=runtime.optimization.unembedding_lr,
        embedding_lr=runtime.optimization.embedding_lr,
        scalar_lr=runtime.optimization.scalar_lr,
        adam_betas=runtime.optimization.adam_betas,
        matrix_lr=runtime.optimization.matrix_lr,
        weight_decay=runtime.optimization.weight_decay,
        optimizer_compile_backend=runtime.compile.optimizer_backend,
        compile_mode=runtime.compile.mode,
    )

    model_for_execution, model_for_eval, microstep_fn, trunk_fn = _build_execution_functions(
        raw_model,
        runtime,
        autocast_ctx,
        grad_accum_steps,
    )
    print(_compile_status(runtime))

    train_loader = make_dataloader(tokenizer, runtime.model.device_batch_size, runtime.model.max_seq_len, "train")
    x, y, epoch = next(train_loader)

    print(f"Time budget: {runtime.time_budget}s")
    print(f"Total batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    def get_lr_multiplier(progress):
        if progress < runtime.optimization.warmup_ratio:
            return progress / runtime.optimization.warmup_ratio if runtime.optimization.warmup_ratio > 0 else 1.0
        if progress < 1.0 - runtime.optimization.warmdown_ratio:
            return 1.0
        cooldown = (1.0 - progress) / runtime.optimization.warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * runtime.optimization.final_lr_frac

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return runtime.optimization.weight_decay * (1 - progress)

    t_start_training = time.time()
    smooth_train_loss = 0.0
    elapsed_training_time = 0.0
    steady_training_time = 0.0
    step = 0
    measured_steps = 0
    step_timer_start = torch.cuda.Event(enable_timing=True)
    step_timer_end = torch.cuda.Event(enable_timing=True)
    target_tok_per_sec = target_tok_per_sec_for_mfu(
        device_peak_flops,
        num_flops_per_token,
        runtime.benchmark.target_mfu_percent,
    )

    while True:
        step_timer_start.record()
        loss_value = None
        for _ in range(grad_accum_steps):
            if runtime.compile.use_compiled_execution:
                # Tell cudagraph-backed modes that each microstep is a fresh replay
                # boundary; without this, tensor lifetime reuse becomes fragile.
                torch.compiler.cudagraph_mark_step_begin()
            if runtime.compile.scope == "microstep":
                train_loss = microstep_fn(x, y)
                loss_value = train_loss.float()
            elif runtime.compile.scope == "trunk":
                with autocast_ctx:
                    cos = raw_model.cos[:, : x.size(1)]
                    sin = raw_model.sin[:, : x.size(1)]
                    hidden = raw_model.transformer.wte(x)
                    hidden = norm(hidden)
                    hidden = trunk_fn(hidden, x, cos, sin)
                    loss = raw_model.compute_loss(hidden, targets=y)
                loss_value = loss.detach().float()
                (loss / grad_accum_steps).backward()
            else:
                with autocast_ctx:
                    loss = model_for_execution(x, y)
                loss_value = loss.detach().float()
                (loss / grad_accum_steps).backward()
            train_loss_f = loss_value.item()
            if not math.isfinite(train_loss_f) or train_loss_f > 100:
                raw_model.zero_grad(set_to_none=True)
                print(f"FAIL: invalid loss {train_loss_f}", flush=True)
                return 1
            x, y, epoch = next(train_loader)

        progress = min(elapsed_training_time / runtime.time_budget, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        raw_model.zero_grad(set_to_none=True)

        step_timer_end.record()
        step_timer_end.synchronize()
        dt = step_timer_start.elapsed_time(step_timer_end) / 1000.0
        elapsed_training_time += dt
        is_benchmark_warmup = runtime.benchmark.enabled and step < runtime.benchmark.warmup_steps

        if (runtime.benchmark.enabled and not is_benchmark_warmup) or (not runtime.benchmark.enabled and step > 10):
            # Exclude cold-start compile steps from benchmark metrics, and give
            # full training runs a short settle period before averaging throughput.
            steady_training_time += dt
            measured_steps += 1

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * min(elapsed_training_time / runtime.time_budget, 1.0)
        tok_per_sec = total_batch_size / dt
        if is_benchmark_warmup:
            tok_per_sec_str = "warmup"
            device_mfu_str = "warmup"
            h100_mfu_str = "warmup"
        else:
            device_mfu = compute_mfu(device_peak_flops, num_flops_per_token, tok_per_sec)
            h100_mfu = compute_mfu(H100_BF16_PEAK_FLOPS, num_flops_per_token, tok_per_sec)
            tok_per_sec_str = f"{tok_per_sec:,.0f}"
            device_mfu_str = f"{device_mfu:.1f}%" if device_mfu is not None else "n/a"
            h100_mfu_str = f"{h100_mfu:.1f}%" if h100_mfu is not None else "n/a"

        if runtime.benchmark.enabled:
            remaining_str = f"{max(runtime.benchmark.steps - (step + 1), 0)} steps"
            progress_str = f"{100 * (step + 1) / runtime.benchmark.steps:.1f}%"
        else:
            remaining_str = f"{max(0.0, runtime.time_budget - elapsed_training_time):.0f}s"
            progress_str = f"{pct_done:.1f}%"

        should_log = (
            step == 0
            or is_benchmark_warmup
            or ((step + 1) % runtime.benchmark.log_interval == 0)
            or (runtime.benchmark.enabled and step + 1 >= runtime.benchmark.steps)
        )
        if should_log:
            print(
                f"step {step:05d} ({progress_str}) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
                f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec_str} | mfu: {device_mfu_str} | "
                f"h100_mfu: {h100_mfu_str} | epoch: {epoch} | remaining: {remaining_str}",
                flush=True,
            )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if runtime.benchmark.enabled and step >= runtime.benchmark.steps:
            break
        if not runtime.benchmark.enabled and elapsed_training_time >= runtime.time_budget:
            break

    total_tokens = step * total_batch_size

    val_bpb = None
    if not runtime.benchmark.enabled:
        model_for_eval.eval()
        with autocast_ctx:
            val_bpb = evaluate_bpb(model_for_eval, tokenizer, runtime.model.device_batch_size)

    t_end = time.time()
    startup_time = t_start_training - t_start
    steady_tok_per_sec = (
        total_batch_size * measured_steps / steady_training_time
        if steady_training_time > 0 and measured_steps > 0
        else None
    )
    # All summary throughput/MFU numbers are reported on the measured steady-state
    # window, not on the full wall-clock run including compile/startup overhead.
    steady_train_flops = num_flops_per_token * steady_tok_per_sec if steady_tok_per_sec is not None else None
    steady_state_mfu = compute_mfu(device_peak_flops, num_flops_per_token, steady_tok_per_sec)
    steady_state_h100_mfu = compute_mfu(H100_BF16_PEAK_FLOPS, num_flops_per_token, steady_tok_per_sec)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    mfu50_gap_percent = (
        runtime.benchmark.target_mfu_percent - steady_state_mfu if steady_state_mfu is not None else None
    )

    print("---")
    print(f"mode:             {'benchmark' if runtime.benchmark.enabled else 'train'}")
    print(f"profile:          {runtime.experiment_profile}")
    print(f"compile_backend:  {runtime.compile.model_backend}")
    print(
        f"compile_mode:     {runtime.compile.mode}"
        if runtime.compile.model_backend == "inductor"
        else "compile_mode:     n/a"
    )
    print(f"compile_scope:    {runtime.compile.scope}")
    print(f"optimizer_compile:{runtime.compile.optimizer_backend}")
    if val_bpb is not None:
        print(f"val_bpb:          {val_bpb:.6f}")
    else:
        print("val_bpb:          skipped")
    print(f"training_seconds: {elapsed_training_time:.1f}")
    print(f"steady_training_seconds: {steady_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"startup_seconds:  {startup_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"tok_per_sec:      {steady_tok_per_sec:,.0f}" if steady_tok_per_sec is not None else "tok_per_sec:      n/a")
    print(
        f"target_tok_per_sec_50_mfu: {target_tok_per_sec:,.0f}"
        if target_tok_per_sec is not None
        else "target_tok_per_sec_50_mfu: n/a"
    )
    print(f"train_tflops:     {steady_train_flops / 1e12:.2f}" if steady_train_flops is not None else "train_tflops:     n/a")
    print(f"mfu_percent:      {steady_state_mfu:.2f}" if steady_state_mfu is not None else "mfu_percent:      n/a")
    print(f"mfu50_gap_percent:{mfu50_gap_percent:.2f}" if mfu50_gap_percent is not None else "mfu50_gap_percent:n/a")
    print(
        f"h100_mfu_percent: {steady_state_h100_mfu:.2f}"
        if steady_state_h100_mfu is not None
        else "h100_mfu_percent: n/a"
    )
    print(f"peak_tflops_est:  {device_peak_flops / 1e12:.2f}" if device_peak_flops is not None else "peak_tflops_est:  n/a")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"measured_steps:   {measured_steps}")
    if runtime.benchmark.enabled and measured_steps == 0:
        print(f"benchmark_note:   use --benchmark-steps > {runtime.benchmark.warmup_steps} for steady-state throughput")
    elif runtime.benchmark.enabled:
        print(f"benchmark_note:   excluded first {runtime.benchmark.warmup_steps} warmup steps from steady-state metrics")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {runtime.model.depth}")
    return 0
