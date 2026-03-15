from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


MODEL_COMPILE_BACKEND_CHOICES = ["auto", "off", "inductor", "cudagraphs", "aot_eager"]
OPTIMIZER_COMPILE_BACKEND_CHOICES = ["auto", "off", "inductor"]
COMPILE_SCOPE_CHOICES = ["model", "microstep", "trunk"]


@dataclass(frozen=True)
class ExperimentProfile:
    aspect_ratio: int
    window_pattern: str
    max_seq_len: int
    depth: int
    device_batch_size: int
    activation_checkpoint: str
    log_interval: int
    benchmark_warmup_steps: int


@dataclass(frozen=True)
class ModelSettings:
    aspect_ratio: int
    head_dim: int
    window_pattern: str
    max_seq_len: int
    depth: int
    activation_checkpoint: str
    default_device_batch_size: int
    device_batch_size: int
    vocab_size: int
    ve_gate_channels: int
    softcap: float


@dataclass(frozen=True)
class CompileSettings:
    model_backend: str
    mode: str
    scope: str
    optimizer_backend: str
    use_compiled_execution: bool
    use_compiled_model: bool
    use_compiled_microstep: bool
    use_compiled_trunk: bool


@dataclass(frozen=True)
class BenchmarkSettings:
    steps: int
    enabled: bool
    warmup_steps: int
    log_interval: int
    target_mfu_percent: float


@dataclass(frozen=True)
class OptimizationSettings:
    embedding_lr: float
    unembedding_lr: float
    matrix_lr: float
    scalar_lr: float
    weight_decay: float
    adam_betas: tuple[float, float]
    warmup_ratio: float
    warmdown_ratio: float
    final_lr_frac: float
    muon_warmup_steps: int


@dataclass(frozen=True)
class RuntimeConfig:
    experiment_profile: str
    time_budget: int
    grad_accum_steps_override: int
    seed: int
    model: ModelSettings
    compile: CompileSettings
    benchmark: BenchmarkSettings
    optimization: OptimizationSettings


THROUGHPUT_PROFILE = ExperimentProfile(
    aspect_ratio=64,
    window_pattern="SSSL",
    max_seq_len=2048,
    depth=8,
    device_batch_size=12,
    activation_checkpoint="none",
    log_interval=1,
    benchmark_warmup_steps=2,
)

MFU50_PROFILE = ExperimentProfile(
    aspect_ratio=85,
    window_pattern="LLLL",
    max_seq_len=4096,
    depth=9,
    device_batch_size=5,
    activation_checkpoint="mlp_only",
    log_interval=10,
    benchmark_warmup_steps=2,
)

EXPERIMENT_PROFILES = {
    "baseline": MFU50_PROFILE,
    "default": MFU50_PROFILE,
    "mfu50": MFU50_PROFILE,
    "throughput": THROUGHPUT_PROFILE,
}

HEAD_DIM = 128
TIME_BUDGET_DEFAULT = 300
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
MUON_WARMUP_STEPS = 300
TARGET_MFU_PERCENT = 50.0
VE_GATE_CHANNELS = 32
SOFTCAP = 15.0
SEED_DEFAULT = 1337


def env_override_int(name: str, default: int) -> int:
    override = os.environ.get(name)
    return int(override) if override is not None else default


def env_override_str(name: str, default: str) -> str:
    override = os.environ.get(name)
    return override if override is not None else default


def env_override_float(name: str, default: float) -> float:
    override = os.environ.get(name)
    return float(override) if override is not None else default


def pick_device_batch_size(default_batch_size: int) -> int:
    return env_override_int("DEVICE_BATCH_SIZE", default_batch_size)


def parse_args(available_inductor_modes: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch pretraining script")
    parser.add_argument(
        "--experiment-profile",
        choices=sorted(EXPERIMENT_PROFILES.keys()),
        default="baseline",
        help="Select the deeper near-50 percent MFU baseline or the preserved throughput-oriented profile.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="Run N optimizer steps, report throughput/MFU, and skip final eval.",
    )
    parser.add_argument(
        "--compile-backend",
        choices=MODEL_COMPILE_BACKEND_CHOICES,
        default="inductor",
        help="Model compile backend. Default is the validated high-throughput inductor path.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=available_inductor_modes,
        default="default",
        help="TorchInductor compile mode. Used only when --compile-backend is inductor.",
    )
    parser.add_argument(
        "--compile-scope",
        choices=COMPILE_SCOPE_CHOICES,
        default="model",
        help="Compile only the model forward, or compile a whole microstep (forward + backward).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps per optimizer step. This is the effective batch-size knob beyond DEVICE_BATCH_SIZE x MAX_SEQ_LEN.",
    )
    parser.add_argument(
        "--optimizer-compile-backend",
        choices=OPTIMIZER_COMPILE_BACKEND_CHOICES,
        default="inductor",
        help="Control optimizer fused-step compilation separately from model compilation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Random seed for model initialization and any stochastic runtime behavior.",
    )
    args = parser.parse_args()
    if args.benchmark_steps < 0:
        parser.error("--benchmark-steps must be >= 0")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be >= 1")
    return args


def build_runtime_config(
    args: argparse.Namespace,
    *,
    model_compile_backend: str,
    optimizer_compile_backend: str,
    vocab_size: int,
) -> RuntimeConfig:
    profile = EXPERIMENT_PROFILES[args.experiment_profile]
    model_settings = ModelSettings(
        aspect_ratio=env_override_int("ASPECT_RATIO", profile.aspect_ratio),
        head_dim=HEAD_DIM,
        window_pattern=env_override_str("WINDOW_PATTERN", profile.window_pattern),
        max_seq_len=env_override_int("MAX_SEQ_LEN", profile.max_seq_len),
        depth=env_override_int("DEPTH", profile.depth),
        activation_checkpoint=env_override_str("ACTIVATION_CHECKPOINT", profile.activation_checkpoint),
        default_device_batch_size=profile.device_batch_size,
        device_batch_size=pick_device_batch_size(profile.device_batch_size),
        vocab_size=vocab_size,
        ve_gate_channels=env_override_int("VE_GATE_CHANNELS", VE_GATE_CHANNELS),
        softcap=env_override_float("SOFTCAP", SOFTCAP),
    )
    compile_settings = CompileSettings(
        model_backend=model_compile_backend,
        mode=args.compile_mode,
        scope=args.compile_scope,
        optimizer_backend=optimizer_compile_backend,
        use_compiled_execution=model_compile_backend != "off",
        use_compiled_model=model_compile_backend != "off" and args.compile_scope == "model",
        use_compiled_microstep=model_compile_backend != "off" and args.compile_scope == "microstep",
        use_compiled_trunk=model_compile_backend != "off" and args.compile_scope == "trunk",
    )
    benchmark_settings = BenchmarkSettings(
        steps=args.benchmark_steps,
        enabled=args.benchmark_steps > 0,
        warmup_steps=env_override_int("BENCHMARK_WARMUP_STEPS", profile.benchmark_warmup_steps),
        log_interval=env_override_int("LOG_INTERVAL", profile.log_interval),
        target_mfu_percent=TARGET_MFU_PERCENT,
    )
    optimization_settings = OptimizationSettings(
        embedding_lr=EMBEDDING_LR,
        unembedding_lr=UNEMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        scalar_lr=SCALAR_LR,
        weight_decay=WEIGHT_DECAY,
        adam_betas=ADAM_BETAS,
        warmup_ratio=WARMUP_RATIO,
        warmdown_ratio=WARMDOWN_RATIO,
        final_lr_frac=FINAL_LR_FRAC,
        muon_warmup_steps=env_override_int("MUON_WARMUP_STEPS", MUON_WARMUP_STEPS),
    )
    return RuntimeConfig(
        experiment_profile=args.experiment_profile,
        time_budget=env_override_int("TIME_BUDGET", TIME_BUDGET_DEFAULT),
        grad_accum_steps_override=args.grad_accum_steps,
        seed=env_override_int("SEED", args.seed),
        model=model_settings,
        compile=compile_settings,
        benchmark=benchmark_settings,
        optimization=optimization_settings,
    )
