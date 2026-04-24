"""autoresearch command-line interface.

Subcommands:

* ``toy-train`` — tiny nanoGPT on synthetic data (CPU-friendly).
* ``info``      — environment + dependency availability.
* ``config``    — pretty-print the loaded config.

All subcommands are designed to work WITHOUT the Railway P2PCLAW API and
WITHOUT an H100 — so they're safe to run in CI / on laptops.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from typing import List, Optional

from . import __version__
from .config import Config, load_config


def _cmd_info(_: argparse.Namespace) -> int:
    info = {
        "autoresearch_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("torch", "numpy", "requests", "transformers", "datasets", "wandb"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[name] = None

    # CUDA status via torch if available
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0

    info["tribunal_implemented"] = False
    info["silicon_integration"] = "partial (inert placeholder)"

    print(json.dumps(info, indent=2))
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.file) if args.file else Config()
    print(json.dumps(asdict(cfg), indent=2))
    return 0


def _cmd_toy_train(args: argparse.Namespace) -> int:
    from .toy_train import toy_train  # lazy: keep --help fast & torch-free

    cfg = load_config(args.config) if args.config else Config()
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.device is not None:
        cfg.device = args.device
    cfg.validate()

    metrics = toy_train(cfg, verbose=args.verbose)
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "losses"},
        indent=2,
    ))
    if metrics.get("loss_first") is not None and metrics.get("loss_last") is not None:
        delta = metrics["loss_first"] - metrics["loss_last"]
        print(f"loss delta (first - last): {delta:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoresearch",
        description="nanoGPT-based autonomous ML research loop (research prototype).",
    )
    p.add_argument("--version", action="version", version=f"autoresearch {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show environment + dependency status.")
    p_info.set_defaults(func=_cmd_info)

    p_cfg = sub.add_parser("config", help="Show the loaded config (defaults if no file).")
    p_cfg.add_argument("-f", "--file", help="Path to a config JSON file.")
    p_cfg.set_defaults(func=_cmd_config)

    p_toy = sub.add_parser("toy-train", help="Run a tiny CPU-friendly nanoGPT training loop.")
    p_toy.add_argument("-c", "--config", help="Path to a config JSON file.")
    p_toy.add_argument("--steps", type=int, help="Override max_steps.")
    p_toy.add_argument("--device", choices=("cpu", "cuda"), help="Override device.")
    p_toy.add_argument("--verbose", action="store_true", help="Print per-step loss.")
    p_toy.set_defaults(func=_cmd_toy_train)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
