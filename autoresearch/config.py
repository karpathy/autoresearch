"""Simple dataclass-based config loader for autoresearch runs.

This is intentionally tiny: a :class:`Config` holds the few knobs the CPU-tier
toy-train path actually reads, and :func:`load_config` accepts either a JSON
file path or a dict and fills in defaults/validates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Mapping, Union


@dataclass
class Config:
    # Model
    vocab_size: int = 65          # matches Karpathy's tinyshakespeare char set; overridden by tokenizer
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 64          # context window
    dropout: float = 0.0

    # Training
    batch_size: int = 8
    max_steps: int = 20
    learning_rate: float = 3e-3
    seed: int = 1337
    device: str = "cpu"           # "cpu" | "cuda"

    # Metrics sidecar (preferred over stdout regex parsing)
    metrics_path: str = "run_metrics.json"

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        for name in ("n_layer", "n_head", "n_embd", "block_size", "batch_size", "max_steps", "vocab_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(source: Union[str, Path, Mapping[str, Any], None] = None) -> Config:
    """Load a :class:`Config`.

    * ``None``  -> all defaults.
    * ``Mapping`` -> overrides applied on top of defaults.
    * ``str``/``Path`` -> JSON file; keys not present in Config are ignored
      with no error (forward-compat).
    """
    data: dict = {}
    if source is None:
        pass
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(source, Mapping):
        data = dict(source)
    else:
        raise TypeError(f"unsupported config source: {type(source).__name__}")

    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in data.items() if k in known})
    cfg.validate()
    return cfg
