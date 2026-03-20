"""Configuration dataclasses for the evaluation infrastructure.

These dataclasses parameterize the walk-forward evaluation, backtesting,
and epoch management. Default values match the current BTC/USD hourly
pipeline so that AssetConfig() produces identical behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters for the backtesting engine."""
    sigma_threshold: float = 0.20        # min |prediction| to open a position
    sigma_full_position: float = 0.50    # prediction level for 100% position
    fee_rate: float = 0.001              # per-trade fee (one side)
    slippage_rate: float = 0.0005        # per-trade slippage estimate


@dataclass(frozen=True)
class WalkForwardConfig:
    """Parameters for walk-forward evaluation."""
    forward_hours: int = 24              # prediction horizon
    time_budget: int = 240               # max seconds for training per window
    epoch_length: int = 30               # evaluations per epoch before holdout rotates
    # Holdout health thresholds: Sharpe >= ok_threshold → OK,
    # >= warn_threshold → CAUTION, below → WARN
    holdout_ok_threshold: float = 0.0
    holdout_warn_threshold: float = -1.0


@dataclass(frozen=True)
class AssetConfig:
    """Complete configuration for evaluating an asset."""
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    eval_count_path: Path = field(default_factory=lambda: Path.home() / ".cache" / "autotrader" / "eval_count")
    salt_env_var: str = "AUTOTRADER_HOLDOUT_SALT"
