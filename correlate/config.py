"""
Configuration for RiskWise correlation analysis pipeline.

Set via environment variables or pass directly to AnalysisConfig.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DBConfig:
    """Postgres connection configuration."""
    host: str = os.getenv("RISKWISE_DB_HOST", "localhost")
    port: int = int(os.getenv("RISKWISE_DB_PORT", "5432"))
    database: str = os.getenv("RISKWISE_DB_NAME", "riskwise")
    user: str = os.getenv("RISKWISE_DB_USER", "riskwise")
    password: str = os.getenv("RISKWISE_DB_PASSWORD", "")
    schema: str = os.getenv("RISKWISE_DB_SCHEMA", "public")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class AnalysisConfig:
    """Parameters controlling the correlation analysis."""

    # Target company/industry
    target_ticker: str = "PEP"          # Stock ticker symbol
    target_name: str = "PepsiCo"        # Human-readable name

    # Database
    db: DBConfig = field(default_factory=DBConfig)

    # Index table discovery — these are SQL patterns the agent can override
    # The pipeline will auto-discover tables matching these patterns
    index_table_pattern: str = os.getenv("RISKWISE_INDEX_TABLE", "rw_indices")
    index_value_column: str = "value"
    index_date_column: str = "date"
    index_name_column: str = "index_name"

    # Analysis parameters
    max_lag_days: int = 90              # Maximum lead/lag window to test
    lag_step_days: int = 1              # Granularity of lag search
    min_overlap_days: int = 60          # Minimum overlapping data points required
    significance_level: float = 0.05    # p-value threshold for statistical tests
    rolling_window_days: int = 60       # Window for rolling correlation
    granger_max_lags: int = 15          # Max lags for Granger causality test
    n_bootstrap: int = 1000             # Bootstrap iterations for confidence intervals

    # Output
    output_dir: str = "reports"
    report_format: str = "markdown"     # "markdown" or "html"

    # Market data
    market_data_period: str = "5y"      # How far back to fetch stock data
    price_column: str = "Close"         # Which price series to use
    use_returns: bool = True            # Analyze log returns instead of raw prices

    # Filtering
    min_abs_correlation: float = 0.15   # Min |r| to include in report
    top_n_indices: int = 20             # Max indices to highlight in exec summary
