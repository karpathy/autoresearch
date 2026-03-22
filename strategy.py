"""
BTC 15-minute direction prediction strategy for Kalshi KXBTC15M contracts.

THIS IS THE ONLY FILE THE AGENT MODIFIES.
"""

import numpy as np
import pandas as pd


class Strategy:

    def on_bar(self, window: pd.DataFrame, context: dict) -> tuple[float, float]:
        minute = context["window_minute"]
        fair = context["fair_price"]

        # Only trade at minutes 0-2: maximum uncertainty = maximum potential edge
        if minute > 2:
            return (fair, 1.0)  # match fair price with impossible threshold = no trade

        close_arr = window["close"].values
        vol_20 = window["volatility_20"].values[-1]

        # --- Signal: Recent price trend over last 15 bars (previous window) ---
        # Autocorrelation: BTC tends to continue its trend over short horizons
        ret_15 = (close_arr[-1] - close_arr[-16]) / close_arr[-16] if len(close_arr) >= 16 else 0.0

        # Normalize by volatility to get a z-score-like measure
        sigma = vol_20 * np.sqrt(15) + 1e-10
        z = ret_15 / sigma

        # Convert z-score to probability with moderate sensitivity
        # Positive z → price trending up → P(up) > 0.5
        probability = 0.5 + np.clip(z * 0.08, -0.35, 0.35)

        # Require substantial edge to trade
        edge_threshold = 0.15

        return (probability, edge_threshold)
