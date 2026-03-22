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

        # Only trade at minutes 0-3
        if minute > 3:
            return (fair, 1.0)

        latest = window.iloc[-1]
        close_arr = window["close"].values

        # --- Signal 1: Mean reversion (contrarian) ---
        # If price overshot recently, expect reversion toward mean
        ret_15 = (close_arr[-1] - close_arr[-16]) / close_arr[-16]
        vol_20 = latest["volatility_20"]
        sigma = vol_20 * np.sqrt(15) + 1e-10
        z_mom = ret_15 / sigma

        # FLIP: negative z (price dropped) → expect reversion UP
        mean_rev_signal = 0.5 - np.clip(z_mom * 0.10, -0.3, 0.3)

        # --- Signal 2: Bollinger Band position ---
        bb_upper = latest["bbands_upper"]
        bb_lower = latest["bbands_lower"]
        bb_range = bb_upper - bb_lower + 1e-10
        bb_pos = (close_arr[-1] - bb_lower) / bb_range  # 0 = at lower band, 1 = at upper

        # Near lower band → expect bounce UP, near upper band → expect drop
        bb_signal = 0.5 + (0.5 - bb_pos) * 0.25

        # --- Ensemble ---
        probability = 0.55 * mean_rev_signal + 0.45 * bb_signal

        edge_threshold = 0.12

        return (probability, edge_threshold)
