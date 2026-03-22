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
        ret_15 = (close_arr[-1] - close_arr[-16]) / close_arr[-16]
        vol_20 = latest["volatility_20"]
        sigma = vol_20 * np.sqrt(15) + 1e-10
        z_mom = ret_15 / sigma

        mean_rev_signal = 0.5 - np.clip(z_mom * 0.10, -0.3, 0.3)

        # --- Signal 2: Bollinger Band position ---
        bb_upper = latest["bbands_upper"]
        bb_lower = latest["bbands_lower"]
        bb_range = bb_upper - bb_lower + 1e-10
        bb_pos = (close_arr[-1] - bb_lower) / bb_range

        bb_signal = 0.5 + (0.5 - bb_pos) * 0.25

        # --- Signal 3: Volume confirmation ---
        # Low volume moves are more likely to revert
        vol_ratio = latest["volume"] / (latest["volume_sma_20"] + 1e-10)
        # If volume is below average, boost the mean reversion signal
        if vol_ratio < 0.8:
            vol_boost = 0.04
        elif vol_ratio > 1.5:
            vol_boost = -0.02  # high volume = trend more likely to stick, reduce contrarian
        else:
            vol_boost = 0.0

        # Apply vol boost in the direction of mean reversion
        if mean_rev_signal > 0.5:
            mean_rev_signal += vol_boost
        else:
            mean_rev_signal -= vol_boost

        # --- Ensemble ---
        probability = 0.55 * mean_rev_signal + 0.45 * bb_signal

        # Tighter threshold: be more selective
        edge_threshold = 0.14

        return (probability, edge_threshold)
