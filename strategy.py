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
        mins_left = context["minutes_remaining"]

        # Only trade at minutes 0-3
        if minute > 3:
            return (fair, 1.0)

        latest = window.iloc[-1]
        close_arr = window["close"].values

        # --- Signal 1: Mean reversion normalized by ATR ---
        ret_7 = (close_arr[-1] - close_arr[-8]) / close_arr[-8]
        atr = latest["atr_14"]
        price = close_arr[-1]
        atr_pct = atr / price + 1e-10
        z_atr = ret_7 / atr_pct

        # Scale sensitivity by time remaining: more time = stronger reversion expected
        time_scale = np.sqrt(mins_left / 15.0)  # 1.0 at minute 0, decays
        mean_rev_signal = 0.5 - np.clip(z_atr * 0.12 * time_scale, -0.35, 0.35)

        # --- Signal 2: Bollinger Band position ---
        bb_upper = latest["bbands_upper"]
        bb_lower = latest["bbands_lower"]
        bb_range = bb_upper - bb_lower + 1e-10
        bb_pos = (close_arr[-1] - bb_lower) / bb_range

        bb_signal = 0.5 + (0.5 - bb_pos) * 0.25

        # --- Signal 3: Volume confirmation ---
        vol_ratio = latest["volume"] / (latest["volume_sma_20"] + 1e-10)
        if vol_ratio < 0.8:
            vol_boost = 0.04
        elif vol_ratio > 1.5:
            vol_boost = -0.02
        else:
            vol_boost = 0.0

        if mean_rev_signal > 0.5:
            mean_rev_signal += vol_boost
        else:
            mean_rev_signal -= vol_boost

        # --- Ensemble ---
        probability = 0.70 * mean_rev_signal + 0.30 * bb_signal

        # --- Adaptive edge threshold ---
        bb_bw = latest["bbands_bandwidth"]
        edge_threshold = np.clip(0.10 + bb_bw * 2.0, 0.10, 0.22)

        return (probability, edge_threshold)
