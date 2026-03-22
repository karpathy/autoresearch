"""
BTC 15-minute direction prediction strategy for Kalshi KXBTC15M contracts.

THIS IS THE ONLY FILE THE AGENT MODIFIES.

The agent can change anything here: probability model, indicators used,
thresholds, edge calculation, ensemble methods, entry timing logic, etc.
"""

import pandas as pd


class Strategy:
    """
    Estimate the probability that BTC will be UP at the end of a Kalshi
    15-minute window, and specify the minimum edge required to trade.

    on_bar() receives:
        window: DataFrame of the last LOOKBACK_MINUTES (60) 1-min candles
            Base:    open, high, low, close, volume
            Returns: returns
            Vol:     volatility_20
            MAs:     sma_20, sma_50, ema_12, ema_26
            RSI:     rsi_14
            MACD:    macd, macd_signal, macd_hist
            BBands:  bbands_lower, bbands_mid, bbands_upper, bbands_bandwidth
            Other:   atr_14, volume_sma_20

        context: dict with Kalshi window info
            window_minute:     int (0-13), current minute within the 15-min window
            window_open_price: float, BTC price at minute 0 of this window
            minutes_remaining: int, minutes until settlement
            fair_price:        float (0-1), binary option fair value P(up)

    Returns: (probability, edge_threshold)
        probability:    float 0-1, model's estimate of P(BTC up at settlement)
        edge_threshold: float 0-1, minimum |probability - fair_price| to trade
    """

    def on_bar(self, window: pd.DataFrame, context: dict) -> tuple[float, float]:
        latest = window.iloc[-1]

        # Simple RSI-based probability estimate
        # RSI < 30 → oversold → higher P(up); RSI > 70 → overbought → lower P(up)
        rsi = latest["rsi_14"]
        probability = 0.5 + (50 - rsi) / 100 * 0.3
        probability = max(0.01, min(0.99, probability))

        # Fixed edge threshold: require 5% edge over fair price to trade
        edge_threshold = 0.05

        return (probability, edge_threshold)
