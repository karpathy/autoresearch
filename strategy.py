"""
BTC 15-minute direction prediction strategy.

THIS IS THE ONLY FILE THE AGENT MODIFIES.

The agent can change anything here: signal logic, indicators used,
thresholds, confidence calculation, ensemble methods, etc.
"""

import pandas as pd


class Strategy:
    """
    Predict whether BTC will be up or down in 15 minutes.

    on_bar() receives a DataFrame of the last LOOKBACK_MINUTES (60) 1-min candles
    with columns:
        Base:    open, high, low, close, volume
        Returns: returns
        Vol:     volatility_20
        MAs:     sma_20, sma_50, ema_12, ema_26
        RSI:     rsi_14
        MACD:    macd, macd_signal, macd_hist
        BBands:  bbands_lower, bbands_mid, bbands_upper, bbands_bandwidth
        Other:   atr_14, volume_sma_20

    Returns: (signal, confidence)
        signal:     1 = predict up, -1 = predict down, 0 = no trade
        confidence: float 0.0 to 1.0 (used as binary contract purchase price)
    """

    def on_bar(self, window: pd.DataFrame) -> tuple[int, float]:
        latest = window.iloc[-1]

        # Simple RSI mean-reversion baseline
        rsi = latest["rsi_14"]

        if rsi < 30:
            return (1, 0.6)   # oversold -> predict up
        elif rsi > 70:
            return (-1, 0.6)  # overbought -> predict down
        else:
            return (0, 0.0)   # no signal
