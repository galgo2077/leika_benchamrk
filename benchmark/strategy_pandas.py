"""
Pandas strategy implementation for the Phase DATA-1 DataFrame benchmark.

Strategy: EMA20/EMA50 crossover + RSI14 filter (Wilder's smoothing).
  Entry: EMA_fast > EMA_slow AND RSI > 50
  Exit:  EMA_fast < EMA_slow OR  RSI < 50

All functions accept a pandas.Series of close prices and return arrays
compatible with VectorBT's Portfolio.from_signals().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMA_FAST  = 20
EMA_SLOW  = 50
RSI_PERIOD = 14
WARMUP    = max(EMA_SLOW, RSI_PERIOD + 1)


def ema(close: pd.Series, period: int) -> pd.Series:
    """Standard EMA via pandas ewm(span). k = 2 / (period + 1), no SMA seed."""
    return close.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, window: int = RSI_PERIOD) -> pd.Series:
    """
    Wilder's RSI matching Leika's RMA formula.
    alpha = 1/window; warm-up via ewm min_periods=window.
    """
    delta    = close.diff()
    avg_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, float("nan"))
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def signals(
    close: pd.Series,
    ema_fast_period: int = EMA_FAST,
    ema_slow_period: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """
    Compute entry/exit boolean arrays plus the raw indicator series.

    Returns:
        entries   — np.ndarray[bool]
        exits     — np.ndarray[bool]
        ema_fast  — pd.Series
        ema_slow  — pd.Series
        rsi_vals  — pd.Series
    """
    ema_fast_s = ema(close, ema_fast_period)
    ema_slow_s = ema(close, ema_slow_period)
    rsi_s      = rsi(close, rsi_period)

    warmup = max(ema_slow_period, rsi_period + 1)
    n      = len(close)
    entries = np.zeros(n, dtype=bool)
    exits   = np.zeros(n, dtype=bool)

    ef = ema_fast_s.values
    es = ema_slow_s.values
    rv = rsi_s.values

    for i in range(warmup, n):
        entries[i] = (ef[i] > es[i]) and (rv[i] > 50.0)
        exits[i]   = (ef[i] < es[i]) or  (rv[i] < 50.0)

    return entries, exits, ema_fast_s, ema_slow_s, rsi_s


def signals_multi(
    close_df: pd.DataFrame,
    ema_fast_period: int = EMA_FAST,
    ema_slow_period: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-asset signal generation. close_df columns = one asset each.
    Returns (entries_df, exits_df) as boolean DataFrames.
    """
    warmup = max(ema_slow_period, rsi_period + 1)
    entries_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
    exits_df   = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
    for col in close_df.columns:
        ents, exts, *_ = signals(
            close_df[col], ema_fast_period, ema_slow_period, rsi_period
        )
        entries_df[col] = ents
        exits_df[col]   = exts
    return entries_df, exits_df
