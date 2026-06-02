"""
Polars strategy implementation for the Phase DATA-1 DataFrame benchmark.

Strategy: EMA20/EMA50 crossover + RSI14 filter via Leika Rust indicators.
  Entry: EMA_fast > EMA_slow AND RSI > 50
  Exit:  EMA_fast < EMA_slow OR  RSI < 50

All functions accept a polars.Series of close prices and return NumPy arrays.
Leika indicators accept polars.Series natively — no .to_numpy() conversion.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import leika

EMA_FAST   = 20
EMA_SLOW   = 50
RSI_PERIOD = 14
WARMUP     = max(EMA_SLOW, RSI_PERIOD + 1)


def ema(close: pl.Series, period: int) -> list:
    """EMA via Leika Rust. Returns list[float | None]; None for warm-up bars."""
    return leika.ema(close, period)


def rsi(close: pl.Series, window: int = RSI_PERIOD) -> list:
    """Wilder's RSI via Leika Rust. Returns list[float | None]; None for warm-up bars."""
    return leika.rsi(close, window)


def signals(
    close: pl.Series,
    ema_fast_period: int = EMA_FAST,
    ema_slow_period: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Compute entry/exit boolean arrays plus raw indicator lists.

    Returns:
        entries   — np.ndarray[bool]
        exits     — np.ndarray[bool]
        ema_fast  — list[float | None]
        ema_slow  — list[float | None]
        rsi_vals  — list[float | None]
    """
    ema_fast_l = leika.ema(close, ema_fast_period)
    ema_slow_l = leika.ema(close, ema_slow_period)
    rsi_l      = leika.rsi(close, rsi_period)

    warmup  = max(ema_slow_period, rsi_period + 1)
    n       = len(close)
    entries = np.zeros(n, dtype=bool)
    exits   = np.zeros(n, dtype=bool)

    for i in range(warmup, n):
        ef, es, rv = ema_fast_l[i], ema_slow_l[i], rsi_l[i]
        if ef is None or es is None or rv is None:
            continue
        entries[i] = (ef > es) and (rv > 50.0)
        exits[i]   = (ef < es) or  (rv < 50.0)

    return entries, exits, ema_fast_l, ema_slow_l, rsi_l


def signals_multi(
    close_series_list: list[pl.Series],
    ema_fast_period: int = EMA_FAST,
    ema_slow_period: int = EMA_SLOW,
    rsi_period: int = RSI_PERIOD,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Multi-asset signal generation.
    Returns list of (entries, exits) arrays, one pair per asset.
    """
    return [
        signals(c, ema_fast_period, ema_slow_period, rsi_period)[:2]
        for c in close_series_list
    ]
