"""
Price generation and indicator logic for the Leika benchmark suite.

All functions are deterministic given a seed.
"""
from __future__ import annotations

import math
import random
from typing import Optional


# ── Price generation ──────────────────────────────────────────────────────────

def generate_prices(n: int, seed: int = 42,
                    s0: float = 100.0, mu: float = 0.0,
                    sigma: float = 0.2, dt: float = 1 / 252) -> list[float]:
    """Standard GBM price series."""
    rng    = random.Random(seed)
    prices = [s0]
    for _ in range(n - 1):
        z  = _box_muller(rng)
        s  = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        prices.append(max(s, 0.001))
    return prices


def generate_prices_extreme(n: int, seed: int = 42,
                             s0: float = 100.0, sigma: float = 0.5,
                             crash_prob: float = 0.003, crash_size: float = 0.18,
                             dt: float = 1 / 252) -> list[float]:
    """
    High-volatility GBM with random crash events.
    sigma=0.50 (2.5× normal), ~0.3% crash probability per bar.
    """
    rng    = random.Random(seed)
    prices = [s0]
    for _ in range(n - 1):
        z    = _box_muller(rng)
        jump = -crash_size if rng.random() < crash_prob else 0.0
        s    = prices[-1] * math.exp(
            (-0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z + jump
        )
        prices.append(max(s, 0.001))
    return prices


def _box_muller(rng: random.Random) -> float:
    u1 = rng.random()
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(max(u1, 1e-15))) * math.cos(2.0 * math.pi * u2)


# ── Pure-Python indicators (mirror Rust — same algorithm) ─────────────────────

def ema_py(data: list[float], period: int) -> list[Optional[float]]:
    """EMA with Wilder smoothing. Warm-up bars → None."""
    if period == 0 or len(data) < period:
        return [None] * len(data)
    k      = 2.0 / (period + 1.0)
    result: list[Optional[float]] = [None] * len(data)
    seed   = sum(data[:period]) / period
    result[period - 1] = seed
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1.0 - k)  # type: ignore[operator]
    return result


def rsi_py(data: list[float], period: int = 14) -> list[Optional[float]]:
    """Wilder RSI. Warm-up bars → None. Values in [0, 100]."""
    if period == 0 or len(data) <= period:
        return [None] * len(data)
    result: list[Optional[float]] = [None] * len(data)
    avg_gain = avg_loss = 0.0
    for i in range(1, period + 1):
        chg = data[i] - data[i - 1]
        if chg > 0:
            avg_gain += chg
        else:
            avg_loss -= chg
    avg_gain /= period
    avg_loss /= period
    rs             = avg_gain / avg_loss if avg_loss > 1e-10 else 1e10
    result[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, len(data)):
        chg  = data[i] - data[i - 1]
        gain = chg if chg > 0 else 0.0
        loss = -chg if chg < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs        = avg_gain / avg_loss if avg_loss > 1e-10 else 1e10
        result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


def macd_py(data: list[float],
            fast: int = 12, slow: int = 26,
            signal: int = 9) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """Pure-Python MACD. Returns (macd_line, signal_line, histogram). Warm-up → None."""
    ema_fast = ema_py(data, fast)
    ema_slow = ema_py(data, slow)

    n    = len(data)
    macd: list[Optional[float]] = [None] * n
    for i in range(n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd[i] = ema_fast[i] - ema_slow[i]  # type: ignore[operator]

    # Signal line: EMA of MACD values — seed from first non-None MACD values
    sig_line: list[Optional[float]] = [None] * n
    macd_vals = [(i, v) for i, v in enumerate(macd) if v is not None]
    if len(macd_vals) >= signal:
        seed_idx = macd_vals[signal - 1][0]
        seed_val = sum(v for _, v in macd_vals[:signal]) / signal
        sig_line[seed_idx] = seed_val
        k = 2.0 / (signal + 1.0)
        for j in range(seed_idx + 1, n):
            if macd[j] is not None and sig_line[j - 1] is not None:
                sig_line[j] = macd[j] * k + sig_line[j - 1] * (1.0 - k)  # type: ignore[operator]

    hist: list[Optional[float]] = [None] * n
    for i in range(n):
        if macd[i] is not None and sig_line[i] is not None:
            hist[i] = macd[i] - sig_line[i]  # type: ignore[operator]

    return macd, sig_line, hist


# ── MACD signal generation ────────────────────────────────────────────────────

def make_macd_signals(
    histogram: list[Optional[float]],
) -> tuple[list[bool], list[bool]]:
    """
    MACD histogram crossover.
    BUY  when histogram crosses ≤0 → >0 (MACD above signal — bullish).
    SELL when histogram crosses ≥0 → <0 (MACD below signal — bearish).
    Returns (entries, exits) as bool lists for leika.Portfolio.from_signals().
    """
    n       = len(histogram)
    entries = [False] * n
    exits   = [False] * n
    for i in range(1, n):
        h0, h1 = histogram[i], histogram[i - 1]
        if h0 is None or h1 is None:
            continue
        if h1 <= 0 and h0 > 0:
            entries[i] = True
        elif h1 >= 0 and h0 < 0:
            exits[i] = True
    return entries, exits


def make_macd_signals_int(histogram: list[Optional[float]]) -> list[int]:
    """Same as make_macd_signals but returns list[int] (1=BUY, -1=SELL, 0=HOLD)."""
    entries, exits = make_macd_signals(histogram)
    return [1 if e else (-1 if x else 0) for e, x in zip(entries, exits)]


# ── Legacy EMA crossover (kept for Python fallback paths) ─────────────────────

def make_signals(prices: list[float],
                 ema_fast: list[Optional[float]],
                 ema_slow: list[Optional[float]],
                 rsi_vals: list[Optional[float]],
                 rsi_buy:  float = 50.0,
                 rsi_sell: float = 70.0) -> list[int]:
    """
    EMA crossover + RSI momentum filter.
    BUY  (1): fast crosses ABOVE slow AND RSI > rsi_buy
    SELL (-1): fast crosses BELOW slow OR  RSI > rsi_sell
    """
    n       = len(prices)
    signals = [0] * n
    for i in range(1, n):
        ef0, ef1 = ema_fast[i], ema_fast[i - 1]
        es0, es1 = ema_slow[i], ema_slow[i - 1]
        r0       = rsi_vals[i]
        if None in (ef0, ef1, es0, es1, r0):
            continue
        cross_up   = ef1 <= es1 and ef0 > es0   # type: ignore[operator]
        cross_down = ef1 >= es1 and ef0 < es0   # type: ignore[operator]
        if cross_up and r0 > rsi_buy:            # type: ignore[operator]
            signals[i] = 1
        elif cross_down or r0 > rsi_sell:        # type: ignore[operator]
            signals[i] = -1
    return signals


def action_to_signal(action: str, in_position: bool) -> int:
    """Convert AI action string to signal int."""
    a = action.upper()
    if a == "BUY" and not in_position:
        return 1
    if a in ("SELL", "CLOSE") and in_position:
        return -1
    return 0


# ── Pure-Python Random Walk generators (mirror Rust randomwalk/engine.rs) ─────
# All return list[float] of length n. Warm-up price = s0 at index 0.

def generate_rw_gbm(n: int, seed: int = 42, s0: float = 100.0,
                    drift: float = 0.0, sigma: float = 0.20,
                    dt: float = 1 / 252) -> list[float]:
    """GBM — same as generate_prices, kept here for uniform dispatch."""
    return generate_prices(n, seed=seed, s0=s0, mu=drift, sigma=sigma, dt=dt)


def generate_rw_gaussian(n: int, seed: int = 42, s0: float = 100.0,
                         sigma: float = 2.0) -> list[float]:
    """Additive Gaussian noise: S_t = S_{t-1} + σ·Z."""
    rng    = random.Random(seed)
    prices = [s0]
    for _ in range(n - 1):
        z = _box_muller(rng)
        prices.append(max(prices[-1] + sigma * z, 0.001))
    return prices


def generate_rw_mean_reversion(n: int, seed: int = 42, s0: float = 100.0,
                                kappa: float = 2.0, theta: float = 100.0,
                                sigma: float = 0.20,
                                dt: float = 1 / 252) -> list[float]:
    """Ornstein-Uhlenbeck: dX = κ(θ−X)dt + σ√dt·Z."""
    rng    = random.Random(seed)
    sig_dt = sigma * math.sqrt(dt)
    prices = [s0]
    x      = s0
    for _ in range(n - 1):
        z  = _box_muller(rng)
        dx = kappa * (theta - x) * dt + sig_dt * z
        x  = max(x + dx, 0.001)
        prices.append(x)
    return prices


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Sample from Poisson(lam) — Knuth algorithm."""
    L = math.exp(-max(lam, 0.0))
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def generate_rw_jump_diffusion(n: int, seed: int = 42, s0: float = 100.0,
                                drift: float = 0.0, sigma: float = 0.20,
                                jump_lambda: float = 3.0,
                                jump_mean: float = -0.05,
                                jump_std: float = 0.10,
                                dt: float = 1 / 252) -> list[float]:
    """Merton Jump Diffusion: GBM + compound Poisson jumps."""
    rng       = random.Random(seed)
    mu_dt     = (drift - 0.5 * sigma ** 2) * dt
    sig_dt    = sigma * math.sqrt(dt)
    lambda_dt = jump_lambda * dt
    prices    = [s0]
    s         = s0
    for _ in range(n - 1):
        z          = _box_muller(rng)
        gbm_factor = math.exp(mu_dt + sig_dt * z)
        n_jumps    = _poisson_sample(rng, lambda_dt)
        jf         = 1.0
        for _ in range(n_jumps):
            j  = _box_muller(rng) * jump_std + jump_mean
            jf *= math.exp(j)
        s = max(s * gbm_factor * max(jf, 1e-6), 0.001)
        prices.append(s)
    return prices


def generate_rw_regime_switching(n: int, seed: int = 42, s0: float = 100.0,
                                  drift_up: float = 0.0, vol_up: float = 0.20,
                                  drift_dn: float = -0.20, vol_dn: float = 0.40,
                                  p_up_dn: float = 0.02, p_dn_up: float = 0.10,
                                  dt: float = 1 / 252) -> list[float]:
    """2-state Markov regime switching (bull/bear)."""
    rng     = random.Random(seed)
    prices  = [s0]
    s       = s0
    in_bull = True
    for _ in range(n - 1):
        if in_bull:
            if rng.random() < p_up_dn:
                in_bull = False
        else:
            if rng.random() < p_dn_up:
                in_bull = True
        drift = drift_up if in_bull else drift_dn
        vol   = vol_up   if in_bull else vol_dn
        z     = _box_muller(rng)
        mu_dt = (drift - 0.5 * vol ** 2) * dt
        s     = max(s * math.exp(mu_dt + vol * math.sqrt(dt) * z), 0.001)
        prices.append(s)
    return prices


# Dispatch map used by runner.py
RW_GENERATORS: dict = {
    "gbm":              generate_rw_gbm,
    "gaussian":         generate_rw_gaussian,
    "mean_reversion":   generate_rw_mean_reversion,
    "jump_diffusion":   generate_rw_jump_diffusion,
    "regime_switching": generate_rw_regime_switching,
}
