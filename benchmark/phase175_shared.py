"""
Phase 1.75 — Shared Data Sectioning benchmark.

Tests the shared-capital multi-asset portfolio engine at scale.
Compares: shared_data_sectioning (mode 0/1/2/3) vs run_batch_symbols (independent reference).

Modes 61–65:
  61 = independent batch reference (run_batch_symbols, 5 assets)
  62 = shared_data_sectioning mode 0 (serial)
  63 = shared_data_sectioning mode 1 (adaptive)
  64 = shared_data_sectioning mode 2 (GPU/adaptive)
  65 = shared_data_sectioning mode 3 (hybrid)

Asset counts tested: 5, 50 (scaled down for quick mode).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import BenchResult, cpu_mem_snapshot

SEED = 42
INIT_CASH = 10_000.0
FEES = 0.001
SIZE_PCT = 0.95
DEFAULT_BARS = 100_000


def _make_signals(bars: int, seed: int, buy_every: int = 20):
    """Generate deterministic entry/exit signals: buy every N bars, sell after 10."""
    import random
    rng = random.Random(seed)
    close = [100.0]
    for _ in range(bars - 1):
        close.append(max(1.0, close[-1] * (1 + rng.gauss(0, 0.01))))
    entries = [False] * bars
    exits = [False] * bars
    i = buy_every
    while i < bars - 11:
        entries[i] = True
        exits[i + 10] = True
        i += buy_every
    return close, entries, exits


def run(mode_num: int, n_assets: int = 5, bars: int = DEFAULT_BARS,
        exec_mode: int = 1, **_) -> BenchResult:
    """Run one Phase 1.75 case."""
    phase = "Phase 1.75"

    # independent batch reference
    if mode_num == 61:
        return _run_independent_batch(mode_num, n_assets, bars, phase)

    # shared_data_sectioning modes 0-3
    mode_map = {62: 0, 63: 1, 64: 2, 65: 3}
    exec_mode = mode_map.get(mode_num, 1)
    return _run_shared(mode_num, n_assets, bars, exec_mode, phase)


def _run_independent_batch(mode_num: int, n_assets: int, bars: int, phase: str) -> BenchResult:
    result = BenchResult(
        mode=mode_num,
        mode_name=f"independent_per_symbol_ref_{n_assets}assets  [round-trip fees]",
        bars=bars,
        phase=phase,
        phase_type="portfolio",
        backend="leika_independent_per_symbol_reference",
        n_assets=n_assets,
        leika_exec_mode=1,
    )
    try:
        import leika
        closes, entries_list, exits_list = [], [], []
        for i in range(n_assets):
            c, e, x = _make_signals(bars, SEED + i)
            closes.append(c)
            entries_list.append(e)
            exits_list.append(x)

        # Use Portfolio.run_batch (independent per-symbol)
        portfolios = []
        for c, e, x in zip(closes, entries_list, exits_list):
            portfolios.append(
                leika.Portfolio.from_signals(c, e, x)
                .init_cash(INIT_CASH)
                .fees(FEES)
            )
        _, mem_pre = cpu_mem_snapshot()
        t0 = time.monotonic()
        batch_results = leika.Portfolio.run_batch(portfolios=portfolios, mode=1)
        result.exec_ms = (time.monotonic() - t0) * 1000
        _, mem_post = cpu_mem_snapshot()
        result.mem_mb = max(mem_pre, mem_post)
        result.throughput_bars_sec = (n_assets * bars) / (result.exec_ms / 1000) if result.exec_ms > 0 else 0.0

        stats = [r.stats_fast() for r in batch_results]
        if stats:
            result.total_return_pct = sum(s.total_return_pct for s in stats) / len(stats)
            result.roi_pct = sum(s.roi_pct for s in stats) / len(stats)
            result.sharpe_ratio = sum(s.sharpe_ratio for s in stats) / len(stats)
            result.sortino_ratio = sum(s.sortino_ratio for s in stats) / len(stats)
            result.calmar_ratio = sum(s.calmar_ratio for s in stats) / len(stats)
            result.max_drawdown_pct = sum(s.max_drawdown_pct for s in stats) / len(stats)
            result.profit_factor = sum(s.profit_factor for s in stats) / len(stats)
            result.win_rate_pct = sum(s.win_rate for s in stats) / len(stats) * 100.0
            result.total_trades = sum(s.total_trades for s in stats)

        result.cash_model = "independent_per_symbol"
        result.shared_data_sectioning = False
        result.split_axis = "symbols"
        result.execution_core = "independent_parallel_batch"
        result.dynamic_sectioning_used = True
        result.sell_at_end_scope = "per_symbol"
        result.dynamic_sectioning_preparation = True
        result.dynamic_sectioning_execution = False
        result.dynamic_sectioning_post_analysis = False

    except Exception as exc:
        result.error = str(exc)
    return result


def _run_shared(mode_num: int, n_assets: int, bars: int, exec_mode: int, phase: str) -> BenchResult:
    mode_label = f"shared_data_sectioning_mode{exec_mode}_{n_assets}assets"
    result = BenchResult(
        mode=mode_num,
        mode_name=mode_label,
        bars=bars,
        phase=phase,
        phase_type="portfolio",
        backend=f"shared_global_cash_mode{exec_mode}",
        n_assets=n_assets,
        leika_exec_mode=exec_mode,
    )
    try:
        import leika
        closes, entries_list, exits_list = [], [], []
        for i in range(n_assets):
            c, e, x = _make_signals(bars, SEED + i)
            closes.append(c)
            entries_list.append(e)
            exits_list.append(x)

        _, mem_pre = cpu_mem_snapshot()
        t0 = time.monotonic()
        shared_result = leika.shared_data_sectioning(
            closes=closes,
            entries=entries_list,
            exits=exits_list,
            symbols=[f"SYM{i}" for i in range(n_assets)],
            init_cash=INIT_CASH,
            fees=FEES,
            size_pct=SIZE_PCT,
            sell_at_end=True,
            mode=exec_mode,
        )
        result.exec_ms = (time.monotonic() - t0) * 1000
        _, mem_post = cpu_mem_snapshot()
        result.mem_mb = max(mem_pre, mem_post)

        stats = shared_result.stats_fast()
        result.total_return_pct = stats.total_return_pct
        result.sharpe_ratio = stats.sharpe_ratio
        result.max_drawdown_pct = stats.max_drawdown_pct
        result.total_trades = stats.total_trades
        result.win_rate_pct = stats.win_rate * 100.0
        result.profit_factor = stats.profit_factor
        result.throughput_bars_sec = (n_assets * bars) / (result.exec_ms / 1000) if result.exec_ms > 0 else 0.0

        result.cash_model = "shared_global_cash"
        result.shared_data_sectioning = True
        result.split_axis = "symbols_for_preparation"
        result.execution_core = "shared_global_cash"
        result.dynamic_sectioning_used = exec_mode >= 1
        result.sell_at_end_scope = "global"
        result.dynamic_sectioning_preparation = exec_mode >= 1
        result.dynamic_sectioning_execution = False  # always sequential
        result.dynamic_sectioning_post_analysis = False

    except Exception as exc:
        result.error = str(exc)
    return result
