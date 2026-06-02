"""
Shared benchmark runner for all 20 modes.

All phases use the same MACD(12,26,9) strategy, same fees/slippage, same seeds.
The only axis that changes per phase is: backend × execution_mode × n_assets × ai_enabled.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("LEIKA_QUIET_INIT", "1")

ProgressFn = Optional[Callable[[int, int, str], None]]

_LARGE_FIELDS = frozenset({"prices", "equity", "ema_fast", "drawdowns", "ai_section_results", "trade_summary"})

def _flush_result(result, path: Path) -> None:
    """Append one result as a JSON line — called after every run to survive mid-benchmark crashes."""
    try:
        d = dataclasses.asdict(result)
        for key in _LARGE_FIELDS:
            d.pop(key, None)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")
            f.flush()
    except Exception:
        pass


def _install_crash_handler(crash_log: Path) -> None:
    """Write a stack trace to crash_log on any unhandled Python exception."""
    def _hook(exc_type, exc_value, exc_tb):
        try:
            crash_log.parent.mkdir(parents=True, exist_ok=True)
            with open(crash_log, "w", encoding="utf-8") as f:
                f.write(f"CRASH {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
                f.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            print(f"\n[crash log] {crash_log}", file=sys.stderr)
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _hook


def _phase_slug(phase_name: str) -> str:
    return phase_name.lower().replace(" ", "").replace(".", "_").replace("/", "_")


def _save_phase_checkpoint(phase_results: list, phase_name: str, out_dir: Path, ts: str) -> None:
    """Write a checkpoint JSON for one completed phase."""
    if not phase_results:
        return
    try:
        slug = _phase_slug(phase_name)
        path = out_dir / f"checkpoint_{slug}_{ts}.json"
        data = []
        for r in phase_results:
            d = dataclasses.asdict(r)
            for key in _LARGE_FIELDS:
                d.pop(key, None)
            data.append(d)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  [checkpoint] {phase_name} → {path.name} ({len(data)} results)")
    except Exception as exc:
        print(f"  [checkpoint] write failed: {exc}", file=sys.stderr)


def _load_checkpoints(out_dir: Path, ts: str) -> list:
    """Reconstruct results list from all checkpoint_*_{ts}.json files."""
    import dataclasses as _dc
    valid_fields = {f.name for f in _dc.fields(BenchResult)}
    results = []
    for path in sorted(out_dir.glob(f"checkpoint_*_{ts}.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for d in data:
                filtered = {k: v for k, v in d.items() if k in valid_fields}
                results.append(BenchResult(**filtered))
            print(f"  loaded {path.name} ({len(data)} results)")
        except Exception as exc:
            print(f"  failed to load {path.name}: {exc}", file=sys.stderr)
    return results


def _tick_progress(progress: ProgressFn, done: int, total: int, label: str) -> None:
    if progress is None:
        return
    try:
        progress(done, total, label)
    except Exception:
        pass


from ai_agent import AiAgent, BenchmarkAiAgent
from ai_context import BenchmarkAiConfig, build_ai_context, build_trade_records, summarize_trade_records
from metrics import BenchResult, compute_metrics, backtest_simple, cpu_mem_snapshot, gpu_snapshot
from strategy import (generate_prices, generate_prices_extreme,
                      macd_py, make_macd_signals, make_macd_signals_int,
                      RW_GENERATORS)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False

import report_gen
import hw_detect
from monitor import LiveMonitor

try:
    import leika
    viz = leika.viz
    _VIZ_AVAILABLE = True
except (ImportError, AttributeError):
    _VIZ_AVAILABLE = False

MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
SEED         = 42
AI_COOLDOWN  = 100
FEES         = 0.001
INIT_CASH    = 10_000.0
N_ASSETS     = 5     # fixed for multi-asset phases

try:
    import leika as _leika
    _LEIKA_OK = True
except ImportError:
    _LEIKA_OK = False


_MC_REFERENCE_CACHE: dict[tuple[int, int, int], dict] = {}
_RW_REFERENCE_CACHE: dict[tuple[int, str, int], dict] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _macd_rust(prices, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIGNAL):
    if _LEIKA_OK:
        return _leika.macd(prices, fast, slow, sig)
    return macd_py(prices, fast, slow, sig)


def _backtest_leika(prices, entries, exits, exec_mode: int = 2):
    """Single-asset Rust backtest — fastest path: PreparedData → run_prepared → stats_fast."""
    if _LEIKA_OK:
        t_prep = time.monotonic()
        data = _leika.PreparedData.from_signals(prices, entries, exits)
        prepare_ms = (time.monotonic() - t_prep) * 1000

        t_run = time.monotonic()
        pf = _leika.Portfolio.run_prepared(data, exec_mode)
        engine_ms = (time.monotonic() - t_run) * 1000

        t_stats = time.monotonic()
        stats = pf.stats_fast()
        stats_ms = (time.monotonic() - t_stats) * 1000

        total_ms = prepare_ms + engine_ms + stats_ms
        return {
            "total_return_pct": stats.total_return_pct,
            "roi_pct": stats.roi_pct,
            "sharpe_ratio":     stats.sharpe_ratio,
            "sortino_ratio":    stats.sortino_ratio,
            "calmar_ratio":     stats.calmar_ratio,
            "max_drawdown_pct": stats.max_drawdown_pct,
            "profit_factor":    stats.profit_factor,
            "portfolio_heat_avg_pct": stats.portfolio_heat_avg_pct,
            "portfolio_heat_max_pct": stats.portfolio_heat_max_pct,
            "win_rate":         stats.win_rate,
            "total_trades":     stats.total_trades,
            "equity":           pf.equity_curve(),
            "summary":          pf.to_dict(level="summary"),
            "prepare_data_ms":  prepare_ms,
            "python_to_rust_conversion_ms": prepare_ms,
            "rust_engine_time_ms": engine_ms,
            "stats_calculation_time_ms": stats_ms,
            "total_timed_ms":   total_ms,
        }
    sigs       = [1 if e else (-1 if x else 0) for e, x in zip(entries, exits)]
    eq, trades = backtest_simple(prices, sigs)
    m          = compute_metrics(eq, trades)
    return {
        "total_return_pct": m["total_return_pct"],
        "roi_pct": m.get("roi_pct", m["total_return_pct"]),
        "sharpe_ratio": m["sharpe_ratio"],
        "sortino_ratio": m.get("sortino_ratio", 0.0),
        "calmar_ratio": m.get("calmar_ratio", 0.0),
        "max_drawdown_pct": m["max_drawdown_pct"],
        "profit_factor": m.get("profit_factor", 0.0),
        "portfolio_heat_avg_pct": m.get("portfolio_heat_avg_pct", 0.0),
        "portfolio_heat_max_pct": m.get("portfolio_heat_max_pct", 0.0),
        "win_rate": 0.0,
        "total_trades": len(trades),
        "equity": eq,
        "python_to_rust_conversion_ms": 0.0,
        "rust_engine_time_ms": 0.0,
        "stats_calculation_time_ms": 0.0,
    }


def _backtest_vbt(prices, entries, exits, use_rust_indicators=False):
    """Single-asset VectorBT backtest. use_rust_indicators=True mixes Rust MACD."""
    try:
        import vectorbt as vbt, numpy as np
        arr = np.array(prices)

        if use_rust_indicators and _LEIKA_OK:
            _, _, histogram = _leika.macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            ents_arr = np.array(entries)
            exts_arr = np.array(exits)
        else:
            ents_arr = np.array(entries)
            exts_arr = np.array(exits)

        pf    = vbt.Portfolio.from_signals(arr, ents_arr, exts_arr,
                                            init_cash=INIT_CASH, fees=FEES, freq="1D")
        _DURATION_METRICS = {
            "period", "max_dd_duration", "avg_dd_duration",
            "max_trade_duration", "avg_trade_duration",
            "avg_win_trade_duration", "avg_loss_trade_duration",
        }
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                if len(prices) >= 500_000:
                    _safe = [m for m in pf.metrics.keys() if m not in _DURATION_METRICS]
                    stats = pf.stats(metrics=_safe)
                else:
                    try:
                        stats = pf.stats()
                    except Exception:
                        _safe = [m for m in pf.metrics.keys() if m not in _DURATION_METRICS]
                        stats = pf.stats(metrics=_safe)
            except Exception:
                stats = {}
        return {
            "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
            "roi_pct": float(stats.get("Return [%]", stats.get("Total Return [%]", 0.0))),
            "sharpe_ratio":     float(stats.get("Sharpe Ratio", 0.0)),
            "sortino_ratio":    float(stats.get("Sortino Ratio", 0.0)),
            "calmar_ratio":     float(stats.get("Calmar Ratio", 0.0)),
            "max_drawdown_pct": abs(float(stats.get("Max Drawdown [%]", 0.0))),
            "profit_factor":    float(stats.get("Profit Factor", 0.0)),
            "portfolio_heat_avg_pct": float(stats.get("Portfolio Heat [%]", 0.0)),
            "portfolio_heat_max_pct": float(stats.get("Max Portfolio Heat [%]", 0.0)),
            "win_rate":         float(stats.get("Win Rate [%]", 0.0)),
            "total_trades":     int(stats.get("Total Trades", 0)),
            "equity":           pf.value().tolist(),
        }
    except ImportError:
        pass
    sigs       = [1 if e else (-1 if x else 0) for e, x in zip(entries, exits)]
    eq, trades = backtest_simple(prices, sigs)
    m          = compute_metrics(eq, trades)
    return {
        "total_return_pct": m["total_return_pct"],
        "roi_pct": m.get("roi_pct", m["total_return_pct"]),
        "sharpe_ratio": m["sharpe_ratio"],
        "sortino_ratio": m.get("sortino_ratio", 0.0),
        "calmar_ratio": m.get("calmar_ratio", 0.0),
        "max_drawdown_pct": m["max_drawdown_pct"],
        "profit_factor": m.get("profit_factor", 0.0),
        "portfolio_heat_avg_pct": m.get("portfolio_heat_avg_pct", 0.0),
        "portfolio_heat_max_pct": m.get("portfolio_heat_max_pct", 0.0),
        "win_rate": 0.0,
        "total_trades": len(trades),
        "equity": eq,
    }


def _signals_from_prices(prices):
    """Compute MACD and derive entry/exit bool arrays."""
    _, _, hist     = macd_py(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    entries, exits = make_macd_signals(hist)
    return entries, exits


def _signals_from_prices_rust(prices):
    """Compute MACD via Rust and derive entry/exit bool arrays."""
    if _LEIKA_OK:
        _, _, hist = _leika.macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    else:
        _, _, hist = macd_py(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    entries, exits = make_macd_signals(hist)
    return entries, exits, hist


def _fill_result(result: BenchResult, m: dict, prices, equity, hist=None):
    result.total_return_pct = m["total_return_pct"]
    result.roi_pct          = m.get("roi_pct", m["total_return_pct"])
    result.sharpe_ratio     = m["sharpe_ratio"]
    result.sortino_ratio     = m.get("sortino_ratio", 0.0)
    result.calmar_ratio      = m.get("calmar_ratio", 0.0)
    result.max_drawdown_pct = m["max_drawdown_pct"]
    result.profit_factor     = m.get("profit_factor", 0.0)
    result.portfolio_heat_avg_pct = m.get("portfolio_heat_avg_pct", 0.0)
    result.portfolio_heat_max_pct = m.get("portfolio_heat_max_pct", 0.0)
    result.win_rate_pct     = m.get("win_rate_pct", m.get("win_rate", 0.0))
    result.total_trades     = m.get("total_trades", 0)
    result.prices           = prices
    result.equity           = equity
    result.drawdowns        = m.get("drawdowns", [])
    result.python_to_rust_conversion_ms = m.get("python_to_rust_conversion_ms", 0.0)
    result.rust_engine_time_ms = m.get("rust_engine_time_ms", result.exec_ms)
    result.engine_time_ms = result.rust_engine_time_ms
    result.stats_calculation_time_ms = m.get("stats_calculation_time_ms", 0.0)
    if result.total_runtime_ms <= 0.0:
        result.total_runtime_ms = (
            result.python_input_time_ms
            + result.python_to_rust_conversion_ms
            + result.rust_engine_time_ms
            + result.stats_calculation_time_ms
            + result.python_export_time_ms
        )
    if hist is not None:
        result.ema_fast = [v if v is not None else float("nan") for v in hist]


def _rel_err(actual: float, reference: float) -> float:
    denom = max(abs(reference), 1e-12)
    return abs(actual - reference) / denom


def _gpu_metric_map(result_dict: dict) -> dict:
    metrics = result_dict.get("gpu_metrics") or {}
    return {
        "backend": result_dict.get("gpu_backend") or result_dict.get("backend") or "",
        "fallback_reason": result_dict.get("fallback_reason") or metrics.get("fallback_reason") or "",
        "kernel_time_ms": float(metrics.get("kernel_time_ms") or 0.0),
        "transfer_time_ms": float(metrics.get("transfer_time_ms") or 0.0),
        "total_gpu_time_ms": float(metrics.get("total_gpu_time_ms") or 0.0),
        "cpu_fallback_time_ms": float(metrics.get("cpu_fallback_time_ms") or 0.0),
        "cpu_start_ms": float(metrics.get("cpu_start_ms") or 0.0),
        "cpu_end_ms": float(metrics.get("cpu_end_ms") or 0.0),
        "gpu_start_ms": float(metrics.get("gpu_start_ms") or 0.0),
        "gpu_end_ms": float(metrics.get("gpu_end_ms") or 0.0),
        "cpu_time_ms": float(metrics.get("cpu_time_ms") or 0.0),
        "gpu_time_ms": float(metrics.get("gpu_time_ms") or 0.0),
        "overlap_ms": float(metrics.get("overlap_ms") or 0.0),
        "overlap_pct": float(metrics.get("overlap_pct") or 0.0),
        "cpu_idle_wait_ms": float(metrics.get("cpu_idle_wait_ms") or 0.0),
        "gpu_idle_wait_ms": float(metrics.get("gpu_idle_wait_ms") or 0.0),
        "hybrid_total_time_ms": float(metrics.get("hybrid_total_time_ms") or 0.0),
    }


def _dist_rel_error(actual: dict, reference: dict, keys: list[str]) -> float:
    errors = []
    for key in keys:
        if key not in actual or key not in reference:
            continue
        errors.append(_rel_err(float(actual[key]), float(reference[key])))
    return max(errors) if errors else 0.0


def _hw_snap():
    g  = gpu_snapshot()
    cp = cpu_mem_snapshot()
    return g, cp


@lru_cache(maxsize=4)
def _resource_plan_for_exec_mode(mode: int):
    if not _LEIKA_OK:
        return None
    try:
        engine = _leika.Engine(mode=mode)
        return engine.resource_plan(), engine.hardware
    except Exception:
        return None


def _apply_resource_context(result: BenchResult) -> None:
    plan_data = None
    exec_mode = getattr(result, "leika_exec_mode", None)
    if exec_mode is not None:
        plan_data = _resource_plan_for_exec_mode(int(exec_mode))
    if plan_data is None and _LEIKA_OK:
        try:
            engine = _leika.Engine()
            plan_data = (engine.resource_plan(), engine.hardware)
        except Exception:
            plan_data = None

    if not plan_data:
        return

    plan, hw = plan_data
    result.cpu_total_threads = int(getattr(hw, "host_logical_threads", getattr(hw, "logical_threads", 0)) or 0)
    result.cpu_physical_cores = int(getattr(hw, "host_physical_cores", getattr(hw, "physical_cores", 0)) or 0)
    result.cpu_workers_selected = int(getattr(plan, "workers", 0) or 0)
    result.segments_selected = int(getattr(plan, "segments", 0) or 0)
    result.ram_total_gb = float(getattr(hw, "ram_gb", 0.0) or 0.0)
    result.ram_available_gb = float(getattr(hw, "available_ram_gb", 0.0) or 0.0)
    result.ram_budget_gb = float(getattr(hw, "safe_ram_budget_gb", 0.0) or 0.0)
    result.ram_peak_gb = float(result.mem_mb / 1024.0) if result.mem_mb > 0 else 0.0
    result.ram_peak_pct = (result.ram_peak_gb / result.ram_total_gb * 100.0) if result.ram_total_gb > 0 else 0.0
    result.section_overhead_ms = max(0.0, result.exec_ms - result.backtest_ms) if result.backtest_ms > 0 else 0.0


def _apply_cash_model(result: BenchResult) -> None:
    """Set cash_model, shared_data_sectioning, split_axis, execution_core, dynamic_sectioning_used."""
    # Respect pre-classification set by phase runners (e.g., shared_global_cash from Phase 1.75).
    if result.cash_model == "shared_global_cash":
        return
    is_leika = result.leika_exec_mode is not None
    exec_mode = result.leika_exec_mode or 0

    if result.phase_type in ("montecarlo", "randomwalk"):
        result.cash_model = "none"
        result.shared_data_sectioning = False
        result.split_axis = "paths"
        result.execution_core = "stochastic_paths"
        result.dynamic_sectioning_used = is_leika and exec_mode >= 1
        result.sell_at_end_scope = "none"
        result.dynamic_sectioning_preparation = False
        result.dynamic_sectioning_execution = False
        result.dynamic_sectioning_post_analysis = False
    elif result.phase == "Phase AI":
        result.cash_model = "none"
        result.shared_data_sectioning = False
        result.split_axis = "ai_sections"
        result.execution_core = "ai_analysis"
        result.dynamic_sectioning_used = False
        result.sell_at_end_scope = "none"
        result.dynamic_sectioning_preparation = False
        result.dynamic_sectioning_execution = False
        result.dynamic_sectioning_post_analysis = False
    elif result.n_assets <= 1:
        result.cash_model = "single_asset_cash"
        result.shared_data_sectioning = False
        result.split_axis = "none"
        result.execution_core = "single_pass"
        result.dynamic_sectioning_used = False
        result.sell_at_end_scope = "single_asset"
        result.dynamic_sectioning_preparation = False
        result.dynamic_sectioning_execution = False
        result.dynamic_sectioning_post_analysis = False
    else:  # n_assets > 1, independent per-symbol batch (current benchmark behavior)
        result.cash_model = "independent_per_symbol"
        result.shared_data_sectioning = False
        result.split_axis = "symbols"
        result.execution_core = "independent_batch"
        result.dynamic_sectioning_used = is_leika and exec_mode >= 1
        result.sell_at_end_scope = "per_symbol"
        result.dynamic_sectioning_preparation = is_leika and exec_mode >= 1
        result.dynamic_sectioning_execution = False
        result.dynamic_sectioning_post_analysis = False


def _aggregate_trade_summary(symbols, entries_sets, exits_sets):
    records = []
    for prices, entries, exits in zip(symbols, entries_sets, exits_sets):
        records.extend(build_trade_records(prices, entries, exits))
    summary = summarize_trade_records(records)
    summary["records"] = records
    return summary


# ── Single-asset runners ──────────────────────────────────────────────────────

def run_vectorbt_baseline(bars: int, mode_num: int, phase: str,
                          extreme: bool = False, progress: ProgressFn = None, **_) -> BenchResult:
    result = BenchResult(mode=mode_num, mode_name="vectorbt_baseline", bars=bars,
                         phase=phase, backend="vectorbt_baseline", n_assets=1)
    try:
        t_input = time.monotonic()
        prices           = generate_prices_extreme(bars) if extreme else generate_prices(bars, seed=SEED)
        entries, exits   = _signals_from_prices(prices)
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        gpu_pre, (_, mem_pre) = _hw_snap()
        t0 = time.monotonic()
        m  = _backtest_vbt(prices, entries, exits, use_rust_indicators=False)
        result.exec_ms = (time.monotonic() - t0) * 1000
        result.rust_engine_time_ms = 0.0
        _, (_, mem_post) = _hw_snap()
        result.mem_mb  = max(mem_pre, mem_post)
        _fill_result(result, m, prices, m["equity"])
        result.throughput_bars_sec = bars / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
    except Exception as exc:
        result.error = str(exc)
    return result


# ── Benchmark Orchestrator ────────────────────────────────────────────────────

DEFAULT_BARS = [1_000, 10_000, 100_000, 1_000_000]
EXTREME_BARS = [1_000, 10_000, 100_000, 500_000, 1_000_000]
PHASE15_BARS = [500, 1_000]
AI_BARS = [500, 1_000]
MC_CANDLE_COUNTS = [1_000, 10_000]
MC_PATH_COUNTS = [100, 500, 10_000, 100_000]
RW_CANDLE_COUNTS = [1_000, 10_000, 100_000]
RW_MODELS_ALL = ["gbm", "gaussian", "mean_reversion", "jump_diffusion", "regime_switching"]

DEFAULT_MODEL = "0xroyce/plutus:latest"
AI_COOLDOWN = 100

# Hard caps for Phase 3 so a bad CLI override does not OOM the benchmark.
MC_MAX_CANDLES_PER_CASE = int(os.environ.get("LEIKA_BENCH_MC_MAX_CANDLES", "20000"))
MC_MAX_PATHS_PER_CASE = int(os.environ.get("LEIKA_BENCH_MC_MAX_PATHS", "20000"))
MC_MAX_TOTAL_POINTS_PER_CASE = int(os.environ.get("LEIKA_BENCH_MC_MAX_POINTS", "10000000"))
MC_MAX_ESTIMATED_BYTES_PER_CASE = int(os.environ.get("LEIKA_BENCH_MC_MAX_BYTES", str(1_000_000_000)))

PHASE_GROUPS = {
    "Phase 1":      {"modes": range(1,  7),  "baseline_mode": 1},
    "Phase 1.5":    {"modes": range(7,  13), "baseline_mode": 7},
    "Phase 2":      {"modes": range(13, 19), "baseline_mode": 13},
    "Phase 2.5":    {"modes": range(19, 25), "baseline_mode": 19},
    "Phase 3":      {"modes": range(25, 31), "baseline_mode": 26},
    "Phase 4":      {"modes": range(31, 37), "baseline_mode": 32},
    "Phase 4.5":    {"modes": range(37, 43), "baseline_mode": 38},
    "Phase AI":     {"modes": range(43, 48), "baseline_mode": 43},
    "Phase DATA-1": {"modes": range(51, 59), "baseline_mode": 51},
}

DEFAULT_RUN_MODES = list(range(1, 43)) + list(range(61, 66))

PHASE_MODE_MAP = {
    "1":    list(range(1,  7)),
    "1.5":  list(range(7,  13)),
    "2":    list(range(13, 19)),
    "2.5":  list(range(19, 25)),
    "3":    list(range(25, 31)),
    "4":    list(range(31, 37)),
    "4.5":  list(range(37, 43)),
    "ai":   list(range(43, 48)),
    "data": list(range(51, 59)),
    "1.75": list(range(61, 66)),
    "all":  list(range(1,  48)) + list(range(51, 59)),
}

PHASE_ALIAS_MAP = {
    "phase1":   "1",
    "phase1_5": "1.5",
    "phase2":   "2",
    "phase2_5": "2.5",
    "phase3":   "3",
    "phase4":   "4",
    "phase4_5": "4.5",
    "phaseai":  "ai",
    "phasedata": "data",
    "phase1_75": "1.75",
}

AI_MODE_IDS      = set(range(13, 25))   # Phases 2 + 2.5
AI_BENCH_MODE_IDS = set(range(43, 48))  # Phase AI
MC_MODE_IDS      = set(range(25, 31))   # Phase 3
RW_MODE_IDS      = set(range(31, 43))   # Phases 4 + 4.5
DATA_MODE_IDS    = set(range(51, 59))   # Phase DATA-1
SHARED_MODE_IDS  = set(range(61, 66))   # Phase 1.75 — Shared Data Sectioning

MODE_LABELS = {
    # Phase 1 — single-asset portfolio
    1:  ("Phase 1", "vectorbt_baseline"),
    2:  ("Phase 1", "vectorbt_rust"),
    3:  ("Phase 1", "leika_mode_0  [CpuOnly]"),
    4:  ("Phase 1", "leika_mode_1  [Adaptive]"),
    5:  ("Phase 1", "leika_mode_2  [GpuAccelerated]"),
    6:  ("Phase 1", "leika_mode_3  [HybridCpuGpu]"),
    # Phase 1.5 — 5-asset portfolio
    7:  ("Phase 1.5", "vectorbt_baseline_5_assets"),
    8:  ("Phase 1.5", "vectorbt_rust_5_assets"),
    9:  ("Phase 1.5", "leika_mode_0_5_assets"),
    10: ("Phase 1.5", "leika_mode_1_5_assets"),
    11: ("Phase 1.5", "leika_mode_2_5_assets"),
    12: ("Phase 1.5", "leika_mode_3_5_assets"),
    # Phase 2 — single-asset + AI
    13: ("Phase 2", "vectorbt_baseline_ai"),
    14: ("Phase 2", "vectorbt_rust_ai"),
    15: ("Phase 2", "leika_mode_0_ai"),
    16: ("Phase 2", "leika_mode_1_ai"),
    17: ("Phase 2", "leika_mode_2_ai"),
    18: ("Phase 2", "leika_mode_3_ai"),
    # Phase 2.5 — 5-asset + AI
    19: ("Phase 2.5", "vectorbt_baseline_5_assets_ai"),
    20: ("Phase 2.5", "vectorbt_rust_5_assets_ai"),
    21: ("Phase 2.5", "leika_mode_0_5_assets_ai"),
    22: ("Phase 2.5", "leika_mode_1_5_assets_ai"),
    23: ("Phase 2.5", "leika_mode_2_5_assets_ai"),
    24: ("Phase 2.5", "leika_mode_3_5_assets_ai"),
    # Phase 3 — Monte Carlo
    25: ("Phase 3", "mc_python_baseline"),
    26: ("Phase 3", "mc_vectorbt_baseline"),
    27: ("Phase 3", "mc_leika_mode_0  [CpuOnly-serial]"),
    28: ("Phase 3", "mc_leika_mode_1  [Adaptive]"),
    29: ("Phase 3", "mc_leika_mode_2  [GpuAccelerated]"),
    30: ("Phase 3", "mc_leika_mode_3  [HybridCpuGpu]"),
    # Phase 4 — Random Walk single
    31: ("Phase 4", "rw_python_baseline"),
    32: ("Phase 4", "rw_vectorbt_baseline"),
    33: ("Phase 4", "rw_leika_mode_0"),
    34: ("Phase 4", "rw_leika_mode_1"),
    35: ("Phase 4", "rw_leika_mode_2"),
    36: ("Phase 4", "rw_leika_mode_3"),
    # Phase 4.5 — Random Walk 5-asset
    37: ("Phase 4.5", "rw_python_baseline_5_assets"),
    38: ("Phase 4.5", "rw_vectorbt_baseline_5_assets"),
    39: ("Phase 4.5", "rw_leika_mode_0_5_assets"),
    40: ("Phase 4.5", "rw_leika_mode_1_5_assets"),
    41: ("Phase 4.5", "rw_leika_mode_2_5_assets"),
    42: ("Phase 4.5", "rw_leika_mode_3_5_assets"),
    # Phase AI — AI benchmark
    43: ("Phase AI", "ai_latency_baseline"),
    44: ("Phase AI", "ai_single_leika_m1"),
    45: ("Phase AI", "ai_multi_leika_m1"),
    46: ("Phase AI", "ai_single_leika_m2"),
    47: ("Phase AI", "ai_multi_leika_m2"),
    # Phase DATA-1 — DataFrame interface benchmark (Pandas→VBT vs Polars→Leika)
    51: ("Phase DATA-1", "vectorbt_pandas_full"),
    52: ("Phase DATA-1", "leika_polars_full_m0  [CpuOnly]"),
    53: ("Phase DATA-1", "leika_polars_full_m1  [Adaptive]"),
    54: ("Phase DATA-1", "leika_polars_full_m2  [GpuAccelerated]"),
    55: ("Phase DATA-1", "vectorbt_pandas_parity"),
    56: ("Phase DATA-1", "leika_polars_parity   [Adaptive]"),
    57: ("Phase DATA-1", "vectorbt_pandas_5assets"),
    58: ("Phase DATA-1", "leika_polars_5assets_m1  [Adaptive]"),
    # Phase 1.75 — Shared Data Sectioning
    61: ("Phase 1.75", "independent_per_symbol_ref_5assets  [round-trip fees]"),
    62: ("Phase 1.75", "shared_data_sectioning_mode0_5assets"),
    63: ("Phase 1.75", "shared_data_sectioning_mode1_5assets"),
    64: ("Phase 1.75", "shared_data_sectioning_mode2_5assets"),
    65: ("Phase 1.75", "shared_data_sectioning_mode3_5assets"),
}


def _parse_phase_filter(value: str) -> list[int]:
    modes: list[int] = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in PHASE_MODE_MAP:
            raise ValueError(f"Unknown phase '{raw}'. Use one or more of: 1, 1.5, 2, 2.5, 3, 4, 4.5, ai, data, all")
        modes.extend(PHASE_MODE_MAP[key])
    return sorted(set(modes))


def _phase_modes_from_aliases(args: argparse.Namespace) -> list[int]:
    modes: list[int] = []
    for attr, phase_key in PHASE_ALIAS_MAP.items():
        if getattr(args, attr, False):
            modes.extend(PHASE_MODE_MAP[phase_key])
    return sorted(set(modes))


DATA_BARS = [1_000, 10_000, 100_000, 1_000_000]

def _datasets_for(mode: int, bar_sizes: list[int], extreme: bool,
                  mc_paths: list[int], rw_models: list[str]) -> list[dict]:
    if mode in AI_BENCH_MODE_IDS:
        return [{"bars": b} for b in AI_BARS]
    if mode in AI_MODE_IDS:
        return [{"bars": b} for b in AI_BARS]
    if mode in MC_MODE_IDS:
        return [{"n_candles": c, "n_paths": p} for c in MC_CANDLE_COUNTS for p in mc_paths]
    if mode in RW_MODE_IDS:
        return [{"n_candles": c, "rw_model": m} for c in RW_CANDLE_COUNTS for m in rw_models]
    if mode in DATA_MODE_IDS:
        sizes = [b for b in DATA_BARS if b <= max(bar_sizes, default=DATA_BARS[-1])] or DATA_BARS
        return [{"bars": b} for b in sizes]
    if mode in SHARED_MODE_IDS:
        sizes = [b for b in bar_sizes if b <= max(bar_sizes, default=100_000)] or bar_sizes
        return [{"bars": b} for b in sizes]
    bars = EXTREME_BARS if extreme and set(bar_sizes) == set(DEFAULT_BARS) else bar_sizes
    return [{"bars": b} for b in bars]


def _dataset_label(ds: dict) -> str:
    if "n_paths" in ds:
        return f"{ds['n_candles']:,} candles × {ds['n_paths']:,} paths"
    if "rw_model" in ds:
        return f"{ds['n_candles']:,} candles  [{ds['rw_model']}]"
    return f"{ds['bars']:,} bars"


def _dataset_progress_units(mode: int, ds: dict) -> int:
    if "n_paths" in ds:
        return int(ds["n_candles"]) * int(ds["n_paths"])
    if mode in RW_MODE_IDS:
        return int(ds["n_candles"]) * (5 if mode >= 31 else 1)
    if mode in AI_BENCH_MODE_IDS:
        return int(ds["bars"]) * (5 if mode in {38, 40} else 1)
    return int(ds.get("bars", ds.get("n_candles", 0)))


def _mc_chunk_paths(n_candles: int, n_paths: int) -> int:
    """Return a safe Phase 3 chunk size that fits the memory budget."""
    if n_candles <= 0 or n_paths <= 0:
        return 0
    by_paths = MC_MAX_PATHS_PER_CASE
    by_points = max(1, MC_MAX_TOTAL_POINTS_PER_CASE // n_candles)
    by_bytes = max(1, MC_MAX_ESTIMATED_BYTES_PER_CASE // max(1, n_candles * 8))
    return max(1, min(n_paths, by_paths, by_points, by_bytes))


def _mc_path_max_drawdown(path: list[float]) -> float:
    """Max drawdown for one path, in percent."""
    if not path:
        return 0.0
    peak = path[0]
    max_dd = 0.0
    for value in path:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _mc_finalize_stats(returns: list[float], max_dds: list[float]) -> dict:
    """Compute aggregate Monte Carlo stats from per-path samples."""
    if not returns:
        return {
            "prob_positive_pct": 0.0,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "std_return_pct": 0.0,
            "median_max_dd_pct": 0.0,
        }
    returns.sort()
    max_dds.sort()
    n = len(returns)
    mean = sum(returns) / n
    prob = sum(1 for r in returns if r > 0) / n * 100.0
    median = returns[n // 2]
    var = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(var) if var > 0 else 0.0
    median_dd = max_dds[len(max_dds) // 2] if max_dds else 0.0
    return {
        "prob_positive_pct": prob,
        "mean_return_pct": mean,
        "median_return_pct": median,
        "std_return_pct": std,
        "median_max_dd_pct": median_dd,
    }


def _compute_speedups(results: list[BenchResult]):
    from collections import defaultdict

    groups: dict = defaultdict(dict)
    for r in results:
        key = (r.phase, r.bars, r.n_paths, r.rw_model)
        groups[key][r.mode] = r

    for (phase, bars, n_paths, rw_model), mode_map in groups.items():
        baseline_mode = None
        for pg_name, pg in PHASE_GROUPS.items():
            if phase == pg_name:
                baseline_mode = pg["baseline_mode"]
                break
        if baseline_mode is None or baseline_mode not in mode_map:
            continue
        baseline_ms = mode_map[baseline_mode].exec_ms
        if baseline_ms <= 0:
            continue
        for r in mode_map.values():
            if r.exec_ms > 0:
                r.speedup_vs_baseline = baseline_ms / r.exec_ms
                if r.cpu_workers_selected > 0:
                    r.parallel_efficiency = r.speedup_vs_baseline / r.cpu_workers_selected
                else:
                    r.parallel_efficiency = 0.0
            else:
                r.parallel_efficiency = 0.0


def export_visualizations(result: BenchResult, base_dir: str):
    if not _VIZ_AVAILABLE or result.error or not result.prices:
        return
    out_dir = Path(base_dir) / str(result.mode) / str(result.bars)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        svg = viz.candles(result.prices, result.prices, result.prices, result.prices)
        (out_dir / "candles.svg").write_text(svg)
        if result.equity:
            (out_dir / "equity.svg").write_text(viz.equity_curve(result.equity))
        if result.drawdowns:
            (out_dir / "drawdown.svg").write_text(viz.drawdown_curve(result.drawdowns))
        if result.ema_fast:
            (out_dir / "indicators.svg").write_text(viz.equity_curve(result.ema_fast))
    except Exception:
        pass


def _run_with_monitor(runner_fn, mode: int, label: str, live: bool,
                      progress=None, warmup: int = 1, timing_runs: int = 1, **kwargs) -> tuple[BenchResult, dict]:
    n_bars = kwargs.get("bars") or kwargs.get("n_candles", 0)
    force_monitor = bool(kwargs.pop("_force_monitor", False))

    # Warmup: identical code path, progress suppressed, results discarded.
    for _ in range(warmup):
        try:
            runner_fn(progress=None, **kwargs)
        except Exception:
            pass

    monitor = None
    if live or force_monitor:
        monitor = LiveMonitor(
            total_bars=n_bars,
            mode=mode,
            interval=0.2 if force_monitor else 0.8,
            label=label,
        )
        monitor.start()

    # First (or only) timed run — this is the canonical result.
    t0 = time.monotonic()
    try:
        result = runner_fn(progress=progress, **kwargs)
    except Exception as exc:
        phase, name = MODE_LABELS.get(mode, ("?", f"mode{mode}"))
        result = BenchResult(mode=mode, mode_name=name, bars=n_bars, error=str(exc))
    elapsed = (time.monotonic() - t0) * 1000

    stats: dict = {}
    if monitor:
        monitor.stop()
        stats = monitor.summary()

    result.exec_ms = result.exec_ms or elapsed
    result.timing_runs = [result.exec_ms]

    # Additional timing runs for stability metrics (no progress callback, no monitor).
    for _ in range(max(0, timing_runs - 1)):
        try:
            t1 = time.monotonic()
            runner_fn(progress=None, **kwargs)
            result.timing_runs.append((time.monotonic() - t1) * 1000)
        except Exception:
            pass

    from metrics import compute_stability_metrics
    compute_stability_metrics(result)

    if stats:
        result.cpu_pct = max(result.cpu_pct, float(stats.get("peak_cpu_pct", 0.0) or 0.0))
        result.mem_mb = max(result.mem_mb, float(stats.get("peak_mem_mb", 0.0) or 0.0))
    result.total_runtime_ms = result.total_runtime_ms or result.exec_ms
    result.warmup_runs = warmup
    _apply_resource_context(result)
    _apply_cash_model(result)
    return result, stats


def _set_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    return old


def _restore_env(name: str, old: Optional[str]) -> None:
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


def _stress_portfolio_batch(asset_count: int, bars: int, exec_mode: int, multiplier: int) -> BenchResult:
    backend = f"stress_portfolio_m{exec_mode}_x{multiplier}"
    result = BenchResult(
        mode=900 + asset_count,
        mode_name=backend,
        bars=bars,
        phase="Dynamic Stress",
        phase_type="portfolio",
        backend=f"leika_mode_{exec_mode}",
        n_assets=asset_count,
        leika_exec_mode=exec_mode,
    )
    if not _LEIKA_OK:
        result.error = "leika module unavailable"
        return result

    old = _set_env("LEIKA_DYNAMIC_SECTION_MULTIPLIER", str(multiplier))
    try:
        import leika

        portfolios = []
        for i in range(asset_count):
            prices = generate_prices(bars, seed=SEED + i)
            entries, exits, _ = _signals_from_prices_rust(prices)
            portfolios.append(
                leika.Portfolio.from_signals(prices, entries, exits)
                .init_cash(INIT_CASH)
                .fees(FEES)
            )

        t0 = time.monotonic()
        _ = leika.Portfolio.run_batch(portfolios=portfolios, mode=exec_mode)
        result.exec_ms = (time.monotonic() - t0) * 1000
        result.throughput_bars_sec = (asset_count * bars) / (result.exec_ms / 1000) if result.exec_ms > 0 else 0.0
        result.cpu_pct = 0.0
        result.mem_mb = 0.0
        result.total_runtime_ms = result.exec_ms
    except Exception as exc:
        result.error = str(exc)
    finally:
        _restore_env("LEIKA_DYNAMIC_SECTION_MULTIPLIER", old)
    return result


def _stress_mc_case(n_paths: int, multiplier: int, exec_mode: int = 2) -> BenchResult:
    old = _set_env("LEIKA_DYNAMIC_SECTION_MULTIPLIER", str(multiplier))
    try:
        result = run_mc_leika(252, n_paths, exec_mode, 910 + n_paths, "Dynamic Stress")
    finally:
        _restore_env("LEIKA_DYNAMIC_SECTION_MULTIPLIER", old)
    return result


def _stress_rw_case(n_paths: int, multiplier: int, exec_mode: int = 2) -> BenchResult:
    old = _set_env("LEIKA_DYNAMIC_SECTION_MULTIPLIER", str(multiplier))
    try:
        result = run_rw_leika(252, "gbm", exec_mode, 920 + n_paths, "Dynamic Stress")
        result.n_paths = n_paths
        if result.exec_ms > 0:
            result.paths_sec = n_paths / (result.exec_ms / 1000.0)
    finally:
        _restore_env("LEIKA_DYNAMIC_SECTION_MULTIPLIER", old)
    return result


def _dynamic_stress_limit_gb() -> float:
    raw = os.environ.get("LEIKA_DYNAMIC_STRESS_RAM_PCT", "70")
    try:
        pct = float(raw)
    except ValueError:
        pct = 70.0
    pct = max(1.0, min(pct, 100.0))
    return pct


def _write_dynamic_audit(results: list[BenchResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Dynamic Sectioning Audit")
    lines.append("")
    lines.append("## Current Implementation Map")
    lines.append("- `ExecutionPlan` creates worker and segment targets in Rust.")
    lines.append("- `SegmentationEngine` is consumed by portfolio batch, Monte Carlo, and Random Walk CPU paths.")
    lines.append("- Python APIs `display_info()`, `detect_hardware()`, `resource_plan()`, and `dynamic_section_report()` surface the plan.")
    lines.append("")
    lines.append("## Engines Using Dynamic Sections")
    lines.append("- Portfolio batch symbols")
    lines.append("- Monte Carlo CPU path fill")
    lines.append("- Random Walk CPU path fill")
    lines.append("")
    lines.append("## Engines Bypassing Dynamic Sections")
    lines.append("- Single-symbol portfolio remains sequential by design")
    lines.append("- VectorBT baselines bypass Leika planning entirely")
    lines.append("")

    def _fmt_pct(v: float) -> str:
        return f"{v:.1f}%"

    lines.append("## CPU Utilization Table")
    lines.append("| Workload | Workers | Segments | CPU% | Threads | Runtime ms | Speedup | Efficiency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r.phase != "Dynamic Stress":
            continue
        lines.append(
            f"| {r.mode_name} | {r.cpu_workers_selected or '—'} | {r.segments_selected or '—'} | "
            f"{_fmt_pct(r.cpu_pct)} | {r.cpu_total_threads or '—'} | {r.exec_ms:.1f} | "
            f"{(r.speedup_vs_baseline or 0.0):.2f}× | {((r.parallel_efficiency or 0.0) * 100.0):.1f}% |"
        )
    lines.append("")
    lines.append("## RAM Utilization Table")
    lines.append("| Workload | RAM total GB | RAM budget GB | RAM peak GB | RAM% | Swap | OOM risk |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for r in results:
        if r.phase != "Dynamic Stress":
            continue
        oom = "yes" if r.ram_budget_gb and r.ram_peak_gb > r.ram_budget_gb else "no"
        lines.append(
            f"| {r.mode_name} | {r.ram_total_gb:.1f} | {r.ram_budget_gb:.1f} | {r.ram_peak_gb:.1f} | "
            f"{r.ram_peak_pct:.1f}% | no | {oom} |"
        )
    lines.append("")
    lines.append("## Optimal Worker / Segment Table")
    lines.append("| Workload | Recommended workers | Recommended segments | Reason |")
    lines.append("|---|---:|---:|---|")
    for r in results:
        if r.phase != "Dynamic Stress":
            continue
        lines.append(
            f"| {r.mode_name} | {r.cpu_workers_selected or '—'} | {r.segments_selected or '—'} | "
            f"Measured run under multiplier sweep; {r.segments_selected or '—'} segments selected. |"
        )
    lines.append("")
    lines.append("## Recommended Planner Changes")
    lines.append("- Keep single-symbol portfolio sequential.")
    lines.append("- Keep worker count capped by visible CPU quota/cpuset and safe RAM budget.")
    lines.append("- Prefer fewer sections for memory-heavy batches.")
    lines.append("")
    lines.append("## Before / After Benchmark Comparison")
    lines.append("- Baseline runs use fixed phase benchmarks; stress runs sweep dynamic section multipliers.")
    lines.append("")
    lines.append("## Remaining Bottlenecks")
    lines.append("- Python object creation dominates the largest stress cases.")
    lines.append("- Memory bandwidth becomes the limit before CPU saturation on the largest batches.")
    lines.append("")
    lines.append("## Vast.ai Recommendations")
    lines.append("- Honour cgroup CPU quota and memory limit rather than host totals.")
    lines.append("- Keep the RAM budget conservative for shared GPUs / oversubscribed nodes.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_dynamic_stress(args: argparse.Namespace) -> None:
    if not _LEIKA_OK:
        raise RuntimeError("leika module is required for dynamic stress runs")

    try:
        hw = hw_detect.detect()
        ram_budget_gb = getattr(hw.mem, "total_gb", 0.0) * (_dynamic_stress_limit_gb() / 100.0)
    except Exception:
        ram_budget_gb = 0.0

    multipliers = [1, 2, 4, 8, 16]
    results: list[BenchResult] = []
    max_runtime_ms = float(os.environ.get("LEIKA_DYNAMIC_STRESS_MAX_RUNTIME_MS", "120000"))

    def _record(result: BenchResult, multiplier: int) -> None:
        _apply_resource_context(result)
        result.ram_budget_gb = result.ram_budget_gb or ram_budget_gb
        result.ram_peak_gb = result.ram_peak_gb or (result.mem_mb / 1024.0 if result.mem_mb else 0.0)
        if result.ram_total_gb <= 0 and getattr(hw, "mem", None):
            result.ram_total_gb = float(hw.mem.total_gb)
            result.ram_available_gb = float(hw.mem.available_gb)
        result.ram_peak_pct = (result.ram_peak_gb / result.ram_total_gb * 100.0) if result.ram_total_gb > 0 else 0.0
        if result.cpu_workers_selected > 0:
            result.segments_selected = max(result.segments_selected, result.cpu_workers_selected * multiplier)
        if result.exec_ms > max_runtime_ms:
            result.error = result.error or f"runtime limit exceeded ({result.exec_ms:.1f} ms > {max_runtime_ms:.1f} ms)"
        if result.ram_budget_gb > 0 and result.ram_peak_gb > result.ram_budget_gb:
            result.error = result.error or f"RAM budget exceeded ({result.ram_peak_gb:.2f} GB > {result.ram_budget_gb:.2f} GB)"
        results.append(result)

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║             LEIKA DYNAMIC SECTIONING STRESS TEST               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    portfolio_assets = [5, 50, 100, 500, 1000]
    for assets in portfolio_assets:
        for mult in multipliers:
            r, stats = _run_with_monitor(
                lambda progress=None, **_: _stress_portfolio_batch(assets, bars=1_000, exec_mode=2, multiplier=mult),
                900 + assets,
                f"portfolio_{assets}",
                False,
                progress=None,
                _force_monitor=True,
            )
            if stats:
                r.cpu_pct = max(r.cpu_pct, float(stats.get("peak_cpu_pct", 0.0) or 0.0))
                r.mem_mb = max(r.mem_mb, float(stats.get("peak_mem_mb", 0.0) or 0.0))
            _record(r, mult)
            print(f"  portfolio {assets:>4} assets × m{mult:<2} : {r.exec_ms:,.1f} ms")
            if r.error:
                print(f"    !! {r.error}")
                break
        if results and results[-1].error:
            break

    mc_paths = [100, 10_000, 100_000, 1_000_000]
    for paths in mc_paths:
        for mult in multipliers:
            r, stats = _run_with_monitor(
                lambda progress=None, **_: _stress_mc_case(paths, multiplier=mult, exec_mode=2),
                910 + paths,
                f"mc_{paths}",
                False,
                progress=None,
                _force_monitor=True,
            )
            if stats:
                r.cpu_pct = max(r.cpu_pct, float(stats.get("peak_cpu_pct", 0.0) or 0.0))
                r.mem_mb = max(r.mem_mb, float(stats.get("peak_mem_mb", 0.0) or 0.0))
            _record(r, mult)
            print(f"  montecarlo {paths:>7,} paths × m{mult:<2} : {r.exec_ms:,.1f} ms")
            if r.error:
                print(f"    !! {r.error}")
                break
        if results and results[-1].error:
            break

    rw_paths = [1, 100, 500, 10_000]
    for paths in rw_paths:
        for mult in multipliers:
            r, stats = _run_with_monitor(
                lambda progress=None, **_: _stress_rw_case(paths, multiplier=mult, exec_mode=2),
                920 + paths,
                f"rw_{paths}",
                False,
                progress=None,
                _force_monitor=True,
            )
            if stats:
                r.cpu_pct = max(r.cpu_pct, float(stats.get("peak_cpu_pct", 0.0) or 0.0))
                r.mem_mb = max(r.mem_mb, float(stats.get("peak_mem_mb", 0.0) or 0.0))
            _record(r, mult)
            print(f"  randomwalk {paths:>7,} paths × m{mult:<2} : {r.exec_ms:,.1f} ms")
            if r.error:
                print(f"    !! {r.error}")
                break
        if results and results[-1].error:
            break

    _compute_speedups(results)
    out_dir = Path(getattr(args, "out_dir", "benchmark_results"))
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    audit_path = _write_dynamic_audit(results, out_dir.parent / "dynamic_sectioning_audit.md")
    report_path = report_gen.generate_report(results, out_dir=str(out_dir), timestamp=ts)
    print(f"\n  Audit : {audit_path}")
    print(f"  Report: {report_path}")


def _ai_stress_case(
    *,
    mode_num: int,
    bars: int,
    n_assets: int,
    ai_mode: str,
    dynamic_sections: bool,
    model: str,
    extreme: bool = False,
) -> BenchResult:
    if ai_mode == "quick-ai":
        config = BenchmarkAiConfig(
            ai_mode=ai_mode,
            model=model,
            max_context_chars=8_000,
            max_output_tokens=256,
            timeout_seconds=30,
            max_ai_cases=None,
        )
    else:
        config = BenchmarkAiConfig(
            ai_mode=ai_mode,
            model=model,
            max_context_chars=16_000,
            max_output_tokens=1_024,
            timeout_seconds=120,
            max_ai_cases=None,
        )
    result = run_with_ai(
        bars=bars,
        mode_num=mode_num,
        phase="AI Dynamic Stress",
        backend=f"ai_{ai_mode}_{'dynamic' if dynamic_sections else 'baseline'}",
        exec_mode=2,
        n_assets=n_assets,
        model=model,
        extreme=extreme,
        ai_config=config,
        dynamic_sections=dynamic_sections,
    )
    result.phase = "AI Dynamic Stress"
    result.phase_type = "portfolio"
    result.mode_name = f"{ai_mode}_{n_assets}assets_{'dynamic' if dynamic_sections else 'baseline'}"
    return result


def run_ai_dynamic_stress(args: argparse.Namespace) -> None:
    if not _LEIKA_OK:
        raise RuntimeError("leika module is required for AI dynamic stress runs")

    model = os.environ.get("LEIKA_MODEL", getattr(args, "model", DEFAULT_MODEL))
    bars = 1_000
    scenarios = [("quick-ai", 1), ("quick-ai", 5), ("full-ai", 1), ("full-ai", 5)]

    results: list[BenchResult] = []
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                LEIKA AI DYNAMIC STRESS TEST                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    for idx, (ai_mode, n_assets) in enumerate(scenarios, start=1):
        dyn = _ai_stress_case(
            mode_num=100 + idx,
            bars=bars,
            n_assets=n_assets,
            ai_mode=ai_mode,
            dynamic_sections=True,
            model=model,
            extreme=getattr(args, "extreme", False),
        )
        base = _ai_stress_case(
            mode_num=200 + idx,
            bars=bars,
            n_assets=n_assets,
            ai_mode=ai_mode,
            dynamic_sections=False,
            model=model,
            extreme=getattr(args, "extreme", False),
        )
        results.extend([dyn, base])
        print(
            f"  {ai_mode:<8} {n_assets} asset(s) | dynamic {dyn.ai_total_time_ms:.1f} ms "
            f"({dyn.ai_calls} calls) vs baseline {base.ai_total_time_ms:.1f} ms "
            f"({base.ai_calls} call)"
        )

    out_dir = Path(getattr(args, "out_dir", "benchmark_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = report_gen.generate_report(results, out_dir=str(out_dir), timestamp=ts)
    print(f"\n  Report: {report_path}")


def run_ai_single_call_baseline(args: argparse.Namespace) -> None:
    if not _LEIKA_OK:
        raise RuntimeError("leika module is required for AI baseline runs")

    model = os.environ.get("LEIKA_MODEL", getattr(args, "model", DEFAULT_MODEL))
    bars = 1_000
    scenarios = [("quick-ai", 1), ("quick-ai", 5), ("full-ai", 1), ("full-ai", 5)]

    results: list[BenchResult] = []
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║              LEIKA AI SINGLE-CALL BASELINE                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    for idx, (ai_mode, n_assets) in enumerate(scenarios, start=1):
        base = _ai_stress_case(
            mode_num=300 + idx,
            bars=bars,
            n_assets=n_assets,
            ai_mode=ai_mode,
            dynamic_sections=False,
            model=model,
            extreme=getattr(args, "extreme", False),
        )
        results.append(base)
        print(f"  {ai_mode:<8} {n_assets} asset(s) | baseline {base.ai_total_time_ms:.1f} ms ({base.ai_calls} call)")

    out_dir = Path(getattr(args, "out_dir", "benchmark_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = report_gen.generate_report(results, out_dir=str(out_dir), timestamp=ts)
    print(f"\n  Report: {report_path}")


def _prompt_choice(prompt: str, options: list[str], default: int = 1) -> int:
    print(f"\n{prompt}")
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    raw = input(f"Select [default {default}]: ").strip()
    if not raw:
        return default
    try:
        choice = int(raw)
        if 1 <= choice <= len(options):
            return choice
    except ValueError:
        pass
    return default


def _prompt_bool(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true", "t"}


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _menu_args() -> argparse.Namespace:
    if not sys.stdin.isatty():
        return argparse.Namespace(
            modes_list=DEFAULT_RUN_MODES,
            bars=",".join(str(b) for b in DEFAULT_BARS),
            mc_paths=",".join(str(p) for p in MC_PATH_COUNTS),
            rw_models=",".join(RW_MODELS_ALL),
            model=DEFAULT_MODEL,
            live=True,
            extreme=False,
            out_dir="benchmark_results",
            no_ai=False,
            test=False,
            benchmark=False,
            mc=False,
            rw=False,
            all=False,
            ai_bench=False,
            phase="",
            modes="",
            phase1=False,
            phase1_5=False,
            phase2=False,
            phase2_5=False,
            phase3=False,
            phase4=False,
            phase4_5=False,
            phaseai=False,
            phasedata=False,
            phase1_75=False,
            warmup=1,
        )

    if _RICH_AVAILABLE:
        console = Console()
        table = Table(title="Leika Benchmark Menu", title_style="bold cyan", show_header=False, box=None)
        table.add_column("#", style="bold cyan", width=3)
        table.add_column("Option", style="bold")
        table.add_row("1", "Portfolio only (Phases 1 + 1.5)")
        table.add_row("2", "Portfolio + AI (Phases 1 + 1.5 + 2 + 2.5)")
        table.add_row("3", "Monte Carlo only (Phase 3)")
        table.add_row("4", "Random Walk only (Phases 4 + 4.5)")
        table.add_row("5", "Everything (including Phase 1.75)")
        table.add_row("6", "Custom phase selection")
        console.print(Panel.fit("Choose what to run. Defaults are safe and fast.", border_style="cyan"))
        console.print(table)
    else:
        print("\nLeika Benchmark Menu")
        print("Choose what to run. Defaults are safe and fast.")
        print("  1. Portfolio only (Phases 1 + 1.5)")
        print("  2. Portfolio + AI (Phases 1 + 1.5 + 2 + 2.5)")
        print("  3. Monte Carlo only (Phase 3)")
        print("  4. Random Walk only (Phases 4 + 4.5)")
        print("  5. Everything (including Phase 1.75)")
        print("  6. Custom phase selection")

    choice = _prompt_choice("What do you want to benchmark?", [
        "Portfolio only (Phases 1 + 1.5)",
        "Portfolio + AI (Phases 1 + 1.5 + 2 + 2.5)",
        "Monte Carlo only (Phase 3)",
        "Random Walk only (Phases 4 + 4.5)",
        "Everything",
        "Custom phase selection",
    ], default=5)

    live = _prompt_bool("Use live progress", True)
    extreme = _prompt_bool("Use extreme dataset", False)
    out_dir = _prompt_text("Output dir", "benchmark_results")

    model = DEFAULT_MODEL
    mc_paths = ",".join(str(p) for p in MC_PATH_COUNTS)
    rw_models = ",".join(RW_MODELS_ALL)
    modes_list: list[int] | None = None
    no_ai = False
    benchmark = False
    mc = False
    rw = False
    ai_bench = False
    phase = ""

    if choice == 1:
        benchmark = True
        modes_list = list(range(1, 13))
        no_ai = True
    elif choice == 2:
        modes_list = list(range(1, 25))
    elif choice == 3:
        mc = True
        modes_list = list(range(25, 31))
    elif choice == 4:
        rw = True
        modes_list = list(range(31, 43))
    elif choice == 5:
        benchmark = True
        mc = True
        rw = True
        ai_bench = True
        modes_list = list(range(1, 48)) + list(range(61, 66))
    else:
        phase = _prompt_text("Enter phases (comma-separated, e.g. 1,3,4.5,ai)", "1,1.5,1.75,2,2.5,3,4,4.5")
        modes_list = _parse_phase_filter(phase)

    return argparse.Namespace(
        modes_list=modes_list,
        bars=",".join(str(b) for b in DEFAULT_BARS),
        mc_paths=mc_paths,
        rw_models=rw_models,
        model=model,
        live=live,
        extreme=extreme,
        out_dir=out_dir,
        no_ai=no_ai,
        test=False,
        benchmark=benchmark,
        mc=mc,
        rw=rw,
        all=choice == 5,
        ai_bench=ai_bench,
        phase=phase,
        modes="",
        phase1=False,
        phase1_5=False,
        phase2=False,
        phase2_5=False,
        phase3=False,
        phase4=False,
        phase4_5=False,
        phaseai=False,
        phasedata=False,
        phase1_75=False,
        warmup=1,
    )


def main():
    import phase1_single
    import phase15_multi
    import phase2_single_ai
    import phase25_multi_ai
    import phase3_montecarlo
    import phase4_randomwalk
    import phase45_rw_multi
    import phase_ai_bench
    import phase_data1
    import phase175_shared

    if len(sys.argv) == 1:
        args = _menu_args()
    else:
        parser = argparse.ArgumentParser(description="Leika benchmark orchestrator")
        parser.add_argument("--bars", default=",".join(str(b) for b in DEFAULT_BARS))
        parser.add_argument("--modes", default="")
        parser.add_argument("--model", default=DEFAULT_MODEL)
        parser.add_argument("--no-ai", action="store_true", help="Skip phases 2 and 2.5")
        parser.add_argument("--extreme", action="store_true", help="High-volatility dataset")
        parser.add_argument("--out-dir", default="benchmark_results")
        parser.add_argument("--test", action="store_true", help="Smoke: mode 3, 1k bars")
        parser.add_argument("--benchmark", action="store_true", help="Phase 1 + 1.5, no AI")
        parser.add_argument("--mc", action="store_true", help="Include Phase 3 (Monte Carlo)")
        parser.add_argument("--rw", action="store_true", help="Include Phase 4 + 4.5 (Random Walk)")
        parser.add_argument("--all", action="store_true", help="All phases: portfolio + MC + RW")
        parser.add_argument("--mc-paths", default=",".join(str(p) for p in MC_PATH_COUNTS), help="MC path counts (Phase 3)")
        parser.add_argument("--rw-models", default=",".join(RW_MODELS_ALL), help="RW model subset (Phase 4/4.5)")
        parser.add_argument("--phase", default="", help="Run only the selected phase(s): 1, 1.5, 2, 2.5, 3, 4, 4.5, ai, all. Comma-separated allowed.")
        parser.add_argument("--phase1", action="store_true", help="Shortcut for --phase 1")
        parser.add_argument("--phase1.5", dest="phase1_5", action="store_true", help="Shortcut for --phase 1.5")
        parser.add_argument("--phase2", action="store_true", help="Shortcut for --phase 2")
        parser.add_argument("--phase2.5", dest="phase2_5", action="store_true", help="Shortcut for --phase 2.5")
        parser.add_argument("--phase3", action="store_true", help="Shortcut for --phase 3")
        parser.add_argument("--phase4", action="store_true", help="Shortcut for --phase 4")
        parser.add_argument("--phase4.5", dest="phase4_5", action="store_true", help="Shortcut for --phase 4.5")
        parser.add_argument("--phaseai", action="store_true", help="Shortcut for --phase ai")
        parser.add_argument("--phasedata", action="store_true", help="Shortcut for --phase data (DataFrame interface benchmark)")
        parser.add_argument("--phase1.75", dest="phase1_75", action="store_true", help="Shortcut for --phase 1.75")
        parser.add_argument("--ai-bench", action="store_true", help="include AI benchmark phases (modes 36-40)")
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--parity-check", action="store_true", help="Run the VectorBT vs Leika parity benchmark")
        parser.add_argument("--quick", action="store_true", help="Use the small parity dataset set")
        parser.add_argument("--dynamic-stress", action="store_true", help="Sweep dynamic section multipliers across large workloads")
        parser.add_argument("--ai-dynamic-stress", action="store_true", help="Compare dynamic multi-section AI vs single-call baseline")
        parser.add_argument("--ai-single-call-baseline", action="store_true", help="Run the single-call AI baseline only")
        parser.add_argument("--warmup", type=int, default=1, metavar="N",
                            help="Warmup runs per case before official timing (default: 1)")
        parser.add_argument("--timing-runs", type=int, default=1, metavar="N",
                            help="Timed runs per case for stability metrics; median is used as official time (default: 1)")
        parser.add_argument("--assemble-from", metavar="TS", default=None,
                            help="Skip running; load checkpoint_*_TS.json and generate final report")
        parser.add_argument("--export-ods", dest="export_ods", action="store_true", default=False,
                            help="Export a LibreOffice Calc .ods workbook (default: on for full runs, "
                                 "must be explicit for --test / --quick)")
        args = parser.parse_args()

    if getattr(args, "parity_check", False):
        phase_arg = getattr(args, "phase", "")
        if phase_arg == "data" or getattr(args, "phasedata", False):
            import phase_data1 as _pd1
            bar_sizes = [int(b) for b in getattr(args, "bars", "1000,10000,100000").split(",") if b.strip()]
            print("\n╔══════════════════════════════════════════════════════════════════╗")
            print("║          LEIKA — Phase DATA-1 Parity Check                      ║")
            print("╚══════════════════════════════════════════════════════════════════╝")
            for bars in bar_sizes:
                res = _pd1.run_parity_check(bars, n_assets=1)
                status = res.get("parity_status", "UNKNOWN")
                icon = "✓" if status == "PASS" else ("~" if status == "PARTIAL" else "✗")
                print(f"\n  {icon} {bars:>9,} bars | {status}")
                if res.get("indicator_diff_note"):
                    print(f"    NOTE: {res['indicator_diff_note']}")
                for note in res.get("notes", []):
                    print(f"    !! {note}")
                if status in ("PASS", "PARTIAL"):
                    print(f"    VBT Pandas total_return_pct : {res.get('vbt_total_return', 0):.4f}%")
                    print(f"    Leika Polars total_return_pct: {res.get('lk_total_return', 0):.4f}%")
                    print(f"    VBT trades: {res.get('vbt_trades', '?')}  Leika trades: {res.get('lk_trades', '?')}")
            print()
        else:
            from parity import run_parity_benchmark
            run_parity_benchmark(args)
        return

    if getattr(args, "dynamic_stress", False):
        run_dynamic_stress(args)
        return

    if getattr(args, "ai_dynamic_stress", False):
        run_ai_dynamic_stress(args)
        return

    if getattr(args, "ai_single_call_baseline", False):
        run_ai_single_call_baseline(args)
        return

    if getattr(args, "modes_list", None) is not None:
        modes = list(args.modes_list)
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║           LEIKA BENCHMARK — Interactive menu selection          ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    elif args.test:
        args.bars = "1000"
        modes = [3]
        args.no_ai = True
        args.live = True
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║       LEIKA SMOKE TEST — leika_mode_0 (CpuOnly), 1k bars        ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    elif args.modes:
        modes = [int(m.strip()) for m in args.modes.split(",") if m.strip()]
    elif any(getattr(args, attr, False) for attr in PHASE_ALIAS_MAP):
        modes = _phase_modes_from_aliases(args)
        selected = [k for k, v in PHASE_ALIAS_MAP.items() if getattr(args, k, False)]
        if selected == ["phase1_5"]:
            args.bars = ",".join(str(b) for b in PHASE15_BARS)
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║      LEIKA BENCHMARK — Phase alias: {','.join(selected):<27} ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    elif args.phase:
        modes = _parse_phase_filter(args.phase)
        if args.phase.strip() in {"1.5", "phase1.5", "phase1_5"}:
            args.bars = ",".join(str(b) for b in PHASE15_BARS)
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║        LEIKA BENCHMARK — Phase filter: {args.phase:<30} ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    elif args.benchmark:
        args.bars = "1000,10000,100000,1000000"
        modes = list(range(1, 13))
        args.no_ai = True
        args.live = True
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║    LEIKA BENCHMARK — Phase 1 + 1.5 (single + 5-asset, no AI)    ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
    else:
        modes = DEFAULT_RUN_MODES

    if args.all:
        modes = list(range(1, 48)) + list(range(61, 66))
    else:
        if args.mc:
            modes += list(range(25, 31))
        if args.rw:
            modes += list(range(31, 43))
        if args.ai_bench:
            modes += list(range(43, 48))

    modes = sorted(set(modes))
    if args.no_ai:
        modes = [m for m in modes if m not in AI_MODE_IDS]
    if os.environ.get("LEIKA_AI_ENABLED", "1") == "0":
        modes = [m for m in modes if m not in AI_MODE_IDS]

    bar_sizes = [int(b.strip()) for b in args.bars.split(",") if b.strip()]
    mc_paths = [int(p.strip()) for p in args.mc_paths.split(",") if p.strip()]
    rw_models = [m.strip() for m in args.rw_models.split(",") if m.strip()]
    model = os.environ.get("LEIKA_MODEL", args.model)
    live = args.live
    extreme = args.extreme
    warmup = max(0, getattr(args, "warmup", 1))
    timing_runs = max(1, getattr(args, "timing_runs", 1))

    total_runs = sum(len(_datasets_for(m, bar_sizes, extreme, mc_paths, rw_models)) for m in modes)

    print(f"\n  Modes   : {modes}")
    print(f"  Bars    : {bar_sizes}" + ("  ⚡ extreme" if extreme else ""))
    if any(m in MC_MODE_IDS for m in modes):
        print(f"  MC paths: {mc_paths}")
    if any(m in RW_MODE_IDS for m in modes):
        print(f"  RW models: {rw_models}")
    print(f"  Model   : {model}")
    print(f"  Warmup  : {warmup} run{'s' if warmup != 1 else ''} discarded per case")
    print(f"  Total   : {total_runs} run(s)\n")

    hw_threads = 1
    hw_ram_gb = 1.0
    try:
        hw = hw_detect.detect()
        hw_threads = hw.cpu.logical_threads
        hw_ram_gb = hw.mem.total_gb
    except Exception:
        pass

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir_path = Path(getattr(args, "out_dir", "benchmark_results"))
    out_dir_path.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir_path / f"partial_{ts}.jsonl"
    _install_crash_handler(out_dir_path / f"crash_{ts}.log")

    # ── Assemble-only mode: rebuild final report from existing checkpoints ─────
    if getattr(args, "assemble_from", None):
        ats = args.assemble_from
        print(f"\n  Assembling report from checkpoints for run {ats} …")
        loaded = _load_checkpoints(out_dir_path, ats)
        if not loaded:
            print(f"  No checkpoint files found matching checkpoint_*_{ats}.json in {out_dir_path}")
            return
        _compute_speedups(loaded)
        _export_ods = getattr(args, "export_ods", False)
        report_path = report_gen.generate_report(
            loaded, out_dir=str(out_dir_path), timestamp=ats, export_ods=_export_ods
        )
        print(f"\n  Report: {report_path}")
        return

    results: list[BenchResult] = []
    agg_cpu = 0.0
    agg_mem = 0.0
    agg_gpu = None

    def _runner_for_mode(mode: int, progress=None, **kw):
        if 1 <= mode <= 6:
            return phase1_single.run(mode, bars=kw["bars"], extreme=extreme, progress=progress)
        if 7 <= mode <= 12:
            return phase15_multi.run(mode, bars=kw["bars"], extreme=extreme, progress=progress)
        if 13 <= mode <= 18:
            return phase2_single_ai.run(mode, bars=kw["bars"], model=model, extreme=extreme, progress=progress)
        if 19 <= mode <= 24:
            return phase25_multi_ai.run(mode, bars=kw["bars"], model=model, extreme=extreme, progress=progress)
        if 25 <= mode <= 30:
            return phase3_montecarlo.run(mode, n_candles=kw["n_candles"], n_paths=kw["n_paths"], progress=progress)
        if 31 <= mode <= 36:
            return phase4_randomwalk.run(mode, n_candles=kw["n_candles"], rw_model=kw["rw_model"], progress=progress)
        if 37 <= mode <= 42:
            return phase45_rw_multi.run(mode, n_candles=kw["n_candles"], rw_model=kw["rw_model"], progress=progress)
        if mode in AI_BENCH_MODE_IDS:
            return phase_ai_bench.run(mode, bars=kw["bars"], model=model, extreme=extreme, progress=progress)
        if mode in DATA_MODE_IDS:
            return phase_data1.run(mode, bars=kw["bars"])
        if 61 <= mode <= 65:
            return phase175_shared.run(mode, bars=kw["bars"])
        raise ValueError(f"Unknown mode {mode}")

    current_phase: str | None = None
    phase_results: list[BenchResult] = []

    # Rich live view only. No tqdm fallback to avoid static progress contamination.
    if _RICH_AVAILABLE:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            overall = progress.add_task("Total", total=total_runs)
            current = progress.add_task("Module", total=1)
            for mode in modes:
                phase, name = MODE_LABELS.get(mode, ("?", f"mode{mode}"))
                if phase != current_phase:
                    if phase_results and current_phase:
                        _save_phase_checkpoint(phase_results, current_phase, out_dir_path, ts)
                        phase_results = []
                    current_phase = phase
                    console.print(f"\n[bold cyan]══════ {phase} ══════[/]")
                console.print(f"[dim]Mode {mode:>2} │ {name}[/]")
                datasets = _datasets_for(mode, bar_sizes, extreme, mc_paths, rw_models)
                for ds in datasets:
                    run_idx = len(results) + 1
                    ds_label = _dataset_label(ds)
                    total_units = _dataset_progress_units(mode, ds)
                    progress.update(current, description=f"{phase} │ Mode {mode:>2} │ {ds_label}", total=total_units, completed=0)
                    def _progress_cb(done: int, total: int, label: str):
                        progress.update(current, completed=min(done, total), total=max(total, 1), description=f"{phase} │ Mode {mode:>2} │ {label}")
                        progress.refresh()

                    result, stats = _run_with_monitor(lambda progress=None, **kw: _runner_for_mode(mode, progress=progress, **kw), mode, name, False, progress=_progress_cb, warmup=warmup, timing_runs=timing_runs, **ds)
                    export_visualizations(result, args.out_dir)
                    if stats.get("peak_cpu_pct", 0):
                        agg_cpu = max(agg_cpu, stats["peak_cpu_pct"])
                    if stats.get("peak_mem_mb", 0):
                        agg_mem = max(agg_mem, stats["peak_mem_mb"])
                    if stats.get("peak_gpu_pct") is not None:
                        agg_gpu = max(agg_gpu or 0, stats["peak_gpu_pct"])

                    status = "✓" if not result.error else "✗"
                    tput = ""
                    if result.throughput_bars_sec > 0:
                        v = result.throughput_bars_sec
                        tput = (f"  {v / 1_000_000_000:.2f}G" if v >= 1e9 else f"  {v / 1_000_000:.1f}M" if v >= 1e6 else f"  {v / 1000:.0f}K")
                        tput += " ops/s"
                    paths_str = (f"  Paths/s:{result.paths_sec / 1e3:.0f}K" if result.paths_sec >= 1000 else "")
                    ai_str = f"  AI:{result.ai_calls}" if result.ai_calls else ""

                    stab_str = ""
                    if len(result.timing_runs) > 1:
                        stab_str = f"  p95:{result.runtime_ms_p95:,.0f}ms  cv:{result.coefficient_of_variation_pct:.1f}%  stab:{result.stability_score:.0f}"
                    console.print(
                        f"  {status}  [{ds_label}]  {result.exec_ms:,.1f}ms"
                        + (f"  │  Return:{result.total_return_pct:+.2f}%  Sharpe:{result.sharpe_ratio:.3f}" if result.phase_type == "portfolio" else "")
                        + f"{tput}{paths_str}{ai_str}{stab_str}"
                    )
                    if result.error:
                        console.print(f"  !! {result.error}")
                    results.append(result)
                    phase_results.append(result)
                    _flush_result(result, partial_path)
                    progress.update(overall, advance=1)
                    progress.update(current, completed=total_units, total=total_units)
                    progress.refresh()
    else:
        # Clean fallback. No static bars.
        for mode in modes:
            phase, name = MODE_LABELS.get(mode, ("?", f"mode{mode}"))
            if phase != current_phase:
                if phase_results and current_phase:
                    _save_phase_checkpoint(phase_results, current_phase, out_dir_path, ts)
                    phase_results = []
                current_phase = phase
                print(f"\n  ══════ {phase} ══════")
            print(f"  Mode {mode:>2} │ {name}")
            datasets = _datasets_for(mode, bar_sizes, extreme, mc_paths, rw_models)
            for ds in datasets:
                ds_label = _dataset_label(ds)
                result, _ = _run_with_monitor(lambda progress=None, **kw: _runner_for_mode(mode, progress=progress, **kw), mode, name, False, progress=None, warmup=warmup, timing_runs=timing_runs, **ds)
                export_visualizations(result, args.out_dir)
                results.append(result)
                phase_results.append(result)
                _flush_result(result, partial_path)
                print(f"  {('✓' if not result.error else '✗')}  [{ds_label}]  {result.exec_ms:,.1f}ms")

    # Save checkpoint for the last phase (not triggered by a phase boundary crossing).
    if phase_results and current_phase:
        _save_phase_checkpoint(phase_results, current_phase, out_dir_path, ts)

    _compute_speedups(results)
    print("\n  ── Speedup vs VectorBT Baseline ─────────────────────────────")
    last_phase = None
    for r in results:
        if r.phase != last_phase:
            print(f"\n  {r.phase}")
            last_phase = r.phase
        sp = f"  {r.speedup_vs_baseline:5.1f}×" if r.speedup_vs_baseline else "      —"
        label = r.mode_name
        if r.rw_model:
            label += f"[{r.rw_model}]"
        if r.phase_type == "montecarlo":
            ds_label = f"{r.bars:,} candles × {r.n_paths:,} paths"
        elif r.phase_type == "randomwalk":
            ds_label = f"{r.bars:,} candles [{r.rw_model}]"
            if r.n_assets > 1:
                ds_label += f" × {r.n_assets}"
        else:
            ds_label = f"{r.bars:,} bars"
        print(f"    Mode {r.mode:>2} {label:<38} [{ds_label}] {r.exec_ms:7.1f}ms{sp}")

    # ODS: on by default for full runs; only with --export-ods for smoke tests.
    _export_ods = getattr(args, "export_ods", False) or not getattr(args, "test", False)

    print("\n  Generating report...")
    report_start = time.monotonic()
    report_path = report_gen.generate_report(
        results, out_dir=args.out_dir, timestamp=ts, export_ods=_export_ods
    )
    report_ms = (time.monotonic() - report_start) * 1000
    for r in results:
        r.report_generation_time_ms = report_ms
    print(f"\n  Report: {report_path}")

if __name__ == "__main__":
    main()


def run_vectorbt_rust(bars: int, mode_num: int, phase: str,
                      extreme: bool = False, progress: ProgressFn = None, **_) -> BenchResult:
    """VectorBT portfolio + Leika Rust MACD indicators."""
    result = BenchResult(mode=mode_num, mode_name="vectorbt_rust", bars=bars,
                         phase=phase, backend="vectorbt_rust", n_assets=1)
    try:
        t_input = time.monotonic()
        prices           = generate_prices_extreme(bars) if extreme else generate_prices(bars, seed=SEED)
        _tick_progress(progress, max(1, bars // 3), bars, f"{max(1, bars // 3):,}/{bars:,} candles")
        entries, exits, hist = _signals_from_prices_rust(prices)
        _tick_progress(progress, max(1, (2 * bars) // 3), bars, f"{max(1, (2 * bars) // 3):,}/{bars:,} candles")
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        gpu_pre, (_, mem_pre) = _hw_snap()
        t0 = time.monotonic()
        m  = _backtest_vbt(prices, entries, exits, use_rust_indicators=True)
        result.exec_ms = (time.monotonic() - t0) * 1000
        _tick_progress(progress, bars, bars, f"{bars:,}/{bars:,} candles")
        _, (_, mem_post) = _hw_snap()
        result.mem_mb  = max(mem_pre, mem_post)
        _fill_result(result, m, prices, m["equity"], hist)
        result.throughput_bars_sec = bars / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
    except Exception as exc:
        result.error = str(exc)
    return result


def run_leika_single(bars: int, exec_mode: int, mode_num: int, phase: str,
                     extreme: bool = False, progress: ProgressFn = None, **_) -> BenchResult:
    """Leika single-asset backtest with specified execution mode (0/1/2/3)."""
    backend = f"leika_mode_{exec_mode}"
    result  = BenchResult(mode=mode_num, mode_name=backend, bars=bars,
                          phase=phase, backend=backend,
                          n_assets=1, leika_exec_mode=exec_mode)
    try:
        t_input = time.monotonic()
        prices           = generate_prices_extreme(bars) if extreme else generate_prices(bars, seed=SEED)
        _tick_progress(progress, max(1, bars // 3), bars, f"{max(1, bars // 3):,}/{bars:,} candles")
        entries, exits, hist = _signals_from_prices_rust(prices)
        _tick_progress(progress, max(1, (2 * bars) // 3), bars, f"{max(1, (2 * bars) // 3):,}/{bars:,} candles")
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        gpu_pre, (_, mem_pre) = _hw_snap()
        t0 = time.monotonic()
        m  = _backtest_leika(prices, entries, exits, exec_mode=exec_mode)
        result.exec_ms = (time.monotonic() - t0) * 1000
        _tick_progress(progress, bars, bars, f"{bars:,}/{bars:,} candles")
        result.python_to_rust_conversion_ms = m.get("python_to_rust_conversion_ms", 0.0)
        result.rust_engine_time_ms = m.get("rust_engine_time_ms", 0.0)
        result.stats_calculation_time_ms = m.get("stats_calculation_time_ms", 0.0)
        _, (_, mem_post) = _hw_snap()
        result.mem_mb  = max(mem_pre, mem_post)
        _fill_result(result, m, prices, m["equity"], hist)
        result.throughput_bars_sec = bars / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
    except Exception as exc:
        result.error = str(exc)
    return result


# ── Multi-asset runners ───────────────────────────────────────────────────────

def run_vectorbt_baseline_5(bars: int, mode_num: int, phase: str,
                             extreme: bool = False, progress: ProgressFn = None, **_) -> BenchResult:
    """VectorBT sequential over 5 independent assets."""
    result = BenchResult(mode=mode_num, mode_name="vectorbt_baseline_5_assets", bars=bars,
                         phase=phase, backend="vectorbt_baseline", n_assets=N_ASSETS)
    try:
        t_input = time.monotonic()
        gen     = generate_prices_extreme if extreme else lambda b, seed=SEED: generate_prices(b, seed=seed)
        symbols = []
        for i in range(N_ASSETS):
            symbols.append(gen(bars, seed=SEED + i))
            _tick_progress(progress, (i + 1) * bars, N_ASSETS * bars, f"{(i + 1) * bars:,}/{N_ASSETS * bars:,} candles")
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        gpu_pre, (_, mem_pre) = _hw_snap()
        t0 = time.monotonic()
        sym_results = []
        for i, sym in enumerate(symbols):
            entries, exits = _signals_from_prices(sym)
            sym_results.append(_backtest_vbt(sym, entries, exits))
            _tick_progress(progress, (N_ASSETS + i + 1) * bars, 2 * N_ASSETS * bars, f"{(N_ASSETS + i + 1) * bars:,}/{2 * N_ASSETS * bars:,} candles")
        result.exec_ms = (time.monotonic() - t0) * 1000
        _, (_, mem_post) = _hw_snap()
        result.mem_mb  = max(mem_pre, mem_post)
        _aggregate(result, sym_results, symbols, bars)
        result.total_runtime_ms = result.exec_ms
    except Exception as exc:
        result.error = str(exc)
    return result


def run_vectorbt_rust_5(bars: int, mode_num: int, phase: str,
                        extreme: bool = False, progress: ProgressFn = None, **_) -> BenchResult:
    """VectorBT portfolio + Rust MACD, 5 assets sequential."""
    result = BenchResult(mode=mode_num, mode_name="vectorbt_rust_5_assets", bars=bars,
                         phase=phase, backend="vectorbt_rust", n_assets=N_ASSETS)
    try:
        t_input = time.monotonic()
        gen     = generate_prices_extreme if extreme else lambda b, seed=SEED: generate_prices(b, seed=seed)
        symbols = []
        for i in range(N_ASSETS):
            symbols.append(gen(bars, seed=SEED + i))
            _tick_progress(progress, (i + 1) * bars, N_ASSETS * bars, f"{(i + 1) * bars:,}/{N_ASSETS * bars:,} candles")
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        gpu_pre, (_, mem_pre) = _hw_snap()
        t0 = time.monotonic()
        sym_results = []
        for i, sym in enumerate(symbols):
            entries, exits, _ = _signals_from_prices_rust(sym)
            sym_results.append(_backtest_vbt(sym, entries, exits, use_rust_indicators=True))
            _tick_progress(progress, (N_ASSETS + i + 1) * bars, 2 * N_ASSETS * bars, f"{(N_ASSETS + i + 1) * bars:,}/{2 * N_ASSETS * bars:,} candles")
        result.exec_ms = (time.monotonic() - t0) * 1000
        _, (_, mem_post) = _hw_snap()
        result.mem_mb  = max(mem_pre, mem_post)
        _aggregate(result, sym_results, symbols, bars)
        result.total_runtime_ms = result.exec_ms
    except Exception as exc:
        result.error = str(exc)
    return result


def run_leika_multi(bars: int, exec_mode: int, mode_num: int, phase: str,
                    extreme: bool = False, progress: ProgressFn = None, **_) -> BenchResult:
    """Leika 5-asset portfolio via run_batch_symbols with specified execution mode."""
    backend = f"leika_mode_{exec_mode}_5_assets"
    result  = BenchResult(mode=mode_num, mode_name=backend, bars=bars,
                          phase=phase, backend=f"leika_mode_{exec_mode}",
                          n_assets=N_ASSETS, leika_exec_mode=exec_mode)
    try:
        t_input = time.monotonic()
        gen     = generate_prices_extreme if extreme else lambda b, seed=SEED: generate_prices(b, seed=seed)
        symbols = []
        for i in range(N_ASSETS):
            symbols.append(gen(bars, seed=SEED + i))
            _tick_progress(progress, (i + 1) * bars, N_ASSETS * bars, f"{(i + 1) * bars:,}/{N_ASSETS * bars:,} candles")
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000

        workers = 1 if exec_mode == 0 else None  # None = use Leika's plan
        gpu_pre, (_, mem_pre) = _hw_snap()
        t0 = time.monotonic()

        if _LEIKA_OK:
            # Parallel dispatch using ThreadPoolExecutor (Rust releases GIL)
            # For mode 0: workers=1 (sequential), mode 1/2: workers=plan.workers
            n_workers = 1 if exec_mode == 0 else N_ASSETS
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_leika_symbol, sym): i
                           for i, sym in enumerate(symbols)}
                sym_results = [{}] * N_ASSETS
                done_assets = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    sym_results[idx] = fut.result()
                    done_assets += 1
                    _tick_progress(progress, (N_ASSETS + done_assets) * bars, 2 * N_ASSETS * bars, f"{(N_ASSETS + done_assets) * bars:,}/{2 * N_ASSETS * bars:,} candles")
        else:
            sym_results = []
            for i, sym in enumerate(symbols):
                sym_results.append(_run_leika_symbol(sym))
                _tick_progress(progress, (N_ASSETS + i + 1) * bars, 2 * N_ASSETS * bars, f"{(N_ASSETS + i + 1) * bars:,}/{2 * N_ASSETS * bars:,} candles")

        result.exec_ms = (time.monotonic() - t0) * 1000
        result.total_runtime_ms = result.exec_ms
        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        valid = [r for r in sym_results if r]
        _aggregate(result, valid, symbols, bars)
    except Exception as exc:
        result.error = str(exc)
    return result


def _run_leika_symbol(prices):
    entries, exits, hist = _signals_from_prices_rust(prices)
    m = _backtest_leika(prices, entries, exits)
    m["hist"] = hist
    return m


# ── Monte Carlo helpers ───────────────────────────────────────────────────────

def _mc_stats_python(paths: list[list[float]], initial_price: float) -> dict:
    """Compute MC distribution stats from raw paths (pure Python)."""
    returns: list[float] = []
    max_dds: list[float] = []
    for p in paths:
        if not p:
            continue
        ret = (p[-1] / initial_price - 1.0) * 100.0
        returns.append(ret)
        peak = p[0]; dd = 0.0
        for v in p:
            if v > peak: peak = v
            d = (peak - v) / peak * 100.0 if peak > 0 else 0.0
            if d > dd: dd = d
        max_dds.append(dd)
    if not returns:
        return {"prob_positive_pct": 0.0, "mean_return_pct": 0.0, "median_return_pct": 0.0,
                "std_return_pct": 0.0, "median_max_dd_pct": 0.0}
    returns.sort()
    n        = len(returns)
    mean     = sum(returns) / n
    prob     = sum(1 for r in returns if r > 0) / n * 100.0
    median   = returns[n // 2]
    var      = sum((r - mean) ** 2 for r in returns) / n
    std      = math.sqrt(var) if var > 0 else 0.0
    max_dds.sort()
    median_dd = max_dds[len(max_dds) // 2] if max_dds else 0.0
    return {"prob_positive_pct": prob, "mean_return_pct": mean,
            "median_return_pct": median, "std_return_pct": std,
            "median_max_dd_pct": median_dd}


def _mc_run_leika_chunk(n_candles: int, n_paths: int, exec_mode: int, seed: int) -> tuple[dict, dict]:
    """Run one bounded Monte Carlo chunk through Leika."""
    if not _LEIKA_OK:
        raise RuntimeError("leika module unavailable")
    mc = _leika.MonteCarlo(
        n_paths=n_paths,
        n_steps=n_candles,
        drift=0.0,
        volatility=0.20,
        initial_price=100.0,
        seed=seed,
        mode=exec_mode,
    )
    r = mc.run(return_paths=False)
    return r, _gpu_metric_map(r)


# ── MC runners ────────────────────────────────────────────────────────────────

def run_mc_baseline(n_candles: int, n_paths: int, mode_num: int, phase: str,
                          progress: ProgressFn = None) -> BenchResult:
    """Python baseline MC: pure Python GBM path generation (no Rust, no VBT)."""
    result = BenchResult(
        mode=mode_num, mode_name="mc_python_baseline",
        bars=n_candles, phase=phase, phase_type="montecarlo",
        backend="python_baseline_mc", n_assets=1, n_paths=n_paths,
    )
    try:
        _, (_, mem_pre) = _hw_snap()
        t_total = time.monotonic()
        total_units = n_paths * n_candles
        done_units = 0

        chunk_paths = _mc_chunk_paths(n_candles, n_paths)
        if chunk_paths <= 0:
            result.error = f"Monte Carlo dataset must be positive, got {n_candles} candles × {n_paths} paths"
            return result

        returns: list[float] = []
        max_dds: list[float] = []
        t_input = time.monotonic()
        for start in range(0, n_paths, chunk_paths):
            stop = min(n_paths, start + chunk_paths)
            chunk = []
            for i in range(start, stop):
                path = generate_prices(n_candles, seed=SEED + i)
                chunk.append(path)
                returns.append((path[-1] / 100.0 - 1.0) * 100.0 if path else 0.0)
                max_dds.append(_mc_path_max_drawdown(path))
                done_units += n_candles
                _tick_progress(progress, done_units, total_units, f"{done_units:,}/{total_units:,} candles")
            if chunk:
                pass
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        t_stats = time.monotonic()
        stats = _mc_finalize_stats(returns, max_dds)
        result.stats_calculation_time_ms = (time.monotonic() - t_stats) * 1000
        result.total_return_pct = stats["mean_return_pct"]
        result.exec_ms = (time.monotonic() - t_total) * 1000
        result.total_runtime_ms = result.exec_ms
        total_ops = n_paths * n_candles
        result.throughput_bars_sec = total_ops / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.paths_sec = n_paths / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
    except Exception as exc:
        result.error = str(exc)
    return result


def run_mc_leika(n_candles: int, n_paths: int, exec_mode: int,
                 mode_num: int, phase: str, progress: ProgressFn = None) -> BenchResult:
    """Leika MonteCarlo engine — exec_mode 0=serial, 1=adaptive, 2=gpu-dispatch."""
    backend = f"leika_mode_{exec_mode}_mc"
    result  = BenchResult(
        mode=mode_num, mode_name=backend,
        bars=n_candles, phase=phase, phase_type="montecarlo",
        backend=f"leika_mode_{exec_mode}", n_assets=1, n_paths=n_paths,
        leika_exec_mode=exec_mode,
    )
    try:
        gpu_pre, (_, mem_pre) = _hw_snap()
        result.python_input_time_ms = 0.0
        t0 = time.monotonic()

        if _LEIKA_OK:
            chunk_paths = _mc_chunk_paths(n_candles, n_paths)
            if chunk_paths <= 0:
                result.error = f"Monte Carlo dataset must be positive, got {n_candles} candles × {n_paths} paths"
                return result

            returns: list[float] = []
            max_dds: list[float] = []
            kernel_time_ms = 0.0
            transfer_time_ms = 0.0
            cpu_fallback_time_ms = 0.0
            fallback_reason = ""
            total_units = n_paths * n_candles
            done_units = 0

            for start in range(0, n_paths, chunk_paths):
                stop = min(n_paths, start + chunk_paths)
                chunk_n = stop - start
                seed = SEED + start
                r, metrics = _mc_run_leika_chunk(n_candles, chunk_n, exec_mode, seed)
                dist = r.get("simulation_distribution", {})
                raw_paths = r.get("raw_paths") or []
                if raw_paths:
                    for path in raw_paths:
                        returns.append((path[-1] / 100.0 - 1.0) * 100.0 if path else 0.0)
                        max_dds.append(_mc_path_max_drawdown(path))
                else:
                    mean_ret = float(dist.get("mean_return_pct", 0.0))
                    returns.extend([mean_ret] * chunk_n)
                    max_dd = float(dist.get("median_max_drawdown_pct", 0.0))
                    max_dds.extend([max_dd] * chunk_n)
                kernel_time_ms += metrics["kernel_time_ms"]
                transfer_time_ms += metrics["transfer_time_ms"]
                cpu_fallback_time_ms += metrics["cpu_fallback_time_ms"]
                fallback_reason = fallback_reason or metrics["fallback_reason"]
                done_units += chunk_n * n_candles
                _tick_progress(progress, done_units, total_units, f"{done_units:,}/{total_units:,} candles")

            stats = _mc_finalize_stats(returns, max_dds)
            result.total_return_pct = stats["mean_return_pct"]
            result.gpu_backend = metrics["backend"] or "Adaptive CPU"
            result.gpu_kernel_time_ms = kernel_time_ms
            result.gpu_transfer_time_ms = transfer_time_ms
            result.gpu_total_time_ms = kernel_time_ms + transfer_time_ms
            result.gpu_cpu_fallback_time_ms = cpu_fallback_time_ms
            result.gpu_fallback_reason = fallback_reason
            result.dynamic_tiling_enabled = bool(r.get("chunk_count", 0))
            result.split_axis = str(r.get("split_axis", ""))
            result.chunk_count = int(r.get("chunk_count", 0) or 0)
            result.chunk_size = int(r.get("chunk_size", 0) or 0)
            result.return_paths = bool(r.get("return_paths", False))
            result.raw_paths_copied = bool(r.get("raw_paths_copied", False))
            result.raw_paths_suppressed = bool(r.get("raw_paths_suppressed", False))
            result.memory_fallback = bool(r.get("memory_fallback", False))
            result.memory_fallback_reason = str(r.get("memory_fallback_reason") or "")
            result.cpu_work_share = float(r.get("cpu_work_share", 0.0) or 0.0)
            result.gpu_work_share = float(r.get("gpu_work_share", 0.0) or 0.0)
            result.vram_budget_mb = float(r.get("vram_budget_mb", 0.0) or 0.0)
            result.vram_peak_mb = float(result.gpu_mem_used_mb or 0.0)
            result.cpu_start_ms = metrics["cpu_start_ms"]
            result.cpu_end_ms = metrics["cpu_end_ms"]
            result.gpu_start_ms = metrics["gpu_start_ms"]
            result.gpu_end_ms = metrics["gpu_end_ms"]
            result.cpu_time_ms = metrics["cpu_time_ms"]
            result.gpu_time_ms = metrics["gpu_time_ms"]
            result.overlap_ms = metrics["overlap_ms"]
            result.overlap_pct = metrics["overlap_pct"]
            result.cpu_idle_wait_ms = metrics["cpu_idle_wait_ms"]
            result.gpu_idle_wait_ms = metrics["gpu_idle_wait_ms"]
            result.hybrid_total_time_ms = metrics["hybrid_total_time_ms"]
        else:
            chunk_paths = _mc_chunk_paths(n_candles, n_paths)
            if chunk_paths <= 0:
                result.error = f"Monte Carlo dataset must be positive, got {n_candles} candles × {n_paths} paths"
                return result
            returns: list[float] = []
            max_dds: list[float] = []
            total_units = n_paths * n_candles
            done_units = 0
            for start in range(0, n_paths, chunk_paths):
                stop = min(n_paths, start + chunk_paths)
                for i in range(start, stop):
                    path = generate_prices(n_candles, seed=SEED + i)
                    returns.append((path[-1] / 100.0 - 1.0) * 100.0 if path else 0.0)
                    max_dds.append(_mc_path_max_drawdown(path))
                    done_units += n_candles
                    _tick_progress(progress, done_units, total_units, f"{done_units:,}/{total_units:,} candles")
            result.total_return_pct = _mc_finalize_stats(returns, max_dds)["mean_return_pct"]
            result.gpu_backend = "Adaptive CPU"
            result.gpu_fallback_reason = "Python fallback"
            result.dynamic_tiling_enabled = False
            result.memory_fallback = False

        result.exec_ms = (time.monotonic() - t0) * 1000
        result.total_runtime_ms = result.exec_ms

        # Numerical error vs CPU reference (mode 1).
        ref_key = (n_candles, n_paths, SEED)
        ref = None
        ref_exec_ms = result.exec_ms
        if exec_mode != 1 and _LEIKA_OK:
            cached = _MC_REFERENCE_CACHE.get(ref_key)
            if cached is None:
                ref_t0 = time.monotonic()
                ref_returns: list[float] = []
                ref_dds: list[float] = []
                chunk_paths = _mc_chunk_paths(n_candles, n_paths)
                for start in range(0, n_paths, chunk_paths):
                    stop = min(n_paths, start + chunk_paths)
                    chunk_n = stop - start
                    r, _ = _mc_run_leika_chunk(n_candles, chunk_n, 1, SEED + start)
                    raw_paths = r.get("raw_paths") or []
                    if raw_paths:
                        for path in raw_paths:
                            ref_returns.append((path[-1] / 100.0 - 1.0) * 100.0 if path else 0.0)
                            ref_dds.append(_mc_path_max_drawdown(path))
                    else:
                        dist = r.get("simulation_distribution", {})
                        ref_returns.extend([float(dist.get("mean_return_pct", 0.0))] * chunk_n)
                        ref_dds.extend([float(dist.get("median_max_drawdown_pct", 0.0))] * chunk_n)
                ref = _mc_finalize_stats(ref_returns, ref_dds)
                ref_exec_ms = (time.monotonic() - ref_t0) * 1000
                _MC_REFERENCE_CACHE[ref_key] = (ref, ref_exec_ms)
            else:
                ref, ref_exec_ms = cached
        if ref is not None:
            actual_dist = float(result.total_return_pct)
            ref_dist = float(ref.get("mean_return_pct", ref.get("total_return_pct", 0.0)))
            result.numerical_error_rel = _rel_err(actual_dist, ref_dist)
            result.cpu_reference_ms = ref_exec_ms
        else:
            result.numerical_error_rel = 0.0
            result.cpu_reference_ms = result.exec_ms

        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        total_ops = n_paths * n_candles
        result.throughput_bars_sec = total_ops / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.paths_sec = n_paths / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        if gpu_pre[0] is not None:
            result.gpu_util_pct   = gpu_pre[0]
            result.gpu_temp_c     = gpu_pre[1]
            result.gpu_mem_used_mb = gpu_pre[2]
    except Exception as exc:
        result.error = str(exc)
    return result


# ── Random Walk runners ───────────────────────────────────────────────────────

def run_rw_baseline(n_candles: int, rw_model: str, mode_num: int, phase: str,
                           progress: ProgressFn = None, **_) -> BenchResult:
    """Python baseline RW: pure Python model-specific generator."""
    result = BenchResult(
        mode=mode_num, mode_name=f"rw_python_baseline_{rw_model}",
        bars=n_candles, phase=phase, phase_type="randomwalk",
        backend="python_baseline_rw", n_assets=1, n_paths=1, rw_model=rw_model,
    )
    try:
        gen = RW_GENERATORS.get(rw_model, generate_prices)
        _, (_, mem_pre) = _hw_snap()
        result.python_input_time_ms = 0.0
        t0 = time.monotonic()
        path = gen(n_candles, seed=SEED)
        _tick_progress(progress, n_candles, n_candles, f"{n_candles:,}/{n_candles:,} candles")
        result.exec_ms = (time.monotonic() - t0) * 1000
        result.total_runtime_ms = result.exec_ms
        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        result.total_return_pct = (path[-1] / path[0] - 1.0) * 100.0 if len(path) > 1 else 0.0
        result.throughput_bars_sec = n_candles / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.paths_sec = 1.0 / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.prices = path
    except Exception as exc:
        result.error = str(exc)
    return result


def run_rw_leika(n_candles: int, rw_model: str, exec_mode: int,
                 mode_num: int, phase: str, progress: ProgressFn = None, **_) -> BenchResult:
    """Leika RandomWalk single-asset — exec_mode 0/1/2."""
    backend = f"leika_mode_{exec_mode}_rw"
    result  = BenchResult(
        mode=mode_num, mode_name=f"{backend}_{rw_model}",
        bars=n_candles, phase=phase, phase_type="randomwalk",
        backend=f"leika_mode_{exec_mode}", n_assets=1, n_paths=1,
        rw_model=rw_model, leika_exec_mode=exec_mode,
    )
    try:
        gpu_pre, (_, mem_pre) = _hw_snap()
        result.python_input_time_ms = 0.0
        t0 = time.monotonic()
        if _LEIKA_OK:
            rw = _leika.RandomWalk(
                model=rw_model,
                n_paths=1,
                n_steps=n_candles,
                drift=0.0,
                volatility=0.20,
                initial_price=100.0,
                seed=SEED,
                mode=exec_mode,
            )
            r = rw.run(return_paths=False)
            dist = r.get("simulation_distribution", {})
            result.total_return_pct = float(dist.get("mean_return_pct", 0.0))
            raw = r.get("raw_paths") or []
            if raw:
                result.prices = raw[0]
            metrics = _gpu_metric_map(r)
            result.gpu_backend = metrics["backend"] or "Adaptive CPU"
            result.gpu_kernel_time_ms = metrics["kernel_time_ms"]
            result.gpu_transfer_time_ms = metrics["transfer_time_ms"]
            result.gpu_total_time_ms = metrics["total_gpu_time_ms"] or (metrics["kernel_time_ms"] + metrics["transfer_time_ms"])
            result.gpu_cpu_fallback_time_ms = metrics["cpu_fallback_time_ms"]
            result.gpu_fallback_reason = metrics["fallback_reason"]
            result.dynamic_tiling_enabled = bool(r.get("chunk_count", 0))
            result.split_axis = str(r.get("split_axis", ""))
            result.chunk_count = int(r.get("chunk_count", 0) or 0)
            result.chunk_size = int(r.get("chunk_size", 0) or 0)
            result.return_paths = bool(r.get("return_paths", False))
            result.raw_paths_copied = bool(r.get("raw_paths_copied", False))
            result.raw_paths_suppressed = bool(r.get("raw_paths_suppressed", False))
            result.memory_fallback = bool(r.get("memory_fallback", False))
            result.memory_fallback_reason = str(r.get("memory_fallback_reason") or "")
            result.cpu_work_share = float(r.get("cpu_work_share", 0.0) or 0.0)
            result.gpu_work_share = float(r.get("gpu_work_share", 0.0) or 0.0)
            result.vram_budget_mb = float(r.get("vram_budget_mb", 0.0) or 0.0)
            result.vram_peak_mb = float(result.gpu_mem_used_mb or 0.0)
            result.cpu_start_ms = metrics["cpu_start_ms"]
            result.cpu_end_ms = metrics["cpu_end_ms"]
            result.gpu_start_ms = metrics["gpu_start_ms"]
            result.gpu_end_ms = metrics["gpu_end_ms"]
            result.cpu_time_ms = metrics["cpu_time_ms"]
            result.gpu_time_ms = metrics["gpu_time_ms"]
            result.overlap_ms = metrics["overlap_ms"]
            result.overlap_pct = metrics["overlap_pct"]
            result.cpu_idle_wait_ms = metrics["cpu_idle_wait_ms"]
            result.gpu_idle_wait_ms = metrics["gpu_idle_wait_ms"]
            result.hybrid_total_time_ms = metrics["hybrid_total_time_ms"]
        else:
            gen  = RW_GENERATORS.get(rw_model, generate_prices)
            path = gen(n_candles, seed=SEED)
            result.total_return_pct = (path[-1] / path[0] - 1.0) * 100.0 if len(path) > 1 else 0.0
            result.prices = path
            result.gpu_backend = "Adaptive CPU"
            result.gpu_fallback_reason = "Python fallback"

        result.exec_ms = (time.monotonic() - t0) * 1000
        result.total_runtime_ms = result.exec_ms

        ref_key = (n_candles, rw_model, SEED)
        ref = None
        ref_exec_ms = result.exec_ms
        if exec_mode != 1 and _LEIKA_OK:
            cached = _RW_REFERENCE_CACHE.get(ref_key)
            if cached is None:
                ref_rw = _leika.RandomWalk(
                    model=rw_model,
                    n_paths=1,
                    n_steps=n_candles,
                    drift=0.0,
                    volatility=0.20,
                    initial_price=100.0,
                    seed=SEED,
                    mode=1,
                )
                ref_t0 = time.monotonic()
                ref = ref_rw.run(return_paths=False)
                ref_exec_ms = (time.monotonic() - ref_t0) * 1000
                _RW_REFERENCE_CACHE[ref_key] = (ref, ref_exec_ms)
            else:
                ref, ref_exec_ms = cached
        if ref is not None:
            ref_dist = float(ref.get("simulation_distribution", {}).get("mean_return_pct", ref.get("mean_return", 0.0)))
            result.numerical_error_rel = _rel_err(float(result.total_return_pct), ref_dist)
            result.cpu_reference_ms = ref_exec_ms
        else:
            result.numerical_error_rel = 0.0
            result.cpu_reference_ms = result.exec_ms

        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        result.throughput_bars_sec = n_candles / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.paths_sec = 1.0 / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        if gpu_pre[0] is not None:
            result.gpu_util_pct   = gpu_pre[0]
            result.gpu_temp_c     = gpu_pre[1]
            result.gpu_mem_used_mb = gpu_pre[2]
    except Exception as exc:
        result.error = str(exc)
    return result


def run_rw_multi_baseline(n_candles: int, rw_model: str, n_assets: int,
                           mode_num: int, phase: str, progress: ProgressFn = None, **_) -> BenchResult:
    """Python baseline RW multi-asset: Python generators run serially."""
    result = BenchResult(
        mode=mode_num, mode_name=f"rw_python_baseline_{n_assets}_assets_{rw_model}",
        bars=n_candles, phase=phase, phase_type="randomwalk",
        backend="python_baseline_rw", n_assets=n_assets, n_paths=n_assets,
        rw_model=rw_model,
    )
    try:
        t_input = time.monotonic()
        gen = RW_GENERATORS.get(rw_model, generate_prices)
        _, (_, mem_pre) = _hw_snap()
        result.python_input_time_ms = 0.0
        t0 = time.monotonic()
        paths = []
        total_units = n_assets * n_candles
        done_units = 0
        for i in range(n_assets):
            path = gen(n_candles, seed=SEED + i)
            paths.append(path)
            done_units += n_candles
            _tick_progress(progress, done_units, total_units, f"{done_units:,}/{total_units:,} candles")
        result.python_input_time_ms = (time.monotonic() - t_input) * 1000
        result.exec_ms = (time.monotonic() - t0) * 1000
        result.total_runtime_ms = result.exec_ms
        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        returns = [(p[-1] / p[0] - 1.0) * 100.0 for p in paths if len(p) > 1]
        result.total_return_pct = sum(returns) / len(returns) if returns else 0.0
        total_ops = n_assets * n_candles
        result.throughput_bars_sec = total_ops / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.paths_sec = n_assets / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
    except Exception as exc:
        result.error = str(exc)
    return result


def run_rw_multi_leika(n_candles: int, rw_model: str, n_assets: int,
                        exec_mode: int, mode_num: int, phase: str, progress: ProgressFn = None, **_) -> BenchResult:
    """Leika RandomWalk multi-asset.

    Mode 0: run each asset serially (n_assets calls with n_paths=1).
    Mode 1/2: run as single call with n_paths=n_assets (Rayon parallel).
    """
    backend = f"leika_mode_{exec_mode}_rw_{n_assets}_assets"
    result  = BenchResult(
        mode=mode_num, mode_name=f"{backend}_{rw_model}",
        bars=n_candles, phase=phase, phase_type="randomwalk",
        backend=f"leika_mode_{exec_mode}", n_assets=n_assets, n_paths=n_assets,
        rw_model=rw_model, leika_exec_mode=exec_mode,
    )
    try:
        t_input = time.monotonic()
        gpu_pre, (_, mem_pre) = _hw_snap()
        result.python_input_time_ms = 0.0
        t0 = time.monotonic()

        if _LEIKA_OK:
            if exec_mode == 0:
                # Mode 0: strictly serial — one path at a time
                returns = []
                kernel_time_ms = 0.0
                transfer_time_ms = 0.0
                cpu_fallback_time_ms = 0.0
                fallback_reason = ""
                total_units = n_assets * n_candles
                done_units = 0
                for i in range(n_assets):
                    rw = _leika.RandomWalk(
                        model=rw_model,
                        n_paths=1,
                        n_steps=n_candles,
                        drift=0.0,
                        volatility=0.20,
                        initial_price=100.0,
                        seed=SEED + i,
                        mode=0,
                    )
                    r = rw.run(return_paths=False)
                    dist = r.get("simulation_distribution", {})
                    returns.append(float(dist.get("mean_return_pct", 0.0)))
                    metrics = _gpu_metric_map(r)
                    kernel_time_ms += metrics["kernel_time_ms"]
                    transfer_time_ms += metrics["transfer_time_ms"]
                    cpu_fallback_time_ms += metrics["cpu_fallback_time_ms"]
                    fallback_reason = fallback_reason or metrics["fallback_reason"]
                    done_units += n_candles
                    _tick_progress(progress, done_units, total_units, f"{done_units:,}/{total_units:,} candles")
                result.total_return_pct = sum(returns) / len(returns) if returns else 0.0
                result.gpu_backend = "Adaptive CPU"
                result.gpu_kernel_time_ms = kernel_time_ms
                result.gpu_transfer_time_ms = transfer_time_ms
                result.gpu_total_time_ms = kernel_time_ms + transfer_time_ms
                result.gpu_cpu_fallback_time_ms = cpu_fallback_time_ms
                result.gpu_fallback_reason = fallback_reason
                result.dynamic_tiling_enabled = False
                result.cpu_work_share = 1.0
                result.gpu_work_share = 0.0
            else:
                # Mode 1/2: all assets in one Rayon-parallel call (n_paths=n_assets)
                rw = _leika.RandomWalk(
                    model=rw_model,
                    n_paths=n_assets,
                    n_steps=n_candles,
                    drift=0.0,
                    volatility=0.20,
                    initial_price=100.0,
                    seed=SEED,
                    mode=exec_mode,
                )
                r = rw.run(return_paths=False)
                dist = r.get("simulation_distribution", {})
                result.total_return_pct = float(dist.get("mean_return_pct", 0.0))
                metrics = _gpu_metric_map(r)
                result.gpu_backend = metrics["backend"] or "Adaptive CPU"
                result.gpu_kernel_time_ms = metrics["kernel_time_ms"]
                result.gpu_transfer_time_ms = metrics["transfer_time_ms"]
                result.gpu_total_time_ms = metrics["total_gpu_time_ms"] or (metrics["kernel_time_ms"] + metrics["transfer_time_ms"])
                result.gpu_cpu_fallback_time_ms = metrics["cpu_fallback_time_ms"]
                result.gpu_fallback_reason = metrics["fallback_reason"]
                result.dynamic_tiling_enabled = bool(r.get("chunk_count", 0))
                result.split_axis = str(r.get("split_axis", ""))
                result.chunk_count = int(r.get("chunk_count", 0) or 0)
                result.chunk_size = int(r.get("chunk_size", 0) or 0)
                result.return_paths = bool(r.get("return_paths", False))
                result.raw_paths_copied = bool(r.get("raw_paths_copied", False))
                result.raw_paths_suppressed = bool(r.get("raw_paths_suppressed", False))
                result.memory_fallback = bool(r.get("memory_fallback", False))
                result.memory_fallback_reason = str(r.get("memory_fallback_reason") or "")
                result.cpu_work_share = float(r.get("cpu_work_share", 0.0) or 0.0)
                result.gpu_work_share = float(r.get("gpu_work_share", 0.0) or 0.0)
                result.vram_budget_mb = float(r.get("vram_budget_mb", 0.0) or 0.0)
                result.vram_peak_mb = float(result.gpu_mem_used_mb or 0.0)
                result.cpu_start_ms = metrics["cpu_start_ms"]
                result.cpu_end_ms = metrics["cpu_end_ms"]
                result.gpu_start_ms = metrics["gpu_start_ms"]
                result.gpu_end_ms = metrics["gpu_end_ms"]
                result.cpu_time_ms = metrics["cpu_time_ms"]
                result.gpu_time_ms = metrics["gpu_time_ms"]
                result.overlap_ms = metrics["overlap_ms"]
                result.overlap_pct = metrics["overlap_pct"]
                result.cpu_idle_wait_ms = metrics["cpu_idle_wait_ms"]
                result.gpu_idle_wait_ms = metrics["gpu_idle_wait_ms"]
                result.hybrid_total_time_ms = metrics["hybrid_total_time_ms"]
        else:
            gen = RW_GENERATORS.get(rw_model, generate_prices)
            paths = [gen(n_candles, seed=SEED + i) for i in range(n_assets)]
            returns = [(p[-1] / p[0] - 1.0) * 100.0 for p in paths if len(p) > 1]
            result.total_return_pct = sum(returns) / len(returns) if returns else 0.0
            result.gpu_backend = "Adaptive CPU"
            result.gpu_fallback_reason = "Python fallback"

        result.exec_ms = (time.monotonic() - t0) * 1000
        result.total_runtime_ms = result.exec_ms

        ref_key = (n_candles, rw_model, n_assets, SEED)
        ref = None
        ref_exec_ms = result.exec_ms
        if exec_mode != 1 and _LEIKA_OK:
            cached = _RW_REFERENCE_CACHE.get(ref_key)
            if cached is None:
                ref_rw = _leika.RandomWalk(
                    model=rw_model,
                    n_paths=n_assets,
                    n_steps=n_candles,
                    drift=0.0,
                    volatility=0.20,
                    initial_price=100.0,
                    seed=SEED,
                    mode=1,
                )
                ref_t0 = time.monotonic()
                ref = ref_rw.run(return_paths=False)
                ref_exec_ms = (time.monotonic() - ref_t0) * 1000
                _RW_REFERENCE_CACHE[ref_key] = (ref, ref_exec_ms)
            else:
                ref, ref_exec_ms = cached
        if ref is not None:
            ref_dist = float(ref.get("simulation_distribution", {}).get("mean_return_pct", ref.get("mean_return", 0.0)))
            result.numerical_error_rel = _rel_err(float(result.total_return_pct), ref_dist)
            result.cpu_reference_ms = ref_exec_ms
        else:
            result.numerical_error_rel = 0.0
            result.cpu_reference_ms = result.exec_ms

        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        total_ops = n_assets * n_candles
        result.throughput_bars_sec = total_ops / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        result.paths_sec = n_assets / (result.exec_ms / 1000) if result.exec_ms > 0 else 0
        if gpu_pre[0] is not None:
            result.gpu_util_pct   = gpu_pre[0]
            result.gpu_temp_c     = gpu_pre[1]
            result.gpu_mem_used_mb = gpu_pre[2]
    except Exception as exc:
        result.error = str(exc)
    return result


def _aggregate(result: BenchResult, sym_results, symbols, bars):
    valid = [r for r in sym_results if r]
    if valid:
        result.total_return_pct = sum(r["total_return_pct"] for r in valid) / len(valid)
        result.roi_pct          = sum(r.get("roi_pct", r["total_return_pct"]) for r in valid) / len(valid)
        result.sharpe_ratio     = sum(r["sharpe_ratio"]     for r in valid) / len(valid)
        result.sortino_ratio    = sum(r.get("sortino_ratio", 0.0) for r in valid) / len(valid)
        result.calmar_ratio     = sum(r.get("calmar_ratio", 0.0) for r in valid) / len(valid)
        result.max_drawdown_pct = sum(r["max_drawdown_pct"] for r in valid) / len(valid)
        result.profit_factor    = sum(r.get("profit_factor", 0.0) for r in valid) / len(valid)
        result.portfolio_heat_avg_pct = sum(r.get("portfolio_heat_avg_pct", 0.0) for r in valid) / len(valid)
        result.portfolio_heat_max_pct = max(r.get("portfolio_heat_max_pct", 0.0) for r in valid)
        result.win_rate_pct     = sum(r.get("win_rate", 0)  for r in valid) / len(valid)
        result.total_trades     = sum(r.get("total_trades", 0) for r in valid)
        best = max(valid, key=lambda r: r["total_return_pct"])
        result.equity  = best.get("equity", [])
        result.prices  = symbols[valid.index(best)]
        result.trade_summary = {}
    n_total = len(valid) * bars
    result.throughput_bars_sec = n_total / (result.exec_ms / 1000) if result.exec_ms > 0 else 0


# ── AI-augmented runners ──────────────────────────────────────────────────────

def run_with_ai(
    bars: int,
    mode_num: int,
    phase: str,
    backend: str,
    exec_mode: Optional[int],
    n_assets: int,
    model: str,
    extreme: bool = False,
    progress: ProgressFn = None,
    ai_config: Optional[BenchmarkAiConfig] = None,
    run_ai: bool = True,
    dynamic_sections: bool = True,
) -> BenchResult:
    """Run the benchmark backend, then analyze the compact result with AI."""
    ai_config = ai_config or BenchmarkAiConfig(
        ai_mode="full-ai",
        model=model,
        max_context_chars=32_000,
        max_output_tokens=2_048,
        timeout_seconds=120,
        max_ai_cases=None,
    )
    if not ai_config.model:
        ai_config.model = model

    result = BenchResult(
        mode=mode_num,
        mode_name=f"{backend}_ai",
        bars=bars,
        phase=phase,
        backend=backend,
        n_assets=n_assets,
        leika_exec_mode=exec_mode,
        ai_enabled=True,
        ai_mode=ai_config.ai_mode,
        ai_model=ai_config.model,
    )
    try:
        gen = generate_prices_extreme if extreme else lambda b, seed=SEED: generate_prices(b, seed=seed)
        symbols = []
        for i in range(n_assets):
            symbols.append(gen(bars, seed=SEED + i))
            _tick_progress(progress, (i + 1) * bars, n_assets * bars, f"{(i + 1) * bars:,}/{n_assets * bars:,} candles")
        agent = BenchmarkAiAgent(ai_config)
        result.ai_model = agent.model

        entries_sets: list[list[bool]] = []
        exits_sets: list[list[bool]] = []
        signal_histories: list[list[Optional[float]]] = []
        for i, sym in enumerate(symbols):
            if _LEIKA_OK:
                _, _, hist = _leika.macd(sym, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            else:
                _, _, hist = macd_py(sym, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            entries, exits = make_macd_signals(hist)
            entries_sets.append(entries)
            exits_sets.append(exits)
            signal_histories.append(hist)
            _tick_progress(progress, (n_assets + i + 1) * bars, 2 * n_assets * bars, f"{(n_assets + i + 1) * bars:,}/{2 * n_assets * bars:,} candles")

        gpu_pre, (_, mem_pre) = _hw_snap()
        backtest_start = time.monotonic()

        sym_results = []
        if "leika" in backend and _LEIKA_OK:
            n_workers = 1 if exec_mode == 0 else n_assets
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {
                    pool.submit(_backtest_leika, symbols[i], entries_sets[i], exits_sets[i]): i
                    for i in range(n_assets)
                }
                sym_results = [{}] * n_assets
                done_assets = 0
                for fut in as_completed(futs):
                    idx = futs[fut]
                    sym_results[idx] = fut.result()
                    done_assets += 1
                    _tick_progress(progress, (2 * n_assets + done_assets) * bars, 3 * n_assets * bars, f"{(2 * n_assets + done_assets) * bars:,}/{3 * n_assets * bars:,} candles")
        else:
            for i in range(n_assets):
                sym_results.append(_backtest_vbt(symbols[i], entries_sets[i], exits_sets[i]))
                _tick_progress(progress, (2 * n_assets + i + 1) * bars, 3 * n_assets * bars, f"{(2 * n_assets + i + 1) * bars:,}/{3 * n_assets * bars:,} candles")

        result.backtest_ms = (time.monotonic() - backtest_start) * 1000.0
        _, (_, mem_post) = _hw_snap()
        result.mem_mb = max(mem_pre, mem_post)
        _aggregate(result, [r for r in sym_results if r], symbols, bars)
        trade_summary = _aggregate_trade_summary(symbols, entries_sets, exits_sets)
        result.trade_summary = trade_summary
        result.total_trades = trade_summary.get("total_trades", result.total_trades)
        result.avg_trade_return_pct = trade_summary.get("avg_trade_return_pct", 0.0)
        result.median_trade_return_pct = trade_summary.get("median_trade_return_pct", 0.0)
        result.best_trade_pct = trade_summary.get("best_trade_pct", 0.0)
        result.worst_trade_pct = trade_summary.get("worst_trade_pct", 0.0)
        result.longest_trade_bars = trade_summary.get("longest_trade_bars", 0)
        result.shortest_trade_bars = trade_summary.get("shortest_trade_bars", 0)

        context_build_start = time.monotonic()
        context = build_ai_context(
            stats={
                "total_return_pct": result.total_return_pct,
                "roi_pct": result.roi_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "profit_factor": result.profit_factor,
                "win_rate_pct": result.win_rate_pct,
                "total_trades": result.total_trades,
            },
            risk_stats={
                "portfolio_heat_avg_pct": result.portfolio_heat_avg_pct,
                "portfolio_heat_max_pct": result.portfolio_heat_max_pct,
                "correlation_risk_score": 0.0,
                "avg_correlation": 0.0,
                "effective_diversification": float(n_assets),
            },
            strategy_info={
                "strategy_name": "MACD_12_26_9",
                "entry_rules": "Histogram crosses above zero",
                "exit_rules": "Histogram crosses below zero",
                "timeframe": "1D",
                "assets": n_assets,
                "bars": bars,
                "fees": FEES,
                "slippage": 0.0,
                "mode": exec_mode if exec_mode is not None else "baseline",
                "backend": backend,
            },
            bars=bars,
            trade_summary=trade_summary,
            prices=symbols[0] if symbols else [],
        )
        result.context_build_ms = (time.monotonic() - context_build_start) * 1000.0

        ai_text = ""
        ai_metrics = None
        if run_ai and agent.available:
            n_windows = max(1, bars // AI_COOLDOWN)
            window_size = bars // n_windows
            _bt_units = 3 * n_assets * bars
            _ai_total = _bt_units + n_windows

            _last_ai_text = ""
            _last_ai_metrics = None
            total_ai_calls = 0
            total_ai_ms = 0.0
            total_prompt_chars = 0
            total_response_chars = 0

            for win_idx in range(n_windows):
                start = win_idx * window_size
                end = bars if win_idx == n_windows - 1 else start + window_size
                win_size = end - start

                win_prices = [sym[start:end] for sym in symbols]
                win_entries_s = [e[start:end] for e in entries_sets]
                win_exits_s = [x[start:end] for x in exits_sets]
                win_trade_summary = _aggregate_trade_summary(win_prices, win_entries_s, win_exits_s)

                p0 = win_prices[0][0] if win_prices and win_prices[0] else 1.0
                p1 = win_prices[0][-1] if win_prices and win_prices[0] else 1.0
                win_return_pct = ((p1 / p0) - 1.0) * 100.0 if p0 > 0 else 0.0
                win_stats = {
                    "total_return_pct": win_return_pct,
                    "roi_pct": win_return_pct,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                    "win_rate_pct": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": win_trade_summary.get("total_trades", 0),
                }
                win_context = build_ai_context(
                    stats=win_stats,
                    risk_stats={
                        "portfolio_heat_avg_pct": 0.0,
                        "portfolio_heat_max_pct": 0.0,
                        "correlation_risk_score": 0.0,
                        "avg_correlation": 0.0,
                        "effective_diversification": float(n_assets),
                    },
                    strategy_info={
                        "strategy_name": "MACD_12_26_9",
                        "entry_rules": "Histogram crosses above zero",
                        "exit_rules": "Histogram crosses below zero",
                        "timeframe": "1D",
                        "assets": n_assets,
                        "bars": win_size,
                        "window": f"{win_idx + 1}/{n_windows}",
                        "fees": FEES,
                        "slippage": 0.0,
                        "mode": exec_mode if exec_mode is not None else "baseline",
                        "backend": backend,
                    },
                    bars=win_size,
                    trade_summary=win_trade_summary,
                    prices=win_prices[0] if win_prices else [],
                )

                _text_win, _metrics_win = agent.analyze_single(win_context)
                _last_ai_text = _text_win
                _last_ai_metrics = _metrics_win
                total_ai_calls += int(_metrics_win.ai_calls or 1)
                total_ai_ms += float(_metrics_win.total_ai_time_ms or 0.0)
                total_prompt_chars += int(_metrics_win.total_prompt_chars or 0)
                total_response_chars += int(_metrics_win.total_response_chars or 0)

                _tick_progress(
                    progress,
                    _bt_units + win_idx + 1,
                    _ai_total,
                    f"AI {win_idx + 1}/{n_windows} windows ({AI_COOLDOWN}-bar cooldown)",
                )

            ai_text = _last_ai_text or ""
            ai_metrics = _last_ai_metrics
            if ai_metrics is None:
                result.ai_fallback = True
                result.ai_skipped_reason = "No AI windows completed"
            else:
                result.ai_calls = total_ai_calls
                result.ai_ms_total = total_ai_ms
                result.ai_total_time_ms = total_ai_ms
                result.ai_context_chars = ai_metrics.context_chars
                result.ai_estimated_tokens = ai_metrics.estimated_tokens
                result.ai_prompt_eval_duration_ms = ai_metrics.prompt_eval_duration_ms
                result.ai_eval_duration_ms = ai_metrics.eval_duration_ms
                result.ai_tokens_per_second = (
                    total_response_chars / max(total_ai_ms / 1000.0, 1e-9) if total_ai_ms > 0 else 0.0
                )
                result.ai_gpu_used = ai_metrics.gpu_used
                result.ai_vram_used_mb = ai_metrics.vram_used_mb
                result.ai_timeout = ai_metrics.timeout
                result.ai_fallback = ai_metrics.fallback or ai_metrics.skipped
                result.ai_skipped_reason = ai_metrics.skipped_reason or ""
                result.ai_sections = ai_metrics.ai_sections
                result.ai_avg_section_ms = total_ai_ms / max(total_ai_calls, 1)
                result.ai_slowest_section = ai_metrics.slowest_section
                result.ai_fastest_section = ai_metrics.fastest_section
                result.ai_total_prompt_chars = total_prompt_chars
                result.ai_total_response_chars = total_response_chars
                result.ai_timeout_count = ai_metrics.timeout_count
                result.ai_fallback_count = ai_metrics.fallback_count
                result.ai_section_parallelism = ai_metrics.section_parallelism
                result.ai_dynamic_sectioning_enabled = ai_metrics.dynamic_sectioning_enabled
                result.ai_section_results = ai_metrics.section_results or []
        else:
            result.ai_fallback = True
            if not run_ai and agent.available:
                result.ai_skipped_reason = "AI skipped: max_ai_cases reached"
            else:
                result.ai_skipped_reason = "Ollama unavailable" if not agent.available else "AI disabled"

        if ai_metrics and ai_metrics.skipped:
            result.ai_fallback = True
            result.ai_skipped_reason = ai_metrics.skipped_reason

        result.total_runtime_ms = result.backtest_ms + result.context_build_ms + result.ai_total_time_ms
        result.exec_ms = result.total_runtime_ms
        result.engine_time_ms = result.backtest_ms
        result.throughput_bars_sec = (n_assets * bars) / (result.backtest_ms / 1000.0) if result.backtest_ms > 0 else 0.0
        result.cpu_pct = 0.0
        if gpu_pre[0] is not None:
            result.gpu_util_pct = gpu_pre[0]
            result.gpu_temp_c = gpu_pre[1]
            result.gpu_mem_used_mb = gpu_pre[2]
        _tick_progress(progress, 3 * n_assets * bars, 3 * n_assets * bars, f"{3 * n_assets * bars:,}/{3 * n_assets * bars:,} candles")

    except Exception as exc:
        result.error = str(exc)
    return result
