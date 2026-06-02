"""Shared result dataclass and performance helpers for the Leika benchmark suite."""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm


@dataclass
class BenchResult:
    mode: int
    mode_name: str
    bars: int
    # -- classification --
    phase: str = ""  # "Phase 1", "Phase 1.5", "Phase 2", "Phase 2.5", ...
    phase_type: str = "portfolio"  # "portfolio" | "montecarlo" | "randomwalk"
    backend: str = ""
    n_assets: int = 1
    leika_exec_mode: Optional[int] = None
    ai_enabled: bool = False
    speedup_vs_baseline: Optional[float] = None
    # -- execution --
    exec_ms: float = 0.0
    python_input_time_ms: float = 0.0
    python_to_rust_conversion_ms: float = 0.0
    engine_time_ms: float = 0.0
    rust_engine_time_ms: float = 0.0
    stats_calculation_time_ms: float = 0.0
    python_export_time_ms: float = 0.0
    report_generation_time_ms: float = 0.0
    backtest_ms: float = 0.0
    context_build_ms: float = 0.0
    total_runtime_ms: float = 0.0
    mem_mb: float = 0.0
    cpu_pct: float = 0.0
    throughput_bars_sec: float = 0.0
    paths_sec: float = 0.0
    cpu_total_threads: int = 0
    cpu_physical_cores: int = 0
    cpu_workers_selected: int = 0
    segments_selected: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_budget_gb: float = 0.0
    ram_peak_gb: float = 0.0
    ram_peak_pct: float = 0.0
    parallel_efficiency: float = 0.0
    section_overhead_ms: float = 0.0
    best_segment_count: int = 0
    worst_segment_count: int = 0
    # -- gpu --
    gpu_util_pct: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_accel_factor: Optional[float] = None
    gpu_backend: str = ""
    gpu_backend_priority: str = ""
    gpu_kernel_time_ms: float = 0.0
    gpu_transfer_time_ms: float = 0.0
    gpu_total_time_ms: float = 0.0
    gpu_cpu_fallback_time_ms: float = 0.0
    gpu_fallback_reason: str = ""
    gpu_h2d_transfer_ms: float = 0.0
    gpu_d2h_transfer_ms: float = 0.0
    gpu_bytes_h2d: int = 0
    gpu_bytes_d2h: int = 0
    cpu_start_ms: float = 0.0
    cpu_end_ms: float = 0.0
    gpu_start_ms: float = 0.0
    gpu_end_ms: float = 0.0
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0
    overlap_ms: float = 0.0
    numerical_error_rel: Optional[float] = None
    cpu_reference_ms: float = 0.0
    # -- dynamic tiling / fallback diagnostics --
    dynamic_tiling_enabled: bool = False
    split_axis: str = ""
    chunk_count: int = 0
    chunk_size: int = 0
    section_multiplier: int = 0
    memory_fallback: bool = False
    memory_fallback_reason: str = ""
    raw_paths_copied: bool = False
    raw_paths_suppressed: bool = False
    return_paths: bool = False
    cpu_work_share: float = 0.0
    gpu_work_share: float = 0.0
    overlap_pct: float = 0.0
    cpu_idle_wait_ms: float = 0.0
    gpu_idle_wait_ms: float = 0.0
    hybrid_total_time_ms: float = 0.0
    vram_budget_mb: float = 0.0
    vram_peak_mb: float = 0.0
    # -- trading --
    total_return_pct: float = 0.0
    roi_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    portfolio_heat_avg_pct: float = 0.0
    portfolio_heat_max_pct: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    avg_trade_return_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    median_trade_return_pct: float = 0.0
    longest_trade_bars: int = 0
    shortest_trade_bars: int = 0
    # -- mc / rw specific --
    n_paths: int = 0
    paths_sec: float = 0.0
    rw_model: str = ""
    scaling_efficiency: Optional[float] = None
    # -- ai --
    ai_mode: str = ""
    ai_model: str = ""
    ai_context_chars: int = 0
    ai_estimated_tokens: int = 0
    ai_prompt_eval_duration_ms: float = 0.0
    ai_eval_duration_ms: float = 0.0
    ai_total_time_ms: float = 0.0
    ai_tokens_per_second: float = 0.0
    ai_gpu_used: bool = False
    ai_vram_used_mb: float = 0.0
    ai_timeout: bool = False
    ai_fallback: bool = False
    ai_skipped_reason: str = ""
    ai_calls: int = 0
    ai_ms_total: float = 0.0
    ai_sections: int = 0
    ai_avg_section_ms: float = 0.0
    ai_slowest_section: str = ""
    ai_fastest_section: str = ""
    ai_total_prompt_chars: int = 0
    ai_total_response_chars: int = 0
    ai_timeout_count: int = 0
    ai_fallback_count: int = 0
    ai_section_parallelism: str = ""
    ai_dynamic_sectioning_enabled: bool = False
    ai_section_results: list[dict] = field(default_factory=list, repr=False)
    error: str = ""
    warmup_runs: int = 0
    # -- execution classification --
    cash_model: str = ""       # none / single_asset_cash / independent_per_symbol / shared_global_cash
    shared_data_sectioning: bool = False  # True only for multi-asset shared-cash portfolios
    execution_core: str = ""   # single_pass / independent_batch / shared_global_cash / stochastic_paths / ai_analysis
    dynamic_sectioning_used: bool = False  # True when engine uses parallel sections for this workload
    sell_at_end_scope: str = ""     # single_asset / per_symbol / global / none
    dynamic_sectioning_preparation: bool = False
    dynamic_sectioning_execution: bool = False   # always False for shared-cash
    dynamic_sectioning_post_analysis: bool = False
    # -- stability metrics (multi-run) --
    timing_runs: list[float] = field(default_factory=list, repr=False)
    runtime_ms_min: float = 0.0
    runtime_ms_median: float = 0.0
    runtime_ms_p95: float = 0.0
    runtime_ms_max: float = 0.0
    runtime_ms_std: float = 0.0
    coefficient_of_variation_pct: float = 0.0
    stability_score: float = 100.0   # 0–100; higher = more stable
    gpu_fallback_count: int = 0
    memory_fallback_count: int = 0
    # -- resource efficiency metrics --
    ops_per_cpu_pct: float = 0.0       # throughput_bars_sec / cpu_pct (when cpu_pct > 0)
    ops_per_ram_gb: float = 0.0        # throughput_bars_sec / mem_mb * 1024
    ops_per_vram_gb: float = 0.0       # throughput_bars_sec / gpu_mem_used_mb * 1024 (when GPU)
    throughput_score: float = 0.0      # normalised throughput (0–100 relative to phase baseline)
    resource_efficiency_score: float = 0.0  # composite: throughput + stability / resource use
    # -- raw data for viz --
    prices: list[float] = field(default_factory=list, repr=False)
    equity: list[float] = field(default_factory=list, repr=False)
    ema_fast: list[float] = field(default_factory=list, repr=False)
    drawdowns: list[float] = field(default_factory=list, repr=False)
    trade_summary: dict = field(default_factory=dict, repr=False)


def compute_stability_metrics(result: "BenchResult") -> None:
    """Populate stability and resource efficiency fields from timing_runs."""
    import statistics as _stats
    runs = result.timing_runs
    if not runs:
        return

    result.runtime_ms_min    = min(runs)
    result.runtime_ms_max    = max(runs)
    result.runtime_ms_median = _stats.median(runs)
    result.exec_ms           = result.runtime_ms_median   # official time = median
    result.runtime_ms_std    = _stats.stdev(runs) if len(runs) >= 2 else 0.0
    p95_idx = max(0, int(len(runs) * 0.95) - 1)
    result.runtime_ms_p95    = sorted(runs)[p95_idx]
    cv = (result.runtime_ms_std / result.runtime_ms_median * 100.0) if result.runtime_ms_median > 0 else 0.0
    result.coefficient_of_variation_pct = round(cv, 2)

    # Stability score: 100 − penalties
    score = 100.0
    if cv > 30:   score -= 40
    elif cv > 15: score -= 20
    elif cv > 5:  score -= 10
    if result.gpu_fallback_count > 0:    score -= 15
    if result.memory_fallback_count > 0: score -= 10
    if result.ai_timeout:                score -= 10
    if result.memory_fallback:           score -= 5
    result.stability_score = max(0.0, round(score, 1))

    # Resource efficiency: ops per unit of resource
    thr = result.throughput_bars_sec
    if result.cpu_pct and result.cpu_pct > 0:
        result.ops_per_cpu_pct = thr / result.cpu_pct
    if result.mem_mb and result.mem_mb > 0:
        result.ops_per_ram_gb = thr / (result.mem_mb / 1024.0)
    if result.gpu_mem_used_mb and result.gpu_mem_used_mb > 0:
        result.ops_per_vram_gb = thr / (result.gpu_mem_used_mb / 1024.0)

    # Resource efficiency score: throughput stability composite (no resource target)
    raw_efficiency = result.stability_score * (1.0 + min(thr / 1e9, 1.0))
    result.resource_efficiency_score = round(min(100.0, raw_efficiency / 2.0), 1)


def gpu_snapshot() -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (util_pct, temp_c, mem_used_mb) or (None, None, None)."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass
    return None, None, None


def cpu_mem_snapshot() -> tuple[float, float]:
    """Returns (cpu_pct, mem_mb) for the current process."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.05)
        mem = psutil.Process(os.getpid()).memory_info().rss / 1e6
        return cpu, mem
    except Exception:
        return 0.0, 0.0


def compute_metrics(equity: list[float], trades: list[dict]) -> dict:
    """Sharpe, Sortino, max drawdown, win rate from equity curve + trade list."""
    if len(equity) < 2:
        return {
            "total_return_pct": 0.0,
            "roi_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "portfolio_heat_avg_pct": 0.0,
            "portfolio_heat_max_pct": 0.0,
            "win_rate_pct": 0.0,
            "drawdowns": [],
        }

    total_return = (equity[-1] / equity[0] - 1.0) * 100.0
    roi_pct = total_return

    rets = [
        (equity[i] - equity[i - 1]) / equity[i - 1]
        for i in range(1, len(equity))
        if equity[i - 1] > 0
    ]

    if len(rets) > 1:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        std_r = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 1e-10 else 0.0
        downside2 = sum((r * r) for r in rets if r < 0) / len(rets)
        d_std = math.sqrt(downside2) if downside2 > 0 else 0.0
        sortino = (mean_r / d_std * math.sqrt(252)) if d_std > 1e-10 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    peak = equity[0] if equity else 0.0
    max_dd = 0.0
    drawdowns = []
    heat_series = []
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
            drawdowns.append(-dd)
            heat_series.append(100.0 - dd)
        else:
            drawdowns.append(0.0)
            heat_series.append(0.0)

    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0
    gross_profit = sum(t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) > 0)
    gross_loss = sum(abs(t.get("pnl", 0.0)) for t in trades if t.get("pnl", 0.0) < 0)
    if gross_loss < 1e-12:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    return {
        "total_return_pct": total_return,
        "roi_pct": roi_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": ((((equity[-1] / equity[0]) ** (1.0 / (len(equity) / 252.0)) - 1.0) / (max_dd / 100.0)) if max_dd > 1e-10 else 0.0),
        "max_drawdown_pct": max_dd,
        "profit_factor": profit_factor if math.isfinite(profit_factor) else float("inf"),
        "portfolio_heat_avg_pct": sum(heat_series) / len(heat_series),
        "portfolio_heat_max_pct": max(heat_series),
        "win_rate_pct": win_rate,
        "drawdowns": drawdowns,
    }


def backtest_simple(
    prices: list[float],
    signals: list[int],
    initial_cash: float = 10_000.0,
    show_progress: bool = False,
    desc: str = "backtest",
) -> tuple[list[float], list[dict]]:
    """Long-only vectorized backtest."""
    cash = initial_cash
    position = 0.0
    entry_p = 0.0
    equity = [initial_cash]
    trades: list[dict] = []

    bar_iter = tqdm(
        zip(prices, signals),
        total=len(prices),
        desc=desc,
        unit="bar",
        leave=False,
        disable=not show_progress,
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} bars [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for price, sig in bar_iter:
        if sig == 1 and position == 0.0 and cash > 0:
            position = cash / price
            entry_p = price
            cash = 0.0
            bar_iter.set_postfix(event="BUY", price=f"{price:.2f}")
        elif sig == -1 and position > 0.0:
            proceeds = position * price
            trades.append({"entry": entry_p, "exit": price, "pnl": proceeds - position * entry_p})
            cash = proceeds
            position = 0.0
            bar_iter.set_postfix(event="SELL", price=f"{price:.2f}")
        equity.append(cash + position * price)

    if position > 0.0:
        proceeds = position * prices[-1]
        trades.append({"entry": entry_p, "exit": prices[-1], "pnl": proceeds - position * entry_p})

    return equity, trades
