"""
AI Benchmark Phases (separate from core portfolio/MC/RW benchmarks).

AI-1: AI-only (measure prompt latency, inference time, tokens/sec)
AI-2: Single-asset backtest + AI analysis
AI-3: Multi-asset (5 assets) + AI analysis

These phases use the same MACD(12,26,9) strategy as Phase 2/2.5 but
measure AI overhead in isolation.
"""
from __future__ import annotations
import time
from runner import run_with_ai, BenchResult, N_ASSETS
from ai_agent import BenchmarkAiAgent
from ai_context import BenchmarkAiConfig
from metrics import gpu_snapshot, cpu_mem_snapshot

AI_BENCH_MODES = {
    43: ("AI-1", "ai_latency_baseline"),
    44: ("AI-2", "ai_single_leika_m1"),
    45: ("AI-3", "ai_multi_leika_m1"),
    46: ("AI-4", "ai_single_leika_m2"),
    47: ("AI-5", "ai_multi_leika_m2"),
}

DEFAULT_MODEL = "0xroyce/plutus:latest"
AI_BENCH_BARS = [500, 1_000]


def run_ai_latency(bars: int, mode_num: int, model: str = DEFAULT_MODEL) -> BenchResult:
    """AI-1: Measure AI prompt latency with no backtest — pure inference overhead."""
    phase = "Phase AI-1"
    result = BenchResult(mode=mode_num, mode_name="ai_latency_baseline",
                         bars=bars, phase=phase, phase_type="portfolio",
                         backend="ai_only", ai_enabled=True)
    config = BenchmarkAiConfig(
        ai_mode="quick-ai",
        model=model,
        max_context_chars=8_000,
        max_output_tokens=512,
        timeout_seconds=60,
        max_ai_cases=1,
    )
    agent = BenchmarkAiAgent(config)
    if not agent.available:
        result.error = "Ollama not available"
        return result

    context = {
        "main_stats": {
            "total_return_pct": 0.0,
            "roi_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_trade_return_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
        },
        "risk_stats": {
            "portfolio_heat_avg_pct": 0.0,
            "portfolio_heat_max_pct": 0.0,
            "correlation_risk_score": 0.0,
            "avg_correlation": 0.0,
            "effective_diversification": 1.0,
        },
        "strategy_info": {
            "strategy_name": "AI_ONLY",
            "entry_rules": "none",
            "exit_rules": "none",
            "timeframe": "benchmark",
            "assets": 1,
            "bars": bars,
            "fees": 0.0,
            "slippage": 0.0,
            "mode": "ai-only",
            "backend": "ai_only",
        },
        "strategy_dna": {
            "trend_following_score": 0.0,
            "mean_reversion_score": 0.0,
            "volatility_sensitivity": 0.0,
            "trade_frequency_score": 0.0,
            "holding_time_score": 0.0,
        },
        "regime_detection": {
            "dominant_regime": "Unknown",
            "regime_confidence": 0.0,
            "high_volatility_pct": 0.0,
            "sideways_pct": 100.0,
            "trending_pct": 0.0,
        },
        "trade_summary": {
            "total_trades": 0,
            "avg_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "longest_trade_bars": 0,
            "shortest_trade_bars": 0,
            "top_best_trades": [],
            "top_worst_trades": [],
        },
    }

    cpu0, mem0 = cpu_mem_snapshot()
    gpu0 = gpu_snapshot()
    t0 = time.monotonic()
    _, metrics = agent.analyze_single(context)
    result.exec_ms = (time.monotonic() - t0) * 1000.0
    result.ai_calls = metrics.ai_calls or (0 if metrics.skipped else 1)
    result.ai_ms_total = metrics.total_ai_time_ms
    result.ai_total_time_ms = metrics.total_ai_time_ms
    result.ai_context_chars = metrics.context_chars
    result.ai_estimated_tokens = metrics.estimated_tokens
    result.ai_prompt_eval_duration_ms = metrics.prompt_eval_duration_ms
    result.ai_eval_duration_ms = metrics.eval_duration_ms
    result.ai_tokens_per_second = metrics.ai_tokens_per_second
    result.ai_gpu_used = metrics.gpu_used
    result.ai_vram_used_mb = metrics.vram_used_mb
    result.ai_timeout = metrics.timeout
    result.ai_fallback = metrics.fallback or metrics.skipped
    result.ai_skipped_reason = metrics.skipped_reason
    result.ai_sections = metrics.ai_sections
    result.ai_avg_section_ms = metrics.avg_section_ms
    result.ai_slowest_section = metrics.slowest_section
    result.ai_fastest_section = metrics.fastest_section
    result.ai_total_prompt_chars = metrics.total_prompt_chars
    result.ai_total_response_chars = metrics.total_response_chars
    result.ai_timeout_count = metrics.timeout_count
    result.ai_fallback_count = metrics.fallback_count
    result.ai_section_parallelism = metrics.section_parallelism
    result.ai_dynamic_sectioning_enabled = metrics.dynamic_sectioning_enabled
    result.ai_section_results = metrics.section_results or []
    cpu1, mem1 = cpu_mem_snapshot()
    gu, gt, gm = gpu_snapshot()
    result.cpu_pct = max(cpu0, cpu1)
    result.mem_mb = max(mem0, mem1)
    if gu is not None:
        result.gpu_util_pct = gu
        result.gpu_mem_used_mb = gm
    return result


def run(mode_num: int, bars: int, model: str = DEFAULT_MODEL,
        extreme: bool = False, **_) -> BenchResult:
    """Dispatch AI benchmark mode."""
    match mode_num:
        case 43:
            return run_ai_latency(bars, mode_num, model)
        case 44:
            return run_with_ai(
                bars, mode_num, "Phase AI-2", "leika_mode_1", 1, 1, model, extreme,
                ai_config=BenchmarkAiConfig(ai_mode="quick-ai", model=model, max_context_chars=8_000, max_output_tokens=256, timeout_seconds=30),
                dynamic_sections=True,
            )
        case 45:
            return run_with_ai(
                bars, mode_num, "Phase AI-3", "leika_mode_1", 1, N_ASSETS, model, extreme,
                ai_config=BenchmarkAiConfig(ai_mode="quick-ai", model=model, max_context_chars=8_000, max_output_tokens=256, timeout_seconds=30),
                dynamic_sections=True,
            )
        case 46:
            return run_with_ai(
                bars, mode_num, "Phase AI-2", "leika_mode_2", 2, 1, model, extreme,
                ai_config=BenchmarkAiConfig(ai_mode="full-ai", model=model, max_context_chars=16_000, max_output_tokens=1_024, timeout_seconds=120),
                dynamic_sections=True,
            )
        case 47:
            return run_with_ai(
                bars, mode_num, "Phase AI-3", "leika_mode_2", 2, N_ASSETS, model, extreme,
                ai_config=BenchmarkAiConfig(ai_mode="full-ai", model=model, max_context_chars=16_000, max_output_tokens=1_024, timeout_seconds=120),
                dynamic_sections=True,
            )
    raise ValueError(f"Unknown AI bench mode {mode_num}")
