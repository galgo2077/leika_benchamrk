"""Compact AI context helpers for benchmark analysis."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Optional


@dataclass
class BenchmarkAiConfig:
    ai_mode: str
    model: str
    max_context_chars: int
    max_output_tokens: int
    timeout_seconds: int
    max_ai_cases: Optional[int] = None


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def build_trade_records(
    prices: list[float],
    entries: list[bool],
    exits: list[bool],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    in_position = False
    entry_bar = 0
    entry_price = 0.0

    for i, price in enumerate(prices):
        if entries[i] and not in_position:
            in_position = True
            entry_bar = i
            entry_price = price
            continue
        if exits[i] and in_position:
            exit_price = price
            duration = max(1, i - entry_bar)
            return_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price else 0.0
            records.append(
                {
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "return_pct": round(return_pct, 4),
                    "duration_bars": duration,
                }
            )
            in_position = False

    if in_position and prices:
        exit_price = prices[-1]
        duration = max(1, len(prices) - 1 - entry_bar)
        return_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price else 0.0
        records.append(
            {
                "entry_bar": entry_bar,
                "exit_bar": len(prices) - 1,
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "return_pct": round(return_pct, 4),
                "duration_bars": duration,
            }
        )

    return records


def summarize_trade_records(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "avg_trade_return_pct": 0.0,
            "median_trade_return_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "longest_trade_bars": 0,
            "shortest_trade_bars": 0,
            "top_best_trades": [],
            "top_worst_trades": [],
        }

    returns = [t["return_pct"] for t in trades]
    longest = max(t["duration_bars"] for t in trades)
    shortest = min(t["duration_bars"] for t in trades)
    top_best = sorted(trades, key=lambda t: t["return_pct"], reverse=True)[:5]
    top_worst = sorted(trades, key=lambda t: t["return_pct"])[:5]
    return {
        "total_trades": len(trades),
        "avg_trade_return_pct": round(sum(returns) / len(returns), 4),
        "median_trade_return_pct": round(median(returns), 4),
        "best_trade_pct": round(max(returns), 4),
        "worst_trade_pct": round(min(returns), 4),
        "longest_trade_bars": longest,
        "shortest_trade_bars": shortest,
        "top_best_trades": top_best,
        "top_worst_trades": top_worst,
    }


def merge_trade_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    for summary in summaries:
        trades.extend(summary.get("records", []))
    return summarize_trade_records(trades)


def infer_regime(prices: list[float]) -> dict[str, Any]:
    if len(prices) < 3:
        return {
            "dominant_regime": "Unknown",
            "regime_confidence": 0.0,
            "high_volatility_pct": 0.0,
            "sideways_pct": 100.0,
            "trending_pct": 0.0,
        }

    returns = [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices)) if prices[i - 1] > 0]
    net_return = (prices[-1] / prices[0] - 1.0) * 100.0 if prices[0] else 0.0
    volatility = (sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0.0
    vol_pct = min(100.0, volatility * 1000.0)
    trending_pct = _clamp(abs(net_return) / max(len(prices) / 250.0, 1.0), 0.0, 100.0)
    sideways_pct = max(0.0, 100.0 - trending_pct - vol_pct * 0.4)
    if net_return > 0.5:
        dominant = "TrendingBull"
    elif net_return < -0.5:
        dominant = "TrendingBear"
    else:
        dominant = "Sideways"
    confidence = _clamp(abs(net_return) / 10.0 + vol_pct / 200.0, 0.0, 1.0)
    return {
        "dominant_regime": dominant,
        "regime_confidence": round(confidence, 4),
        "high_volatility_pct": round(vol_pct, 4),
        "sideways_pct": round(sideways_pct, 4),
        "trending_pct": round(trending_pct, 4),
    }


def build_strategy_dna(stats: dict[str, Any], trades: dict[str, Any], bars: int) -> dict[str, float]:
    total_trades = max(1, int(trades.get("total_trades", 0)))
    trade_rate = total_trades / max(bars, 1)
    trend_score = _clamp((stats.get("sharpe_ratio", 0.0) + stats.get("total_return_pct", 0.0) / 50.0) / 2.0)
    mean_rev_score = _clamp(1.0 - abs(stats.get("worst_trade_pct", 0.0)) / 20.0)
    vol_sens = _clamp(stats.get("max_drawdown_pct", 0.0) / 50.0)
    trade_freq = _clamp(trade_rate * 1000.0)
    holding_score = _clamp(trades.get("median_trade_return_pct", 0.0) / 10.0 if total_trades else 0.0)
    return {
        "trend_following_score": round(trend_score, 4),
        "mean_reversion_score": round(mean_rev_score, 4),
        "volatility_sensitivity": round(vol_sens, 4),
        "trade_frequency_score": round(trade_freq, 4),
        "holding_time_score": round(holding_score, 4),
    }


def build_ai_context(
    *,
    stats: dict[str, Any],
    risk_stats: dict[str, Any],
    strategy_info: dict[str, Any],
    bars: int,
    trade_summary: dict[str, Any],
    prices: Optional[list[float]] = None,
) -> dict[str, Any]:
    dna = build_strategy_dna(stats, trade_summary, bars)
    regime = infer_regime(prices or [])
    context = {
        "main_stats": {
            "total_return_pct": round(stats.get("total_return_pct", 0.0), 4),
            "roi_pct": round(stats.get("roi_pct", stats.get("total_return_pct", 0.0)), 4),
            "sharpe_ratio": round(stats.get("sharpe_ratio", 0.0), 4),
            "sortino_ratio": round(stats.get("sortino_ratio", 0.0), 4),
            "calmar_ratio": round(stats.get("calmar_ratio", 0.0), 4),
            "max_drawdown_pct": round(stats.get("max_drawdown_pct", 0.0), 4),
            "win_rate": round(stats.get("win_rate_pct", 0.0), 4),
            "profit_factor": round(stats.get("profit_factor", 0.0), 4)
            if math.isfinite(float(stats.get("profit_factor", 0.0)))
            else stats.get("profit_factor", 0.0),
            "total_trades": int(stats.get("total_trades", trade_summary.get("total_trades", 0))),
            "avg_trade_return_pct": round(trade_summary.get("avg_trade_return_pct", 0.0), 4),
            "best_trade_pct": round(trade_summary.get("best_trade_pct", 0.0), 4),
            "worst_trade_pct": round(trade_summary.get("worst_trade_pct", 0.0), 4),
        },
        "risk_stats": {
            "portfolio_heat_avg_pct": round(risk_stats.get("portfolio_heat_avg_pct", 0.0), 4),
            "portfolio_heat_max_pct": round(risk_stats.get("portfolio_heat_max_pct", 0.0), 4),
            "correlation_risk_score": round(risk_stats.get("correlation_risk_score", 0.0), 4),
            "avg_correlation": round(risk_stats.get("avg_correlation", 0.0), 4),
            "effective_diversification": round(risk_stats.get("effective_diversification", 0.0), 4),
        },
        "strategy_info": strategy_info,
        "strategy_dna": dna,
        "regime_detection": regime,
        "trade_summary": {
            "total_trades": trade_summary.get("total_trades", 0),
            "avg_trade_return_pct": trade_summary.get("avg_trade_return_pct", 0.0),
            "median_trade_return_pct": trade_summary.get("median_trade_return_pct", 0.0),
            "best_trade_pct": trade_summary.get("best_trade_pct", 0.0),
            "worst_trade_pct": trade_summary.get("worst_trade_pct", 0.0),
            "longest_trade_bars": trade_summary.get("longest_trade_bars", 0),
            "shortest_trade_bars": trade_summary.get("shortest_trade_bars", 0),
            "top_best_trades": trade_summary.get("top_best_trades", [])[:5],
            "top_worst_trades": trade_summary.get("top_worst_trades", [])[:5],
        },
    }
    return context


def compact_context(context: dict[str, Any], max_context_chars: int) -> tuple[str, int, int, bool]:
    """Serialize context and compress it if needed."""
    def dump(obj: dict[str, Any]) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)

    raw = dump(context)
    if len(raw) <= max_context_chars:
        return raw, len(raw), estimate_tokens(raw), False

    compressed = copy.deepcopy(context)
    trade_summary = compressed.get("trade_summary", {})
    trade_summary["top_best_trades"] = []
    trade_summary["top_worst_trades"] = []
    compressed["trade_summary"] = trade_summary
    compressed["strategy_info"] = {
        k: v
        for k, v in compressed.get("strategy_info", {}).items()
        if k in {"strategy_name", "timeframe", "assets", "bars", "mode", "backend"}
    }
    raw = dump(compressed)
    if len(raw) <= max_context_chars:
        return raw, len(raw), estimate_tokens(raw), True

    minimal = {
        "main_stats": compressed.get("main_stats", {}),
        "risk_stats": compressed.get("risk_stats", {}),
        "strategy_info": compressed.get("strategy_info", {}),
    }
    raw = dump(minimal)
    if len(raw) <= max_context_chars:
        return raw, len(raw), estimate_tokens(raw), True

    raise ValueError("AI skipped: context exceeds configured max_context_chars")
