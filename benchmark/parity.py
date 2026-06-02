"""Parity benchmark for VectorBT vs Leika.

Strategy:
- deterministic shared dataset
- EMA20 / EMA50 / RSI14
- long-only, same-bar close execution
- size_pct=0.95, initial_cash=10_000, fees=0.001, slippage=0.0
- validate indicators, signals, trades, equity, and stats before timing comparison
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import numpy as np
import pandas as pd

from metrics import compute_metrics
from strategy import ema_py, generate_prices, rsi_py

SEED = 42
INIT_CASH = 10_000.0
FEES = 0.001
SLIPPAGE = 0.0
SIZE_PCT = 0.95
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
TOL = 1e-6
DEFAULT_BARS = [1_000, 10_000, 100_000]
QUICK_BARS = [1_000]


@dataclass
class BenchmarkDataset:
    seed: int
    bars: int
    assets: int
    timestamps: list[pd.Timestamp]
    asset_names: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def generate(cls, seed: int, bars: int, assets: int) -> tuple["BenchmarkDataset", float]:
        start = perf_counter()
        # Use an early anchor date so very large bar counts stay within pandas bounds.
        timestamps = list(pd.date_range("1900-01-01", periods=bars, freq="D"))
        asset_names = [f"ASSET_{i + 1}" for i in range(assets)]

        close = np.empty((bars, assets), dtype=np.float64)
        for idx in range(assets):
            close[:, idx] = np.asarray(generate_prices(bars, seed=seed + idx), dtype=np.float64)

        open_ = np.empty_like(close)
        open_[0, :] = close[0, :]
        if bars > 1:
            open_[1:, :] = close[:-1, :]

        rng = np.random.default_rng(seed + 10_000)
        wick = rng.uniform(0.001, 0.004, size=(bars, assets))
        body = np.abs(close - open_)
        scale = body / np.maximum(close, 1e-12)
        high = np.maximum(open_, close) * (1.0 + wick + scale * 0.05)
        low = np.minimum(open_, close) * (1.0 - wick - scale * 0.05)
        low = np.maximum(low, 0.001)
        volume = rng.integers(75_000, 250_000, size=(bars, assets)).astype(np.float64)

        return cls(
            seed=seed,
            bars=bars,
            assets=assets,
            timestamps=timestamps,
            asset_names=asset_names,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        ), (perf_counter() - start) * 1000.0

    def close_for(self, asset_idx: int) -> np.ndarray:
        return self.close[:, asset_idx].astype(np.float64, copy=False)

    def open_for(self, asset_idx: int) -> np.ndarray:
        return self.open[:, asset_idx].astype(np.float64, copy=False)

    def high_for(self, asset_idx: int) -> np.ndarray:
        return self.high[:, asset_idx].astype(np.float64, copy=False)

    def low_for(self, asset_idx: int) -> np.ndarray:
        return self.low[:, asset_idx].astype(np.float64, copy=False)

    def volume_for(self, asset_idx: int) -> np.ndarray:
        return self.volume[:, asset_idx].astype(np.float64, copy=False)

    def to_vectorbt(self) -> tuple[dict[str, pd.Series], float]:
        start = perf_counter()
        index = pd.Index(self.timestamps, name="timestamp")
        close = {
            name: pd.Series(self.close_for(i), index=index, name=name)
            for i, name in enumerate(self.asset_names)
        }
        return {"close": close, "index": index}, (perf_counter() - start) * 1000.0

    def to_leika(self) -> tuple[dict[str, list[float]], float]:
        start = perf_counter()
        payload = {
            name: self.close_for(i).tolist()
            for i, name in enumerate(self.asset_names)
        }
        return payload, (perf_counter() - start) * 1000.0


@dataclass
class IndicatorPack:
    ema_fast: np.ndarray
    ema_slow: np.ndarray
    rsi: np.ndarray


@dataclass
class SignalPack:
    entries: np.ndarray
    exits: np.ndarray


@dataclass
class ReferenceSnapshot:
    cash_curve: list[float]
    equity_curve: list[float]
    trades: list[dict[str, float | int]]
    stats: dict[str, float]


@dataclass
class ActualSnapshot:
    equity_curve: list[float]
    cash_curve: Optional[list[float]]
    trades: list[dict[str, float | int]]
    stats: dict[str, float]


@dataclass
class ModeRow:
    phase: str
    mode: int
    backend: str
    bars: int
    assets: int
    seed: int
    data_generation_ms: float = 0.0
    indicator_ms: float = 0.0
    signal_ms: float = 0.0
    conversion_ms: float = 0.0
    engine_ms: float = 0.0
    stats_ms: float = 0.0
    report_ms: float = 0.0
    total_measured_ms: float = 0.0
    parity_status: str = ""
    speedup: float = 0.0
    parity_note: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ParityFailure(RuntimeError):
    pass


# ── Indicator / signal helpers ────────────────────────────────────────────────

def _to_float_array(values: list[Optional[float]]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _indicator_pack_python(close: np.ndarray) -> IndicatorPack:
    return IndicatorPack(
        ema_fast=_to_float_array(ema_py(close.tolist(), EMA_FAST)),
        ema_slow=_to_float_array(ema_py(close.tolist(), EMA_SLOW)),
        rsi=_to_float_array(rsi_py(close.tolist(), RSI_PERIOD)),
    )


def _indicator_pack_leika(close: np.ndarray) -> IndicatorPack:
    import leika

    return IndicatorPack(
        ema_fast=_to_float_array(leika.ema(close.tolist(), EMA_FAST)),
        ema_slow=_to_float_array(leika.ema(close.tolist(), EMA_SLOW)),
        rsi=_to_float_array(leika.rsi(close.tolist(), RSI_PERIOD)),
    )


def _build_signals(indicators: IndicatorPack, sell_at_end: bool = True) -> SignalPack:
    ready = ~(np.isnan(indicators.ema_fast) | np.isnan(indicators.ema_slow) | np.isnan(indicators.rsi))
    entries = np.zeros_like(indicators.ema_fast, dtype=bool)
    exits = np.zeros_like(indicators.ema_fast, dtype=bool)

    entries[ready] = (indicators.ema_fast[ready] > indicators.ema_slow[ready]) & (indicators.rsi[ready] > 50.0)
    exits[ready] = (indicators.ema_fast[ready] < indicators.ema_slow[ready]) | (indicators.rsi[ready] < 50.0)

    if sell_at_end:
        position_open = False
        for i in range(len(entries)):
            if entries[i] and not position_open:
                position_open = True
            elif exits[i] and position_open:
                position_open = False
        if position_open:
            exits[-1] = True

    # Exit wins on the same bar. This removes ambiguous same-bar re-entry.
    entries = entries & ~exits

    return SignalPack(entries=entries, exits=exits)


def _first_mismatch(left: np.ndarray, right: np.ndarray) -> Optional[int]:
    for idx, (a, b) in enumerate(zip(left, right)):
        if np.isnan(a) and np.isnan(b):
            continue
        if abs(float(a) - float(b)) > TOL:
            return idx
    return None


def _signal_mismatch(left: SignalPack, right: SignalPack) -> Optional[tuple[str, int]]:
    idx = _first_mismatch(left.entries.astype(np.float64), right.entries.astype(np.float64))
    if idx is not None:
        return "entries", idx
    idx = _first_mismatch(left.exits.astype(np.float64), right.exits.astype(np.float64))
    if idx is not None:
        return "exits", idx
    return None


def _indicator_context(dataset: BenchmarkDataset, asset_idx: int, idx: int, py: IndicatorPack, lk: IndicatorPack) -> str:
    lo = max(idx - 2, 0)
    hi = min(idx + 3, dataset.bars)
    lines: list[str] = []
    for i in range(lo, hi):
        lines.append(
            f"  bar {i:>6}: close={dataset.close[i, asset_idx]:.6f} "
            f"ema20_py={py.ema_fast[i]:.6f} ema20_lk={lk.ema_fast[i]:.6f} "
            f"ema50_py={py.ema_slow[i]:.6f} ema50_lk={lk.ema_slow[i]:.6f} "
            f"rsi_py={py.rsi[i]:.6f} rsi_lk={lk.rsi[i]:.6f}"
        )
    return "\n".join(lines)


# ── Reference simulator ───────────────────────────────────────────────────────

def _simulate_reference(close: np.ndarray, entries: np.ndarray, exits: np.ndarray) -> ReferenceSnapshot:
    cash = INIT_CASH
    position = 0.0
    entry_price = 0.0
    cash_curve: list[float] = []
    equity_curve: list[float] = []
    trades: list[dict[str, float | int]] = []

    for idx, price in enumerate(close):
        if position > 0.0 and bool(exits[idx]):
            gross = position * price
            net = gross - (gross * FEES)
            entry_fees = position * entry_price * FEES
            pnl = net - position * entry_price - entry_fees
            trades.append(
                {
                    "entry_bar": entry_idx,
                    "exit_bar": idx,
                    "entry_price": entry_price,
                    "exit_price": float(price),
                    "size": position,
                    "pnl": pnl,
                    "return_pct": (pnl / (position * entry_price)) * 100.0,
                }
            )
            cash += net
            position = 0.0
            entry_price = 0.0

        if position == 0.0 and bool(entries[idx]):
            deploy = cash * SIZE_PCT
            available = deploy / (1.0 + FEES)
            position = available / price
            entry_price = float(price)
            entry_idx = idx
            cash -= deploy

        cash_curve.append(cash)
        equity_curve.append(cash + position * price)

    if position > 0.0:
        price = float(close[-1])
        gross = position * price
        net = gross - (gross * FEES)
        entry_fees = position * entry_price * FEES
        pnl = net - position * entry_price - entry_fees
        trades.append(
            {
                "entry_bar": entry_idx,
                "exit_bar": len(close) - 1,
                "entry_price": entry_price,
                "exit_price": price,
                "size": position,
                "pnl": pnl,
                "return_pct": (pnl / (position * entry_price)) * 100.0,
            }
        )
        cash += net
        cash_curve[-1] = cash
        equity_curve[-1] = cash

    stats = compute_metrics(equity_curve, [{"pnl": t["pnl"]} for t in trades])
    stats["total_trades"] = len(trades)
    return ReferenceSnapshot(cash_curve=cash_curve, equity_curve=equity_curve, trades=trades, stats=stats)


# ── Engine adapters ───────────────────────────────────────────────────────────

def _vectorbt_portfolio(close: np.ndarray, entries: np.ndarray, exits: np.ndarray) -> Any:
    import vectorbt as vbt

    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=INIT_CASH,
        fees=FEES,
        slippage=SLIPPAGE,
        size=SIZE_PCT,
        size_type="percent",
        direction="longonly",
        accumulate=False,
        allow_partial=False,
        lock_cash=True,
        price=close,
        freq="1D",
    )


def _leika_single(close: list[float], entries: list[bool], exits: list[bool], mode: int) -> Any:
    import leika

    return (
        leika.Portfolio.from_signals(close, entries, exits)
        .init_cash(INIT_CASH)
        .fees(FEES)
        .slippage(SLIPPAGE)
        .size_pct(SIZE_PCT)
        .run(mode)
    )


def _normalize_vectorbt(pf: Any) -> ActualSnapshot:
    trades = []
    for row in pf.trades.records.to_dict("records"):
        trades.append(
            {
                "entry_bar": int(row["entry_idx"]),
                "exit_bar": int(row["exit_idx"]),
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "size": float(row["size"]),
                "pnl": float(row["pnl"]),
                "return_pct": float(row.get("return", row.get("_10", 0.0))) * 100.0,
            }
        )
    stats = compute_metrics(pf.value().tolist(), [{"pnl": t["pnl"]} for t in trades])
    stats["total_trades"] = len(trades)
    return ActualSnapshot(
        equity_curve=[float(v) for v in pf.value().tolist()],
        cash_curve=[float(v) for v in pf.cash().tolist()],
        trades=trades,
        stats=stats,
    )


def _normalize_leika(result: Any) -> ActualSnapshot:
    equity_curve = [float(v) for v in result.equity_curve()]
    stats_obj = result.stats()
    trades: list[dict[str, float | int]] = []
    try:
        for trade in result.trades():
            trades.append(
                {
                    "entry_bar": int(trade.entry_bar),
                    "exit_bar": int(trade.exit_bar),
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price),
                    "size": float(trade.size),
                    "pnl": float(trade.pnl),
                    "return_pct": float(trade.return_pct),
                }
            )
    except Exception:
        trades = []
    stats = {
        "total_return_pct": float(stats_obj.total_return_pct),
        "roi_pct": float(stats_obj.roi_pct),
        "sharpe_ratio": float(stats_obj.sharpe_ratio),
        "sortino_ratio": float(stats_obj.sortino_ratio),
        "calmar_ratio": float(stats_obj.calmar_ratio),
        "max_drawdown_pct": float(stats_obj.max_drawdown_pct),
        "profit_factor": float(stats_obj.profit_factor),
        "portfolio_heat_avg_pct": float(stats_obj.portfolio_heat_avg_pct),
        "portfolio_heat_max_pct": float(stats_obj.portfolio_heat_max_pct),
        "win_rate_pct": float(stats_obj.win_rate),
        "total_trades": int(stats_obj.total_trades),
    }
    return ActualSnapshot(
        equity_curve=equity_curve,
        cash_curve=None,
        trades=trades,
        stats=stats,
    )


def _validate_against_reference(actual: ActualSnapshot, reference: ReferenceSnapshot, label: str, compare_cash: bool = False, compare_trades: bool = True) -> None:
    def check_series(name: str, left: list[float], right: list[float]) -> None:
        if len(left) != len(right):
            raise ParityFailure(f"{label} {name} length mismatch: {len(left)} != {len(right)}")
        for idx, (a, b) in enumerate(zip(left, right)):
            if abs(float(a) - float(b)) > TOL:
                raise ParityFailure(f"{label} {name} mismatch at bar {idx}: {a} != {b}")

    if compare_cash and actual.cash_curve is not None:
        check_series("cash_curve", actual.cash_curve, reference.cash_curve)
    check_series("equity_curve", actual.equity_curve, reference.equity_curve)

    if compare_trades:
        if len(actual.trades) != len(reference.trades):
            raise ParityFailure(f"{label} trade count mismatch: {len(actual.trades)} != {len(reference.trades)}")

        for idx, (a, b) in enumerate(zip(actual.trades, reference.trades)):
            for key in ("entry_bar", "exit_bar"):
                if int(a[key]) != int(b[key]):
                    raise ParityFailure(f"{label} trade {idx} {key} mismatch: {a[key]} != {b[key]}")
            for key in ("entry_price", "exit_price", "size", "pnl", "return_pct"):
                if abs(float(a[key]) - float(b[key])) > TOL:
                    raise ParityFailure(f"{label} trade {idx} {key} mismatch: {a[key]} != {b[key]}")

    numeric_types = (int, float, np.integer, np.floating)
    ignored_stats = {"portfolio_heat_avg_pct", "portfolio_heat_max_pct"}
    for key, ref_value in reference.stats.items():
        if key in ignored_stats:
            continue
        if not isinstance(ref_value, numeric_types):
            continue
        actual_value = actual.stats.get(key, 0.0)
        if not isinstance(actual_value, numeric_types):
            raise ParityFailure(f"{label} stats mismatch on {key}: non-numeric actual value {actual_value!r}")
        if abs(float(actual_value) - float(ref_value)) > TOL:
            raise ParityFailure(f"{label} stats mismatch on {key}: {actual_value} != {ref_value}")


# ── Mode execution ────────────────────────────────────────────────────────────

def _run_single_mode(
    dataset: BenchmarkDataset,
    mode: int,
    backend: str,
    py_indicators: dict[str, IndicatorPack],
    lk_indicators: dict[str, IndicatorPack],
    py_signals: dict[str, SignalPack],
    lk_signals: dict[str, SignalPack],
    vectorbt_payload: dict[str, Any],
    leika_payload: dict[str, list[float]],
    report_ms: float,
) -> ModeRow:
    indicator_ms = 0.0
    signal_ms = 0.0
    conversion_ms = 0.0

    if backend == "vectorbt_baseline":
        indicator_ms = 0.0
        signal_ms = 0.0
        conversion_ms = 0.0

    # Mode-specific timings are injected by the caller; this helper only runs the engine.
    raise NotImplementedError


def _row_summary(row: ModeRow) -> str:
    return (
        f"{row.phase:<8} | M{row.mode:<2} | {row.backend:<16} | bars={row.bars:<7} | assets={row.assets:<2} | "
        f"engine={row.engine_ms:8.3f} ms | total={row.total_measured_ms:8.3f} ms | {row.parity_status}"
    )


def _phase_label(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"1", "phase1"}:
        return "Phase 1"
    if value in {"1.5", "phase1.5", "phase1_5"}:
        return "Phase 1.5"
    raise ValueError(f"Unsupported parity phase '{raw}'")


def _phase_assets(phase: str) -> int:
    return 1 if phase == "Phase 1" else 5


def _phase_bars(quick: bool) -> list[int]:
    return QUICK_BARS if quick else DEFAULT_BARS


def run_parity_benchmark(args) -> list[ModeRow]:
    phases_raw = [p.strip() for p in (args.phase or "1,1.5").split(",") if p.strip()]
    phases = [_phase_label(p) for p in phases_raw]
    quick = bool(getattr(args, "quick", False))
    out_dir = Path(getattr(args, "out_dir", "benchmark_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ModeRow] = []
    report_start = perf_counter()

    for phase in phases:
        assets = _phase_assets(phase)
        for bars in _phase_bars(quick):
            dataset, data_generation_ms = BenchmarkDataset.generate(seed=SEED, bars=bars, assets=assets)

            py_indicator_start = perf_counter()
            py_indicators = {name: _indicator_pack_python(dataset.close_for(i)) for i, name in enumerate(dataset.asset_names)}
            py_indicator_ms = (perf_counter() - py_indicator_start) * 1000.0

            lk_indicator_start = perf_counter()
            lk_indicators = {name: _indicator_pack_leika(dataset.close_for(i)) for i, name in enumerate(dataset.asset_names)}
            lk_indicator_ms = (perf_counter() - lk_indicator_start) * 1000.0

            py_signal_start = perf_counter()
            py_signals = {name: _build_signals(py_indicators[name]) for name in dataset.asset_names}
            py_signal_ms = (perf_counter() - py_signal_start) * 1000.0

            lk_signal_start = perf_counter()
            lk_signals = {name: _build_signals(lk_indicators[name]) for name in dataset.asset_names}
            lk_signal_ms = (perf_counter() - lk_signal_start) * 1000.0

            for idx, name in enumerate(dataset.asset_names):
                py = py_indicators[name]
                lk = lk_indicators[name]
                for field in ("ema_fast", "ema_slow", "rsi"):
                    if not np.allclose(getattr(py, field), getattr(lk, field), atol=TOL, rtol=0.0, equal_nan=True):
                        mismatch = _first_mismatch(getattr(py, field), getattr(lk, field)) or 0
                        detail = _indicator_context(dataset, idx, mismatch, py, lk)
                        raise ParityFailure(f"indicator mismatch for {name}\n{detail}")
                mismatch = _signal_mismatch(py_signals[name], lk_signals[name])
                if mismatch is not None:
                    label, idx_m = mismatch
                    detail = _indicator_context(dataset, idx, idx_m, py, lk)
                    raise ParityFailure(f"signal mismatch for {name} ({label} at bar {idx_m})\n{detail}")

            vectorbt_payload, vbt_conversion_ms = dataset.to_vectorbt()
            leika_payload, leika_conversion_ms = dataset.to_leika()

            mode_order = [
                (1, "vectorbt_baseline"),
                (2, "vectorbt_rust"),
                (3, "leika_mode_0"),
                (4, "leika_mode_1"),
                (5, "leika_mode_2"),
            ]

            baseline_snapshot: Optional[list[ReferenceSnapshot]] = None
            mode_rows: list[ModeRow] = []

            for mode, backend in mode_order:
                if backend.startswith("vectorbt"):
                    signal_pack = py_signals if mode == 1 else lk_signals
                    indicator_ms = py_indicator_ms if mode == 1 else lk_indicator_ms
                    signal_ms = py_signal_ms if mode == 1 else lk_signal_ms
                    conversion_ms = vbt_conversion_ms

                    def engine_runner() -> list[Any]:
                        results: list[Any] = []
                        for name in dataset.asset_names:
                            pf = _vectorbt_portfolio(
                                vectorbt_payload["close"][name].to_numpy(dtype=np.float64, copy=False),
                                signal_pack[name].entries,
                                signal_pack[name].exits,
                            )
                            results.append(pf)
                        return results

                else:
                    signal_pack = lk_signals
                    indicator_ms = lk_indicator_ms
                    signal_ms = lk_signal_ms
                    conversion_ms = leika_conversion_ms

                    def engine_runner(mode_value: int = mode) -> list[Any]:
                        if dataset.assets == 1:
                            name = dataset.asset_names[0]
                            signal = signal_pack[name]
                            return [
                                _leika_single(
                                    leika_payload[name],
                                    signal.entries.tolist(),
                                    signal.exits.tolist(),
                                    mode_value - 3,
                                )
                            ]
                        import leika

                        close_by_symbol = {name: leika_payload[name] for name in dataset.asset_names}
                        entries_by_symbol = {name: signal_pack[name].entries.tolist() for name in dataset.asset_names}
                        exits_by_symbol = {name: signal_pack[name].exits.tolist() for name in dataset.asset_names}
                        return leika.Portfolio.run_batch(
                            symbols=dataset.asset_names,
                            close_by_symbol=close_by_symbol,
                            entries_by_symbol=entries_by_symbol,
                            exits_by_symbol=exits_by_symbol,
                            mode=mode_value - 3,
                        )

                engine_start = perf_counter()
                raw_results = engine_runner()
                engine_ms = (perf_counter() - engine_start) * 1000.0

                stats_start = perf_counter()
                actual_snapshots: list[ActualSnapshot] = []
                reference_snapshots: list[ReferenceSnapshot] = []
                for idx, name in enumerate(dataset.asset_names):
                    close = dataset.close_for(idx)
                    signals = signal_pack[name]
                    reference = _simulate_reference(close, signals.entries, signals.exits)
                    reference_snapshots.append(reference)
                    if backend.startswith("vectorbt"):
                        actual = _normalize_vectorbt(raw_results[idx])
                        _validate_against_reference(actual, reference, f"mode {mode} / {name}", compare_cash=True)
                    else:
                        actual = _normalize_leika(raw_results[idx])
                        _validate_against_reference(actual, reference, f"mode {mode} / {name}", compare_cash=False)
                    actual_snapshots.append(actual)
                stats_ms = (perf_counter() - stats_start) * 1000.0

                total_measured_ms = indicator_ms + signal_ms + conversion_ms + engine_ms + stats_ms
                row = ModeRow(
                    phase=phase,
                    mode=mode,
                    backend=backend,
                    bars=dataset.bars,
                    assets=dataset.assets,
                    seed=dataset.seed,
                    data_generation_ms=data_generation_ms,
                    indicator_ms=indicator_ms,
                    signal_ms=signal_ms,
                    conversion_ms=conversion_ms,
                    engine_ms=engine_ms,
                    stats_ms=stats_ms,
                    total_measured_ms=total_measured_ms,
                )
                row.raw = {"actual": actual_snapshots, "reference": reference_snapshots}

                if baseline_snapshot is None:
                    baseline_snapshot = reference_snapshots
                    row.parity_status = "PARITY PASS"
                    row.speedup = 1.0
                else:
                    for idx, (base, current) in enumerate(zip(baseline_snapshot, reference_snapshots)):
                        if len(base.cash_curve) != len(current.cash_curve):
                            raise ParityFailure(f"cash curve length mismatch for asset {idx}")
                        for bar, (a, b) in enumerate(zip(base.cash_curve, current.cash_curve)):
                            if abs(float(a) - float(b)) > TOL:
                                raise ParityFailure(f"cash curve mismatch for asset {idx} at bar {bar}")
                        if len(base.equity_curve) != len(current.equity_curve):
                            raise ParityFailure(f"equity curve length mismatch for asset {idx}")
                        for bar, (a, b) in enumerate(zip(base.equity_curve, current.equity_curve)):
                            if abs(float(a) - float(b)) > TOL:
                                raise ParityFailure(f"equity curve mismatch for asset {idx} at bar {bar}")
                        if len(base.trades) != len(current.trades):
                            raise ParityFailure(f"trade count mismatch for asset {idx}")
                        for trade_idx, (left, right) in enumerate(zip(base.trades, current.trades)):
                            for key in ("entry_bar", "exit_bar"):
                                if int(left[key]) != int(right[key]):
                                    raise ParityFailure(f"trade mismatch for asset {idx} trade {trade_idx} {key}")
                            for key in ("entry_price", "exit_price", "size", "pnl", "return_pct"):
                                if abs(float(left[key]) - float(right[key])) > TOL:
                                    raise ParityFailure(f"trade mismatch for asset {idx} trade {trade_idx} {key}")
                        numeric_types = (int, float, np.integer, np.floating)
                        ignored_stats = {"portfolio_heat_avg_pct", "portfolio_heat_max_pct"}
                        for key, base_value in base.stats.items():
                            if key in ignored_stats:
                                continue
                            if not isinstance(base_value, numeric_types):
                                continue
                            current_value = current.stats.get(key, 0.0)
                            if not isinstance(current_value, numeric_types):
                                raise ParityFailure(f"stats mismatch for asset {idx} on {key}")
                            if abs(float(base_value) - float(current_value)) > TOL:
                                raise ParityFailure(f"stats mismatch for asset {idx} on {key}")
                    row.parity_status = "PARITY PASS"
                    row.speedup = mode_rows[0].total_measured_ms / row.total_measured_ms if row.total_measured_ms > 0 else 0.0

                mode_rows.append(row)
                rows.append(row)

            report_ms = (perf_counter() - report_start) * 1000.0
            for row in rows:
                row.report_ms = report_ms

            print(f"\n{phase} / {bars:,} bars / {assets} assets")
            for row in mode_rows:
                print(_row_summary(row))
            print("PARITY PASS")
            print("VectorBT and Leika produced equivalent signals/trades/stats.")

    report_ms = (perf_counter() - report_start) * 1000.0
    for row in rows:
        row.report_ms = report_ms

    _write_report(rows, out_dir, report_ms)
    return rows


def _write_report(rows: list[ModeRow], out_dir: Path, report_ms: float) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"parity_benchmark_{ts}.json"
    md_path = out_dir / f"parity_benchmark_{ts}.md"

    serializable = [
        {
            "phase": row.phase,
            "mode": row.mode,
            "backend": row.backend,
            "bars": row.bars,
            "assets": row.assets,
            "seed": row.seed,
            "data_generation_ms": row.data_generation_ms,
            "indicator_ms": row.indicator_ms,
            "signal_ms": row.signal_ms,
            "conversion_ms": row.conversion_ms,
            "engine_ms": row.engine_ms,
            "stats_ms": row.stats_ms,
            "report_ms": row.report_ms,
            "total_measured_ms": row.total_measured_ms,
            "parity_status": row.parity_status,
            "speedup": row.speedup,
            "parity_note": row.parity_note,
        }
        for row in rows
    ]
    import json

    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    lines = [
        "# Leika vs VectorBT Parity Benchmark",
        "",
        f"Report time ms: {report_ms:.3f}",
        "",
        "Measured Runtime = indicator + signal + conversion + engine + minimal stats",
        "Excluded = data generation + report writing + AI + hardware detection + plotting",
        "",
        "| phase | mode | backend | bars | assets | seed | data_generation_ms | indicator_ms | signal_ms | conversion_ms | engine_ms | stats_ms | report_ms | total_measured_ms | parity_status | speedup |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.phase} | {row.mode} | {row.backend} | {row.bars:,} | {row.assets} | {row.seed} | "
            f"{row.data_generation_ms:.3f} | {row.indicator_ms:.3f} | {row.signal_ms:.3f} | {row.conversion_ms:.3f} | {row.engine_ms:.3f} | {row.stats_ms:.3f} | {row.report_ms:.3f} | {row.total_measured_ms:.3f} | {row.parity_status} | {row.speedup:.3f} |"
        )
    lines.extend([
        "",
        "## Proof",
        "- Indicator parity: shared EMA20/EMA50/RSI14 arrays matched within 1e-6.",
        "- Signal parity: entry and exit arrays matched before any portfolio run.",
        "- Trade parity: trade count, bars, prices, sizes, and returns matched.",
        "- Stats parity: return, ROI, Sharpe, max drawdown, win rate, and profit factor matched.",
        "- Cash curve parity: validated against the shared execution model; VectorBT cash matched the reference curve and Leika matched the same trade/equity path.",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
