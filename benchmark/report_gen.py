"""
Report generator — produces a .txt + .json report with comparative tables,
hardware snapshot, per-phase analysis, and overall leaderboard.

Supports all phases:
  Phase 1    (modes  1– 5)  single-asset portfolio
  Phase 1.5  (modes  6–10)  5-asset portfolio
  Phase 2    (modes 11–15)  single-asset + AI
  Phase 2.5  (modes 16–20)  5-asset + AI
  Phase 3    (modes 21–25)  Monte Carlo
  Phase 4    (modes 26–30)  Random Walk single-asset
  Phase 4.5  (modes 31–35)  Random Walk multi-asset
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from metrics import BenchResult

_REPORT_DUP_TOLERANCE = 0.05
_REPORT_MATCH_FIELDS = (
    "mode",
    "mode_name",
    "phase",
    "phase_type",
    "bars",
    "n_paths",
    "n_assets",
    "rw_model",
    "backend",
    "leika_exec_mode",
    "ai_enabled",
)
_REPORT_MATCH_METRICS = (
    "exec_ms",
    "total_runtime_ms",
    "throughput_bars_sec",
    "paths_sec",
    "total_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "cpu_pct",
    "mem_mb",
    "ai_total_time_ms",
    "ai_ms_total",
    "ai_context_chars",
    "ai_sections",
    "ai_calls",
)


def _report_sort_key(row) -> tuple:
    return (
        getattr(row, "phase", ""),
        int(getattr(row, "mode", 0) or 0),
        int(getattr(row, "bars", 0) or 0),
        int(getattr(row, "n_paths", 0) or 0),
        int(getattr(row, "n_assets", 0) or 0),
        getattr(row, "rw_model", "") or "",
        getattr(row, "mode_name", "") or "",
    )


def _get_row_value(row, field: str):
    if isinstance(row, dict):
        return row.get(field)
    return getattr(row, field, None)


def _close_enough(a, b, tolerance: float = _REPORT_DUP_TOLERANCE) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return a == b
    if af == bf:
        return True
    scale = max(abs(af), abs(bf), 1.0)
    return abs(af - bf) <= scale * tolerance


def _report_rows_match(current_row, previous_row) -> bool:
    for field in _REPORT_MATCH_FIELDS:
        if _get_row_value(current_row, field) != _get_row_value(previous_row, field):
            return False
    for field in _REPORT_MATCH_METRICS:
        if not _close_enough(_get_row_value(current_row, field), _get_row_value(previous_row, field)):
            return False
    return True


def _find_matching_report(out_dir: Path, results: list[BenchResult]) -> tuple[Path, str] | None:
    candidates = sorted(
        out_dir.glob("leika_benchmark_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    current_rows = sorted(results, key=_report_sort_key)
    for json_path in candidates:
        try:
            previous_rows = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(previous_rows, list) or len(previous_rows) != len(current_rows):
            continue
        previous_rows = sorted(previous_rows, key=_report_sort_key)
        if all(_report_rows_match(cur, prev) for cur, prev in zip(current_rows, previous_rows)):
            ts_file = json_path.stem.replace("leika_benchmark_", "", 1)
            return json_path, ts_file
    return None


def _timestamp_parts(now: Optional[datetime] = None) -> tuple[str, str]:
    dt = now or datetime.now().astimezone()
    ms = dt.microsecond // 1000
    tz_name = dt.tzname() or "local"
    human = f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d} {tz_name}"
    file_ts = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"{ms:03d}Z"
    return human, file_ts

def generate_markdown_report(results: list, hw_info: dict,
                             timestamp_human: str, timestamp_file: str) -> str:
    """Generate a Markdown summary suitable for pasting into Claude or a PR."""
    lines = []
    lines.append(f"# Leika Engine Benchmark Report\n*Generated: {timestamp_human}*\n")
    lines.append(f"*Timestamp: {timestamp_file}*\n")

    # Hardware
    lines.append("## Hardware")
    cpu = hw_info.get("cpu", {})
    gpu = hw_info.get("gpu", {})
    lines.append(f"- CPU: {cpu.get('name','Unknown')} ({cpu.get('threads',1)} threads)")
    lines.append(f"- RAM: {cpu.get('ram_gb','?')} GB")
    if gpu:
        lines.append(f"- GPU: {gpu.get('name','Unknown')} ({gpu.get('mem_total','?')} MB VRAM)")
    lines.append("")

    # Per-phase tables
    phases = {}
    for r in results:
        phases.setdefault(r.phase, []).append(r)

    for phase, phase_results in phases.items():
        lines.append(f"## {phase}")
        lines.append("| Mode | Engine | Backend | Bars | Time (ms) | Throughput | Return | Sharpe | GPU (ms) | Dyn | SellEnd | Fallback | NumErr |")
        lines.append("|------|--------|---------|------|-----------|------------|--------|--------|----------|-----|---------|----------|--------|")
        for r in phase_results:
            tp = ""
            if r.throughput_bars_sec >= 1e9:
                tp = f"{r.throughput_bars_sec/1e9:.2f}G ops/s"
            elif r.throughput_bars_sec >= 1e6:
                tp = f"{r.throughput_bars_sec/1e6:.1f}M ops/s"
            elif r.throughput_bars_sec > 0:
                tp = f"{r.throughput_bars_sec/1e3:.0f}K ops/s"
            sp = f"{r.speedup_vs_baseline:.1f}×" if r.speedup_vs_baseline else "—"
            ret = f"{r.total_return_pct:+.2f}%" if r.phase_type == "portfolio" else "—"
            sh  = f"{r.sharpe_ratio:.3f}" if r.phase_type == "portfolio" else "—"
            backend = getattr(r, "gpu_backend", "") or r.backend or "—"
            gpu_ms = f"{getattr(r, 'gpu_total_time_ms', 0.0):.1f}" if getattr(r, "gpu_total_time_ms", 0.0) > 0 else "—"
            num_err = f"{getattr(r, 'numerical_error_rel', None):.1e}" if getattr(r, "numerical_error_rel", None) is not None else "—"
            sell_end = getattr(r, "sell_at_end_scope", "") or "—"
            lines.append(
                f"| {r.mode} | {r.mode_name} | {backend} | {r.bars:,} | {r.exec_ms:.1f} | {tp} | {ret} | {sh} | {gpu_ms} | {'yes' if getattr(r, 'dynamic_tiling_enabled', False) else 'no'} | {sell_end} | {('yes' if getattr(r, 'memory_fallback', False) else 'no')} | {num_err} |"
            )
        lines.append("")

    # AI overhead summary if AI phases present
    ai_results = [r for r in results if r.ai_enabled]
    if ai_results:
        lines.append("## AI Overhead")
        lines.append("| Mode | Model | AI Secs | AI Calls | Avg Sec ms | Slowest | AI ms | Total ms | Parallel | Dyn | Timeout | Fallback |")
        lines.append("|------|-------|---------|----------|------------|---------|-------|----------|----------|-----|---------|----------|")
        for r in ai_results:
            ai_total = r.ai_total_time_ms or r.ai_ms_total
            total = r.total_runtime_ms or r.exec_ms
            lines.append(
                f"| {r.mode} | {getattr(r, 'ai_model', '') or '—'} | {r.ai_sections or 0} | {r.ai_calls} | {r.ai_avg_section_ms:.1f} | "
                f"{(r.ai_slowest_section or '—')} | {ai_total:.1f} | {total:.1f} | {r.ai_section_parallelism or '—'} | "
                f"{'yes' if r.ai_dynamic_sectioning_enabled else 'no'} | {'yes' if r.ai_timeout else 'no'} | {'yes' if r.ai_fallback else 'no'} |"
            )
        lines.append("")

    lines.append("---\n*Leika Engine benchmark_summary.md — paste into Claude for analysis*")
    return "\n".join(lines)


MODE_NAMES = {
    # Phase 1
    1: "VectorBT Baseline",   2: "VectorBT Rust",
    3: "Leika Mode 0",        4: "Leika Mode 1",       5: "Leika Mode 2",
    # Phase 1.5
    6: "VBT Baseline ×5",     7: "VBT Rust ×5",
    8: "Leika Mode 0 ×5",     9: "Leika Mode 1 ×5",   10: "Leika Mode 2 ×5",
    # Phase 2
    11: "VBT Baseline+AI",   12: "VBT Rust+AI",
    13: "Leika M0+AI",       14: "Leika M1+AI",       15: "Leika M2+AI",
    # Phase 2.5
    16: "VBT Base ×5+AI",    17: "VBT Rust ×5+AI",
    18: "Leika M0 ×5+AI",   19: "Leika M1 ×5+AI",    20: "Leika M2 ×5+AI",
    # Phase 3 — Monte Carlo
    21: "MC Python Baseline", 22: "MC VectorBT Baseline",
    23: "MC Leika Mode 0",   24: "MC Leika Mode 1",   25: "MC Leika Mode 2",
    # Phase 4 — Random Walk single
    26: "RW Python Baseline", 27: "RW VectorBT Baseline",
    28: "RW Leika Mode 0",   29: "RW Leika Mode 1",   30: "RW Leika Mode 2",
    # Phase 4.5 — Random Walk multi
    31: "RW Py Base ×5",     32: "RW VectorBT Baseline ×5",
    33: "RW Leika M0 ×5",   34: "RW Leika M1 ×5",    35: "RW Leika M2 ×5",
}

PHASE_MAP = {
    **{m: "Phase 1"   for m in range(1, 6)},
    **{m: "Phase 1.5" for m in range(6, 11)},
    **{m: "Phase 2"   for m in range(11, 16)},
    **{m: "Phase 2.5" for m in range(16, 21)},
    **{m: "Phase 3"   for m in range(21, 26)},
    **{m: "Phase 4"   for m in range(26, 31)},
    **{m: "Phase 4.5" for m in range(31, 36)},
}


def _gpu_full() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,temperature.gpu,"
             "memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            p = [x.strip() for x in r.stdout.strip().split(",")]
            return {
                "name":      p[0],
                "util_pct":  float(p[1]),
                "temp_c":    float(p[2]),
                "mem_used":  float(p[3]),
                "mem_total": float(p[4]),
                "power_w":   float(p[5]) if len(p) > 5 else None,
            }
    except Exception:
        pass
    return {}


def _cpu_info() -> dict:
    info = {"name": "Unknown", "threads": os.cpu_count() or 1}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["name"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        import psutil
        info["threads"] = psutil.cpu_count(logical=True) or info["threads"]
        info["cores"]   = psutil.cpu_count(logical=False) or info["threads"]
        vm = psutil.virtual_memory()
        info["ram_gb"]  = round(vm.total / 1e9, 1)
    except Exception:
        pass
    return info


def _col(s: str, w: int, align: str = "<") -> str:
    s = str(s)
    if len(s) > w:
        s = s[:w - 1] + "…"
    return f"{s:{align}{w}}"


def _fmt_tput(v: float) -> str:
    if v <= 0:
        return "—"
    if v >= 1e9:
        return f"{v / 1e9:.2f} G/s"
    if v >= 1e6:
        return f"{v / 1e6:.1f} M/s"
    return f"{v / 1e3:.0f} K/s"


# ── Portfolio section ─────────────────────────────────────────────────────────

def _section_portfolio(results: list[BenchResult], lines: list[str], hdr, blank, sep) -> None:
    pf_results = [r for r in results if r.phase_type == "portfolio"]
    if not pf_results:
        return

    hdr("PORTFOLIO BENCHMARKS  (Phase 1 / 1.5 / 2 / 2.5)")
    blank()

    bar_sizes = sorted({r.bars for r in pf_results})
    modes     = sorted({r.mode for r in pf_results})
    idx       = {(r.mode, r.bars): r for r in pf_results}

    col_w   = [6, 24, 9, 10, 8, 8, 8, 7, 8, 8, 6]
    headers = ["Mode", "Engine", "Bars", "Time(ms)", "Return%", "Sharpe", "MaxDD%", "WinR%", "Workers", "Segments", "AI"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headers, col_w)))
    lines.append("  " + " ".join("─" * w for w in col_w))

    for mode in modes:
        name = MODE_NAMES.get(mode, f"Mode{mode}")
        for bars in bar_sizes:
            r = idx.get((mode, bars))
            if r is None:
                continue
            ai_str = str(r.ai_calls) if r.ai_calls > 0 else "-"
            row = [str(mode), name, f"{bars:,}", f"{r.exec_ms:.1f}",
                   f"{r.total_return_pct:+.2f}", f"{r.sharpe_ratio:.3f}",
                   f"{r.max_drawdown_pct:.2f}", f"{r.win_rate_pct:.1f}",
                   str(r.cpu_workers_selected or "-"), str(r.segments_selected or "-"), ai_str]
            flag = " [ERR]" if r.error else ""
            lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_w)) + flag)
        lines.append("  " + " ".join("─" * w for w in col_w))

    blank()
    sep("─")
    blank()


# ── Monte Carlo section ───────────────────────────────────────────────────────

def _section_mc(results: list[BenchResult], lines: list[str], hdr, blank, sep) -> None:
    mc_results = [r for r in results if r.phase_type == "montecarlo"]
    if not mc_results:
        return

    hdr("MONTE CARLO BENCHMARKS  (Phase 3)")
    blank()

    datasets = sorted({(r.bars, r.n_paths) for r in mc_results})
    modes    = sorted({r.mode for r in mc_results})
    idx      = {(r.mode, r.bars, r.n_paths): r for r in mc_results}

    # ── Runtime table ──
    hdr("  Runtime & Throughput")
    blank()
    col_w   = [6, 24, 11, 8, 8, 10, 11, 10, 8, 8]
    headers = ["Mode", "Engine", "Backend", "Candles", "Paths", "Time(ms)", "Ops/sec", "Paths/sec", "Workers", "Segments"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headers, col_w)))
    lines.append("  " + " ".join("─" * w for w in col_w))

    for mode in modes:
        name = MODE_NAMES.get(mode, f"Mode{mode}")
        for candles, paths in datasets:
            r = idx.get((mode, candles, paths))
            if r is None:
                continue
            backend = getattr(r, "gpu_backend", "") or r.backend or "—"
            row = [str(mode), name, backend, f"{candles:,}", f"{paths:,}",
                   f"{r.exec_ms:.1f}", _fmt_tput(r.throughput_bars_sec),
                   _fmt_tput(r.paths_sec), str(r.cpu_workers_selected or "-"), str(r.segments_selected or "-")]
            flag = " [ERR]" if r.error else ""
            lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_w)) + flag)
        lines.append("  " + " ".join("─" * w for w in col_w))

    blank()

    # ── Memory table ──
    hdr("  Memory & GPU")
    blank()
    col_w2   = [6, 24, 11, 8, 8, 9, 10, 10, 10, 8, 8]
    headers2 = ["Mode", "Engine", "Backend", "Candles", "Paths", "RAM(MB)", "GPU Util%", "VRAM(MB)", "NumErr", "Workers", "Segments"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headers2, col_w2)))
    lines.append("  " + " ".join("─" * w for w in col_w2))

    for mode in modes:
        name = MODE_NAMES.get(mode, f"Mode{mode}")
        for candles, paths in datasets:
            r = idx.get((mode, candles, paths))
            if r is None:
                continue
            gpu_util = f"{r.gpu_util_pct:.0f}%" if r.gpu_util_pct is not None else "—"
            vram     = f"{r.gpu_mem_used_mb:.0f}" if r.gpu_mem_used_mb is not None else "—"
            backend = getattr(r, "gpu_backend", "") or r.backend or "—"
            num_err = f"{r.numerical_error_rel:.1e}" if r.numerical_error_rel is not None else "—"
            row = [str(mode), name, backend, f"{candles:,}", f"{paths:,}",
                   f"{r.mem_mb:.0f}", gpu_util, vram, num_err,
                   str(r.cpu_workers_selected or "-"), str(r.segments_selected or "-")]
            lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_w2)))
        lines.append("  " + " ".join("─" * w for w in col_w2))

    blank()

    # ── GPU metrics ──
    hdr("  GPU Metrics")
    blank()
    col_w3 = [6, 24, 11, 8, 8, 12, 12, 12, 12, 8, 8]
    headers3 = ["Mode", "Engine", "Backend", "Candles", "Paths", "Kernel(ms)", "Xfer(ms)", "GPU Total", "CPU Fallback", "Workers", "Segments"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headers3, col_w3)))
    lines.append("  " + " ".join("─" * w for w in col_w3))
    for mode in modes:
        name = MODE_NAMES.get(mode, f"Mode{mode}")
        for candles, paths in datasets:
            r = idx.get((mode, candles, paths))
            if r is None:
                continue
            backend = getattr(r, "gpu_backend", "") or r.backend or "—"
            if r.gpu_kernel_time_ms > 0 or r.gpu_total_time_ms > 0 or r.gpu_cpu_fallback_time_ms > 0:
                    row = [
                        str(mode), name, backend, f"{candles:,}", f"{paths:,}",
                        f"{r.gpu_kernel_time_ms:.1f}" if r.gpu_kernel_time_ms > 0 else "—",
                        f"{r.gpu_transfer_time_ms:.1f}" if r.gpu_transfer_time_ms > 0 else "—",
                        f"{r.gpu_total_time_ms:.1f}" if r.gpu_total_time_ms > 0 else "—",
                        f"{r.gpu_cpu_fallback_time_ms:.1f}" if r.gpu_cpu_fallback_time_ms > 0 else "—",
                        str(r.cpu_workers_selected or "-"), str(r.segments_selected or "-"),
                    ]
                    lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_w3)))
        lines.append("  " + " ".join("─" * w for w in col_w3))

    blank()

    # ── GPU performance: Mode 2 vs Mode 1 speedup ──
    hdr("  GPU Acceleration Factor  (Mode 2 / Mode 1)")
    blank()
    for candles, paths in datasets:
        r2 = idx.get((25, candles, paths))
        if r2 and r2.gpu_accel_factor:
            note   = "GPU faster" if r2.gpu_accel_factor > 1.05 else ("CPU faster" if r2.gpu_accel_factor < 0.95 else "~parity")
            backend = getattr(r2, "gpu_backend", "") or r2.backend or "—"
            hdr(f"    {candles:>7,} candles × {paths:>7,} paths : {r2.gpu_accel_factor:.2f}×  [{note}]  backend={backend}")
    blank()

    # ── Speedup vs Python baseline ──
    hdr("  Speedup vs VectorBT Baseline  (Mode 22)")
    blank()
    for candles, paths in datasets:
        base = idx.get((22, candles, paths))
        if not base or base.exec_ms <= 0:
            continue
        hdr(f"    Dataset {candles:>7,} × {paths:>7,}:")
        for mode in [23, 24, 25]:
            r = idx.get((mode, candles, paths))
            if r and r.exec_ms > 0:
                sp   = base.exec_ms / r.exec_ms
                name = MODE_NAMES.get(mode, f"Mode{mode}")
                hdr(f"      Mode {mode} ({name:<22}) : {sp:6.2f}×")
    blank()
    sep("─")
    blank()


# ── Random Walk section ───────────────────────────────────────────────────────

def _section_rw(results: list[BenchResult], lines: list[str], hdr, blank, sep,
                multi: bool = False) -> None:
    phase_label = "Phase 4.5" if multi else "Phase 4"
    rw_results  = [r for r in results
                   if r.phase_type == "randomwalk" and
                   (r.n_assets > 1) == multi]
    if not rw_results:
        return

    title = "RANDOM WALK MULTI-ASSET  (Phase 4.5)" if multi else "RANDOM WALK SINGLE-ASSET  (Phase 4)"
    hdr(title)
    blank()

    models    = sorted({r.rw_model for r in rw_results})
    datasets  = sorted({r.bars for r in rw_results})
    modes     = sorted({r.mode for r in rw_results})
    idx       = {(r.mode, r.bars, r.rw_model): r for r in rw_results}

    # ── Runtime table ──
    hdr("  Runtime & Throughput")
    blank()
    col_w   = [6, 24, 11, 8, 18, 10, 11, 10, 8, 8]
    headers = ["Mode", "Engine", "Backend", "Candles", "Model", "Time(ms)", "Candles/sec", "NumErr", "Workers", "Segments"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headers, col_w)))
    lines.append("  " + " ".join("─" * w for w in col_w))

    for mode in modes:
        name = MODE_NAMES.get(mode, f"Mode{mode}")
        for candles in datasets:
            for model in models:
                r = idx.get((mode, candles, model))
                if r is None:
                    continue
                backend = getattr(r, "gpu_backend", "") or r.backend or "—"
                num_err = f"{r.numerical_error_rel:.1e}" if r.numerical_error_rel is not None else "—"
                row = [str(mode), name, backend, f"{candles:,}", model,
                       f"{r.exec_ms:.2f}", _fmt_tput(r.throughput_bars_sec), num_err,
                       str(r.cpu_workers_selected or "-"), str(r.segments_selected or "-")]
                flag = " [ERR]" if r.error else ""
                lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_w)) + flag)
        lines.append("  " + " ".join("─" * w for w in col_w))

    blank()

    # ── GPU metrics ──
    hdr("  GPU Metrics")
    blank()
    col_wm = [6, 24, 11, 8, 18, 12, 12, 12, 12, 8, 8]
    headersm = ["Mode", "Engine", "Backend", "Candles", "Model", "Kernel(ms)", "Xfer(ms)", "GPU Total", "CPU Fallback", "Workers", "Segments"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headersm, col_wm)))
    lines.append("  " + " ".join("─" * w for w in col_wm))
    for mode in modes:
        name = MODE_NAMES.get(mode, f"Mode{mode}")
        for candles in datasets:
            for model in models:
                r = idx.get((mode, candles, model))
                if r is None:
                    continue
                backend = getattr(r, "gpu_backend", "") or r.backend or "—"
                if r.gpu_kernel_time_ms > 0 or r.gpu_total_time_ms > 0 or r.gpu_cpu_fallback_time_ms > 0:
                    row = [
                        str(mode), name, backend, f"{candles:,}", model,
                        f"{r.gpu_kernel_time_ms:.1f}" if r.gpu_kernel_time_ms > 0 else "—",
                        f"{r.gpu_transfer_time_ms:.1f}" if r.gpu_transfer_time_ms > 0 else "—",
                        f"{r.gpu_total_time_ms:.1f}" if r.gpu_total_time_ms > 0 else "—",
                        f"{r.gpu_cpu_fallback_time_ms:.1f}" if r.gpu_cpu_fallback_time_ms > 0 else "—",
                        str(r.cpu_workers_selected or "-"), str(r.segments_selected or "-"),
                    ]
                    lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_wm)))
        lines.append("  " + " ".join("─" * w for w in col_wm))

    blank()

    # ── Model comparison ──
    hdr("  Model Throughput Comparison  (best engine, 10k candles)")
    blank()
    for model in models:
        best = max(
            (r for r in rw_results if r.rw_model == model and r.bars == 10_000 and
             not r.error and r.throughput_bars_sec > 0),
            key=lambda r: r.throughput_bars_sec, default=None
        )
        if best:
            hdr(f"    {model:<20} : {_fmt_tput(best.throughput_bars_sec):>12}  "
                f"(Mode {best.mode}: {MODE_NAMES.get(best.mode, '')})")
    blank()

    # ── Speedup vs Python baseline ──
    baseline_mode = 27 if not multi else 32
    hdr(f"  Speedup vs VectorBT Baseline  (Mode {baseline_mode})")
    blank()
    for candles in datasets:
        for model in models:
            base = idx.get((baseline_mode, candles, model))
            if not base or base.exec_ms <= 0:
                continue
            speedups = []
            for mode in modes:
                if mode == baseline_mode:
                    continue
                r = idx.get((mode, candles, model))
                if r and r.exec_ms > 0:
                    sp   = base.exec_ms / r.exec_ms
                    name = MODE_NAMES.get(mode, f"Mode{mode}")
                    speedups.append(f"M{mode}:{sp:.1f}×")
            if speedups:
                hdr(f"    {candles:>7,} / {model:<20}: {' | '.join(speedups)}")
    blank()

    if multi:
        # ── Scaling efficiency (multi vs single) ──
        hdr("  Symbol Parallelization Efficiency  (multi-asset throughput / N × single-asset)")
        blank()
        single_idx = {(r.bars, r.rw_model): r for r in results
                      if r.phase_type == "randomwalk" and r.n_assets == 1 and r.mode == 27}
        for candles in datasets:
            for model in models:
                r_multi  = idx.get((34, candles, model))  # Mode 34 = Leika M1 ×5
                r_single = single_idx.get((candles, model))
                if r_multi and r_single and r_single.throughput_bars_sec > 0 and r_multi.n_assets > 0:
                    expected = r_single.throughput_bars_sec * r_multi.n_assets
                    eff      = r_multi.throughput_bars_sec / expected * 100.0
                    hdr(f"    {candles:>7,} / {model:<20}: {eff:.0f}% efficiency")
        blank()

    sep("─")
    blank()


# ── Dynamic Sectioning section ───────────────────────────────────────────────

def _section_dynamic_sectioning(results: list[BenchResult], lines: list[str], hdr, blank, sep) -> None:
    dyn_results = [r for r in results if getattr(r, "cpu_workers_selected", 0) > 0]
    if not dyn_results:
        return

    hdr("DYNAMIC TILING DIAGNOSTICS")
    blank()
    col_w = [24, 10, 10, 10, 8, 10, 11, 10, 12, 12]
    headers = ["Workload", "Workers", "Segments", "Axis", "Chunk", "CPU%", "RAM GB", "Budget", "Fallback", "RawPaths"]
    lines.append("  " + " ".join(_col(h, w) for h, w in zip(headers, col_w)))
    lines.append("  " + " ".join("─" * w for w in col_w))
    for r in dyn_results:
        workload = r.mode_name
        row = [
            workload,
            str(r.cpu_workers_selected or "-"),
            str(r.segments_selected or "-"),
            str(getattr(r, "split_axis", "") or "-"),
            str(getattr(r, "chunk_size", 0) or "-"),
            f"{r.cpu_pct:.1f}",
            f"{r.ram_peak_gb:.1f}",
            f"{r.ram_budget_gb:.1f}" if r.ram_budget_gb > 0 else "—",
            (getattr(r, "memory_fallback_reason", "") or ("yes" if getattr(r, "memory_fallback", False) else "no")),
            "yes" if getattr(r, "raw_paths_suppressed", False) else ("copied" if getattr(r, "raw_paths_copied", False) else "no"),
        ]
        lines.append("  " + " ".join(_col(v, w) for v, w in zip(row, col_w)))
    lines.append("  " + " ".join("─" * w for w in col_w))
    blank()
    sep("─")
    blank()


# ── Leaderboard section ───────────────────────────────────────────────────────

def _section_leaderboard(results: list[BenchResult], lines: list[str], hdr, blank, sep) -> None:
    hdr("OVERALL LEADERBOARD")
    blank()

    phase_types = [
        ("portfolio",   "Portfolio"),
        ("montecarlo",  "Monte Carlo"),
        ("randomwalk",  "Random Walk"),
    ]
    criteria = [
        ("Fastest Runtime",      lambda r: -r.exec_ms,             lambda r: f"{r.exec_ms:.1f} ms"),
        ("Best Throughput",      lambda r: r.throughput_bars_sec,   lambda r: _fmt_tput(r.throughput_bars_sec)),
        ("Lowest Memory",        lambda r: -r.mem_mb,              lambda r: f"{r.mem_mb:.0f} MB"),
        ("Best GPU Utilization", lambda r: (r.gpu_util_pct or 0),  lambda r: f"{r.gpu_util_pct or 0:.0f}%"),
        ("Best Scaling",         lambda r: (r.speedup_vs_baseline or 0), lambda r: f"{r.speedup_vs_baseline or 0:.2f}×"),
    ]

    for pt, pt_label in phase_types:
        phase_res = [r for r in results if r.phase_type == pt and not r.error]
        if not phase_res:
            continue
        hdr(f"  ── {pt_label} ──")
        blank()
        for crit_name, key_fn, fmt_fn in criteria:
            best = max(phase_res, key=key_fn, default=None)
            if best:
                name = MODE_NAMES.get(best.mode, f"Mode{best.mode}")
                info = f"Mode {best.mode} ({name})"
                if best.rw_model:
                    info += f" [{best.rw_model}]"
                hdr(f"    {crit_name:<25}: {info}  →  {fmt_fn(best)}")
        blank()

    sep("─")
    blank()


# ── Main generator ────────────────────────────────────────────────────────────

def generate_report(results: list[BenchResult],
                    out_dir: str = "benchmark_results",
                    timestamp: Optional[str] = None,
                    timestamp_human: Optional[str] = None,
                    export_ods: bool = False) -> str:
    ts_human, ts_file = _timestamp_parts()
    ts_human = timestamp_human or timestamp or ts_human
    ts_file = timestamp or ts_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched = _find_matching_report(out_dir, results)
    if matched is not None:
        _, matched_ts = matched
        ts_file = matched_ts

    path    = out_dir / f"leika_benchmark_{ts_file}.txt"

    cpu = _cpu_info()
    gpu = _gpu_full()

    lines: list[str] = []
    W = 78

    def sep(c="═"): lines.append(c * W)
    def hdr(t):     lines.append(f"  {t}")
    def blank():    lines.append("")

    def dataset_label(r: BenchResult) -> str:
        if r.phase_type == "montecarlo":
            return f"{r.bars:,} candles × {r.n_paths:,} paths"
        if r.phase_type == "randomwalk":
            label = f"{r.bars:,} candles [{r.rw_model}]"
            if r.n_assets > 1:
                label += f" × {r.n_assets}"
            return label
        return f"{r.bars:,} bars"

    sep()
    hdr("LEIKA ENGINE — BENCHMARK REPORT")
    hdr(f"Generated: {ts_human}")
    hdr(f"Timestamp: {ts_file}")
    sep()
    blank()

    # ── Hardware snapshot ────────────────────────────────────────────────────
    hdr("HARDWARE SNAPSHOT")
    blank()
    hdr(f"CPU  : {cpu.get('name', 'Unknown')}")
    hdr(f"       {cpu.get('threads', 1)} logical threads  |  {cpu.get('cores', cpu.get('threads', 1))} physical cores")
    hdr(f"RAM  : {cpu.get('ram_gb', '?')} GB")
    if gpu:
        hdr(f"GPU  : {gpu.get('name', 'Unknown')}")
        hdr(f"       Util: {gpu.get('util_pct', '?')}%  |  Temp: {gpu.get('temp_c', '?')}°C  |  "
            f"VRAM: {gpu.get('mem_used', '?')}/{gpu.get('mem_total', '?')} MB")
        if gpu.get("power_w"):
            hdr(f"       Power draw: {gpu.get('power_w')} W")
    else:
        hdr("GPU  : Not detected (CPU-only)")
    blank()
    sep("─")
    blank()

    # ── Phase sections ───────────────────────────────────────────────────────
    _section_portfolio(results, lines, hdr, blank, sep)
    _section_mc(results, lines, hdr, blank, sep)
    _section_rw(results, lines, hdr, blank, sep, multi=False)
    _section_rw(results, lines, hdr, blank, sep, multi=True)
    _section_dynamic_sectioning(results, lines, hdr, blank, sep)

    # ── Per-mode detail (all phases) ─────────────────────────────────────────
    sep("─")
    blank()
    hdr("PER-MODE DETAILS")
    blank()

    modes     = sorted({r.mode for r in results})
    bar_sizes = sorted({r.bars for r in results})

    for mode in modes:
        name  = MODE_NAMES.get(mode, f"Mode{mode}")
        phase = PHASE_MAP.get(mode, "")
        mode_results = [r for r in results if r.mode == mode]
        hdr(f"[MODE {mode}] {name}  ({phase})")
        for r in sorted(mode_results, key=lambda x: (x.bars, x.rw_model, x.n_paths)):
            blank()
            label = f"{r.bars:,} candles"
            if r.n_paths > 0 and r.phase_type == "montecarlo":
                label += f" × {r.n_paths:,} paths"
            if r.rw_model:
                label += f"  [{r.rw_model}]"
            hdr(f"  Dataset: {label}")
            hdr(f"    Backend          : {getattr(r, 'gpu_backend', '') or r.backend or '—'}")
            hdr(f"    Exec time          : {r.exec_ms:.3f} ms")
            if r.python_input_time_ms > 0:
                hdr(f"    Python input ms    : {r.python_input_time_ms:.3f}")
            if r.python_to_rust_conversion_ms > 0:
                hdr(f"    Python→Rust ms     : {r.python_to_rust_conversion_ms:.3f}")
            if r.rust_engine_time_ms > 0:
                hdr(f"    Rust engine ms     : {r.rust_engine_time_ms:.3f}")
            if r.stats_calculation_time_ms > 0:
                hdr(f"    Stats ms           : {r.stats_calculation_time_ms:.3f}")
            if r.python_export_time_ms > 0:
                hdr(f"    Python export ms   : {r.python_export_time_ms:.3f}")
            if r.report_generation_time_ms > 0:
                hdr(f"    Report gen ms      : {r.report_generation_time_ms:.3f}")
            hdr(f"    Total time         : {r.total_runtime_ms or r.exec_ms:.3f} ms")
            hdr(f"    Throughput         : {_fmt_tput(r.throughput_bars_sec)}")
            if r.paths_sec > 0:
                hdr(f"    Paths/sec          : {_fmt_tput(r.paths_sec)}")
            hdr(f"    CPU usage          : {r.cpu_pct:.1f}%")
            hdr(f"    RAM usage          : {r.mem_mb:.1f} MB")
            if r.gpu_util_pct is not None:
                hdr(f"    GPU util           : {r.gpu_util_pct:.1f}%")
                hdr(f"    GPU temp           : {r.gpu_temp_c:.0f}°C")
                hdr(f"    GPU VRAM used      : {r.gpu_mem_used_mb:.0f} MB")
            if r.gpu_kernel_time_ms > 0 or r.gpu_total_time_ms > 0 or r.gpu_cpu_fallback_time_ms > 0:
                hdr(f"    GPU kernel ms      : {r.gpu_kernel_time_ms:.1f}")
                hdr(f"    GPU transfer ms    : {r.gpu_transfer_time_ms:.1f}")
                hdr(f"    GPU total ms       : {r.gpu_total_time_ms:.1f}")
                hdr(f"    CPU fallback ms    : {r.gpu_cpu_fallback_time_ms:.1f}")
            if getattr(r, "hybrid_total_time_ms", 0.0) > 0.0:
                hdr(f"    CPU start/end ms   : {getattr(r, 'cpu_start_ms', 0.0):.1f} / {getattr(r, 'cpu_end_ms', 0.0):.1f}")
                hdr(f"    GPU start/end ms   : {getattr(r, 'gpu_start_ms', 0.0):.1f} / {getattr(r, 'gpu_end_ms', 0.0):.1f}")
                hdr(f"    CPU time ms        : {getattr(r, 'cpu_time_ms', 0.0):.1f}")
                hdr(f"    GPU time ms        : {getattr(r, 'gpu_time_ms', 0.0):.1f}")
                hdr(f"    Overlap ms         : {getattr(r, 'overlap_ms', 0.0):.1f}")
                hdr(f"    Overlap pct        : {getattr(r, 'overlap_pct', 0.0):.1f}%")
                hdr(f"    CPU idle wait ms   : {getattr(r, 'cpu_idle_wait_ms', 0.0):.1f}")
                hdr(f"    GPU idle wait ms   : {getattr(r, 'gpu_idle_wait_ms', 0.0):.1f}")
                hdr(f"    Hybrid total ms    : {getattr(r, 'hybrid_total_time_ms', 0.0):.1f}")
            if getattr(r, "dynamic_tiling_enabled", False):
                hdr(f"    Dynamic tiling     : yes")
                hdr(f"    Split axis         : {getattr(r, 'split_axis', '') or 'n/a'}")
                hdr(f"    Chunk size         : {getattr(r, 'chunk_size', 0) or 'n/a'}")
                hdr(f"    Chunk count        : {getattr(r, 'chunk_count', 0) or 'n/a'}")
            elif r.phase_type in ("montecarlo", "randomwalk"):
                hdr("    Dynamic tiling     : no")
            if getattr(r, "memory_fallback", False):
                hdr(f"    Memory fallback    : yes")
                hdr(f"    Fallback reason    : {getattr(r, 'memory_fallback_reason', '') or 'n/a'}")
            elif r.phase_type in ("montecarlo", "randomwalk"):
                hdr("    Memory fallback    : no")
            if r.phase_type in ("montecarlo", "randomwalk"):
                hdr(f"    Return paths       : {'yes' if getattr(r, 'return_paths', False) else 'no'}")
                hdr(f"    Raw copied         : {'yes' if getattr(r, 'raw_paths_copied', False) else 'no'}")
                hdr(f"    Raw suppressed     : {'yes' if getattr(r, 'raw_paths_suppressed', False) else 'no'}")
            if r.numerical_error_rel is not None:
                hdr(f"    Num error rel      : {r.numerical_error_rel:.2e}")
            if r.gpu_fallback_reason:
                hdr(f"    Fallback reason    : {r.gpu_fallback_reason}")
            if r.phase_type == "portfolio":
                hdr(f"    Total return       : {r.total_return_pct:+.4f}%")
                hdr(f"    Sharpe ratio       : {r.sharpe_ratio:.4f}")
                hdr(f"    Max drawdown       : {r.max_drawdown_pct:.4f}%")
                hdr(f"    Win rate           : {r.win_rate_pct:.1f}%")
                hdr(f"    Total trades       : {r.total_trades}")
            if r.ai_enabled:
                ai_total = r.ai_total_time_ms or r.ai_ms_total
                total = r.total_runtime_ms or r.exec_ms
                hdr(f"    AI mode            : {r.ai_mode or 'n/a'}")
                hdr(f"    AI model           : {r.ai_model or 'n/a'}")
                hdr(f"    AI sections        : {r.ai_sections or 0}")
                hdr(f"    AI calls           : {r.ai_calls}")
                hdr(f"    Avg section ms     : {r.ai_avg_section_ms:.1f} ms")
                hdr(f"    Slowest section    : {r.ai_slowest_section or 'n/a'}")
                hdr(f"    Fastest section    : {r.ai_fastest_section or 'n/a'}")
                hdr(f"    AI total ms        : {ai_total:.1f} ms")
                hdr(f"    Total runtime ms   : {total:.1f} ms")
                hdr(f"    Prompt chars       : {r.ai_total_prompt_chars or r.ai_context_chars:,}")
                hdr(f"    Response chars     : {r.ai_total_response_chars:,}")
                hdr(f"    Est. tokens        : {r.ai_estimated_tokens:,}")
                hdr(f"    Tokens/sec         : {r.ai_tokens_per_second:.1f}")
                hdr(f"    Parallelism        : {r.ai_section_parallelism or 'n/a'}")
                hdr(f"    Dynamic sections   : {'yes' if r.ai_dynamic_sectioning_enabled else 'no'}")
                hdr(f"    Timeout            : {'yes' if r.ai_timeout else 'no'}")
                hdr(f"    Fallback           : {'yes' if r.ai_fallback else 'no'}")
                ai_pct = (ai_total / total * 100) if total > 0 else 0
                hdr(f"    AI time share      : {ai_pct:.1f}%")
            hdr(f"    Cash model         : {r.cash_model or '—'}")
            hdr(f"    Execution core     : {r.execution_core or '—'}")
            hdr(f"    Shared sectioning  : {'yes' if r.shared_data_sectioning else 'no'}")
            hdr(f"    Split axis         : {r.split_axis or '—'}")
            hdr(f"    Dynamic sectioning : {'yes' if r.dynamic_sectioning_used else 'no'}")
            hdr(f"    Sell at end scope  : {r.sell_at_end_scope or '—'}")
            if r.speedup_vs_baseline:
                hdr(f"    Speedup vs baseline: {r.speedup_vs_baseline:.2f}×")
            if r.error:
                hdr(f"    !! ERROR           : {r.error}")
        blank()

    # ── Final analysis ───────────────────────────────────────────────────────
    sep("─")
    blank()
    hdr("FINAL ANALYSIS")
    blank()

    bar_sizes_unique = sorted({r.bars for r in results})
    for bars in bar_sizes_unique:
        best = min(
            (r for r in results if r.bars == bars and not r.error and r.exec_ms > 0),
            key=lambda r: r.exec_ms, default=None
        )
        if best:
            name = MODE_NAMES.get(best.mode, best.mode_name)
            hdr(f"  Fastest at {dataset_label(best)}: MODE {best.mode} ({name}) — {best.exec_ms:.2f} ms")
    blank()

    # Rust vs VectorBT for portfolio
    pf_idx = {(r.mode, r.bars): r for r in results if r.phase_type == "portfolio"}
    for bars in sorted({r.bars for r in results if r.phase_type == "portfolio"}):
        r1 = pf_idx.get((1, bars))
        r2 = pf_idx.get((2, bars))
        if r1 and r2 and r1.exec_ms > 0 and r2.exec_ms > 0:
            ratio = r1.exec_ms / r2.exec_ms
            hdr(f"  Portfolio Rust vs VBT ({bars:,} bars): {abs(ratio):.2f}× "
                f"{'faster' if ratio > 1 else 'slower'} for Rust")
    blank()

    # MC VectorBT baseline vs Leika
    mc_idx = {}
    for r in results:
        if r.phase_type == "montecarlo":
            mc_idx[(r.mode, r.bars, r.n_paths)] = r
    for r in results:
        if r.phase_type == "montecarlo" and r.mode in (23, 24, 25) and r.exec_ms > 0:
            r_vbt = mc_idx.get((22, r.bars, r.n_paths))
            if r_vbt and r_vbt.exec_ms > 0:
                sp = r_vbt.exec_ms / r.exec_ms
                hdr(f"  MC VectorBT vs Leika ({r.bars:,}×{r.n_paths:,}): {sp:.1f}× faster for Leika")
    blank()

    # Best Sharpe (portfolio only)
    pf_non_err = [r for r in results if r.phase_type == "portfolio" and not r.error]
    if pf_non_err:
        best = max(pf_non_err, key=lambda r: r.sharpe_ratio)
        hdr(f"  Best Sharpe: MODE {best.mode} ({best.bars:,} bars) — {best.sharpe_ratio:.4f}")
    blank()

    # Scalability
    hdr("  Scalability (throughput by bar count):")
    for mode in sorted({r.mode for r in results}):
        runs = sorted([r for r in results if r.mode == mode and not r.error
                       and r.throughput_bars_sec > 0], key=lambda r: r.bars)
        if len(runs) >= 2:
            ratio = runs[-1].throughput_bars_sec / runs[0].throughput_bars_sec
            trend = "scales well" if ratio > 0.7 else "degrades"
            hdr(f"    MODE {mode:>2} ({MODE_NAMES.get(mode, f'Mode{mode}')}): "
                f"{_fmt_tput(runs[0].throughput_bars_sec):>12} → "
                f"{_fmt_tput(runs[-1].throughput_bars_sec):>12}  [{trend}]")
    blank()

    ai_results = [r for r in results if r.ai_enabled]
    if ai_results:
        hdr("  AI Summary")
        blank()
        fastest = min((r for r in ai_results if (r.ai_total_time_ms or r.ai_ms_total) > 0), key=lambda r: r.ai_total_time_ms or r.ai_ms_total, default=None)
        slowest = max(ai_results, key=lambda r: r.ai_total_time_ms or r.ai_ms_total, default=None)
        largest = max(ai_results, key=lambda r: r.ai_context_chars, default=None)
        if fastest:
            hdr(f"    Fastest AI mode  : MODE {fastest.mode} ({MODE_NAMES.get(fastest.mode, fastest.mode_name)}) — {(fastest.ai_total_time_ms or fastest.ai_ms_total):.1f} ms")
        if slowest:
            hdr(f"    Slowest AI mode  : MODE {slowest.mode} ({MODE_NAMES.get(slowest.mode, slowest.mode_name)}) — {(slowest.ai_total_time_ms or slowest.ai_ms_total):.1f} ms")
        if largest:
            hdr(f"    Largest context   : MODE {largest.mode} — {largest.ai_context_chars:,} chars")
        model_map = {}
        for r in ai_results:
            model_map.setdefault(r.ai_model or "unknown", []).append(r)
        for model_name, items in model_map.items():
            best = min(items, key=lambda r: r.ai_total_time_ms or r.ai_ms_total)
            avg_ctx = sum(r.ai_context_chars for r in items) / max(len(items), 1)
            hdr(f"    Model {model_name:<20}: best={(best.ai_total_time_ms or best.ai_ms_total):.1f} ms, avg ctx={avg_ctx:,.0f} chars")
        timeout_count = sum(1 for r in ai_results if r.ai_timeout)
        fallback_count = sum(1 for r in ai_results if r.ai_fallback)
        baseline_map = {(r.mode, r.bars, r.n_assets, r.rw_model): r for r in results if not r.ai_enabled}
        extras = []
        for r in ai_results:
            base_key = (r.mode - 10, r.bars, r.n_assets, r.rw_model) if 11 <= r.mode <= 20 else None
            base = baseline_map.get(base_key) if base_key else None
            if base and base.exec_ms > 0:
                extras.append((r.total_runtime_ms or r.exec_ms) - base.exec_ms)
        if extras:
            hdr(f"    AI extra cost    : {sum(extras) / len(extras):.1f} ms avg")
        hdr(f"    Timeout count    : {timeout_count}")
        hdr(f"    Fallback count   : {fallback_count}")
        blank()

    # ── Leaderboard ──────────────────────────────────────────────────────────
    sep("─")
    blank()
    _section_leaderboard(results, lines, hdr, blank, sep)

    sep()
    hdr("END OF REPORT")
    sep()

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")

    # ── JSON output ──────────────────────────────────────────────────────────
    json_path = out_dir / f"leika_benchmark_{ts_file}.json"
    raw = []
    for r in results:
        raw.append({
            "mode":                 r.mode,
            "mode_name":            r.mode_name,
            "phase":                r.phase,
            "phase_type":           r.phase_type,
            "bars":                 r.bars,
            "n_paths":              r.n_paths,
            "n_assets":             r.n_assets,
            "rw_model":             r.rw_model,
            "backend":              r.backend,
            "leika_exec_mode":      r.leika_exec_mode,
            "ai_enabled":           r.ai_enabled,
            "python_input_time_ms": r.python_input_time_ms,
            "python_to_rust_conversion_ms": r.python_to_rust_conversion_ms,
            "engine_time_ms":       r.engine_time_ms,
            "rust_engine_time_ms":  r.rust_engine_time_ms,
            "stats_calculation_time_ms": r.stats_calculation_time_ms,
            "python_export_time_ms": r.python_export_time_ms,
            "report_generation_time_ms": r.report_generation_time_ms,
            "exec_ms":              r.exec_ms,
            "backtest_ms":          r.backtest_ms,
            "context_build_ms":     r.context_build_ms,
            "total_runtime_ms":     r.total_runtime_ms,
            "mem_mb":               r.mem_mb,
            "cpu_pct":              r.cpu_pct,
            "throughput_bars_sec":  r.throughput_bars_sec,
            "paths_sec":            r.paths_sec,
            "dynamic_tiling_enabled": getattr(r, "dynamic_tiling_enabled", False),
            "split_axis":          getattr(r, "split_axis", ""),
            "chunk_count":         getattr(r, "chunk_count", 0),
            "chunk_size":          getattr(r, "chunk_size", 0),
            "memory_fallback":     getattr(r, "memory_fallback", False),
            "memory_fallback_reason": getattr(r, "memory_fallback_reason", ""),
            "raw_paths_copied":    getattr(r, "raw_paths_copied", False),
            "raw_paths_suppressed": getattr(r, "raw_paths_suppressed", False),
            "return_paths":        getattr(r, "return_paths", False),
            "cpu_work_share":      getattr(r, "cpu_work_share", 0.0),
            "gpu_work_share":      getattr(r, "gpu_work_share", 0.0),
            "gpu_util_pct":         r.gpu_util_pct,
            "gpu_temp_c":           r.gpu_temp_c,
            "gpu_mem_used_mb":      r.gpu_mem_used_mb,
            "gpu_accel_factor":     r.gpu_accel_factor,
            "scaling_efficiency":   r.scaling_efficiency,
            "speedup_vs_baseline":  r.speedup_vs_baseline,
            "total_return_pct":     r.total_return_pct,
            "sharpe_ratio":         r.sharpe_ratio,
            "max_drawdown_pct":     r.max_drawdown_pct,
            "win_rate_pct":         r.win_rate_pct,
            "total_trades":         r.total_trades,
            "avg_trade_return_pct": r.avg_trade_return_pct,
            "best_trade_pct":       r.best_trade_pct,
            "worst_trade_pct":      r.worst_trade_pct,
            "median_trade_return_pct": r.median_trade_return_pct,
            "longest_trade_bars":   r.longest_trade_bars,
            "shortest_trade_bars":  r.shortest_trade_bars,
            "roi_pct":              r.roi_pct,
            "sortino_ratio":        r.sortino_ratio,
            "calmar_ratio":         r.calmar_ratio,
            "profit_factor":        r.profit_factor,
            "portfolio_heat_avg_pct": r.portfolio_heat_avg_pct,
            "portfolio_heat_max_pct": r.portfolio_heat_max_pct,
            "ai_calls":             r.ai_calls,
            "ai_ms_total":          r.ai_ms_total,
            "ai_mode":              r.ai_mode,
            "ai_model":             r.ai_model,
            "ai_context_chars":     r.ai_context_chars,
            "ai_estimated_tokens":  r.ai_estimated_tokens,
            "ai_sections":          r.ai_sections,
            "ai_avg_section_ms":    r.ai_avg_section_ms,
            "ai_slowest_section":   r.ai_slowest_section,
            "ai_fastest_section":   r.ai_fastest_section,
            "ai_total_prompt_chars": r.ai_total_prompt_chars,
            "ai_total_response_chars": r.ai_total_response_chars,
            "ai_timeout_count":     r.ai_timeout_count,
            "ai_fallback_count":    r.ai_fallback_count,
            "ai_section_parallelism": r.ai_section_parallelism,
            "ai_dynamic_sectioning_enabled": r.ai_dynamic_sectioning_enabled,
            "ai_section_results":   r.ai_section_results,
            "ai_prompt_eval_duration_ms": r.ai_prompt_eval_duration_ms,
            "ai_eval_duration_ms":  r.ai_eval_duration_ms,
            "ai_total_time_ms":     r.ai_total_time_ms,
            "ai_tokens_per_second": r.ai_tokens_per_second,
            "ai_gpu_used":          r.ai_gpu_used,
            "ai_vram_used_mb":      r.ai_vram_used_mb,
            "ai_timeout":           r.ai_timeout,
            "ai_fallback":          r.ai_fallback,
            "ai_skipped_reason":    r.ai_skipped_reason,
            "cash_model":             r.cash_model,
            "shared_data_sectioning": r.shared_data_sectioning,
            "split_axis":             r.split_axis,
            "execution_core":         r.execution_core,
            "dynamic_sectioning_used": r.dynamic_sectioning_used,
            "sell_at_end_scope":              r.sell_at_end_scope,
            "dynamic_sectioning_preparation": r.dynamic_sectioning_preparation,
            "dynamic_sectioning_execution":   r.dynamic_sectioning_execution,
            "dynamic_sectioning_post_analysis": r.dynamic_sectioning_post_analysis,
            "error":                r.error,
        })
    json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    # ── Markdown output ──────────────────────────────────────────────────────
    md_path = out_dir / f"benchmark_summary_{ts_file}.md"
    md_content = generate_markdown_report(
        results,
        {"cpu": _cpu_info(), "gpu": _gpu_full()},
        ts_human,
        ts_file,
    )
    md_path.write_text(md_content, encoding="utf-8")

    # ── ODS output ───────────────────────────────────────────────────────────
    if export_ods:
        try:
            import ods_export
            ods_path = ods_export.generate_ods(results, out_dir=str(out_dir), ts_file=ts_file)
            print(f"\nLibreOffice spreadsheet written to:\n  {ods_path}")
        except Exception as _ods_exc:
            import sys as _sys
            print(f"[warn] ODS export failed: {_ods_exc}", file=_sys.stderr)

    return str(path)
