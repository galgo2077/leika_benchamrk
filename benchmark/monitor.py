"""Live system resource monitor for Leika benchmark runs.

Usage:
    mon = LiveMonitor(total_bars=100_000, mode=2)
    mon.start()
    result = run_something()
    mon.stop()
    stats = mon.summary()
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

from metrics import cpu_mem_snapshot, gpu_snapshot

BOLD   = "\033[1m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
RESET  = "\033[0m"

_CLEAR_LINE = "\033[2K\033[1G"


class LiveMonitor:
    """Background thread that samples system stats every `interval` seconds.

    Prints a live 2-line status block to stdout during the run.
    Clears the block cleanly when stop() is called.
    """

    def __init__(
        self,
        total_bars: int = 0,
        mode: int = 0,
        interval: float = 1.0,
        label: str = "",
    ) -> None:
        self.total_bars   = total_bars
        self.mode         = mode
        self.interval     = interval
        self.label        = label or f"Mode {mode}"
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)
        self._lines       = 0
        self._start_time  = 0.0

        # Peak stats (filled during run)
        self.peak_cpu_pct:  float          = 0.0
        self.peak_mem_mb:   float          = 0.0
        self.peak_gpu_pct:  Optional[float] = None
        self.peak_gpu_temp: Optional[float] = None
        self._samples = 0

        # Logical thread count for display
        self._n_threads = os.cpu_count() or 1

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._start_time = time.monotonic()
        print()                     # blank line before live block
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        # Erase the live block
        if self._lines > 0:
            for _ in range(self._lines):
                sys.stdout.write("\033[F" + _CLEAR_LINE)
            sys.stdout.write("\033[F" + _CLEAR_LINE)  # erase the blank line we added
            sys.stdout.flush()
        self._lines = 0

    # ── sampling ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_and_print()

    def _sample_and_print(self) -> None:
        elapsed = time.monotonic() - self._start_time
        cpu_pct, mem_mb = cpu_mem_snapshot()
        gpu_util, gpu_temp, _gpu_mem = gpu_snapshot()

        self._samples += 1
        self.peak_cpu_pct = max(self.peak_cpu_pct, cpu_pct)
        self.peak_mem_mb  = max(self.peak_mem_mb,  mem_mb)
        if gpu_util is not None:
            self.peak_gpu_pct = max(self.peak_gpu_pct or 0.0, gpu_util)
        if gpu_temp is not None:
            self.peak_gpu_temp = gpu_temp

        mem_gb  = mem_mb / 1024.0
        gpu_str = f"GPU: {gpu_util:.0f}%  │  " if gpu_util is not None else ""
        cpu_bar = _mini_bar(cpu_pct)
        gpu_bar = _mini_bar(gpu_util) if gpu_util is not None else ""

        lines = [
            f"  {BOLD}{CYAN}[LEIKA RUNNING]{RESET}  {self.label}",
            f"  CPU: {cpu_pct:5.1f}% {cpu_bar}  │  RAM: {mem_gb:.2f} GB  │  "
            f"{gpu_str}Threads: {self._n_threads} active",
            f"  Elapsed: {elapsed:.1f}s",
        ]

        # Erase previous live block
        if self._lines > 0:
            for _ in range(self._lines):
                sys.stdout.write("\033[F" + _CLEAR_LINE)
            sys.stdout.flush()

        for line in lines:
            print(line)
        sys.stdout.flush()

        self._lines = len(lines)

    # ── result ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "peak_cpu_pct":  self.peak_cpu_pct,
            "peak_mem_mb":   self.peak_mem_mb,
            "peak_gpu_pct":  self.peak_gpu_pct,
            "peak_gpu_temp": self.peak_gpu_temp,
            "samples":       self._samples,
        }


def _mini_bar(pct: Optional[float], width: int = 8) -> str:
    if pct is None:
        return " " * width
    filled = int(width * min(pct, 100.0) / 100.0)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def detect_bottleneck(
    results: list,
    hw_threads: int,
    hw_ram_gb: float,
    peak_cpu_pct: float = 0.0,
    peak_mem_mb: float = 0.0,
    peak_gpu_pct: Optional[float] = None,
) -> str:
    """Classify the primary performance bottleneck from benchmark results."""
    if not results:
        return "unknown"

    ram_used_gb = peak_mem_mb / 1024.0
    ram_ratio   = ram_used_gb / max(hw_ram_gb, 1.0)

    if peak_gpu_pct is not None and peak_gpu_pct > 70.0:
        return "GPU-accelerated"
    if peak_cpu_pct > 75.0:
        return "CPU-bound"
    if ram_ratio > 0.65:
        return "memory-bound"
    # Fallback: single-threaded Python overhead
    return "CPU-bound"


def print_bottleneck_report(
    results: list,
    hw_threads: int,
    hw_ram_gb: float,
    peak_cpu_pct: float = 0.0,
    peak_mem_mb: float  = 0.0,
    peak_gpu_pct: Optional[float] = None,
) -> None:
    """Print bottleneck analysis and recommendations."""
    bottleneck = detect_bottleneck(
        results, hw_threads, hw_ram_gb,
        peak_cpu_pct, peak_mem_mb, peak_gpu_pct
    )

    sep = "═" * 62

    print()
    print(sep)
    print()
    print(f"  {BOLD}SYSTEM HEALTH REPORT{RESET}")
    print()

    # Stats
    sampled = peak_cpu_pct > 0.0 or peak_mem_mb > 0.0
    no_sample_note = "  (runs completed faster than monitor interval)" if not sampled else ""
    print(f"  Peak CPU:     {peak_cpu_pct:.1f}%{no_sample_note}")
    print(f"  Peak RAM:     {peak_mem_mb / 1024:.2f} GB  "
          f"({peak_mem_mb / 1024 / max(hw_ram_gb, 1) * 100:.0f}% of {hw_ram_gb:.0f} GB)")
    if peak_gpu_pct is not None:
        print(f"  Peak GPU:     {peak_gpu_pct:.1f}%")
    else:
        print(f"  GPU:          not detected / not used")
    print(f"  CPU threads:  {hw_threads}")
    print()

    # Benchmark results summary
    ok_runs  = [r for r in results if not getattr(r, "error", "")]
    err_runs = [r for r in results if getattr(r, "error", "")]
    print(f"  Runs:         {len(ok_runs)} passed / {len(err_runs)} failed")

    if ok_runs:
        avg_ms  = sum(r.exec_ms for r in ok_runs) / len(ok_runs)
        max_thr = max((r.throughput_bars_sec for r in ok_runs), default=0.0)
        print(f"  Avg exec:     {avg_ms:,.1f} ms")
        print(f"  Peak thruput: {max_thr:,.0f} bars/sec")

    print()

    # Bottleneck classification
    bottleneck_map = {
        "CPU-bound":       (YELLOW, "CPU-bound", "Add more threads or use Mode 2 (full saturation)."),
        "memory-bound":    (YELLOW, "Memory-bound", "Reduce batch size or upgrade RAM."),
        "GPU-accelerated": (GREEN,  "GPU-accelerated", "GPU fully utilized — pipeline healthy."),
        "unknown":         (CYAN,   "Unknown", "Run a larger benchmark for clearer signal."),
    }
    color, label, advice = bottleneck_map.get(bottleneck, (CYAN, bottleneck, ""))

    print(f"  Bottleneck:   {color}{BOLD}{label}{RESET}")
    print(f"  Recommendation: {advice}")
    print()

    # Scale assessment
    if len(ok_runs) == len(results) and not err_runs:
        print(f"  {GREEN}{BOLD}✓ Benchmark PASSED — Leika Engine fully operational.{RESET}")
    else:
        print(f"  {YELLOW}⚠  {len(err_runs)} run(s) failed. Check output above.{RESET}")

    print()
    print(sep)
    print()
