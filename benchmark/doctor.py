"""
leika doctor — Full system diagnostics and readiness check.

Checks:
  1. Hardware profile (CPU / RAM / GPU)
  2. Rust engine binary
  3. PyO3 Python bindings
  4. Rayon thread pool config
  5. Thread environment (OpenBLAS / MKL oversubscription)
  6. GPU + Docker GPU access
  7. Python dependencies
  8. Scale readiness (1k / 10k / 100k / 1M+)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
import hw_detect

BOLD   = "\033[1m"
GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
RESET  = "\033[0m"

_pass_n = 0
_fail_n = 0
_warn_n = 0


def _ok(label: str, detail: str = "") -> None:
    global _pass_n
    _pass_n += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  {GREEN}✓{RESET}  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    global _fail_n
    _fail_n += 1
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {RED}✗{RESET}  {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    global _warn_n
    _warn_n += 1
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {YELLOW}⚠{RESET}  {label}{suffix}")


def _section(title: str) -> None:
    pad = max(0, 52 - len(title))
    print()
    print(f"{BOLD}{CYAN}── {title} {'─' * pad}{RESET}")
    print()


# ── 1. Hardware ───────────────────────────────────────────────────────────────

def check_hardware() -> hw_detect.HardwareSnapshot:
    _section("Hardware Profile")
    hw = hw_detect.detect()
    hw_detect.print_hardware(hw)

    if hw.score >= 60:
        _ok(f"Hardware score: {hw.score}/100", "large-scale runs enabled")
    elif hw.score >= 30:
        _warn(f"Hardware score: {hw.score}/100", "medium-scale OK; large runs may be slow")
    else:
        _warn(f"Hardware score: {hw.score}/100", "only small-scale runs recommended")

    return hw


# ── 2. Rust binary ────────────────────────────────────────────────────────────

def check_rust_binary() -> None:
    _section("Rust Engine Binary")

    # cargo
    try:
        r = subprocess.run(["cargo", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok("cargo", r.stdout.strip())
        else:
            _fail("cargo", "found but failed")
    except FileNotFoundError:
        _fail("cargo", "not installed — install Rust from https://rustup.rs")
        return
    except subprocess.TimeoutExpired:
        _fail("cargo", "timeout")
        return

    # leika_engine binary
    try:
        r = subprocess.run(["leika_engine", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            # Extract first non-empty line for display (old binaries print full report)
            first_line = next((l.strip() for l in r.stdout.splitlines() if l.strip()), "ok")
            display = first_line[:60] if len(first_line) > 60 else first_line
            _ok("leika_engine binary", display)
        else:
            _warn("leika_engine binary", "in PATH but --version not recognized")
    except FileNotFoundError:
        _warn("leika_engine binary", "not in PATH — run: cargo build --release")

    # cargo check (fast compilation check)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"    Running cargo check (project: {proj_root}) ...")
    t0 = time.monotonic()
    check_env = {**os.environ, "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"}
    try:
        r = subprocess.run(
            ["cargo", "check", "--quiet"],
            capture_output=True, text=True, timeout=90,
            cwd=proj_root, env=check_env,
        )
        elapsed = (time.monotonic() - t0) * 1000
        if r.returncode == 0:
            _ok(f"cargo check — no compilation errors ({elapsed:.0f}ms)")
        else:
            snippet = r.stderr[:300].strip() if r.stderr else "no stderr"
            _fail("cargo check", snippet)
    except FileNotFoundError:
        _warn("cargo check", "cargo not available")
    except subprocess.TimeoutExpired:
        _warn("cargo check", "timeout — run manually: cargo check")


# ── 3. PyO3 bindings ─────────────────────────────────────────────────────────

def check_pyo3() -> bool:
    _section("PyO3 Python Bindings")

    try:
        import leika
        _ok("import leika", f"v{leika.__version__}")
    except ImportError as exc:
        _fail("import leika", str(exc))
        _warn("Fix", "run: maturin develop --release")
        return False

    try:
        engine = leika.Engine()
        hw = engine.hardware
        assert hw.logical_threads >= 1 and hw.ram_gb > 0
        _ok("Engine() + HardwareProfile", f"{hw.logical_threads} threads, {hw.ram_gb:.1f} GB RAM")
    except Exception as exc:
        _fail("Engine() instantiation", str(exc))
        return False

    try:
        plan = engine.resource_plan()
        assert plan.workers >= 1
        _ok("Engine().resource_plan()", f"{plan.workers} workers, {plan.segments} segments")
    except Exception as exc:
        _fail("resource_plan()", str(exc))

    try:
        prices = [float(i + 1) for i in range(100)]
        ema20 = leika.ema(prices, 20)
        rsi14 = leika.rsi(prices, 14)
        assert len(ema20) == 100 and len(rsi14) == 100
        _ok("leika.ema() + leika.rsi() — 100 bars")
    except Exception as exc:
        _fail("indicator functions", str(exc))

    try:
        mc = leika.MonteCarlo(n_paths=10, n_steps=10, seed=42)
        assert mc is not None
        _ok("MonteCarlo() instantiation")
    except Exception as exc:
        _fail("MonteCarlo()", str(exc))

    return True


# ── 3b. Execution mode smoke tests ───────────────────────────────────────────

def check_exec_modes() -> None:
    _section("Execution Mode Smoke Tests (Mode 0 / 1 / 2)")

    try:
        import leika
        from strategy import generate_prices, ema_py, rsi_py, make_signals
        from metrics  import backtest_simple
    except ImportError as exc:
        _warn("Mode smoke tests skipped", str(exc))
        return

    BARS = 500

    def _run_mode(mode: int) -> tuple[bool, float, str]:
        import time
        try:
            engine = leika.Engine(mode=mode)
            prices = generate_prices(BARS, seed=42)
            try:
                ef = leika.ema(prices, 20)
                es = leika.ema(prices, 50)
                rs = leika.rsi(prices, 14)
            except Exception:
                ef = ema_py(prices, 20)
                es = ema_py(prices, 50)
                rs = rsi_py(prices, 14)
            sigs = make_signals(prices, ef, es, rs)
            t0 = time.monotonic()
            eq, trades = backtest_simple(prices, sigs)
            ms = (time.monotonic() - t0) * 1000
            assert len(eq) > 0
            plan = engine.resource_plan()
            return True, ms, f"{plan.workers} workers, {plan.strategy}"
        except Exception as exc:
            return False, 0.0, str(exc)

    for mode, label in [(0, "CPU-only"), (1, "Hybrid (CPU+GPU)"), (2, "HPC stress")]:
        ok, ms, detail = _run_mode(mode)
        tag = f"Mode {mode} — {label}"
        if ok:
            _ok(tag, f"{ms:.1f}ms  │  {detail}")
        else:
            _fail(tag, detail)


# ── 4. Rayon thread pool ──────────────────────────────────────────────────────

def check_rayon(hw: hw_detect.HardwareSnapshot) -> None:
    _section("Rayon Thread Pool")

    expected = max(1, hw.cpu.logical_threads - 2)

    try:
        import leika
        plan = leika.Engine().resource_plan()
        workers = plan.workers

        if workers == expected:
            _ok(f"Thread pool: {workers} workers", "all logical threads - 2 for OS/AI")
        elif workers >= expected - 2:
            _warn(f"Thread pool: {workers} workers", f"expected ~{expected} — slight underuse")
        else:
            _fail(
                f"Thread pool: {workers} workers",
                f"expected ~{expected} — possible artificial cap (check RAYON_NUM_THREADS)",
            )
    except Exception as exc:
        _warn("Could not verify thread pool via PyO3", str(exc))

    # Check env override
    override = os.environ.get("RAYON_NUM_THREADS", "")
    if override:
        try:
            val = int(override)
            if val < expected:
                _warn(f"RAYON_NUM_THREADS={val}", f"lower than optimal ({expected})")
            else:
                _ok(f"RAYON_NUM_THREADS={val}")
        except ValueError:
            _warn(f"RAYON_NUM_THREADS='{override}'", "invalid integer")
    else:
        _ok("RAYON_NUM_THREADS not set", "engine auto-detects thread count")


# ── 5. Thread environment ─────────────────────────────────────────────────────

def check_thread_env() -> None:
    _section("Thread Environment (Anti-oversubscription)")

    vars_expected = {
        "OMP_NUM_THREADS":      "1",
        "MKL_NUM_THREADS":      "1",
        "OPENBLAS_NUM_THREADS": "1",
        "BLIS_NUM_THREADS":     "1",
        "NUMEXPR_NUM_THREADS":  "1",
    }

    all_set = True
    for var, ideal in vars_expected.items():
        val = os.environ.get(var, "")
        if not val:
            _warn(f"{var} not set", "set to 1 to prevent OpenBLAS/MKL fighting Rayon")
            all_set = False
        elif val == ideal:
            _ok(f"{var}=1")
        else:
            _warn(f"{var}={val}", "non-1 may cause thread contention with Rayon")
            all_set = False

    if not all_set:
        print()
        print(f"  {YELLOW}Tip: Add to your shell profile or run before benchmark:{RESET}")
        print(f"  {YELLOW}  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1{RESET}")
        print(f"  {YELLOW}         BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1{RESET}")


# ── 6. GPU ────────────────────────────────────────────────────────────────────

def check_gpu(hw: hw_detect.HardwareSnapshot) -> None:
    _section("GPU & Acceleration")

    if hw.gpu is None:
        _warn("No GPU detected", "CPU-only mode — GPU score: 0")
        return

    _ok(f"GPU: {hw.gpu.name}")
    _ok(f"VRAM: {hw.gpu.vram_gb:.0f} GB")
    _ok(f"Backend: {hw.gpu.backend}")
    if hw.gpu.driver:
        _ok(f"Driver: {hw.gpu.driver}")

    if hw.gpu.backend == "CUDA":
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                util, mem_used, temp = parts[0], parts[1], parts[2]
                _ok(f"nvidia-smi live: util={util}%  mem={mem_used}MiB  temp={temp}°C")
            else:
                _fail("nvidia-smi", "command failed")
        except FileNotFoundError:
            _fail("nvidia-smi", "not found — CUDA driver may not be installed")

    # Docker GPU passthrough (optional — skip gracefully if docker absent)
    try:
        r = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            raise FileNotFoundError
        _ok("Docker available", r.stdout.strip())

        if hw.gpu.backend == "CUDA":
            print("    Checking Docker GPU passthrough (nvidia-container-toolkit)...")
            r2 = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all",
                 "ubuntu:22.04", "ls", "/dev/nvidia0"],
                capture_output=True, text=True, timeout=20,
            )
            if r2.returncode == 0:
                _ok("Docker GPU passthrough: /dev/nvidia0 accessible")
            else:
                _warn("Docker GPU passthrough", "GPU not accessible inside Docker — "
                      "install nvidia-container-toolkit")
    except FileNotFoundError:
        _warn("Docker", "not found — skip if not using Docker deployment")
    except subprocess.TimeoutExpired:
        _warn("Docker GPU check", "timeout")


# ── 7. Python dependencies ────────────────────────────────────────────────────

def check_python_deps() -> None:
    _section("Python Dependencies")

    required = [
        ("polars",   "polars"),
        ("pyarrow",  "pyarrow"),
        ("numpy",    "numpy"),
        ("psutil",   "psutil"),
    ]
    optional = [
        ("pandas",     "pandas"),
        ("matplotlib", "matplotlib"),
        ("rich",       "rich"),
        ("vectorbt",   "vectorbt"),
        ("ollama",     "ollama"),
    ]

    for label, mod in required:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            _ok(f"{label}  v{ver}")
        except ImportError:
            _fail(f"{label}", "not installed (required)")

    for label, mod in optional:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            _ok(f"{label}  v{ver}  (optional)")
        except ImportError:
            _warn(f"{label}", "not installed — some modes unavailable")


# ── 8. Scale readiness ────────────────────────────────────────────────────────

def check_scale_readiness(hw: hw_detect.HardwareSnapshot) -> None:
    _section("Scale Readiness")

    tiers: list[tuple[str, bool, str]] = [
        ("1k bars",    True,
         "always"),
        ("10k bars",   hw.cpu.logical_threads >= 4,
         f"need ≥4 threads, have {hw.cpu.logical_threads}"),
        ("100k bars",  hw.cpu.logical_threads >= 8 and hw.mem.total_gb >= 8.0,
         f"need ≥8 threads + 8 GB RAM"),
        ("1M+ bars",   hw.cpu.logical_threads >= 16 and hw.mem.total_gb >= 32.0,
         f"need ≥16 threads + 32 GB RAM"),
    ]

    for tier, ok, reason in tiers:
        if ok:
            _ok(tier, reason)
        else:
            _warn(tier, f"under-spec — {reason}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(hw: hw_detect.HardwareSnapshot) -> None:
    print()
    print("═" * 62)
    print()
    print(f"  {BOLD}LEIKA DOCTOR — SUMMARY{RESET}")
    print()
    print(f"  {GREEN}Passed:   {_pass_n}{RESET}")
    print(f"  {YELLOW}Warnings: {_warn_n}{RESET}")
    print(f"  {RED}Failed:   {_fail_n}{RESET}")
    print()

    # Hardware tier label
    t = hw.cpu.logical_threads
    r = hw.mem.total_gb
    if t >= 64 and r >= 128:
        tier_color, tier_label = GREEN,  "LARGE SCALE ENABLED"
    elif t >= 16 and r >= 32:
        tier_color, tier_label = YELLOW, "MEDIUM SCALE ENABLED"
    elif t >= 8 and r >= 8:
        tier_color, tier_label = YELLOW, "SMALL-MEDIUM SCALE"
    else:
        tier_color, tier_label = RED,    "LIMITED HARDWARE"

    print(f"  Mode: {BOLD}{tier_color}{tier_label}{RESET}")
    print()

    if _fail_n == 0:
        print(f"  {BOLD}{GREEN}✓ System ready for Leika Engine.{RESET}")
        print(f"    Run: ./scripts/leika run --test")
    else:
        print(f"  {BOLD}{RED}✗ {_fail_n} critical issue(s). Fix before running.{RESET}")

    print()
    print("═" * 62)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║           LEIKA DOCTOR — SYSTEM DIAGNOSTICS         ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}")

    hw = check_hardware()
    check_rust_binary()
    check_pyo3()
    check_exec_modes()
    check_rayon(hw)
    check_thread_env()
    check_gpu(hw)
    check_python_deps()
    check_scale_readiness(hw)
    print_summary(hw)


if __name__ == "__main__":
    main()
