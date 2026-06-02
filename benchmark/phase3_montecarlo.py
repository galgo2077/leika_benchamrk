"""Phase 3 — Monte Carlo simulation (modes 25–30).

Dataset matrix:
  n_candles : 1_000 | 10_000
  n_paths   : 100 | 500 | 10_000 | 100_000

Engines per dataset (6):
  25 — Python Baseline    (pure Python GBM paths)
  26 — VectorBT Baseline  (leika.MonteCarlo, default Rayon pool)
  27 — Leika Mode 0       (CpuOnly — serial, one path at a time)
  28 — Leika Mode 1       (Adaptive — full Rayon parallel)
  29 — Leika Mode 2       (GpuAccelerated — GPU or Rayon fallback)
  30 — Leika Mode 3       (HybridCpuGpu — Rayon CPU + GPU concurrently)

The caller iterates over (n_candles, n_paths) pairs; this module routes to the
correct engine based on mode_num.
"""
from runner import run_mc_baseline, run_mc_leika


def run(mode_num: int, n_candles: int, n_paths: int, **_) -> object:
    phase = "Phase 3"
    match mode_num:
        case 25: return run_mc_baseline(n_candles, n_paths, mode_num, phase, progress=_.get("progress"))
        case 26: return run_mc_leika(n_candles, n_paths, exec_mode=1, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 27: return run_mc_leika(n_candles, n_paths, exec_mode=0, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 28: return run_mc_leika(n_candles, n_paths, exec_mode=1, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 29: return run_mc_leika(n_candles, n_paths, exec_mode=2, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 30: return run_mc_leika(n_candles, n_paths, exec_mode=3, mode_num=mode_num, phase=phase, progress=_.get("progress"))
    raise ValueError(f"Unknown mode {mode_num} for Phase 3")
