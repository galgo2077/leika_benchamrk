"""Phase 4.5 — Random Walk multi-asset portfolio (modes 37–42).

Dataset matrix:
  rw_model  : gbm | gaussian | mean_reversion | jump_diffusion | regime_switching
  n_candles : 1_000 | 10_000 | 100_000
  n_assets  : 5

Engines per (model × candle) pair (6):
  37 — Python Baseline    (Python generators, serially over 5 assets)
  38 — VectorBT Baseline  (leika.RandomWalk n_paths=5, default Rayon pool)
  39 — Leika Mode 0       (serial: 5 × n_paths=1 calls)
  40 — Leika Mode 1       (parallel: n_paths=5 single call — Rayon)
  41 — Leika Mode 2       (GpuAccelerated parallel)
  42 — Leika Mode 3       (HybridCpuGpu — Rayon CPU + GPU concurrently)

Total: 5 models × 3 candle sizes × 6 engines = 90 executions.
"""
from runner import run_rw_multi_baseline, run_rw_multi_leika

N_ASSETS = 5


def run(mode_num: int, n_candles: int, rw_model: str, **_) -> object:
    phase = "Phase 4.5"
    match mode_num:
        case 37: return run_rw_multi_baseline(n_candles, rw_model, N_ASSETS, mode_num, phase, progress=_.get("progress"))
        case 38: return run_rw_multi_leika(n_candles, rw_model, N_ASSETS, exec_mode=1, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 39: return run_rw_multi_leika(n_candles, rw_model, N_ASSETS, exec_mode=0, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 40: return run_rw_multi_leika(n_candles, rw_model, N_ASSETS, exec_mode=1, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 41: return run_rw_multi_leika(n_candles, rw_model, N_ASSETS, exec_mode=2, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 42: return run_rw_multi_leika(n_candles, rw_model, N_ASSETS, exec_mode=3, mode_num=mode_num, phase=phase, progress=_.get("progress"))
    raise ValueError(f"Unknown mode {mode_num} for Phase 4.5")
