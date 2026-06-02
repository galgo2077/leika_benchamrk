"""Phase 4 — Random Walk single-asset (modes 31–36).

Dataset matrix:
  rw_model  : gbm | gaussian | mean_reversion | jump_diffusion | regime_switching
  n_candles : 1_000 | 10_000 | 100_000
  n_paths   : 1 (single price series per asset)

Engines per (model × candle) pair (6):
  31 — Python Baseline    (pure Python, model-specific generator)
  32 — VectorBT Baseline  (leika.RandomWalk, default Rayon pool)
  33 — Leika Mode 0       (CpuOnly  — trivially serial for n_paths=1)
  34 — Leika Mode 1       (Adaptive — full Rayon)
  35 — Leika Mode 2       (GpuAccelerated)
  36 — Leika Mode 3       (HybridCpuGpu — Rayon CPU + GPU concurrently)

Total: 5 models × 3 candle sizes × 6 engines = 90 executions.
"""
from runner import run_rw_baseline, run_rw_leika


def run(mode_num: int, n_candles: int, rw_model: str, **_) -> object:
    phase = "Phase 4"
    match mode_num:
        case 31: return run_rw_baseline(n_candles, rw_model, mode_num, phase, progress=_.get("progress"))
        case 32: return run_rw_leika(n_candles, rw_model, exec_mode=1, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 33: return run_rw_leika(n_candles, rw_model, exec_mode=0, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 34: return run_rw_leika(n_candles, rw_model, exec_mode=1, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 35: return run_rw_leika(n_candles, rw_model, exec_mode=2, mode_num=mode_num, phase=phase, progress=_.get("progress"))
        case 36: return run_rw_leika(n_candles, rw_model, exec_mode=3, mode_num=mode_num, phase=phase, progress=_.get("progress"))
    raise ValueError(f"Unknown mode {mode_num} for Phase 4")
