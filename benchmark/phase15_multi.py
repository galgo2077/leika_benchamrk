"""Phase 1.5 — 5-asset portfolio (modes 7–12)."""
from runner import run_vectorbt_baseline_5, run_vectorbt_rust_5, run_leika_multi

def run(mode_num: int, bars: int, extreme: bool = False, **_):
    phase = "Phase 1.5"
    match mode_num:
        case 7:  return run_vectorbt_baseline_5(bars, mode_num, phase, extreme, progress=_.get("progress"))
        case 8:  return run_vectorbt_rust_5(bars, mode_num, phase, extreme, progress=_.get("progress"))
        case 9:  return run_leika_multi(bars, exec_mode=0, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
        case 10: return run_leika_multi(bars, exec_mode=1, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
        case 11: return run_leika_multi(bars, exec_mode=2, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
        case 12: return run_leika_multi(bars, exec_mode=3, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
    raise ValueError(f"Unknown mode {mode_num} for phase1.5")
