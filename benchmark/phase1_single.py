"""Phase 1 — Single-asset portfolio (modes 1–6)."""
from runner import run_vectorbt_baseline, run_vectorbt_rust, run_leika_single

def run(mode_num: int, bars: int, extreme: bool = False, **_):
    phase = "Phase 1"
    match mode_num:
        case 1: return run_vectorbt_baseline(bars, mode_num, phase, extreme, progress=_.get("progress"))
        case 2: return run_vectorbt_rust(bars, mode_num, phase, extreme, progress=_.get("progress"))
        case 3: return run_leika_single(bars, exec_mode=0, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
        case 4: return run_leika_single(bars, exec_mode=1, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
        case 5: return run_leika_single(bars, exec_mode=2, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
        case 6: return run_leika_single(bars, exec_mode=3, mode_num=mode_num, phase=phase, extreme=extreme, progress=_.get("progress"))
    raise ValueError(f"Unknown mode {mode_num} for phase1")
