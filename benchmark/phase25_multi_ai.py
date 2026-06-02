"""Phase 2.5 — 5-asset portfolio + AI (modes 19–24)."""
from runner import run_with_ai


def run(mode_num: int, bars: int, model: str = "0xroyce/plutus:latest",
        extreme: bool = False, **kwargs):
    phase = "Phase 2.5"
    progress = kwargs.get("progress")
    ai_config = kwargs.get("ai_config")
    run_ai = kwargs.get("run_ai", True)
    match mode_num:
        case 19: return run_with_ai(bars, mode_num, phase, "vectorbt_baseline", None, 5, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 20: return run_with_ai(bars, mode_num, phase, "vectorbt_rust",     None, 5, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 21: return run_with_ai(bars, mode_num, phase, "leika_mode_0",      0,    5, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 22: return run_with_ai(bars, mode_num, phase, "leika_mode_1",      1,    5, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 23: return run_with_ai(bars, mode_num, phase, "leika_mode_2",      2,    5, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 24: return run_with_ai(bars, mode_num, phase, "leika_mode_3",      3,    5, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
    raise ValueError(f"Unknown mode {mode_num} for phase2.5")
