"""Phase 2 — Single-asset + AI (modes 13–18)."""
from runner import run_with_ai


def run(mode_num: int, bars: int, model: str = "0xroyce/plutus:latest",
        extreme: bool = False, **kwargs):
    phase = "Phase 2"
    progress = kwargs.get("progress")
    ai_config = kwargs.get("ai_config")
    run_ai = kwargs.get("run_ai", True)
    match mode_num:
        case 13: return run_with_ai(bars, mode_num, phase, "vectorbt_baseline", None, 1, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 14: return run_with_ai(bars, mode_num, phase, "vectorbt_rust",     None, 1, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 15: return run_with_ai(bars, mode_num, phase, "leika_mode_0",      0,    1, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 16: return run_with_ai(bars, mode_num, phase, "leika_mode_1",      1,    1, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 17: return run_with_ai(bars, mode_num, phase, "leika_mode_2",      2,    1, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
        case 18: return run_with_ai(bars, mode_num, phase, "leika_mode_3",      3,    1, model, extreme, progress=progress, ai_config=ai_config, run_ai=run_ai)
    raise ValueError(f"Unknown mode {mode_num} for phase2")
