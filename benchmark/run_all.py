from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_venv() -> None:
    try:
        import vectorbt  # noqa: F401
        return
    except Exception:
        pass

    venv_python = Path(__file__).resolve().parents[1] / ".venv/bin/python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("LEIKA_BENCH_VENV_BOOTSTRAPPED") == "1":
        return

    os.environ["LEIKA_BENCH_VENV_BOOTSTRAPPED"] = "1"
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap_venv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner import main


if __name__ == "__main__":
    main()
