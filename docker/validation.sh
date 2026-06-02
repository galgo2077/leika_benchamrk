#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Leika Benchmark validation"

python3 - <<'PY'
import leika
assert leika is not None
print(f"import leika OK: {getattr(leika, '__version__', 'unknown')}")
PY

python3 - <<PY
from compileall import compile_dir
from pathlib import Path
root = Path(r"$ROOT")
if not compile_dir(str(root / "benchmark"), quiet=1):
    raise SystemExit(1)
print("py_compile OK")
PY

test -f "$ROOT/benchmark/run_all.py"
test -f "$ROOT/benchmark/runner.py"
test -f "$ROOT/benchmark/report_gen.py"
test -f "$ROOT/benchmark/requirements.txt"

echo "validation OK"
