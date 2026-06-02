#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmark_results/$(date +%Y%m%d_%H%M%S)}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT/benchmark_reports/$(date +%Y%m%d_%H%M%S)}"
HISTORY_DIR="${HISTORY_DIR:-$ROOT/benchmark_history/$(date +%Y%m%d_%H%M%S)}"
DOCS_DIR="${DOCS_DIR:-$ROOT/benchmark_docs}"

mkdir -p "$RESULTS_DIR" "$REPORTS_DIR" "$HISTORY_DIR" "$DOCS_DIR"

python3 - <<'PY'
import leika
print(f"leika {getattr(leika, '__version__', 'unknown')} OK")
PY

exec "$ROOT/scripts/benchmark" --out-dir "$RESULTS_DIR" "$@"
