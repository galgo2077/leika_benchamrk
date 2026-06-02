#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RESULTS_DIR="benchmark_results/$(date +"%Y%m%d_%H%M%S")"

mkdir -p "$RESULTS_DIR"

echo ""
echo "┌──────────────────────────────────────────────────────┐"
echo "│             LEIKA BENCHMARK SUITE                    │"
echo "└──────────────────────────────────────────────────────┘"
echo ""
echo "  Results:  $RESULTS_DIR"
echo ""

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    . "$SCRIPT_DIR/.venv/bin/activate"
fi

exec "$SCRIPT_DIR/scripts/benchmark" --out-dir "$RESULTS_DIR" "$@"
