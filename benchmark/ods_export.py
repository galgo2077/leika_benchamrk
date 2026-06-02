"""
ODS spreadsheet exporter for Leika benchmark results.

Pure stdlib — zipfile + XML strings.  No odfpy / openpyxl / pandas needed.
Produces a LibreOffice Calc-compatible .ods workbook with 12 sheets.
"""
from __future__ import annotations

import dataclasses
import io
import math
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Set

# Fields excluded from Raw Results (large arrays, not useful as cells).
_LARGE_FIELDS = frozenset({
    "prices", "equity", "ema_fast", "drawdowns",
    "ai_section_results", "trade_summary",
})

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _e(v: Any) -> str:
    """XML-escape a value to a safe string."""
    s = "" if v is None else str(v)
    return (s.replace("&", "&amp;")
              .replace("<", "&lt;")
              .replace(">", "&gt;")
              .replace('"', "&quot;"))


def _f(v: Any) -> Optional[float]:
    """Return a finite float or None (handles None, nan, inf, bool)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def _disp(fv: float) -> str:
    """Human-readable display string for a float cell."""
    if fv == int(fv) and abs(fv) < 1e12:
        return str(int(fv))
    return f"{fv:.6g}"


# ---------------------------------------------------------------------------
# ODS file constants
# ---------------------------------------------------------------------------

_MIMETYPE = "application/vnd.oasis.opendocument.spreadsheet"

_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'xmlns:number="urn:oasis:names:tc:opendocument:xmlns:datastyle:1.0"'
)

_MANIFEST = """\
<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
  <manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>
  <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
  <manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

_STYLES = """\
<?xml version="1.0" encoding="UTF-8"?>
<office:document-styles
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
    xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
    office:version="1.2">
  <office:styles>
    <style:default-style style:family="table-cell">
      <style:text-properties fo:font-family="Liberation Sans" fo:font-size="10pt"/>
    </style:default-style>
    <style:style style:name="Default" style:family="table-cell"/>
  </office:styles>
</office:document-styles>"""

_AUTO_STYLES = """\
  <office:automatic-styles>
    <style:style style:name="co-opt" style:family="table-column">
      <style:table-column-properties style:use-optimal-width="true" fo:break-before="auto"/>
    </style:style>
    <style:style style:name="ce-hdr" style:family="table-cell" style:parent-style-name="Default">
      <style:table-cell-properties fo:background-color="#dae8fc"/>
      <style:text-properties fo:font-weight="bold"/>
    </style:style>
    <style:style style:name="ce-dat" style:family="table-cell" style:parent-style-name="Default"/>
  </office:automatic-styles>"""


# ---------------------------------------------------------------------------
# Sheet builder
# ---------------------------------------------------------------------------

class _Sheet:
    """Accumulates XML for one ODS sheet."""

    def __init__(self, name: str) -> None:
        self.name = name[:31]   # ODS sheet-name limit for compatibility
        self._parts: List[str] = []
        self._n_cols: int = 0

    # -- public API ----------------------------------------------------------

    def add_header_row(self, headers: List[str]) -> None:
        self._n_cols = len(headers)
        self._parts.append(
            f'<table:table-column table:style-name="co-opt" '
            f'table:number-columns-repeated="{max(self._n_cols, 1)}" '
            f'table:default-cell-style-name="ce-dat"/>'
        )
        self._parts.append("<table:table-header-rows><table:table-row>")
        for h in headers:
            self._parts.append(
                f'<table:table-cell table:style-name="ce-hdr" '
                f'office:value-type="string"><text:p>{_e(h)}</text:p></table:table-cell>'
            )
        self._parts.append("</table:table-row></table:table-header-rows>")

    def add_row(self, values: List[Any], num_cols: Set[int]) -> None:
        self._parts.append("<table:table-row>")
        for ci, val in enumerate(values):
            self._parts.append(_cell(val, ci in num_cols))
        # Pad to column count so LibreOffice renders empty trailing cells.
        if len(values) < self._n_cols:
            gap = self._n_cols - len(values)
            self._parts.append(
                f'<table:table-cell table:number-columns-repeated="{gap}"/>'
            )
        self._parts.append("</table:table-row>")

    def to_xml(self) -> str:
        inner = "\n".join(self._parts)
        return f'<table:table table:name="{_e(self.name)}">\n{inner}\n</table:table>'


def _cell(val: Any, numeric: bool) -> str:
    """Build the XML for a single ODS table cell."""
    if val is None or val == "":
        return '<table:table-cell table:style-name="ce-dat"/>'
    if numeric:
        fv = _f(val)
        if fv is None:
            return '<table:table-cell table:style-name="ce-dat"/>'
        return (
            f'<table:table-cell table:style-name="ce-dat" '
            f'office:value-type="float" office:value="{fv}">'
            f'<text:p>{_disp(fv)}</text:p></table:table-cell>'
        )
    return (
        f'<table:table-cell table:style-name="ce-dat" '
        f'office:value-type="string"><text:p>{_e(str(val))}</text:p></table:table-cell>'
    )


# ---------------------------------------------------------------------------
# ODS document
# ---------------------------------------------------------------------------

class _OdsDoc:
    def __init__(self) -> None:
        self.sheets: List[_Sheet] = []

    def _content_xml(self) -> str:
        sheets_xml = "\n".join(s.to_xml() for s in self.sheets)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<office:document-content {_NS} office:version="1.2">\n'
            '  <office:scripts/>\n'
            '  <office:font-face-decls/>\n'
            f'{_AUTO_STYLES}\n'
            '  <office:body>\n'
            '    <office:spreadsheet>\n'
            f'{sheets_xml}\n'
            '    </office:spreadsheet>\n'
            '  </office:body>\n'
            '</office:document-content>\n'
        )

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # mimetype MUST be first and uncompressed per ODF spec.
            zinfo = zipfile.ZipInfo("mimetype")
            zinfo.compress_type = zipfile.ZIP_STORED
            zf.writestr(zinfo, _MIMETYPE)
            zf.writestr("META-INF/manifest.xml", _MANIFEST)
            zf.writestr("styles.xml", _STYLES)
            zf.writestr("content.xml", self._content_xml())
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Column definitions (headers + numeric-column index sets)
# ---------------------------------------------------------------------------

_PF_HEADERS = [
    "Phase", "Mode Number", "Engine Name", "Execution Mode",
    "Bars", "Assets", "Runtime ms", "Ops/s", "Speedup vs VectorBT",
    "Return %", "ROI %", "Sharpe", "Sortino", "Calmar",
    "Max Drawdown %", "Win Rate %", "Profit Factor", "Total Trades",
    "Dynamic Sectioning Used", "Workers", "Sections", "Chunk Size",
    "CPU %", "RAM MB", "Backend", "Stability Score", "Notes",
]
_PF_NUM: Set[int] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     19, 20, 21, 22, 23, 25}


def _pf_row(r: Any) -> List[Any]:
    dyn = "yes" if r.cpu_workers_selected > 0 else "no"
    return [
        r.phase, r.mode, r.mode_name, r.leika_exec_mode,
        r.bars, r.n_assets,
        r.exec_ms, r.throughput_bars_sec, r.speedup_vs_baseline,
        r.total_return_pct, r.roi_pct, r.sharpe_ratio,
        r.sortino_ratio, r.calmar_ratio, r.max_drawdown_pct,
        r.win_rate_pct, r.profit_factor, r.total_trades,
        dyn,
        r.cpu_workers_selected if r.cpu_workers_selected > 0 else None,
        r.segments_selected if r.segments_selected > 0 else None,
        None,  # Chunk Size — not tracked per-run
        r.cpu_pct, r.mem_mb,
        r.backend, 1.0,   # Stability Score — single timed run, nominal
        r.error or None,
    ]


# ---- Monte Carlo -----------------------------------------------------------

_MC_HEADERS = [
    "Phase", "Mode Number", "Engine Name", "Execution Mode",
    "Candles", "Paths", "Runtime ms", "Paths/sec", "Candles/sec",
    "Speedup vs Python Baseline", "Speedup vs Leika Mode 1",
    "Mean Return %", "Median Return %", "P05 Return %", "P95 Return %",
    "Std Return %", "Probability Positive %", "Median Max Drawdown %",
    "Dynamic Sectioning Used", "Workers", "Sections", "Chunk Size",
    "Return Paths", "Raw Paths Copied", "Memory Fallback", "Memory Fallback Reason",
    "CPU %", "RAM MB", "GPU %", "VRAM MB",
    "GPU Kernel Time ms", "GPU Transfer Time ms", "Backend", "Notes",
]
_MC_NUM: Set[int] = {
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31,
}


def _mc_row(r: Any, speedup_vs_m1: Optional[float] = None) -> List[Any]:
    fallback = "yes" if r.gpu_fallback_reason else "no"
    return [
        r.phase, r.mode, r.mode_name, r.leika_exec_mode,
        r.bars, r.n_paths,
        r.exec_ms, r.paths_sec, r.throughput_bars_sec,
        r.speedup_vs_baseline, speedup_vs_m1,
        r.total_return_pct,
        None, None, None, None, None, None,   # distribution stats not in BenchResult
        "yes" if r.cpu_workers_selected > 0 else "no",
        r.cpu_workers_selected if r.cpu_workers_selected > 0 else None,
        r.segments_selected if r.segments_selected > 0 else None,
        None,           # Chunk Size
        r.n_paths or None, None,   # Return Paths, Raw Paths Copied
        fallback, r.gpu_fallback_reason or None,
        r.cpu_pct, r.mem_mb,
        r.gpu_util_pct,
        r.gpu_mem_used_mb,
        r.gpu_kernel_time_ms if r.gpu_kernel_time_ms > 0 else None,
        r.gpu_transfer_time_ms if r.gpu_transfer_time_ms > 0 else None,
        r.backend,
        r.error or None,
    ]


# ---- Random Walk -----------------------------------------------------------

_RW_HEADERS = [
    "Phase", "Mode Number", "Engine Name", "Execution Mode",
    "Model", "Assets", "Candles", "Paths", "Runtime ms", "Paths/sec", "Candles/sec",
    "Speedup", "Mean Return %", "P05 Return %", "P95 Return %",
    "Probability Positive %", "Std Return %",
    "Dynamic Sectioning Used", "Workers", "Sections", "Chunk Size",
    "Return Paths", "Raw Paths Copied", "Memory Fallback",
    "CPU %", "RAM MB", "GPU %", "VRAM MB", "Backend", "Notes",
]
_RW_NUM: Set[int] = {
    1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 19, 20, 21, 22, 24, 25, 26, 27,
}


def _rw_row(r: Any) -> List[Any]:
    fallback = "yes" if r.gpu_fallback_reason else "no"
    return [
        r.phase, r.mode, r.mode_name, r.leika_exec_mode,
        r.rw_model, r.n_assets, r.bars,
        r.n_paths if r.n_paths > 0 else None,
        r.exec_ms, r.paths_sec, r.throughput_bars_sec,
        r.speedup_vs_baseline,
        r.total_return_pct,
        None, None, None, None,   # distribution stats not in BenchResult
        "yes" if r.cpu_workers_selected > 0 else "no",
        r.cpu_workers_selected if r.cpu_workers_selected > 0 else None,
        r.segments_selected if r.segments_selected > 0 else None,
        None, None, None,         # Chunk Size, Return Paths, Raw Paths Copied
        fallback,
        r.cpu_pct, r.mem_mb,
        r.gpu_util_pct, r.gpu_mem_used_mb,
        r.backend, r.error or None,
    ]


# ---- AI Benchmarks ---------------------------------------------------------

_AI_HEADERS = [
    "Phase", "Mode", "Engine", "AI Mode", "Model",
    "AI Sections", "AI Calls", "Ollama Concurrency", "Context Chars",
    "Estimated Tokens", "Total AI Time ms", "Avg Section Time ms",
    "Slowest Section", "Tokens/sec", "Timeout Count", "Fallback Count",
    "Backtest Time ms", "AI Overhead ms", "AI Overhead %",
    "GPU Used", "VRAM Used MB", "Notes",
]
_AI_NUM: Set[int] = {1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20}


def _ai_row(r: Any) -> List[Any]:
    ai_total = r.ai_total_time_ms or r.ai_ms_total
    total = r.total_runtime_ms or r.exec_ms
    overhead_pct = (ai_total / total * 100.0) if total > 0 else None
    return [
        r.phase, r.mode, r.mode_name, r.ai_mode, r.ai_model,
        r.ai_sections or None, r.ai_calls or None,
        None,   # Ollama Concurrency — not tracked
        r.ai_context_chars or None,
        r.ai_estimated_tokens or None,
        ai_total or None, r.ai_avg_section_ms or None,
        r.ai_slowest_section or None,
        r.ai_tokens_per_second or None,
        r.ai_timeout_count or None,
        r.ai_fallback_count or None,
        r.backtest_ms or r.exec_ms or None,
        ai_total or None,
        overhead_pct,
        "yes" if r.ai_gpu_used else "no",
        r.ai_vram_used_mb if r.ai_vram_used_mb else None,
        r.error or None,
    ]


# ---- Dynamic Tiling --------------------------------------------------------

_DYN_HEADERS = [
    "Workload", "Phase", "Mode", "Dynamic Tiling Enabled", "Split Axis",
    "Workers", "Sections", "Section Multiplier", "Chunk Size",
    "CPU Work Share %", "GPU Work Share %", "CPU Time ms", "GPU Time ms",
    "GPU Kernel Time ms", "GPU Transfer Time ms", "Hybrid Total Time ms",
    "Overlap %", "CPU Idle Wait ms", "GPU Idle Wait ms",
    "Memory Fallback", "Raw Paths Copied", "Recommended Mode", "Decision Reason",
]
_DYN_NUM: Set[int] = {2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20}


def _dyn_row(r: Any) -> List[Any]:
    has_gpu = r.gpu_total_time_ms > 0
    cpu_ms = (r.exec_ms - r.gpu_total_time_ms) if has_gpu else r.exec_ms
    fallback = "yes" if r.gpu_fallback_reason else "no"
    return [
        r.mode_name, r.phase, r.mode,
        "yes" if r.cpu_workers_selected > 0 else "no",
        None,   # Split Axis — not tracked in BenchResult
        r.cpu_workers_selected if r.cpu_workers_selected > 0 else None,
        r.segments_selected if r.segments_selected > 0 else None,
        None, None,   # Section Multiplier, Chunk Size
        None, None,   # CPU Work Share %, GPU Work Share %
        cpu_ms if cpu_ms and cpu_ms > 0 else None,
        r.gpu_total_time_ms if has_gpu else None,
        r.gpu_kernel_time_ms if r.gpu_kernel_time_ms > 0 else None,
        r.gpu_transfer_time_ms if r.gpu_transfer_time_ms > 0 else None,
        r.exec_ms,
        None, None, None,  # Overlap %, CPU Idle Wait ms, GPU Idle Wait ms
        fallback,
        None,   # Raw Paths Copied
        None,   # Recommended Mode
        r.gpu_fallback_reason or None,
    ]


# ---- GPU Metrics -----------------------------------------------------------

_GPU_HEADERS = [
    "Phase", "Mode", "Engine", "Workload", "Backend",
    "GPU Util %", "GPU Temp C", "VRAM Used MB",
    "GPU Kernel Time ms", "GPU Transfer Time ms",
    "GPU Total Time ms", "CPU Fallback Time ms",
    "GPU Accel Factor", "Numerical Error",
]
_GPU_NUM: Set[int] = {1, 5, 6, 7, 8, 9, 10, 11, 12, 13}


def _gpu_row(r: Any) -> List[Any]:
    if r.phase_type == "montecarlo" and r.n_paths > 0:
        workload = f"{r.bars:,} candles × {r.n_paths:,} paths"
    elif r.phase_type == "randomwalk":
        workload = f"{r.bars:,} candles [{r.rw_model}]"
    else:
        workload = f"{r.bars:,} bars"
    backend = getattr(r, "gpu_backend", "") or r.backend or ""
    return [
        r.phase, r.mode, r.mode_name, workload, backend,
        r.gpu_util_pct,
        r.gpu_temp_c,
        r.gpu_mem_used_mb,
        r.gpu_kernel_time_ms if r.gpu_kernel_time_ms > 0 else None,
        r.gpu_transfer_time_ms if r.gpu_transfer_time_ms > 0 else None,
        r.gpu_total_time_ms if r.gpu_total_time_ms > 0 else None,
        r.gpu_cpu_fallback_time_ms if r.gpu_cpu_fallback_time_ms > 0 else None,
        r.gpu_accel_factor,
        r.numerical_error_rel,
    ]


# ---- Resource Usage --------------------------------------------------------

_RES_HEADERS = [
    "Phase", "Mode", "Engine", "Bars", "Assets", "Paths",
    "CPU Threads", "Workers", "CPU Util %",
    "RAM Total MB", "RAM Budget MB", "RAM Peak MB", "RAM Used %",
    "GPU Util %", "VRAM Used MB",
    "Backend", "Fallback Reason",
    "Cash Model", "Execution Core", "Shared Data Sectioning",
    "Dynamic Sectioning Used", "Split Axis",
    "Sell At End Scope",
]
_RES_NUM: Set[int] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}


def _res_row(r: Any) -> List[Any]:
    ram_total_mb = r.ram_total_gb * 1024.0 if r.ram_total_gb else None
    ram_budget_mb = r.ram_budget_gb * 1024.0 if r.ram_budget_gb else None
    backend = getattr(r, "gpu_backend", "") or r.backend or ""
    return [
        r.phase, r.mode, r.mode_name,
        r.bars, r.n_assets,
        r.n_paths if r.n_paths > 0 else None,
        r.cpu_total_threads if r.cpu_total_threads > 0 else None,
        r.cpu_workers_selected if r.cpu_workers_selected > 0 else None,
        r.cpu_pct,
        ram_total_mb,
        ram_budget_mb,
        r.mem_mb if r.mem_mb > 0 else None,
        r.ram_peak_pct if r.ram_peak_pct > 0 else None,
        r.gpu_util_pct,
        r.gpu_mem_used_mb,
        backend,
        r.gpu_fallback_reason or None,
        getattr(r, "cash_model", None) or None,
        getattr(r, "execution_core", None) or None,
        "yes" if getattr(r, "shared_data_sectioning", False) else "no",
        "yes" if getattr(r, "dynamic_sectioning_used", False) else "no",
        getattr(r, "split_axis", None) or None,
        getattr(r, "sell_at_end_scope", None) or None,
    ]


# ---- Stability Metrics -----------------------------------------------------

_STAB_HEADERS = [
    "Phase", "Mode", "Engine", "Workload",
    "Warmup Runs", "Repeat Runs",
    "Runtime Min ms", "Runtime Median ms", "Runtime Mean ms",
    "Runtime P95 ms", "Runtime Max ms", "Runtime Std ms",
    "Coefficient of Variation %", "Stability Score",
    "Noisy Result", "Fallback Count", "Timeout Count",
]
_STAB_NUM: Set[int] = {1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}


def _stab_row(r: Any) -> List[Any]:
    if r.phase_type == "montecarlo" and r.n_paths > 0:
        workload = f"{r.bars:,} candles × {r.n_paths:,} paths"
    elif r.phase_type == "randomwalk":
        workload = f"{r.bars:,} candles [{r.rw_model}]"
    else:
        workload = f"{r.bars:,} bars"
    ms = r.exec_ms if r.exec_ms > 0 else None
    # Only one timed run per result — min/median/mean/p95/max are all the same.
    # Std and CV cannot be computed from a single measurement.
    return [
        r.phase, r.mode, r.mode_name, workload,
        r.warmup_runs, 1,
        ms, ms, ms, ms, ms,
        None, None,   # Std, CV — unknown from single run
        1.0,          # Stability Score — nominal
        "no",         # Noisy Result — unknown from single run
        r.ai_fallback_count or None,
        r.ai_timeout_count or None,
    ]


# ---- Summary ---------------------------------------------------------------

_SUM_HEADERS = [
    "Phase", "Best Engine", "Best Mode", "Best Runtime ms",
    "Baseline Runtime ms", "Speedup vs VectorBT",
    "Workload", "Bars", "Assets", "Paths", "Backend", "Notes",
]
_SUM_NUM: Set[int] = {2, 3, 4, 5, 7, 8, 9}


def _summary_rows(results: list) -> List[List[Any]]:
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in results:
        key = (r.phase, r.bars, r.n_paths, r.rw_model)
        groups[key].append(r)

    rows = []
    for (phase, bars, n_paths, rw_model), group in sorted(groups.items()):
        valid = [r for r in group if not r.error and r.exec_ms > 0]
        if not valid:
            continue
        best = min(valid, key=lambda r: r.exec_ms)
        baseline = min(group, key=lambda r: r.mode)

        if rw_model:
            workload = f"{bars:,} candles [{rw_model}]"
        elif n_paths > 0:
            workload = f"{bars:,} candles × {n_paths:,} paths"
        else:
            workload = f"{bars:,} bars"

        baseline_ms = baseline.exec_ms if not baseline.error else None
        rows.append([
            phase,
            best.mode_name,
            best.mode,
            best.exec_ms,
            baseline_ms,
            best.speedup_vs_baseline,
            workload,
            bars,
            best.n_assets,
            n_paths if n_paths > 0 else None,
            best.backend,
            best.error or None,
        ])
    return rows


# ---- Raw Results -----------------------------------------------------------

def _raw_headers(results: list) -> List[str]:
    if not results:
        return []
    d = dataclasses.asdict(results[0])
    return [k for k in d if k not in _LARGE_FIELDS]


def _raw_num_cols(headers: List[str], results: list) -> Set[int]:
    if not results:
        return set()
    d = dataclasses.asdict(results[0])
    return {
        i for i, h in enumerate(headers)
        if isinstance(d.get(h), (int, float)) and not isinstance(d.get(h), bool)
    }


def _raw_row(r: Any, headers: List[str]) -> List[Any]:
    d = dataclasses.asdict(r)
    return [d.get(h) for h in headers]


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def generate_ods(results: list,
                 out_dir: str = "benchmark_results",
                 ts_file: str = "") -> str:
    """
    Build and write leika_benchmark_<ts_file>.ods to out_dir.

    Returns the absolute path to the written file.
    Missing metrics are written as blank cells — nothing crashes.
    """
    from datetime import datetime, timezone
    if not ts_file:
        ts_file = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S000Z")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ods_path = out_path / f"leika_benchmark_{ts_file}.ods"

    doc = _OdsDoc()

    # ── Sheet 1: Summary ────────────────────────────────────────────────────
    sh = _Sheet("Summary")
    sh.add_header_row(_SUM_HEADERS)
    for row in _summary_rows(results):
        sh.add_row(row, _SUM_NUM)
    doc.sheets.append(sh)

    # ── Sheet 2: Phase 1 Portfolio Single ───────────────────────────────────
    sh = _Sheet("Phase 1 Portfolio Single")
    sh.add_header_row(_PF_HEADERS)
    for r in results:
        if r.phase == "Phase 1" and r.phase_type == "portfolio":
            sh.add_row(_pf_row(r), _PF_NUM)
    doc.sheets.append(sh)

    # ── Sheet 3: Phase 1.5 Portfolio Multi Asset ────────────────────────────
    sh = _Sheet("Phase 1.5 Portfolio Multi Asset")
    sh.add_header_row(_PF_HEADERS)
    for r in results:
        if r.phase == "Phase 1.5" and r.phase_type == "portfolio":
            sh.add_row(_pf_row(r), _PF_NUM)
    doc.sheets.append(sh)

    # ── Sheet 4: Phase 3 Monte Carlo ────────────────────────────────────────
    mc_idx = {(r.mode, r.bars, r.n_paths): r
              for r in results if r.phase_type == "montecarlo"}
    sh = _Sheet("Phase 3 Monte Carlo")
    sh.add_header_row(_MC_HEADERS)
    for r in results:
        if r.phase_type == "montecarlo":
            # Speedup vs MC Leika mode 1 (Adaptive) = mode 28 in current scheme
            m1 = mc_idx.get((28, r.bars, r.n_paths))
            sp_m1 = (
                (m1.exec_ms / r.exec_ms)
                if (m1 and r.exec_ms > 0 and not m1.error and m1.exec_ms > 0)
                else None
            )
            sh.add_row(_mc_row(r, sp_m1), _MC_NUM)
    doc.sheets.append(sh)

    # ── Sheet 5: Phase 4 Random Walk ────────────────────────────────────────
    sh = _Sheet("Phase 4 Random Walk")
    sh.add_header_row(_RW_HEADERS)
    for r in results:
        if r.phase_type == "randomwalk" and r.n_assets <= 1:
            sh.add_row(_rw_row(r), _RW_NUM)
    doc.sheets.append(sh)

    # ── Sheet 6: Phase 4.5 Random Walk Multi Asset ──────────────────────────
    sh = _Sheet("Phase 4.5 Random Walk Multi")
    sh.add_header_row(_RW_HEADERS)
    for r in results:
        if r.phase_type == "randomwalk" and r.n_assets > 1:
            sh.add_row(_rw_row(r), _RW_NUM)
    doc.sheets.append(sh)

    # ── Sheet 7: AI Benchmarks ──────────────────────────────────────────────
    sh = _Sheet("AI Benchmarks")
    sh.add_header_row(_AI_HEADERS)
    for r in results:
        if r.ai_enabled:
            sh.add_row(_ai_row(r), _AI_NUM)
    doc.sheets.append(sh)

    # ── Sheet 8: Dynamic Tiling ─────────────────────────────────────────────
    sh = _Sheet("Dynamic Tiling")
    sh.add_header_row(_DYN_HEADERS)
    for r in results:
        if r.cpu_workers_selected > 0 or r.segments_selected > 0:
            sh.add_row(_dyn_row(r), _DYN_NUM)
    doc.sheets.append(sh)

    # ── Sheet 9: GPU Metrics ────────────────────────────────────────────────
    sh = _Sheet("GPU Metrics")
    sh.add_header_row(_GPU_HEADERS)
    for r in results:
        has_gpu_data = (
            r.gpu_util_pct is not None
            or r.gpu_kernel_time_ms > 0
            or r.gpu_total_time_ms > 0
            or r.gpu_mem_used_mb is not None
        )
        if has_gpu_data:
            sh.add_row(_gpu_row(r), _GPU_NUM)
    doc.sheets.append(sh)

    # ── Sheet 10: Resource Usage ────────────────────────────────────────────
    sh = _Sheet("Resource Usage")
    sh.add_header_row(_RES_HEADERS)
    for r in results:
        sh.add_row(_res_row(r), _RES_NUM)
    doc.sheets.append(sh)

    # ── Sheet 11: Stability Metrics ─────────────────────────────────────────
    sh = _Sheet("Stability Metrics")
    sh.add_header_row(_STAB_HEADERS)
    for r in results:
        sh.add_row(_stab_row(r), _STAB_NUM)
    doc.sheets.append(sh)

    # ── Sheet 12: Raw Results ───────────────────────────────────────────────
    raw_hdrs = _raw_headers(results)
    raw_num = _raw_num_cols(raw_hdrs, results)
    sh = _Sheet("Raw Results")
    sh.add_header_row(raw_hdrs)
    for r in results:
        sh.add_row(_raw_row(r, raw_hdrs), raw_num)
    doc.sheets.append(sh)

    ods_path.write_bytes(doc.to_bytes())
    return str(ods_path)
