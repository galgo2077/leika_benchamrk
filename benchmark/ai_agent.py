"""AI Trading Agent and compact benchmark-analysis client."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from ai_context import BenchmarkAiConfig, compact_context, estimate_tokens
from metrics import gpu_snapshot
try:
    from leika.ai_dynamic import AIAnalysisEngine, AIAnalysisResult
except Exception:
    @dataclass
    class AIAnalysisResult:  # type: ignore[no-redef]
        summary: str = ""
        final_text: str = ""
        sections: dict = field(default_factory=dict)
        ai_metrics: dict = field(default_factory=dict)

    class AIAnalysisEngine:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            pass

        def analyze(self, context: dict) -> AIAnalysisResult:
            return AIAnalysisResult(
                summary="AI analysis unavailable in this build.",
                final_text="AI analysis unavailable in this build.",
                sections={},
                ai_metrics={
                    "section_count": 0,
                    "ai_calls": 0,
                    "total_ai_time_ms": 0.0,
                    "average_section_time_ms": 0.0,
                    "slowest_section_ms": 0.0,
                    "total_prompt_chars": 0,
                    "total_response_chars": 0,
                    "timeouts": 0,
                    "fallbacks": 1,
                    "dynamic_sectioning_enabled": False,
                },
            )

SYSTEM_PROMPT = """\
You are an autonomous Quantitative Algorithmic Trading Agent (Senior Level).

Your objective is to maximize:
- Sharpe Ratio
- Sortino Ratio
- Risk-adjusted returns

You operate on financial time series data using:
- MACD (Moving Average Convergence Divergence): fast=12, slow=26, signal=9

CRITICAL TRADING RULE

MACD is ONLY an advisory tool.
It does NOT determine your actions alone.

Your final decision must be based on:
- price action
- market structure
- support/resistance levels
- volatility regime
- momentum shifts (MACD histogram direction and magnitude)
"""


@dataclass
class BenchmarkAiMetrics:
    model: str
    ai_mode: str
    context_chars: int = 0
    estimated_tokens: int = 0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    total_ai_time_ms: float = 0.0
    ai_tokens_per_second: float = 0.0
    gpu_used: bool = False
    vram_used_mb: float = 0.0
    timeout: bool = False
    fallback: bool = False
    skipped: bool = False
    skipped_reason: str = ""
    ai_sections: int = 0
    ai_calls: int = 0
    avg_section_ms: float = 0.0
    slowest_section: str = ""
    fastest_section: str = ""
    total_prompt_chars: int = 0
    total_response_chars: int = 0
    timeout_count: int = 0
    fallback_count: int = 0
    section_parallelism: str = ""
    dynamic_sectioning_enabled: bool = False
    section_results: list[dict] = field(default_factory=list)


class AiAgent:
    def __init__(self, model: str = "0xroyce/plutus:latest", host: str = "localhost", port: int = 11434):
        self.model = model
        self.base_url = f"http://{host}:{port}"
        self.available = self._ping()
        self.call_count = 0
        self.total_ms = 0.0

    def _ping(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def decide(
        self,
        prices: list[float],
        macd_line: list[Optional[float]],
        signal_line: list[Optional[float]],
        histogram: list[Optional[float]],
        fast_period: int,
        slow_period: int,
        signal_period: int,
        cycle: int,
        total_cycles: int,
    ) -> dict:
        """Legacy trading helper kept for compatibility with the older AI demos."""
        if not self.available:
            return {"action": "HOLD", "fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}

        recent = prices[-24:] if len(prices) >= 24 else prices
        hist_r = [v for v in (histogram[-24:] if len(histogram) >= 24 else histogram) if v is not None]
        macd_r = [v for v in (macd_line[-24:] if len(macd_line) >= 24 else macd_line) if v is not None]
        sig_r = [v for v in (signal_line[-24:] if len(signal_line) >= 24 else signal_line) if v is not None]

        price_chg = ((recent[-1] / recent[0]) - 1.0) * 100.0 if len(recent) > 1 else 0.0
        hi = max(recent)
        lo = min(recent)

        macd_last = f"{macd_r[-1]:.6f}" if macd_r else "N/A"
        hist_last = f"{hist_r[-1]:.6f}" if hist_r else "N/A"
        hist_dir = "rising" if len(hist_r) >= 2 and hist_r[-1] > hist_r[-2] else "falling" if len(hist_r) >= 2 else "N/A"

        user_msg = (
            f"Cycle {cycle}/{total_cycles} | Last 24 candles\n"
            f"Price: {recent[-1]:.4f} | Change: {price_chg:+.2f}% | Range: {lo:.4f}–{hi:.4f}\n"
            f"MACD({fast_period},{slow_period},{signal_period}) last: {macd_last}\n"
            f"MACD histogram: {hist_last} ({hist_dir})\n"
            f"\nDecide now."
        )

        payload = json.dumps(
            {
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": user_msg,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 350},
            }
        ).encode()

        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                text = data.get("response", "")
        except Exception:
            self.total_ms += (time.monotonic() - t0) * 1000
            return {"action": "HOLD", "fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period}

        self.call_count += 1
        self.total_ms += (time.monotonic() - t0) * 1000
        return self._parse(text, fast_period, slow_period, signal_period)

    def _parse(self, text: str, fast_period: int, slow_period: int, signal_period: int) -> dict:
        action = "HOLD"
        new_fast = fast_period
        new_slow = slow_period
        new_signal = signal_period

        for line in text.splitlines():
            upper = line.upper()
            if "ACTION:" in upper:
                if "BUY" in upper:
                    action = "BUY"
                elif "SELL" in upper:
                    action = "SELL"
                elif "CLOSE" in upper:
                    action = "SELL"
                break

        for line in text.splitlines():
            low = line.lower()
            if "current fast_period:" in low or "fast period:" in low:
                try:
                    tok = line.split(":")[-1].strip().split()[0]
                    val = int("".join(c for c in tok if c.isdigit()))
                    if 5 <= val <= 50:
                        new_fast = val
                except Exception:
                    pass
            if "current slow_period:" in low or "slow period:" in low:
                try:
                    tok = line.split(":")[-1].strip().split()[0]
                    val = int("".join(c for c in tok if c.isdigit()))
                    if 15 <= val <= 200:
                        new_slow = val
                except Exception:
                    pass
            if "current signal_period:" in low or "signal period:" in low:
                try:
                    tok = line.split(":")[-1].strip().split()[0]
                    val = int("".join(c for c in tok if c.isdigit()))
                    if 3 <= val <= 30:
                        new_signal = val
                except Exception:
                    pass

        return {"action": action, "fast_period": new_fast, "slow_period": new_slow, "signal_period": new_signal}


class BenchmarkAiAgent:
    """Bounded benchmark analysis client.

    It sends compact structured context only and records request/response timing.
    """

    def __init__(self, config: BenchmarkAiConfig, host: str = "localhost", port: int = 11434):
        self.config = config
        self.base_url = f"http://{host}:{port}"
        self.host = host
        self.port = port
        self.available = self._ping()
        self.model = self._resolve_model()
        self.call_count = 0
        self.total_ms = 0.0

    def _ping(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def _available_models(self) -> list[dict]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
            return list(data.get("models", []))
        except Exception:
            return []

    def _model_size_key(self, model: dict) -> float:
        size = model.get("details", {}).get("parameter_size") or model.get("size") or ""
        text = str(size).lower().strip()
        if not text:
            return float("inf")
        num = ""
        for ch in text:
            if ch.isdigit() or ch == ".":
                num += ch
        try:
            return float(num)
        except Exception:
            return float("inf")

    def _resolve_model(self) -> str:
        if self.config.model:
            return self.config.model
        if self.config.ai_mode == "quick-ai":
            models = self._available_models()
            if models:
                smallest = min(models, key=self._model_size_key)
                return smallest.get("name") or smallest.get("model") or "qwen2.5:0.5b"
            return "qwen2.5:0.5b"
        return "0xroyce/plutus:latest"

    def _build_engine(self, dynamic_sections: bool) -> AIAnalysisEngine:
        return AIAnalysisEngine(
            model=self.model,
            host=self.host,
            port=self.port,
            ai_mode=self.config.ai_mode,
            dynamic_sections=dynamic_sections,
            timeout_seconds=self.config.timeout_seconds,
            max_context_chars=self.config.max_context_chars,
            max_output_tokens=self.config.max_output_tokens,
            allow_parallel=True,
        )

    def _metrics_from_result(self, result: AIAnalysisResult, metrics: BenchmarkAiMetrics) -> BenchmarkAiMetrics:
        ai_metrics = result.ai_metrics
        metrics.ai_sections = int(ai_metrics.get("section_count", len(result.sections) or 0))
        metrics.ai_calls = int(ai_metrics.get("ai_calls", metrics.ai_sections))
        metrics.avg_section_ms = float(ai_metrics.get("average_section_time_ms", 0.0))
        metrics.slowest_section = str(ai_metrics.get("slowest_section", ""))
        metrics.fastest_section = str(ai_metrics.get("fastest_section", ""))
        metrics.total_prompt_chars = int(ai_metrics.get("total_prompt_chars", 0))
        metrics.total_response_chars = int(ai_metrics.get("total_response_chars", 0))
        metrics.timeout_count = int(ai_metrics.get("timeouts", 0))
        metrics.fallback_count = int(ai_metrics.get("fallbacks", 0))
        metrics.section_parallelism = str(ai_metrics.get("section_parallelism", ""))
        metrics.dynamic_sectioning_enabled = bool(ai_metrics.get("dynamic_sectioning_enabled", False))
        metrics.section_results = list(result.sections.values())
        metrics.total_ai_time_ms = float(ai_metrics.get("total_ai_time_ms", metrics.total_ai_time_ms))
        metrics.timeout = metrics.timeout_count > 0
        metrics.fallback = metrics.fallback_count > 0
        if metrics.total_ai_time_ms > 0 and metrics.ai_calls > 0:
            metrics.ai_tokens_per_second = metrics.total_response_chars / max(metrics.total_ai_time_ms / 1000.0, 1e-9)
        return metrics

    def analyze(self, context: dict) -> tuple[str, BenchmarkAiMetrics]:
        """Dynamic multi-section AI analysis."""
        metrics = BenchmarkAiMetrics(model=self.model, ai_mode=self.config.ai_mode)
        try:
            compact, context_chars, estimated_tokens, compressed = compact_context(context, self.config.max_context_chars)
        except ValueError as exc:
            metrics.skipped = True
            metrics.skipped_reason = str(exc)
            return metrics.skipped_reason, metrics

        metrics.context_chars = context_chars
        metrics.estimated_tokens = estimated_tokens

        engine = self._build_engine(dynamic_sections=True)
        gpu_before = gpu_snapshot()
        t0 = time.monotonic()
        try:
            analysis = engine.analyze(context)
        except Exception as exc:
            self.total_ms += (time.monotonic() - t0) * 1000
            metrics.fallback = True
            metrics.dynamic_sectioning_enabled = True
            metrics.skipped_reason = str(exc)
            return "AI skipped: inference failure", metrics

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.call_count += int(analysis.ai_metrics.get("ai_calls", 0) or 0)
        self.total_ms += elapsed_ms

        metrics = self._metrics_from_result(analysis, metrics)
        metrics.prompt_eval_duration_ms = float(analysis.ai_metrics.get("slowest_section_ms", 0.0))
        metrics.eval_duration_ms = float(analysis.ai_metrics.get("average_section_time_ms", 0.0))
        metrics.total_ai_time_ms = float(analysis.ai_metrics.get("total_ai_time_ms", elapsed_ms)) or elapsed_ms
        metrics.ai_tokens_per_second = (
            metrics.total_response_chars / max(metrics.total_ai_time_ms / 1000.0, 1e-9)
            if metrics.total_ai_time_ms > 0
            else estimate_tokens(analysis.summary) / max(elapsed_ms / 1000.0, 1e-9)
        )
        metrics.gpu_used = any(v is not None and v > 0 for v in gpu_before)
        if gpu_before[2] is not None:
            metrics.vram_used_mb = float(gpu_before[2])

        return analysis.final_text or analysis.summary, metrics

    def analyze_single(self, context: dict) -> tuple[str, BenchmarkAiMetrics]:
        metrics = BenchmarkAiMetrics(model=self.model, ai_mode=self.config.ai_mode)

        try:
            compact, context_chars, estimated_tokens, compressed = compact_context(context, self.config.max_context_chars)
        except ValueError as exc:
            metrics.skipped = True
            metrics.skipped_reason = str(exc)
            return metrics.skipped_reason, metrics

        metrics.context_chars = context_chars
        metrics.estimated_tokens = estimated_tokens

        prompt = (
            "You are reviewing a Leika benchmark run. "
            "Use only the structured JSON context. "
            "Do not ask for raw candles, logs, or full trade lists. "
            "Return concise bullet points for strengths, weaknesses, and one improvement.\n\n"
            f"CONTEXT_JSON={compact}"
        )

        payload = json.dumps(
            {
                "model": self.model,
                "system": "You are a concise quantitative benchmark analyst.",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": self.config.max_output_tokens,
                },
            }
        ).encode()

        gpu_before = gpu_snapshot()
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                body = resp.read()
                data = json.loads(body)
                text = str(data.get("response", "")).strip()
                prompt_eval_ns = int(data.get("prompt_eval_duration") or 0)
                eval_ns = int(data.get("eval_duration") or 0)
                total_ns = int(data.get("total_duration") or 0)
                prompt_eval_count = int(data.get("prompt_eval_count") or 0)
                eval_count = int(data.get("eval_count") or 0)
        except urllib.error.URLError:
            self.total_ms += (time.monotonic() - t0) * 1000
            metrics.timeout = True
            metrics.fallback = True
            return "AI skipped: timeout", metrics
        except Exception:
            self.total_ms += (time.monotonic() - t0) * 1000
            metrics.fallback = True
            return "AI skipped: inference failure", metrics

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.call_count += 1
        self.total_ms += elapsed_ms

        metrics.prompt_eval_duration_ms = prompt_eval_ns / 1_000_000.0
        metrics.eval_duration_ms = eval_ns / 1_000_000.0
        metrics.total_ai_time_ms = (total_ns / 1_000_000.0) if total_ns > 0 else elapsed_ms
        metrics.total_ai_time_ms = metrics.total_ai_time_ms or elapsed_ms
        metrics.ai_tokens_per_second = (
            eval_count / (metrics.eval_duration_ms / 1000.0)
            if metrics.eval_duration_ms > 1e-9 and eval_count > 0
            else estimate_tokens(text) / max(metrics.total_ai_time_ms / 1000.0, 1e-9)
        )
        metrics.gpu_used = any(v is not None and v > 0 for v in gpu_before)
        if gpu_before[2] is not None:
            metrics.vram_used_mb = float(gpu_before[2])
        metrics.ai_sections = 1
        metrics.ai_calls = 1
        metrics.avg_section_ms = metrics.total_ai_time_ms
        metrics.slowest_section = "baseline"
        metrics.fastest_section = "baseline"
        metrics.total_prompt_chars = len(prompt)
        metrics.total_response_chars = len(text)
        metrics.timeout_count = 1 if metrics.timeout else 0
        metrics.fallback_count = 1 if metrics.fallback else 0
        metrics.section_parallelism = "queued"
        metrics.dynamic_sectioning_enabled = False
        metrics.section_results = [{
            "section": "baseline",
            "model": self.model,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            "estimated_prompt_tokens": estimate_tokens(prompt),
            "latency_ms": metrics.total_ai_time_ms,
            "tokens_per_second": metrics.ai_tokens_per_second,
            "timeout": metrics.timeout,
            "fallback": metrics.fallback,
        }]

        return text, metrics
