"""
Hardware detection for the benchmark suite.
Probes CPU, RAM, GPU/VRAM via psutil + subprocess, and calls Leika PyO3 bindings.
Prints a formatted Resource Planner output.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CpuInfo:
    name: str
    logical_threads: int
    physical_cores: int


@dataclass
class MemInfo:
    total_gb: float
    available_gb: float


@dataclass
class GpuInfo:
    name: str
    vram_gb: float
    backend: str   # CUDA | ROCm | CPU
    driver: str = ""


@dataclass
class HardwareSnapshot:
    cpu: CpuInfo
    mem: MemInfo
    gpu: Optional[GpuInfo]
    host_logical_threads: int = 0
    host_physical_cores: int = 0
    host_ram_gb: float = 0.0
    safe_ram_budget_gb: float = 0.0
    safe_ram_budget_pct: float = 70.0
    cpu_quota_cores: Optional[float] = None
    cpuset_cores: Optional[int] = None
    in_container: bool = False
    cgroup_version: Optional[int] = None
    score: int = 0
    leika_workers: int = 0
    leika_segments: int = 0
    leika_ai_workers: int = 0
    leika_throughput_m: float = 0.0
    leika_strategy: str = "CPU-Only"


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_cpu() -> CpuInfo:
    try:
        import psutil
        logical   = psutil.cpu_count(logical=True) or 1
        physical  = psutil.cpu_count(logical=False) or logical
    except ImportError:
        logical  = os.cpu_count() or 1
        physical = logical

    # Try to get CPU name
    name = "Unknown CPU"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            name = r.stdout.strip()
        elif platform.system() == "Windows":
            r = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True
            )
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and l.strip() != "Name"]
            name = lines[0] if lines else "Unknown CPU"
    except Exception:
        pass

    return CpuInfo(name=name, logical_threads=logical, physical_cores=physical)


def _detect_mem() -> MemInfo:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return MemInfo(
            total_gb=vm.total / 1e9,
            available_gb=vm.available / 1e9,
        )
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return MemInfo(total_gb=kb / 1e6, available_gb=kb / 1e6)
        except Exception:
            pass
    return MemInfo(total_gb=0.0, available_gb=0.0)


def _detect_gpu() -> Optional[GpuInfo]:
    # NVIDIA CUDA
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            name    = parts[0]
            vram_gb = float(parts[1]) / 1024.0
            driver  = parts[2] if len(parts) > 2 else ""
            return GpuInfo(name=name, vram_gb=vram_gb, backend="CUDA", driver=driver)
    except Exception:
        pass

    # AMD ROCm
    try:
        r = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            name    = "AMD GPU"
            vram_gb = 0.0
            for line in r.stdout.splitlines():
                if "Card series" in line:
                    name = line.split(":", 1)[1].strip()
                if "Total Memory" in line:
                    try:
                        vram_gb = float(line.split()[-1]) / 1_048_576
                    except Exception:
                        pass
            return GpuInfo(name=name, vram_gb=vram_gb, backend="ROCm")
    except Exception:
        pass

    return None


def _score_hardware(cpu: CpuInfo, mem: MemInfo, gpu: Optional[GpuInfo]) -> int:
    # Mirrors the fixed Rust formula in leika_engine/src/hardware/profile.rs:
    # linear scale up to 128 threads (40 pts max), log bonus beyond.
    t = float(cpu.logical_threads)
    if t <= 128.0:
        cpu_pts = (t / 128.0) * 40.0
    else:
        import math
        cpu_pts = min(40.0 + math.log(t / 128.0) * 5.0, 50.0)
    # RAM: reference 512 GB (mirrors fixed Rust formula)
    ram_pts = min(mem.total_gb / 512.0, 1.0) * 30.0
    gpu_pts = (min((gpu.vram_gb if gpu else 0.0) / 24.0, 1.0) * 30.0) if gpu else 0.0
    return round(cpu_pts + ram_pts + gpu_pts)


def _compute_leika_plan(cpu: CpuInfo, gpu: Optional[GpuInfo], hw_score: int):
    """Compute the Resource Planner output (mirrors plan.rs logic).

    NO WORKER CAP — matches Rust which removed the 28-thread ceiling.
    On an 80-thread machine: workers = 78, not 28.
    """
    workers  = max(1, cpu.logical_threads - 2)  # no upper cap
    segments = workers * 2

    if gpu:
        if gpu.vram_gb >= 16:
            ai_workers = 3
        elif gpu.vram_gb >= 8:
            ai_workers = 2
        else:
            ai_workers = 1
    else:
        ai_workers = 1

    backend_factor = 1.0
    if gpu:
        if gpu.backend == "CUDA":
            backend_factor = 1.5
        elif gpu.backend == "ROCm":
            backend_factor = 1.4

    throughput = workers * (hw_score / 100.0) * backend_factor
    strategy = "GPU-Accelerated" if gpu else "CPU-Only"

    return workers, segments, ai_workers, throughput, strategy


# ── Main detection ────────────────────────────────────────────────────────────

def detect() -> HardwareSnapshot:
    """Detect all hardware and compute Leika resource plan."""
    cpu = _detect_cpu()
    mem = _detect_mem()
    gpu = _detect_gpu()

    try:
        import leika
        engine   = leika.Engine()
        hw_obj   = engine.hardware
        plan_obj = engine.resource_plan()

        cpu = CpuInfo(
            name=hw_obj.cpu_name,
            logical_threads=hw_obj.logical_threads,
            physical_cores=hw_obj.physical_cores,
        )
        mem_total = hw_obj.ram_gb
        if gpu is None and hw_obj.gpu_name:
            gpu = GpuInfo(
                name=hw_obj.gpu_name,
                vram_gb=hw_obj.vram_gb or 0.0,
                backend=hw_obj.gpu_backend or "CPU",
                driver=hw_obj.driver_version or "",
            )

        return HardwareSnapshot(
            cpu=cpu,
            mem=MemInfo(total_gb=mem_total, available_gb=mem_total * 0.7),
            gpu=gpu,
            host_logical_threads=hw_obj.host_logical_threads,
            host_physical_cores=hw_obj.host_physical_cores,
            host_ram_gb=hw_obj.host_ram_gb,
            safe_ram_budget_gb=hw_obj.safe_ram_budget_gb,
            safe_ram_budget_pct=hw_obj.safe_ram_budget_pct,
            cpu_quota_cores=hw_obj.cpu_quota_cores,
            cpuset_cores=hw_obj.cpuset_cores,
            in_container=hw_obj.in_container,
            cgroup_version=hw_obj.cgroup_version,
            score=hw_obj.hardware_score,
            leika_workers=plan_obj.workers,
            leika_segments=plan_obj.segments,
            leika_ai_workers=plan_obj.ai_workers,
            leika_throughput_m=plan_obj.expected_throughput_m,
            leika_strategy=plan_obj.strategy,
        )
    except (ImportError, AttributeError):
        pass

    score = _score_hardware(cpu, mem, gpu)
    workers, segments, ai_workers, throughput, strategy = _compute_leika_plan(cpu, gpu, score)

    return HardwareSnapshot(
        cpu=cpu,
        mem=mem,
        gpu=gpu,
        host_logical_threads=cpu.logical_threads,
        host_physical_cores=cpu.physical_cores,
        host_ram_gb=mem.total_gb,
        safe_ram_budget_gb=mem.total_gb * 0.7,
        safe_ram_budget_pct=70.0,
        score=score,
        leika_workers=workers,
        leika_segments=segments,
        leika_ai_workers=ai_workers,
        leika_throughput_m=throughput,
        leika_strategy=strategy,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_hardware(hw: HardwareSnapshot) -> None:
    """Print hardware and Resource Planner output."""
    sep = "═" * 58
    thin = "─" * 58

    print()
    print(sep)
    print("  LEIKA — HARDWARE PROFILE & RESOURCE PLANNER")
    print(sep)
    print()

    print("  CPU:")
    print(f"    {hw.cpu.name}")
    if hw.host_logical_threads and hw.host_logical_threads != hw.cpu.logical_threads:
        print(f"    Host:    {hw.host_logical_threads} Threads  |  {hw.host_physical_cores} Physical Cores")
        print(f"    Visible: {hw.cpu.logical_threads} Threads  |  {hw.cpu.physical_cores} Physical Cores")
    else:
        print(f"    {hw.cpu.logical_threads} Threads  |  {hw.cpu.physical_cores} Physical Cores")
    print()

    print("  RAM:")
    if hw.host_ram_gb and hw.host_ram_gb != hw.mem.total_gb:
        print(f"    Host:    {hw.host_ram_gb:.0f} GB Total")
        print(f"    Visible: {hw.mem.total_gb:.0f} GB Total  |  {hw.mem.available_gb:.0f} GB Available")
    else:
        print(f"    {hw.mem.total_gb:.0f} GB Total  |  {hw.mem.available_gb:.0f} GB Available")
    print(f"    Safe budget: {hw.safe_ram_budget_gb:.1f} GB ({hw.safe_ram_budget_pct:.0f}%)")
    if hw.in_container:
        print(f"    Container: yes (cgroup v{hw.cgroup_version or '?'})")
    print()

    if hw.gpu:
        print(f"  GPU ({hw.gpu.backend}):")
        print(f"    {hw.gpu.name}")
        print(f"    {hw.gpu.vram_gb:.0f} GB VRAM")
        if hw.gpu.driver:
            print(f"    Driver: {hw.gpu.driver}")
    else:
        print("  GPU:")
        print("    None detected — CPU-only mode")
    print()

    print(f"  Hardware Score:  {hw.score}/100")
    print()
    print(thin)
    print()
    print("  Resource Planner:")
    print()
    print(f"    Strategy:              {hw.leika_strategy}")
    print(f"    Dynamic Modules:       {hw.leika_segments}")
    print(f"    Monte Carlo Workers:   {hw.leika_workers}")
    print(f"    Random Walk Workers:   {hw.leika_workers}")
    print(f"    AI Workers:            {hw.leika_ai_workers}")
    print(f"    Est. Throughput:       ~{hw.leika_throughput_m:.1f} M bars/sec")
    print()
    print(sep)
    print()


def to_dict(hw: HardwareSnapshot) -> dict:
    """Serialize hardware snapshot to a JSON-serializable dict."""
    return {
        "cpu_name":            hw.cpu.name,
        "cpu_threads":         hw.cpu.logical_threads,
        "cpu_cores":           hw.cpu.physical_cores,
        "ram_gb":              round(hw.mem.total_gb, 1),
        "ram_available_gb":    round(hw.mem.available_gb, 1),
        "host_ram_gb":         round(hw.host_ram_gb, 1),
        "safe_ram_budget_gb":  round(hw.safe_ram_budget_gb, 1),
        "safe_ram_budget_pct":  round(hw.safe_ram_budget_pct, 1),
        "gpu_name":            hw.gpu.name if hw.gpu else None,
        "gpu_vram_gb":         round(hw.gpu.vram_gb, 1) if hw.gpu else 0.0,
        "gpu_backend":         hw.gpu.backend if hw.gpu else "CPU",
        "hw_score":            hw.score,
        "leika_strategy":      hw.leika_strategy,
        "leika_workers":       hw.leika_workers,
        "leika_segments":      hw.leika_segments,
        "leika_ai_workers":    hw.leika_ai_workers,
        "leika_throughput_m":  round(hw.leika_throughput_m, 2),
    }


if __name__ == "__main__":
    hw = detect()
    print_hardware(hw)
