"""
GPU monitor for RoboFleet Nexus.

Queries nvidia-smi for live GPU metrics and classifies each device into
a simulation capability profile. Handles both WSL environments and
bare-metal Linux identically.

Supported device profiles (auto-detected by VRAM):
  LAPTOP_DEV       ≤ 4 GB   RTX A1000, RTX 3050 Ti, etc.
  WORKSTATION_DEV  ≤ 10 GB  RTX 4060, RTX 3060, etc.
  WORKSTATION_HIGH ≤ 20 GB  RTX 5080 (16 GB), RTX 4080, etc.
  PRODUCTION       > 20 GB  RTX 6000 Ada, L40S, A100, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SimProfile(str, Enum):
    LAPTOP_DEV = "LAPTOP_DEV"          # ≤ 4 GB  — CI / mock sim only
    WORKSTATION_DEV = "WORKSTATION_DEV"  # ≤ 10 GB — light Isaac Sim
    WORKSTATION_HIGH = "WORKSTATION_HIGH"  # ≤ 20 GB — full Isaac Sim
    PRODUCTION = "PRODUCTION"          # > 20 GB — production cluster
    UNKNOWN = "UNKNOWN"


def _classify(vram_mb: int) -> SimProfile:
    if vram_mb <= 6_500:
        return SimProfile.LAPTOP_DEV
    if vram_mb <= 10_500:
        return SimProfile.WORKSTATION_DEV
    if vram_mb <= 20_000:
        return SimProfile.WORKSTATION_HIGH
    return SimProfile.PRODUCTION


@dataclass
class GpuDevice:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    utilization_pct: int
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    driver_version: str
    cuda_version: str
    profile: SimProfile
    wsl: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def vram_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return round(self.vram_used_mb / self.vram_total_mb * 100, 1)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "vram_pct": self.vram_pct,
            "utilization_pct": self.utilization_pct,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
            "power_limit_w": self.power_limit_w,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "profile": self.profile.value,
            "wsl": self.wsl,
            "timestamp": self.timestamp,
        }


_NVIDIA_SMI_QUERY = (
    "--query-gpu="
    "index,name,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,"
    "power.draw,power.limit,"
    "driver_version"
    " --format=csv,noheader,nounits"
)


def _is_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _parse_safe_float(value: str) -> float:
    try:
        return float(value.strip().replace(" W", "").replace(" MiB", "").replace("[N/A]", "0").replace("N/A", "0"))
    except (ValueError, TypeError):
        return 0.0


def _parse_safe_int(value: str) -> int:
    try:
        return int(value.strip().replace(" MiB", "").replace(" %", "").replace(" C", ""))
    except ValueError:
        return 0


def query_gpus() -> list[GpuDevice]:
    """
    Synchronously query all NVIDIA GPUs via nvidia-smi.
    Returns an empty list if nvidia-smi is unavailable (CPU-only / VM).
    """
    if not shutil.which("nvidia-smi"):
        logger.warning("nvidia-smi not found — returning empty GPU list")
        return []

    wsl = _is_wsl()

    try:
        result = subprocess.run(
            ["nvidia-smi", *_NVIDIA_SMI_QUERY.split()],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi timed out")
        return []
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        logger.warning("nvidia-smi error: %s", result.stderr.strip())
        return []

    devices: list[GpuDevice] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue

        (
            idx_s, name,
            mem_total_s, mem_used_s, mem_free_s,
            util_s, temp_s,
            power_draw_s, power_limit_s,
            driver_v,
        ) = parts[:10]

        vram_total = _parse_safe_int(mem_total_s)
        device = GpuDevice(
            index=_parse_safe_int(idx_s),
            name=name,
            vram_total_mb=vram_total,
            vram_used_mb=_parse_safe_int(mem_used_s),
            vram_free_mb=_parse_safe_int(mem_free_s),
            utilization_pct=_parse_safe_int(util_s),
            temperature_c=_parse_safe_int(temp_s),
            power_draw_w=_parse_safe_float(power_draw_s),
            power_limit_w=_parse_safe_float(power_limit_s),
            driver_version=driver_v.strip(),
            cuda_version="N/A",
            profile=_classify(vram_total),
            wsl=wsl,
        )
        devices.append(device)

    return devices


async def query_gpus_async() -> list[GpuDevice]:
    """Non-blocking wrapper — runs nvidia-smi in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, query_gpus)


async def gpu_poll_loop(
    broadcast_fn,          # async callable(list[dict])
    interval_seconds: float = 3.0,
) -> None:
    """
    Background task: polls GPU metrics every `interval_seconds` and
    calls `broadcast_fn` with the serialised list.

    Usage in FastAPI lifespan:
        asyncio.create_task(gpu_poll_loop(manager.emit_gpu_snapshot))
    """
    while True:
        try:
            devices = await query_gpus_async()
            await broadcast_fn([d.to_dict() for d in devices])
        except Exception as exc:
            logger.exception("GPU poll error: %s", exc)
        await asyncio.sleep(interval_seconds)
