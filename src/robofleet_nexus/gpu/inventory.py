from __future__ import annotations

import json
import shutil
import subprocess
from pydantic import BaseModel, Field


class GpuDevice(BaseModel):
    index: int
    name: str
    uuid: str | None = None
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    utilization_gpu_pct: float | None = None
    temperature_c: float | None = None


class GpuInventory(BaseModel):
    available: bool
    source: str
    devices: list[GpuDevice] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def detect_nvidia_gpus() -> GpuInventory:
    """
    Detect NVIDIA GPUs using nvidia-smi.

    This intentionally avoids pynvml for the first implementation so the core
    package remains lightweight and works in CI without NVIDIA libraries.
    """
    if shutil.which("nvidia-smi") is None:
        return GpuInventory(
            available=False,
            source="nvidia-smi",
            warnings=["nvidia-smi not found on PATH"],
        )

    query = ",".join(
        [
            "index",
            "name",
            "uuid",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "temperature.gpu",
        ]
    )

    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return GpuInventory(
            available=False,
            source="nvidia-smi",
            warnings=[f"nvidia-smi query failed: {exc}"],
        )

    devices: list[GpuDevice] = []

    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue

        index, name, uuid, mem_total, mem_used, util, temp = parts

        devices.append(
            GpuDevice(
                index=int(index),
                name=name,
                uuid=uuid,
                memory_total_mb=int(mem_total),
                memory_used_mb=int(mem_used),
                utilization_gpu_pct=float(util),
                temperature_c=float(temp),
            )
        )

    return GpuInventory(
        available=bool(devices),
        source="nvidia-smi",
        devices=devices,
    )


def inventory_as_json() -> str:
    return json.dumps(detect_nvidia_gpus().model_dump(), indent=2)
