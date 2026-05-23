from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SimulationProfileName(str, Enum):
    ci_mock = "ci_mock"
    laptop_dev = "laptop_dev"
    workstation_l40s = "workstation_l40s"
    production_gpu_node = "production_gpu_node"


class SimulationProfile(BaseModel):
    name: SimulationProfileName
    description: str
    default_min_vram_gb: int
    supports_isaac_sim: bool
    supports_isaac_lab: bool
    supports_synthetic_data: bool
    notes: list[str]


SIMULATION_PROFILES: dict[SimulationProfileName, SimulationProfile] = {
    SimulationProfileName.ci_mock: SimulationProfile(
        name=SimulationProfileName.ci_mock,
        description="CI-safe mock profile with no NVIDIA GPU requirement.",
        default_min_vram_gb=0,
        supports_isaac_sim=False,
        supports_isaac_lab=False,
        supports_synthetic_data=False,
        notes=[
            "Used for unit tests and GitHub Actions.",
            "Does not launch Isaac Sim or Isaac Lab.",
        ],
    ),
    SimulationProfileName.laptop_dev: SimulationProfile(
        name=SimulationProfileName.laptop_dev,
        description="Developer laptop profile for lightweight planning and mock orchestration.",
        default_min_vram_gb=6,
        supports_isaac_sim=False,
        supports_isaac_lab=False,
        supports_synthetic_data=False,
        notes=[
            "Suitable for API development, scheduler testing, and GPU visibility checks.",
            "Not intended for full Isaac Sim workloads.",
        ],
    ),
    SimulationProfileName.workstation_l40s: SimulationProfile(
        name=SimulationProfileName.workstation_l40s,
        description="Single-node NVIDIA L40S workstation profile.",
        default_min_vram_gb=32,
        supports_isaac_sim=True,
        supports_isaac_lab=True,
        supports_synthetic_data=True,
        notes=[
            "Suitable for headless Isaac Sim planning.",
            "Designed for GPU-aware simulation orchestration experiments.",
        ],
    ),
    SimulationProfileName.production_gpu_node: SimulationProfile(
        name=SimulationProfileName.production_gpu_node,
        description="Production Kubernetes GPU node profile.",
        default_min_vram_gb=40,
        supports_isaac_sim=True,
        supports_isaac_lab=True,
        supports_synthetic_data=True,
        notes=[
            "Intended for future Kubernetes scheduling integration.",
            "Assumes NVIDIA container runtime and GPU device plugin/operator path.",
        ],
    ),
}


def list_simulation_profiles() -> list[SimulationProfile]:
    return list(SIMULATION_PROFILES.values())
