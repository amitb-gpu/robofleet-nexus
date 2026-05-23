from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class IsaacBackend(str, Enum):
    isaac_sim = "isaac_sim"
    isaac_lab = "isaac_lab"
    mock = "mock"


class SimulationResources(BaseModel):
    gpu_count: int = 1
    min_vram_gb: int = 8
    timeout_minutes: int = 30


class SimulationParameters(BaseModel):
    scenario: str
    robot_model: str
    num_runs: int = 1
    random_seed_start: int = 0
    domain_randomization: bool = False
    collect_synthetic_data: bool = False


class SimulationSafetyPolicy(BaseModel):
    allow_physical_robot_commands: bool = False
    human_approval_required: bool = False


class IsaacSimulationJob(BaseModel):
    job_id: str
    backend: IsaacBackend = IsaacBackend.mock
    resources: SimulationResources = Field(default_factory=SimulationResources)
    parameters: SimulationParameters
    safety: SimulationSafetyPolicy = Field(default_factory=SimulationSafetyPolicy)
    metadata: dict[str, str] = Field(default_factory=dict)


class SimulationPlan(BaseModel):
    job_id: str
    accepted: bool
    backend: IsaacBackend
    launch_mode: str
    reason: str
    planned_command: list[str]
    warnings: list[str] = Field(default_factory=list)
