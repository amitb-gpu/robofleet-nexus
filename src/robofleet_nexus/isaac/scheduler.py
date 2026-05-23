from __future__ import annotations

from robofleet_nexus.gpu.inventory import GpuInventory, detect_nvidia_gpus
from robofleet_nexus.isaac.job_spec import IsaacBackend, IsaacSimulationJob, SimulationPlan


def plan_simulation_job(
    job: IsaacSimulationJob,
    gpu_inventory: GpuInventory | None = None,
) -> SimulationPlan:
    """
    Create a safe execution plan for an Isaac simulation job.

    This is intentionally a planner, not a launcher. Launching Isaac Sim or Isaac
    Lab should remain policy-gated and auditable.
    """
    inventory = gpu_inventory or detect_nvidia_gpus()

    if job.safety.allow_physical_robot_commands:
        return SimulationPlan(
            job_id=job.job_id,
            accepted=False,
            backend=job.backend,
            launch_mode="blocked",
            reason="Simulation jobs may not enable physical robot commands.",
            planned_command=[],
            warnings=["Physical robot command path requested by job safety policy."],
        )

    if job.backend in {IsaacBackend.isaac_sim, IsaacBackend.isaac_lab}:
        if not inventory.available:
            return SimulationPlan(
                job_id=job.job_id,
                accepted=False,
                backend=job.backend,
                launch_mode="blocked",
                reason="NVIDIA GPU is required but no GPU inventory is available.",
                planned_command=[],
                warnings=inventory.warnings,
            )

        eligible_devices = [
            device
            for device in inventory.devices
            if device.memory_total_mb is not None
            and device.memory_total_mb >= job.resources.min_vram_gb * 1024
        ]

        if len(eligible_devices) < job.resources.gpu_count:
            return SimulationPlan(
                job_id=job.job_id,
                accepted=False,
                backend=job.backend,
                launch_mode="blocked",
                reason="Insufficient eligible NVIDIA GPU capacity for requested simulation job.",
                planned_command=[],
                warnings=[
                    f"required_gpus={job.resources.gpu_count}",
                    f"eligible_gpus={len(eligible_devices)}",
                    f"min_vram_gb={job.resources.min_vram_gb}",
                ],
            )

    if job.backend == IsaacBackend.isaac_sim:
        command = [
            "isaac-sim.sh",
            "--headless",
            "--/app/window/drawMouse=false",
            "--/app/renderer/resolution/width=1280",
            "--/app/renderer/resolution/height=720",
        ]
        launch_mode = "isaac_sim_headless_dry_run"
    elif job.backend == IsaacBackend.isaac_lab:
        command = [
            "python",
            "scripts/reinforcement_learning/train.py",
            f"--task={job.parameters.scenario}",
        ]
        launch_mode = "isaac_lab_dry_run"
    else:
        command = [
            "python",
            "-m",
            "robofleet_nexus.isaac.mock_runner",
            f"--job-id={job.job_id}",
        ]
        launch_mode = "mock_dry_run"

    return SimulationPlan(
        job_id=job.job_id,
        accepted=True,
        backend=job.backend,
        launch_mode=launch_mode,
        reason="Simulation job passed resource and safety planning checks.",
        planned_command=command,
    )
