from robofleet_nexus.gpu.inventory import GpuDevice, GpuInventory
from robofleet_nexus.isaac.job_spec import IsaacBackend, IsaacSimulationJob, SimulationParameters
from robofleet_nexus.isaac.scheduler import plan_simulation_job


def test_mock_simulation_plan_does_not_require_gpu() -> None:
    job = IsaacSimulationJob(
        job_id="sim-test-001",
        backend=IsaacBackend.mock,
        parameters=SimulationParameters(
            scenario="warehouse_navigation",
            robot_model="nova_carter",
        ),
    )

    inventory = GpuInventory(available=False, source="test")

    plan = plan_simulation_job(job, gpu_inventory=inventory)

    assert plan.accepted is True
    assert plan.launch_mode == "mock_dry_run"
    assert "mock_runner" in " ".join(plan.planned_command)


def test_isaac_sim_plan_requires_gpu() -> None:
    job = IsaacSimulationJob(
        job_id="sim-test-002",
        backend=IsaacBackend.isaac_sim,
        parameters=SimulationParameters(
            scenario="warehouse_navigation",
            robot_model="nova_carter",
        ),
    )

    inventory = GpuInventory(
        available=False,
        source="test",
        warnings=["no test GPU"],
    )

    plan = plan_simulation_job(job, gpu_inventory=inventory)

    assert plan.accepted is False
    assert plan.launch_mode == "blocked"
    assert "GPU" in plan.reason


def test_isaac_sim_plan_accepts_eligible_gpu() -> None:
    job = IsaacSimulationJob(
        job_id="sim-test-003",
        backend=IsaacBackend.isaac_sim,
        parameters=SimulationParameters(
            scenario="warehouse_navigation",
            robot_model="nova_carter",
        ),
    )

    inventory = GpuInventory(
        available=True,
        source="test",
        devices=[
            GpuDevice(
                index=0,
                name="NVIDIA L40S",
                uuid="GPU-test",
                memory_total_mb=49140,
                memory_used_mb=1024,
                utilization_gpu_pct=10.0,
                temperature_c=45.0,
            )
        ],
    )

    plan = plan_simulation_job(job, gpu_inventory=inventory)

    assert plan.accepted is True
    assert plan.launch_mode == "isaac_sim_headless_dry_run"
    assert "isaac-sim.sh" in plan.planned_command[0]


def test_physical_robot_command_path_is_blocked() -> None:
    job = IsaacSimulationJob(
        job_id="sim-test-004",
        backend=IsaacBackend.mock,
        parameters=SimulationParameters(
            scenario="warehouse_navigation",
            robot_model="nova_carter",
        ),
        safety={"allow_physical_robot_commands": True},
    )

    inventory = GpuInventory(available=True, source="test")

    plan = plan_simulation_job(job, gpu_inventory=inventory)

    assert plan.accepted is False
    assert plan.launch_mode == "blocked"
    assert "physical robot commands" in plan.reason
