from pathlib import Path

import pytest

from robofleet_nexus.isaac.job_spec import IsaacBackend
from robofleet_nexus.isaac.loader import SimulationJobLoadError, load_simulation_job


def test_load_yaml_simulation_job() -> None:
    job = load_simulation_job(Path("examples/simulations/isaac_warehouse_nav.yaml"))

    assert job.job_id == "sim-warehouse-nav-001"
    assert job.backend == IsaacBackend.mock
    assert job.parameters.scenario == "warehouse_navigation"
    assert job.parameters.robot_model == "nova_carter"


def test_loader_rejects_unknown_extension(tmp_path: Path) -> None:
    job_file = tmp_path / "job.txt"
    job_file.write_text("not yaml")

    with pytest.raises(SimulationJobLoadError):
        load_simulation_job(job_file)
