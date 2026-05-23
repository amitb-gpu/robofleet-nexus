import json
from pathlib import Path

from typer.testing import CliRunner

from robofleet_nexus.cli.main import app


runner = CliRunner()


def test_cli_plans_simulation_from_yaml() -> None:
    result = runner.invoke(
        app,
        ["simulations", "plan", "examples/simulations/isaac_warehouse_nav.yaml"],
    )

    assert result.exit_code == 0

    payload = json.loads(result.stdout)

    assert payload["job_id"] == "sim-warehouse-nav-001"
    assert payload["accepted"] is True
    assert payload["backend"] == "mock"
    assert payload["launch_mode"] == "mock_dry_run"


def test_cli_fail_on_blocked_returns_nonzero(tmp_path: Path) -> None:
    job_file = tmp_path / "blocked.yaml"
    job_file.write_text(
        """
job_id: blocked-physical-command-test
backend: mock
parameters:
  scenario: warehouse_navigation
  robot_model: nova_carter
safety:
  allow_physical_robot_commands: true
  human_approval_required: true
"""
    )

    result = runner.invoke(
        app,
        ["simulations", "plan", str(job_file), "--fail-on-blocked"],
    )

    assert result.exit_code == 2

    payload = json.loads(result.stdout)

    assert payload["accepted"] is False
    assert payload["launch_mode"] == "blocked"
