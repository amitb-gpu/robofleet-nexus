from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from robofleet_nexus.isaac.loader import SimulationJobLoadError, load_simulation_job
from robofleet_nexus.isaac.scheduler import plan_simulation_job

app = typer.Typer(help="RoboFleet Nexus CLI")
simulations_app = typer.Typer(help="Simulation orchestration commands")
app.add_typer(simulations_app, name="simulations")

console = Console()


@app.command()
def version() -> None:
    console.print("RoboFleet Nexus v0.1.0")


@app.command()
def doctor() -> None:
    console.print("[green]Local package import OK.[/green]")
    console.print("Next checks: ROS2 availability, NVIDIA GPU visibility, Isaac Sim bridge readiness.")


@simulations_app.command("plan")
def plan_simulation(
    job_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a YAML or JSON Isaac simulation job specification.",
    ),
    fail_on_blocked: bool = typer.Option(
        False,
        "--fail-on-blocked",
        help="Exit with code 2 if the simulation plan is blocked.",
    ),
) -> None:
    """
    Load a simulation job spec and produce an auditable dry-run execution plan.
    """
    try:
        job = load_simulation_job(job_file)
        plan = plan_simulation_job(job)
    except SimulationJobLoadError as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print_json(data=plan.model_dump(mode="json"))

    if fail_on_blocked and not plan.accepted:
        raise typer.Exit(code=2)
