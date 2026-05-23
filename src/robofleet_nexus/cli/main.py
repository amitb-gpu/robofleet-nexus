from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="RoboFleet Nexus CLI")
console = Console()


@app.command()
def version() -> None:
    console.print("RoboFleet Nexus v0.1.0")


@app.command()
def doctor() -> None:
    console.print("[green]Local package import OK.[/green]")
    console.print("Next checks: ROS2 availability, NVIDIA GPU visibility, Isaac Sim bridge readiness.")
