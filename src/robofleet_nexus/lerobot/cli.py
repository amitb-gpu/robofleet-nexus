"""
CLI commands for LeRobot v3 dataset export.

Add to cli/main.py:
    from robofleet_nexus.lerobot.cli import dataset_app
    app.add_typer(dataset_app, name="dataset")

Usage:
    robofleet dataset export --robot-id bot_001
    robofleet dataset export --robot-id bot_001 --task pick_and_place --output ./my_datasets
    robofleet dataset inspect ./my_datasets/bot_001/5eps_20260524_090000
    robofleet dataset buffer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
dataset_app = typer.Typer(help="LeRobot v3 dataset commands.")


@dataset_app.command("export")
def cmd_export(
    robot_id: str = typer.Option(..., "--robot-id", help="Robot ID to export episodes for"),
    task: str = typer.Option("unspecified", "--task", help="Task label"),
    fps: float = typer.Option(10.0, "--fps", help="Nominal recording FPS"),
    output: Optional[str] = typer.Option(None, "--output", help="Output directory"),
    include_open: bool = typer.Option(False, "--include-open", help="Export the current open episode too"),
    nexus_url: str = typer.Option("http://localhost:8000", "--nexus-url"),
) -> None:
    """Export completed episodes for a robot to a LeRobot v3 dataset."""
    import httpx

    payload = {
        "robot_id": robot_id,
        "task": task,
        "fps": fps,
        "include_open_episode": include_open,
    }
    if output:
        payload["output_dir"] = output

    console.print(f"[bold cyan]Exporting dataset for [white]{robot_id}[/white]...[/bold cyan]")

    try:
        r = httpx.post(f"{nexus_url}/dataset/export", json=payload, timeout=30.0)
    except httpx.ConnectError:
        console.print(f"[red]Cannot reach Nexus at {nexus_url}[/red]")
        raise typer.Exit(1)

    if r.status_code == 404:
        detail = r.json().get("detail", "No episodes found.")
        console.print(f"[yellow]{detail}[/yellow]")
        raise typer.Exit(1)

    if r.status_code != 200:
        console.print(f"[red]Export failed: {r.status_code} {r.text[:200]}[/red]")
        raise typer.Exit(1)

    summary = r.json()
    _print_export_summary(summary)


@dataset_app.command("buffer")
def cmd_buffer(
    nexus_url: str = typer.Option("http://localhost:8000", "--nexus-url"),
) -> None:
    """Show the current episode buffer status."""
    import httpx

    try:
        r = httpx.get(f"{nexus_url}/dataset/buffer", timeout=10.0)
    except httpx.ConnectError:
        console.print(f"[red]Cannot reach Nexus at {nexus_url}[/red]")
        raise typer.Exit(1)

    data = r.json()
    robots = data.get("robots", [])
    if not robots:
        console.print("[yellow]No telemetry in buffer yet.[/yellow]")
        return

    table = Table(title="Episode Buffer Status")
    table.add_column("Robot ID", style="cyan")
    table.add_column("Buffered events", justify="right")
    table.add_column("Joint state frames", justify="right")
    table.add_column("Completed episodes", justify="right")

    for r_stats in robots:
        table.add_row(
            r_stats["robot_id"],
            str(r_stats["buffered_events"]),
            str(r_stats["joint_state_events"]),
            str(r_stats["completed_episodes"]),
        )
    console.print(table)


@dataset_app.command("inspect")
def cmd_inspect(
    dataset_dir: str = typer.Argument(..., help="Path to exported dataset directory"),
) -> None:
    """Inspect an exported LeRobot v3 dataset."""
    import pandas as pd

    root = Path(dataset_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        console.print(f"[red]No dataset found at {root}[/red]")
        raise typer.Exit(1)

    info = json.loads(info_path.read_text())

    console.print(f"\n[bold cyan]Dataset:[/bold cyan] {root}")
    console.print(f"  Robot ID:       {info.get('robot_id', '?')}")
    console.print(f"  Robot type:     {info.get('robot_type', '?')}")
    console.print(f"  Total episodes: {info.get('total_episodes', '?')}")
    console.print(f"  Total frames:   {info.get('total_frames', '?')}")
    console.print(f"  FPS:            {info.get('fps', '?')}")

    # Show first episode as sample
    chunk_dir = root / "data" / "chunk-000"
    parquets = sorted(chunk_dir.glob("*.parquet")) if chunk_dir.exists() else []
    if parquets:
        console.print(f"\n[bold]Sample — {parquets[0].name}:[/bold]")
        df = pd.read_parquet(parquets[0])
        console.print(f"  Frames:   {len(df)}")
        console.print(f"  Columns:  {list(df.columns)}")
        console.print(f"  Duration: {df['timestamp'].max():.2f}s")
        console.print(f"\n  First frame:")
        first = df.iloc[0]
        console.print(f"    action:             {first['action']}")
        console.print(f"    observation.state:  {first['observation.state']}")
        console.print(f"    observation.pose:   {first['observation.pose']}")
        console.print(f"    observation.battery:{first['observation.battery']}")


@dataset_app.command("list")
def cmd_list(
    nexus_url: str = typer.Option("http://localhost:8000", "--nexus-url"),
) -> None:
    """List all dataset exports from this session."""
    import httpx

    try:
        r = httpx.get(f"{nexus_url}/dataset/exports", timeout=10.0)
    except httpx.ConnectError:
        console.print(f"[red]Cannot reach Nexus at {nexus_url}[/red]")
        raise typer.Exit(1)

    data = r.json()
    exports = data.get("exports", [])
    if not exports:
        console.print("[yellow]No exports this session.[/yellow]")
        return

    table = Table(title="Dataset Exports")
    table.add_column("Robot ID", style="cyan")
    table.add_column("Episodes", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Output directory")

    for exp in exports:
        table.add_row(
            exp.get("robot_id", "?"),
            str(exp.get("total_episodes", "?")),
            str(exp.get("total_frames", "?")),
            exp.get("output_dir", "?"),
        )
    console.print(table)


def _print_export_summary(summary: dict) -> None:
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"  Robot ID:       {summary.get('robot_id', '?')}")
    console.print(f"  Total episodes: {summary.get('total_episodes', '?')}")
    console.print(f"  Total frames:   {summary.get('total_frames', '?')}")
    console.print(f"  Output:         [cyan]{summary.get('output_dir', '?')}[/cyan]")

    episodes = summary.get("episodes", [])
    if episodes:
        console.print(f"\n  Episodes:")
        for ep in episodes:
            console.print(
                f"    [{ep.get('episode_index', '?')}] "
                f"{ep.get('length', '?')} frames  "
                f"{ep.get('duration_seconds', '?'):.1f}s  "
                f"{ep.get('task', '?')}"
            )
