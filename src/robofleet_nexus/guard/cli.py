"""
CLI for RoboFleet Guard.

    robofleet guard demo [--post http://localhost:8000] [--realtime]
    robofleet guard fit-baseline <lerobot_dataset_dir> --out guard_config/
    robofleet guard verify-audit <audit.jsonl>
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from robofleet_nexus.audit.hash_chain import AuditRecord, HashChainAuditLog
from robofleet_nexus.guard.baseline import ObservationBaseline
from robofleet_nexus.guard.core import GuardDecision, Verdict, ViolationKind
from robofleet_nexus.guard.envelope import SafetyEnvelope

guard_app = typer.Typer(help="Runtime safety guard for learned robot policies")
console = Console()

VERDICT_STYLE = {Verdict.PASS: "green", Verdict.CLAMP: "yellow", Verdict.HALT: "bold red"}


@guard_app.command("demo")
def demo(
    out: Annotated[
        Path, typer.Option("--out", help="Directory for the dataset, config and audit log.")
    ] = Path("guard_demo"),
    post: Annotated[
        str, typer.Option("--post", help="RoboFleet Nexus URL to send interventions to.")
    ] = "",
    realtime: Annotated[
        bool, typer.Option("--realtime", help="Run at 30 Hz instead of as fast as possible.")
    ] = False,
) -> None:
    """Run a policy with injected faults through the guard. No ROS2 or GPU needed."""
    from robofleet_nexus.guard.demo import HZ, run_demo
    from robofleet_nexus.guard.events import decision_to_event

    client = None
    if post:
        import httpx

        client = httpx.Client(base_url=post, timeout=5.0)

    def forward(decision: GuardDecision) -> None:
        nonlocal client
        if client is None:
            return
        try:
            client.post("/telemetry", json=decision_to_event(decision, source="mock"))
        except Exception as exc:  # noqa: BLE001 - the demo should keep running
            console.print(
                f"[yellow]Could not reach {post} ({exc}). Is the API running? "
                "Continuing without posting.[/yellow]\n"
                "[dim]Start it with: uvicorn robofleet_nexus.api.main:app --port 8000[/dim]"
            )
            client.close()
            client = None

    def pace(_step: int) -> None:
        if realtime:
            time.sleep(1.0 / HZ)

    console.print("[bold]RoboFleet Guard demo[/bold]: 6-joint arm, 30 Hz, 4 injected faults\n")
    result = run_demo(out, on_intervention=forward, on_step=pace)
    if client is not None:
        client.close()

    table = Table(title="Interventions", show_lines=False)
    table.add_column("Step", justify="right")
    table.add_column("Verdict")
    table.add_column("Why")
    previous_kinds: set[str] = set()
    for d in result.interventions:
        kinds = {v.kind.value for v in d.violations}
        # Collapse runs of identical clamps so the table stays readable.
        if d.verdict is Verdict.CLAMP and kinds == previous_kinds:
            continue
        previous_kinds = kinds
        style = VERDICT_STYLE[d.verdict]
        table.add_row(str(d.step), f"[{style}]{d.verdict.value.upper()}[/{style}]",
                      d.violations[0].detail if d.violations else "")
    console.print(table)

    s = result.guard.stats
    latched = s.violations_by_kind.get(ViolationKind.LATCHED.value, 0)
    console.print(
        f"\n{s.steps} steps: [green]{s.passed} passed[/green], "
        f"[yellow]{s.clamped} clamped[/yellow], [red]{s.halted} halted[/red] "
        f"({latched} of them holding position while latched), {s.resets} operator reset"
    )
    chain = "[green]intact[/green]" if result.chain_ok else "[red]BROKEN[/red]"
    console.print(f"Audit chain: {len(result.audit.records())} records, {chain}")
    console.print(f"Wrote dataset, baseline.json, envelope.yaml and audit.jsonl to {out}/")


@guard_app.command("fit-baseline")
def fit_baseline(
    dataset: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="LeRobot v3 dataset directory.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("guard_config"),
    control_hz: Annotated[float, typer.Option("--hz", help="Policy control rate.")] = 30.0,
    column: Annotated[str, typer.Option("--column")] = "observation.state",
) -> None:
    """Fit an OOD baseline and derive a starting safety envelope from demonstrations."""
    baseline = ObservationBaseline.from_lerobot_dataset(dataset, column=column)
    envelope = SafetyEnvelope.from_baseline(baseline, control_hz=control_hz)
    b = baseline.save(out / "baseline.json")
    e = envelope.to_yaml(out / "envelope.yaml")
    console.print(f"Fitted on {baseline.num_frames} frames, {baseline.dim} dimensions.")
    console.print(f"  {b}\n  {e}")
    console.print("[yellow]Review envelope.yaml before use: derived limits are a starting "
                  "point, not a certified safety configuration.[/yellow]")


@guard_app.command("verify-audit")
def verify_audit(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="audit.jsonl file.")],
) -> None:
    """Recompute the hash chain of an exported audit log."""
    records = [
        AuditRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ok, bad = HashChainAuditLog.verify_records(records)
    if ok:
        console.print(f"[green]Chain intact[/green]: {len(records)} records verified.")
        return
    console.print(f"[red]Chain broken at record {bad}[/red]: it or an earlier record was altered.")
    raise typer.Exit(code=1)
