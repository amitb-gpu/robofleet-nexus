from __future__ import annotations

from fastapi import FastAPI
from robofleet_nexus.gpu.inventory import GpuInventory, detect_nvidia_gpus
from robofleet_nexus.isaac.job_spec import IsaacSimulationJob, SimulationPlan
from robofleet_nexus.isaac.scheduler import plan_simulation_job
from robofleet_nexus.isaac.profiles import SimulationProfile, list_simulation_profiles

from robofleet_nexus.audit.hash_chain import HashChainAuditLog
from robofleet_nexus.diagnostics.rules import evaluate_event
from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent

app = FastAPI(
    title="RoboFleet Nexus",
    description="Robotics orchestration layer for telemetry, diagnostics, simulation workflows, and auditability.",
    version="0.1.0",
)

AUDIT_LOG = HashChainAuditLog()
EVENTS: list[RobotEvent] = []
FINDINGS: list[DiagnosticFinding] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telemetry", response_model=list[DiagnosticFinding])
def ingest_telemetry(event: RobotEvent) -> list[DiagnosticFinding]:
    EVENTS.append(event)

    AUDIT_LOG.append(
        event_type="telemetry.ingested",
        actor="api",
        payload=event.model_dump(mode="json"),
    )

    findings = evaluate_event(event)
    FINDINGS.extend(findings)

    for finding in findings:
        AUDIT_LOG.append(
            event_type="diagnostic.finding.created",
            actor="diagnostic_engine",
            payload=finding.model_dump(mode="json"),
        )

    return findings


@app.get("/diagnostics", response_model=list[DiagnosticFinding])
def diagnostics() -> list[DiagnosticFinding]:
    return FINDINGS


@app.get("/audit")
def audit() -> dict[str, object]:
    return {"records": [record.model_dump() for record in AUDIT_LOG.records()]}

@app.get("/gpu/inventory", response_model=GpuInventory)
def gpu_inventory() -> GpuInventory:
    inventory = detect_nvidia_gpus()

    AUDIT_LOG.append(
        event_type="gpu.inventory.checked",
        actor="api",
        payload=inventory.model_dump(mode="json"),
    )

    return inventory


@app.post("/simulations/plan", response_model=SimulationPlan)
def plan_simulation(job: IsaacSimulationJob) -> SimulationPlan:
    plan = plan_simulation_job(job)

    AUDIT_LOG.append(
        event_type="simulation.plan.created",
        actor="simulation_scheduler",
        payload={
            "job": job.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
        },
    )

    return plan

@app.get("/simulations/profiles", response_model=list[SimulationProfile])
def simulation_profiles() -> list[SimulationProfile]:
    profiles = list_simulation_profiles()

    AUDIT_LOG.append(
        event_type="simulation.profiles.listed",
        actor="api",
        payload={"profile_count": len(profiles)},
    )

    return profiles
