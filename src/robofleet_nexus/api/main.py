from __future__ import annotations

from fastapi import FastAPI, BackgroundTasks
from robofleet_nexus.gpu.inventory import GpuInventory, detect_nvidia_gpus
from robofleet_nexus.isaac.job_spec import IsaacSimulationJob, SimulationPlan
from robofleet_nexus.isaac.scheduler import plan_simulation_job
from robofleet_nexus.isaac.profiles import SimulationProfile, list_simulation_profiles
from robofleet_nexus.audit.hash_chain import HashChainAuditLog
from robofleet_nexus.diagnostics.rules import evaluate_event
from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent
from robofleet_nexus.api.ws_routes import ws_router, lifespan, on_telemetry_ingested
from robofleet_nexus.api.ws_manager import manager as ws_manager
from robofleet_nexus.rca.agent import run_rca
from robofleet_nexus.lerobot.episode_buffer import episode_buffer
from robofleet_nexus.lerobot.routes import dataset_router

app = FastAPI(
    lifespan=lifespan,
    title="RoboFleet Nexus",
    description="Robotics orchestration layer for telemetry, diagnostics, simulation workflows, and auditability.",
    version="0.1.0",
)
app.include_router(ws_router)
app.include_router(dataset_router)

AUDIT_LOG = HashChainAuditLog()
EVENTS: list[RobotEvent] = []
FINDINGS: list[DiagnosticFinding] = []
RCA_RESULTS: list[dict] = []

# Cooldown tracker: robot_id → set of finding titles already RCA'd this session
_RCA_SEEN: dict[str, set[str]] = {}
_RCA_COOLDOWN_SECONDS = 900  # only re-run RCA for same finding after 15 minutes
_RCA_LAST_FIRED: dict[str, float] = {}


async def _run_and_broadcast_rca(robot_id: str, findings: list[DiagnosticFinding]) -> None:
    """Background task: run RCA and push result to dashboard."""
    relevant_events = [e for e in EVENTS if e.robot_id == robot_id]
    rca = await run_rca(robot_id, findings, relevant_events)
    if rca and "error" not in rca:
        RCA_RESULTS.append(rca)
        await ws_manager.broadcast({"type": "rca_result", "rca": rca})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/telemetry", response_model=list[DiagnosticFinding])
async def ingest_telemetry(
    event: RobotEvent, background_tasks: BackgroundTasks
) -> list[DiagnosticFinding]:
    EVENTS.append(event)
    AUDIT_LOG.append(
        event_type="telemetry.ingested",
        actor="api",
        payload=event.model_dump(mode="json"),
    )

    # Feed LeRobot episode buffer
    episode_buffer.ingest(event)

    findings = evaluate_event(event)
    FINDINGS.extend(findings)
    for finding in findings:
        AUDIT_LOG.append(
            event_type="diagnostic.finding.created",
            actor="diagnostic_engine",
            payload=finding.model_dump(mode="json"),
        )
    await on_telemetry_ingested(event.robot_id, event.model_dump(mode="json"))

    if findings:
        import time
        robot_id = event.robot_id
        now = time.time()
        last = _RCA_LAST_FIRED.get(robot_id, 0)
        new_titles = {f.title for f in findings}
        seen = _RCA_SEEN.get(robot_id, set())
        truly_new = new_titles - seen
        cooldown_expired = (now - last) > _RCA_COOLDOWN_SECONDS
        if truly_new or cooldown_expired:
            _RCA_SEEN[robot_id] = seen | new_titles
            _RCA_LAST_FIRED[robot_id] = now
            background_tasks.add_task(_run_and_broadcast_rca, event.robot_id, findings)

    return findings


@app.get("/diagnostics", response_model=list[DiagnosticFinding])
def diagnostics() -> list[DiagnosticFinding]:
    return FINDINGS


@app.get("/rca", response_model=list[dict])
def rca_results() -> list[dict]:
    return RCA_RESULTS


@app.post("/rca/analyze")
async def trigger_rca(robot_id: str, background_tasks: BackgroundTasks) -> dict:
    """Manually trigger RCA for a robot against all its current findings."""
    robot_findings = [f for f in FINDINGS if f.robot_id == robot_id]
    if not robot_findings:
        return {"status": "no_findings", "robot_id": robot_id}
    background_tasks.add_task(_run_and_broadcast_rca, robot_id, robot_findings)
    return {"status": "rca_triggered", "robot_id": robot_id, "findings": len(robot_findings)}


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
