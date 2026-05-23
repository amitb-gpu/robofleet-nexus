from __future__ import annotations

from fastapi import FastAPI

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
