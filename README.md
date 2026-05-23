# RoboFleet Nexus

Production-style robotics orchestration for ROS2, NVIDIA Isaac workflows, telemetry, diagnostics, GPU-aware simulation scheduling, and auditability.

## What this project is

RoboFleet Nexus is an orchestration layer that sits above robotics infrastructure. It does not replace physics engines, robot middleware, or simulation frameworks. Instead, it connects telemetry, diagnostics, simulation jobs, GPU resources, and policy-governed agentic workflows into one production-oriented control plane.

The project is designed for robotics teams that need better visibility into:

- robot health
- ROS2 event streams
- simulation runs
- GPU-backed workloads
- diagnostic workflows
- safety boundaries
- decision audit trails

## Why this exists

Modern robotics stacks are powerful but fragmented. A team may use ROS2 for robot communication, Isaac Sim for simulation, Isaac Lab for robot learning, Kubernetes for infrastructure, and separate tools for observability.

RoboFleet Nexus provides a unified orchestration layer above those systems.

## Core capabilities

- Robot telemetry ingestion
- ROS2 adapter architecture
- Isaac Sim job orchestration model
- GPU-aware simulation scheduling
- Diagnostic rule engine
- Agentic RCA workflow design
- Safety policy enforcement
- Tamper-evident audit logs
- REST API
- CLI
- Docker and Kubernetes-ready deployment path

## Current status

MVP in progress.

The first release focuses on:

- typed telemetry schemas
- REST ingestion API
- deterministic diagnostic rules
- structured findings
- tamper-evident audit logging
- mock ROS2 and Isaac Sim adapters

## Architecture

```mermaid
flowchart TB
    ROS2[ROS2 / Robot Events] --> Ingest[Telemetry Ingestion]
    Isaac[Isaac Sim / Isaac Lab Jobs] --> Ingest
    GPU[NVIDIA GPU Metrics] --> Ingest
    Ingest --> Core[RoboFleet Nexus Core]
    Core --> Diagnostics[Diagnostic Engine]
    Core --> Agent[Agentic Orchestrator]
    Core --> Policy[Policy Engine]
    Core --> Audit[Tamper-Evident Audit Log]
    Core --> API[REST API / CLI / Dashboard]
