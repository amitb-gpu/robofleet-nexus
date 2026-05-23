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

## NVIDIA robotics orchestration layer

RoboFleet Nexus treats NVIDIA robotics infrastructure as a set of orchestration targets rather than something to reimplement.

The platform is designed to sit above:

- NVIDIA Isaac Sim for simulation workflows
- Isaac Lab for robot learning and experiment management
- ROS2 bridges for robot middleware integration
- NVIDIA GPUs for simulation, synthetic data, and acceleration workloads
- Kubernetes GPU nodes for future production scheduling

The current scheduler does not directly launch physical robot commands. It first creates a structured simulation plan, checks GPU capacity, applies safety policy, and records the decision in the audit log.

Example behavior:

- A lightweight mock simulation job can run in CI or on a laptop.
- A full Isaac Sim job requesting 16 GB of VRAM is blocked on a 6 GB RTX A1000 laptop GPU.
- The same job can be accepted on a workstation profile such as an NVIDIA L40S node.

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

## Current working capabilities

The current implementation includes a tested vertical slice of the orchestration control plane:

- FastAPI service with health, telemetry, diagnostics, audit, GPU inventory, and simulation planning endpoints
- Typed Pydantic schemas for robot telemetry, diagnostic findings, GPU inventory, and Isaac simulation jobs
- Deterministic diagnostic rule engine for robot and infrastructure events
- Tamper-evident audit log using chained record hashes
- NVIDIA GPU inventory through `nvidia-smi`
- GPU-capacity-aware Isaac Sim / Isaac Lab job planning
- Simulation environment profiles for CI, laptop development, L40S workstation, and production GPU nodes
- CI-friendly mock simulation path that does not require Isaac Sim or a physical robot
- Unit tests, Ruff linting, and strict mypy type checking

## Current status

MVP in progress. The first working vertical slice is live: telemetry ingestion, diagnostics, audit logging, NVIDIA GPU inventory, and GPU-aware Isaac simulation planning.

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

## CLI simulation planning examples

Plan a CI-safe mock simulation job:

```bash
robofleet simulations plan examples/simulations/isaac_warehouse_nav.yaml
