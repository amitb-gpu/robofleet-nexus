# NVIDIA Isaac Architecture

RoboFleet Nexus is designed as an orchestration and observability layer above NVIDIA robotics infrastructure. It does not attempt to replace Isaac Sim, Isaac Lab, ROS2, or Kubernetes GPU scheduling. Instead, it provides a production-style control plane for planning, validating, auditing, and eventually dispatching robotics simulation and diagnostic workflows.

## Positioning

RoboFleet Nexus sits above the robotics runtime stack:

```text
RoboFleet Nexus
  ├── telemetry ingestion
  ├── diagnostics and RCA
  ├── policy-gated agentic orchestration
  ├── GPU-aware simulation planning
  ├── audit trails
  └── future Kubernetes scheduling

NVIDIA / Robotics Backends
  ├── Isaac Sim
  ├── Isaac Lab
  ├── Isaac ROS / ROS2
  ├── Replicator / synthetic data generation
  └── NVIDIA GPU infrastructure

Isaac Sim remains the simulation backend. Isaac Lab remains the robot learning and experiment framework. ROS2 remains the robot middleware. RoboFleet Nexus is the orchestration layer that reasons across them.

Design Principle

The platform follows one core rule:

The orchestrator may plan, validate, schedule, audit, and recommend. It must not directly bypass simulator, ROS2, or safety-control boundaries.

This is especially important for agentic workflows. LLM-assisted diagnostics may produce structured recommendations, but physical robot commands remain policy-gated and human-reviewable.

NVIDIA Components
Isaac Sim

Isaac Sim is the primary simulation backend target. RoboFleet Nexus treats Isaac Sim jobs as schedulable workloads:

simulation job spec
→ GPU inventory check
→ safety policy check
→ launch plan
→ audit record
→ future dispatch

The current implementation produces a dry-run launch plan instead of directly launching Isaac Sim.

Isaac Lab

Isaac Lab is treated as a robot learning and experiment-management backend. It is useful for reinforcement learning, imitation learning, motion planning, and sim-to-real workflows.

RoboFleet Nexus does not implement robot learning itself. Instead, it provides orchestration primitives for:

experiment job definitions
resource requirements
run counts
random seeds
domain-randomization settings
output collection
auditability
Isaac ROS / ROS2

ROS2 remains the middleware layer for robot communication. RoboFleet Nexus should integrate through ROS2 adapters rather than bypassing ROS2 semantics.

Planned ROS2 integration targets:

topics for telemetry streams
services for request/response diagnostics
actions for long-running robot or simulation tasks
parameters for configuration state
rosbag metadata for replay and failure analysis

Isaac Sim's ROS2 bridge is the expected simulation-to-ROS integration path.

Replicator and Synthetic Data

Isaac Sim Replicator and related synthetic-data generation workflows are future orchestration targets.

RoboFleet Nexus should model synthetic-data jobs using structured fields:

scenario
robot_model
sensor_configuration
domain_randomization
annotation_outputs
num_frames
random_seed
storage_target
gpu_requirements

The orchestrator should track synthetic-data runs as first-class jobs rather than informal scripts.

NVIDIA GPU Infrastructure

The current implementation detects NVIDIA GPUs using nvidia-smi and exposes GPU inventory through the API.

Current endpoint:

GET /gpu/inventory

Example result from a developer laptop:

{
  "available": true,
  "source": "nvidia-smi",
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA RTX A1000 6GB Laptop GPU",
      "memory_total_mb": 6144,
      "memory_used_mb": 4554,
      "utilization_gpu_pct": 1.0,
      "temperature_c": 45.0
    }
  ],
  "warnings": []
}

This allows the scheduler to reject oversized simulation workloads before launch.

Example:

requested Isaac Sim job: min_vram_gb=16
available laptop GPU:   6 GB
scheduler decision:     blocked
reason:                 insufficient eligible NVIDIA GPU capacity

This is intentional. The platform should make resource constraints explicit and auditable.

Environment Profiles

RoboFleet Nexus models different execution environments explicitly:

Profile	Purpose	Isaac Sim	Isaac Lab	Synthetic Data
ci_mock	GitHub Actions and unit tests	No	No	No
laptop_dev	API development and lightweight GPU checks	No	No	No
workstation_l40s	Single-node NVIDIA L40S simulation workstation	Yes	Yes	Yes
production_gpu_node	Future Kubernetes GPU node	Yes	Yes	Yes

This prevents the common mistake of treating a laptop, CI runner, workstation, and production GPU cluster as equivalent.

Current API Surface
GPU Inventory
GET /gpu/inventory

Purpose:

detect local NVIDIA GPUs
expose memory, utilization, and temperature
support scheduling decisions
provide infrastructure observability
Simulation Profiles
GET /simulations/profiles

Purpose:

describe supported execution targets
distinguish CI, laptop, workstation, and production GPU environments
make simulation assumptions explicit
Simulation Planning
POST /simulations/plan

Purpose:

validate a proposed simulation job
enforce safety policy
check GPU eligibility
produce a dry-run launch command
record the decision in the audit log
Scheduling Model

The scheduler is intentionally conservative.

A simulation job is accepted only if:

The safety policy does not request physical robot command paths.
The backend is supported.
GPU inventory is available when required.
Enough eligible GPUs meet the requested VRAM threshold.
The job can be represented as an auditable launch plan.

The scheduler currently plans but does not execute. That separation is deliberate.

plan first
audit always
execute later
Safety Boundaries

RoboFleet Nexus should never let an agentic planner directly issue physical robot commands.

Blocked by policy:

physical robot command path enabled from simulation job
safety override requests
direct actuator commands
unreviewed restart of robot-control services
unbounded simulation launch loops

Allowed without human approval:

read-only telemetry inspection
GPU inventory checks
simulation dry-run planning
diagnostic finding creation
audit-log reads
mock simulation planning

Conditionally allowed:

Isaac Sim job launch
Isaac Lab experiment launch
Kubernetes job creation
ROS2 node restart
simulation batch cancellation

These should require explicit policy rules and audit records.

Future Kubernetes Path

The production path should use Kubernetes GPU scheduling rather than local ad-hoc launch scripts.

Future targets:

NVIDIA Container Toolkit
NVIDIA GPU Operator
NVIDIA device plugin
DCGM Exporter
Prometheus/Grafana metrics
Kubernetes Jobs for simulation runs
node labels for GPU type and capacity
storage-backed artifact collection

In this model, RoboFleet Nexus becomes the control plane that decides whether a simulation job should be admitted and where it should run.

Roadmap
Phase 1 — Current
FastAPI control plane
telemetry ingestion
diagnostic findings
tamper-evident audit log
NVIDIA GPU inventory
Isaac Sim / Isaac Lab job planning
simulation profiles
CI checks
Phase 2 — Isaac Dry-Run Examples
checked-in Isaac simulation job examples
CLI command for planning simulation jobs
JSON/YAML job loading
mock simulation run output
audit records for simulation planning
Phase 3 — Local Isaac Integration
optional local Isaac Sim path detection
explicit headless launch configuration
dry-run versus execute mode
result directory tracking
simulation run registry
Phase 4 — ROS2 / Isaac Bridge Integration
ROS2 topic health adapter
Isaac Sim ROS2 bridge readiness checks
rosbag metadata ingestion
simulated robot telemetry replay
failure correlation across ROS2 and simulation events
Phase 5 — Kubernetes GPU Scheduling
Kubernetes Job manifest generation
GPU resource requests
NVIDIA GPU Operator assumptions
DCGM metrics integration
simulation queue and admission control
Non-Goals

RoboFleet Nexus does not attempt to:

replace Isaac Sim physics
replace Isaac Lab learning workflows
replace ROS2
directly control physical robots without policy approval
implement a full simulator
hide GPU/resource constraints
make LLMs autonomous robot operators
Summary

RoboFleet Nexus is an NVIDIA-aware robotics orchestration layer.

It provides the missing production control-plane logic around simulation jobs, GPU resources, robot telemetry, diagnostics, policy, and auditability.

The long-term goal is not to compete with Isaac. The goal is to make Isaac-centered robotics workflows more observable, schedulable, auditable, and production-ready.
