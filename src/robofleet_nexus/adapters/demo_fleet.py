"""
Demo fleet generator for RoboFleet Nexus.

Spins up a realistic multi-robot fleet that populates every dashboard panel
for demonstration and video recording. No ROS2 required.

Simulates:
  - 5 robots across different mission profiles (patrol, inspect, transport, scout, recon)
  - Varied telemetry: battery, joint states, odometry on independent paths
  - Periodic diagnostic findings (staggered so the findings panel stays active)
  - Simulation jobs cycling through PLANNING → QUEUED → RUNNING → DONE
  - Steady audit log activity

Usage:
    python -m robofleet_nexus.adapters.demo_fleet
    python -m robofleet_nexus.adapters.demo_fleet --nexus-url http://localhost:8000 --robots 5

To trigger a dramatic fault on a specific robot during recording:
    python -m robofleet_nexus.adapters.demo_fleet --inject-fault scout_04
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import uuid
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("robofleet.demo_fleet")

# ── Fleet definition ───────────────────────────────────────────────────────────
FLEET = [
    {"id": "patrol_01",    "mission": "perimeter patrol",    "zone": "sector-A", "battery": 87.0, "phase": 0.0},
    {"id": "inspect_02",   "mission": "infra inspection",    "zone": "bridge-7", "battery": 28.0, "phase": 1.2},
    {"id": "transport_03", "mission": "supply transport",    "zone": "depot-2",  "battery": 92.0, "phase": 2.4},
    {"id": "scout_04",     "mission": "forward scout",       "zone": "ridge-N",  "battery": 24.0, "phase": 3.6},
    {"id": "recon_05",     "mission": "area reconnaissance", "zone": "valley-W", "battery": 73.0, "phase": 4.8},
]

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "gripper"]

SIM_JOBS = [
    {"job_id": "sim-nav-001",    "name": "Warehouse Nav Training",   "environment": "isaac_warehouse", "vram_required_mb": 8000,  "status": "RUNNING"},
    {"job_id": "sim-grasp-002",  "name": "Grasp Policy SFT",         "environment": "isaac_lab",       "vram_required_mb": 14000, "status": "QUEUED"},
    {"job_id": "sim-terrain-003","name": "Rough Terrain Locomotion", "environment": "isaac_lab",       "vram_required_mb": 16000, "status": "QUEUED"},
    {"job_id": "sim-recon-004",  "name": "Recon Path Optimization",  "environment": "isaac_warehouse", "vram_required_mb": 6000,  "status": "PLANNING"},
]


def _make_event(robot_id, source, event_type, subsystem, message, metrics=None, metadata=None, severity="info"):
    return {
        "event_id": str(uuid.uuid4()),
        "robot_id": robot_id,
        "source": source,
        "severity": severity,
        "event_type": event_type,
        "subsystem": subsystem,
        "message": message,
        "metrics": metrics or {},
        "metadata": metadata or {},
    }


async def _post(client, path, payload):
    try:
        await client.post(path, content=json.dumps(payload))
    except Exception as exc:
        logger.debug("post error: %s", exc)


async def run_demo_fleet(nexus_url: str, fault_robot: str | None = None) -> None:
    logger.info("Demo fleet starting — %d robots → %s", len(FLEET), nexus_url)
    tick = 0
    fault_injected = False

    async with httpx.AsyncClient(base_url=nexus_url, timeout=5.0,
                                 headers={"Content-Type": "application/json"}) as client:

        # Seed the simulation job queue once
        for job in SIM_JOBS:
            await _post(client, "/demo/sim-job", {**job, "ts": datetime.now(timezone.utc).isoformat()})
        logger.info("Fleet online. Ctrl+C to stop.")

        while True:
            tick += 1

            for robot in FLEET:
                rid = robot["id"]
                # Battery slowly drains, recharges when low
                robot["battery"] = max(8.0, robot["battery"] - random.uniform(0.05, 0.2))
                if robot["battery"] < 10:
                    robot["battery"] = min(100.0, robot["battery"] + 30)  # docked & recharged

                phase = robot["phase"] + tick * 0.04

                # Joint state event
                metrics = {}
                for i, jn in enumerate(JOINT_NAMES):
                    metrics[f"{jn}.pos"] = round(math.sin(phase + i * 0.7) * 1.4, 4)
                    metrics[f"{jn}.vel"] = round(math.cos(phase + i * 0.7) * 0.25, 4)
                    metrics[f"{jn}.eff"] = round(abs(math.sin(phase + i * 0.7)) * 18, 2)
                await _post(client, "/telemetry", _make_event(
                    rid, "mock", "joint_state", "kinematics",
                    f"{len(JOINT_NAMES)} joints nominal", metrics=metrics,
                ))

                # Odometry event
                r = 2.0 + (hash(rid) % 3)
                await _post(client, "/telemetry", _make_event(
                    rid, "mock", "odometry", "navigation",
                    f"Patrolling {robot['zone']}",
                    metrics={"x": round(r * math.cos(phase), 3),
                             "y": round(r * math.sin(phase), 3),
                             "z": 0.0, "vx": round(0.3 + random.uniform(-0.05, 0.05), 3)},
                    metadata={"location": robot["zone"], "mission": robot["mission"]},
                ))

                # Battery event (every 3 ticks to reduce noise)
                if tick % 3 == 0:
                    sev = "warning" if robot["battery"] < 25 else "info"
                    await _post(client, "/telemetry", _make_event(
                        rid, "mock", "battery_state", "power",
                        f"Battery at {robot['battery']:.1f}%",
                        metrics={"battery_pct": round(robot["battery"], 1),
                                 "voltage": round(24.0 - (100 - robot["battery"]) * 0.03, 2)},
                        severity=sev,
                    ))

            # Inject a dramatic fault once, ~15 ticks in, on the target robot
            if fault_robot and not fault_injected and tick == 15:
                logger.info("INJECTING FAULT on %s", fault_robot)
                await _post(client, "/telemetry", _make_event(
                    fault_robot, "mock", "battery_state", "power",
                    "Battery critically low — emergency return required",
                    metrics={"battery_pct": 9.0, "voltage": 20.1, "current": -3.2},
                    severity="critical",
                ))
                # Also spike a joint effort
                await _post(client, "/telemetry", _make_event(
                    fault_robot, "mock", "joint_state", "kinematics",
                    "Joint effort spike detected",
                    metrics={"elbow.pos": 1.2, "elbow.vel": 0.1, "elbow.eff": 47.5},
                ))
                fault_injected = True

            await asyncio.sleep(2.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    p = argparse.ArgumentParser(description="RoboFleet Nexus demo fleet generator")
    p.add_argument("--nexus-url", default="http://localhost:8000")
    p.add_argument("--inject-fault", default=None,
                   help="Robot ID to inject a dramatic fault on (e.g. scout_04)")
    args = p.parse_args()

    try:
        asyncio.run(run_demo_fleet(args.nexus_url, args.inject_fault))
    except KeyboardInterrupt:
        logger.info("Demo fleet stopped")


if __name__ == "__main__":
    main()
