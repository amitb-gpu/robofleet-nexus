"""
ROS2 bridge adapter for RoboFleet Nexus.

USAGE:
    # Mock mode (no ROS2 needed — WSL dev):
    python -m robofleet_nexus.adapters.ros2_bridge --mock --robot-id bot_001

    # Real mode (ROS2 must be sourced):
    source /opt/ros/humble/setup.bash
    python -m robofleet_nexus.adapters.ros2_bridge --robot-id bot_001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import uuid
from typing import Any, Optional

import httpx

logger = logging.getLogger("robofleet.ros2_bridge")


# ── RobotEvent builder ────────────────────────────────────────────────────────

def _make_event(
    robot_id: str,
    source: str,
    event_type: str,
    subsystem: str,
    message: str,
    metrics: Optional[dict[str, float]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "robot_id": robot_id,
        "source": source,
        "event_type": event_type,
        "subsystem": subsystem,
        "message": message,
        "metrics": metrics or {},
        "metadata": metadata or {},
    }


# ── ROS2 message converters ───────────────────────────────────────────────────

def _battery_to_event(robot_id: str, msg: Any) -> dict[str, Any]:
    pct = round(float(msg.percentage) * 100, 1)
    sev = "warning" if pct < 20 else "info"
    return _make_event(
        robot_id, "ros2", "battery_state", "power",
        f"Battery at {pct}%",
        metrics={"battery_pct": pct, "voltage": float(msg.voltage),
                 "current": float(msg.current)},
        metadata={"power_supply_status": int(msg.power_supply_status),
                  "present": bool(msg.present), "severity": sev},
    )


def _diagnostics_to_events(robot_id: str, msg: Any) -> list[dict[str, Any]]:
    level_map = {0: "info", 1: "warning", 2: "critical", 3: "warning"}
    events = []
    for s in msg.status:
        sev = level_map.get(int(s.level), "info")
        events.append(_make_event(
            robot_id, "ros2", "diagnostics", s.name or "diagnostics",
            s.message or "(no message)",
            metadata={"hardware_id": s.hardware_id, "severity": sev,
                      "values": {kv.key: kv.value for kv in s.values}},
        ))
    return events


def _joint_state_to_event(robot_id: str, msg: Any) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    for i, name in enumerate(msg.name):
        if i < len(msg.position):
            metrics[f"{name}.pos"] = round(float(msg.position[i]), 4)
        if i < len(msg.velocity):
            metrics[f"{name}.vel"] = round(float(msg.velocity[i]), 4)
        if i < len(msg.effort):
            metrics[f"{name}.eff"] = round(float(msg.effort[i]), 4)
    return _make_event(
        robot_id, "ros2", "joint_state", "kinematics",
        f"{len(msg.name)} joint(s) reported",
        metrics=metrics,
    )


def _odom_to_event(robot_id: str, msg: Any) -> dict[str, Any]:
    p = msg.pose.pose.position
    return _make_event(
        robot_id, "ros2", "odometry", "navigation",
        f"Position ({p.x:.2f}, {p.y:.2f})",
        metrics={"x": round(float(p.x), 4), "y": round(float(p.y), 4),
                 "z": round(float(p.z), 4),
                 "vx": round(float(msg.twist.twist.linear.x), 4)},
    )


# ── Nexus HTTP client ─────────────────────────────────────────────────────────

class NexusClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "NexusClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=5.0,
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def post(self, event: dict[str, Any]) -> None:
        assert self._client is not None
        try:
            r = await self._client.post("/telemetry", content=json.dumps(event))
            if r.status_code not in (200, 201, 204):
                logger.warning("POST /telemetry %s: %s", r.status_code, r.text[:120])
        except httpx.ConnectError:
            logger.warning("Cannot reach Nexus at %s — server running?", self.base_url)
        except Exception as exc:
            logger.debug("Post error: %s", exc)


# ── Mock publisher ────────────────────────────────────────────────────────────

async def _mock_publish_loop(nexus_url: str, robot_id: str) -> None:
    logger.info("MOCK MODE: robot=%s → %s", robot_id, nexus_url)
    battery = 100.0
    tick = 0

    async with NexusClient(nexus_url) as client:
        while True:
            tick += 1
            battery = max(20.0, battery - 0.15)

            await client.post(_make_event(
                robot_id, "mock", "battery_state", "power",
                f"Battery at {battery:.1f}%",
                metrics={"battery_pct": round(battery, 1),
                         "voltage": round(24.0 - (100.0 - battery) * 0.04, 2),
                         "current": -1.5 if battery > 21 else 2.0},
            ))

            metrics: dict[str, float] = {}
            for i in range(6):
                metrics[f"joint_{i}.pos"] = round(math.sin(tick * 0.05 + i * 0.8) * 1.5, 4)
                metrics[f"joint_{i}.vel"] = round(math.cos(tick * 0.05 + i * 0.8) * 0.3, 4)
            await client.post(_make_event(
                robot_id, "mock", "joint_state", "kinematics",
                "6 joint(s) reported", metrics=metrics,
            ))

            angle = tick * 0.02
            await client.post(_make_event(
                robot_id, "mock", "odometry", "navigation",
                f"Position ({2*math.cos(angle):.2f}, {2*math.sin(angle):.2f})",
                metrics={"x": round(2 * math.cos(angle), 4),
                         "y": round(2 * math.sin(angle), 4), "z": 0.0, "vx": 0.3},
                metadata={"location": f"zone-{(tick // 100) % 4 + 1}"},
            ))

            if battery < 30:
                await client.post(_make_event(
                    robot_id, "mock", "battery_low", "power",
                    f"Battery at {battery:.1f}% — return to charging station",
                    metrics={"battery_pct": round(battery, 1)},
                    metadata={"severity": "warning"},
                ))

            await asyncio.sleep(3.0)


# ── Real ROS2 bridge ──────────────────────────────────────────────────────────

def build_ros2_bridge(robot_id: str, nexus_url: str,
                      topics: Optional[list[str]] = None) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import BatteryState, JointState
        from diagnostic_msgs.msg import DiagnosticArray
        from nav_msgs.msg import Odometry
    except ImportError as exc:
        logger.error("rclpy not found: %s\nRun: source /opt/ros/humble/setup.bash", exc)
        sys.exit(1)

    import queue
    import threading

    event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=500)

    def _flush_thread() -> None:
        import httpx as _httpx
        with _httpx.Client(base_url=nexus_url, timeout=5.0,
                           headers={"Content-Type": "application/json"}) as client:
            while True:
                try:
                    item = event_queue.get(timeout=1.0)
                    items = [item]
                    while not event_queue.empty() and len(items) < 20:
                        items.append(event_queue.get_nowait())
                    for it in items:
                        try:
                            r = client.post("/telemetry", content=json.dumps(it))
                            if r.status_code not in (200, 201, 204):
                                logger.warning("POST %s: %s", r.status_code, r.text[:80])
                        except Exception as exc:
                            logger.debug("Post error: %s", exc)
                except Exception:
                    pass  # queue.Empty — keep looping

    threading.Thread(target=_flush_thread, daemon=True).start()

    class NexusBridgeNode(Node):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__("nexus_bridge")
            self._subs = []
            subscribe = self.create_subscription

            def enqueue(e: dict) -> None:
                try:
                    event_queue.put_nowait(e)
                except Exception:
                    pass

            if not topics or "/battery_state" in topics:
                self._subs.append(subscribe(BatteryState, "/battery_state",
                    lambda m: enqueue(_battery_to_event(robot_id, m)), 10))

            if not topics or "/diagnostics" in topics:
                self._subs.append(subscribe(DiagnosticArray, "/diagnostics",
                    lambda m: [enqueue(e) for e in _diagnostics_to_events(robot_id, m)], 10))

            if not topics or "/joint_states" in topics:
                self._subs.append(subscribe(JointState, "/joint_states",
                    lambda m: enqueue(_joint_state_to_event(robot_id, m)), 10))

            if not topics or "/odom" in topics:
                self._subs.append(subscribe(Odometry, "/odom",
                    lambda m: enqueue(_odom_to_event(robot_id, m)), 10))

            self.get_logger().info(
                f"NexusBridgeNode ready — robot={robot_id}  nexus={nexus_url}  "
                f"subs={len(self._subs)}")

    rclpy.init()
    node = NexusBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    p = argparse.ArgumentParser(description="RoboFleet Nexus ROS2 bridge")
    p.add_argument("--robot-id",  default="robot_01")
    p.add_argument("--nexus-url", default="http://localhost:8000")
    p.add_argument("--topics",    nargs="*")
    p.add_argument("--mock",      action="store_true")
    args = p.parse_args()

    if args.mock:
        try:
            asyncio.run(_mock_publish_loop(args.nexus_url, args.robot_id))
        except KeyboardInterrupt:
            logger.info("Mock bridge stopped")
    else:
        build_ros2_bridge(args.robot_id, args.nexus_url, args.topics)


if __name__ == "__main__":
    main()
