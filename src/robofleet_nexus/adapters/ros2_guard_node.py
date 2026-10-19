"""
ROS2 guard node: puts RoboFleet Guard between a learned policy and the robot.

    policy ──/policy/joint_command──▶ [ guard node ] ──/joint_command──▶ robot / Isaac Sim
                                          ▲      │
                       /joint_states ─────┘      ├──▶ /guard/status   (std_msgs/String, JSON)
                                                 └──▶ POST /telemetry (RoboFleet Nexus)

The policy publishes its proposed joint targets to ``/policy/joint_command``
instead of directly to the robot. The node checks each one against the safety
envelope and observation baseline, forwards the safe command to ``/joint_command``
(the topic Isaac Sim's ROS2 joint controller listens on by default), and reports
every intervention to RoboFleet Nexus.

After a halt the node keeps publishing the last safe command (hold position) and
ignores the policy until an operator publishes their name to ``/guard/reset``:

    ros2 topic pub --once /guard/reset std_msgs/String "{data: 'amit'}"

USAGE:
    source /opt/ros/jazzy/setup.bash
    PYTHONPATH=/path/to/robofleet-nexus/src /usr/bin/python3 -m \\
        robofleet_nexus.adapters.ros2_guard_node \\
        --robot-id arm_01 --envelope envelope.yaml --baseline baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

from robofleet_nexus.audit.hash_chain import HashChainAuditLog
from robofleet_nexus.guard.baseline import ObservationBaseline
from robofleet_nexus.guard.core import ActionGuard, GuardDecision, Verdict
from robofleet_nexus.guard.envelope import SafetyEnvelope
from robofleet_nexus.guard.events import decision_to_event

logger = logging.getLogger("robofleet.guard_node")

CLAMP_EVENT_MIN_INTERVAL_S = 1.0


def ordered_values(
    names: Sequence[str], values: Sequence[float], wanted: Sequence[str]
) -> list[float] | None:
    """
    Reorder a JointState field into envelope joint order.

    If the message carries joint names, values are matched by name and None is
    returned when any envelope joint is missing. If the message has no names, the
    values are taken positionally and must match the envelope length.
    """
    if not values:
        return None
    if names:
        by_name = {n: float(v) for n, v in zip(names, values)}
        if not all(w in by_name for w in wanted):
            return None
        return [by_name[w] for w in wanted]
    if len(values) != len(wanted):
        return None
    return [float(v) for v in values]


class TelemetryForwarder:
    """Posts guard events to RoboFleet Nexus from a background thread."""

    def __init__(self, nexus_url: str | None) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=500)
        self._last_clamp_post = 0.0
        if nexus_url:
            threading.Thread(target=self._run, args=(nexus_url,), daemon=True).start()
        self._enabled = bool(nexus_url)

    def submit(self, decision: GuardDecision) -> None:
        if not self._enabled:
            return
        if decision.verdict is Verdict.CLAMP:
            now = time.monotonic()
            if now - self._last_clamp_post < CLAMP_EVENT_MIN_INTERVAL_S:
                return  # clamps can fire every control step; send at most one per second
            self._last_clamp_post = now
        try:
            self._queue.put_nowait(decision_to_event(decision, source="ros2"))
        except queue.Full:
            logger.warning("telemetry queue full; dropping guard event")

    def _run(self, nexus_url: str) -> None:
        import httpx

        with httpx.Client(base_url=nexus_url, timeout=5.0) as client:
            while True:
                event = self._queue.get()
                try:
                    r = client.post("/telemetry", json=event)
                    if r.status_code >= 300:
                        logger.warning("POST /telemetry %s: %s", r.status_code, r.text[:120])
                except httpx.HTTPError as exc:
                    logger.debug("POST /telemetry failed: %s", exc)


def run_guard_node(args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
    except ImportError as exc:
        logger.error("ROS2 not found: %s\nRun: source /opt/ros/jazzy/setup.bash", exc)
        sys.exit(1)

    envelope = SafetyEnvelope.from_yaml(args.envelope)
    baseline = ObservationBaseline.load(args.baseline) if args.baseline else None
    forwarder = TelemetryForwarder(args.nexus_url or None)
    audit = HashChainAuditLog()
    guard = ActionGuard(
        envelope,
        robot_id=args.robot_id,
        baseline=baseline,
        audit_log=audit,
        on_intervention=forwarder.submit,
    )
    joint_names = envelope.joint_names

    class GuardNode(Node):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__("robofleet_guard")
            self._observation: list[float] | None = None
            self._effort: list[float] | None = None
            self._cmd_pub = self.create_publisher(JointState, args.command_topic, 10)
            self._status_pub = self.create_publisher(String, "/guard/status", 10)
            self.create_subscription(JointState, args.state_topic, self._on_state, 10)
            self.create_subscription(JointState, args.policy_topic, self._on_policy, 10)
            self.create_subscription(String, "/guard/reset", self._on_reset, 10)
            self.get_logger().info(
                f"RoboFleet Guard ready: robot={guard.robot_id} joints={len(joint_names)} "
                f"policy={args.policy_topic} -> command={args.command_topic} "
                f"ood={'on' if baseline else 'off'}"
            )

        def _on_state(self, msg: Any) -> None:
            self._observation = ordered_values(msg.name, msg.position, joint_names)
            self._effort = ordered_values(msg.name, msg.effort, joint_names)

        def _on_policy(self, msg: Any) -> None:
            action = ordered_values(msg.name, msg.position, joint_names)
            if action is None:
                self.get_logger().warning(
                    "policy command is missing envelope joints; treating as malformed"
                )
                action = []
            decision = guard.check(action, observation=self._observation, effort=self._effort)
            if decision.safe_action is not None:
                out = JointState()
                out.header.stamp = self.get_clock().now().to_msg()
                out.name = list(joint_names)
                out.position = [float(v) for v in decision.safe_action]
                self._cmd_pub.publish(out)
            if decision.verdict is not Verdict.PASS or decision.violations:
                self._publish_status(decision)

        def _on_reset(self, msg: Any) -> None:
            operator = str(msg.data).strip()
            try:
                guard.reset(operator=operator, note="reset via /guard/reset")
            except ValueError as exc:
                self.get_logger().error(str(exc))
                return
            self.get_logger().info(f"guard reset by operator '{operator}'")

        def _publish_status(self, decision: GuardDecision) -> None:
            status = String()
            status.data = json.dumps(
                {
                    "robot_id": decision.robot_id,
                    "step": decision.step,
                    "verdict": decision.verdict.value,
                    "latched": decision.latched,
                    "violations": [v.detail for v in decision.violations],
                }
            )
            self._status_pub.publish(status)

    rclpy.init()
    node = GuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        ok, bad = audit.verify()
        logger.info(
            "guard stats: %s | audit records=%d chain_ok=%s%s",
            guard.stats.model_dump(), len(audit.records()), ok,
            "" if ok else f" (first bad record {bad})",
        )
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    p = argparse.ArgumentParser(description="RoboFleet Guard ROS2 node")
    p.add_argument("--robot-id", default="robot_01")
    p.add_argument("--envelope", required=True, help="Safety envelope YAML")
    p.add_argument("--baseline", help="Observation baseline JSON (enables OOD checks)")
    p.add_argument("--policy-topic", default="/policy/joint_command")
    p.add_argument("--state-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/joint_command")
    p.add_argument("--nexus-url", default="http://localhost:8000",
                   help="RoboFleet Nexus API; pass '' to disable telemetry")
    run_guard_node(p.parse_args())


if __name__ == "__main__":
    main()
