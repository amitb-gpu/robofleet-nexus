"""
Fake 6-joint arm and scripted policy for testing RoboFleet Guard over ROS2.

No Isaac Sim, GPU, or trained policy needed. It plays two roles:

  robot   publishes /joint_states and follows whatever arrives on /joint_command
  policy  publishes /policy/joint_command along a nominal pick-and-place motion,
          with faults injected on a fixed schedule

so the full loop runs through the real guard node:

    fake policy ──/policy/joint_command──▶ guard node ──/joint_command──▶ fake robot
         ▲                                     │                              │
         └──── /guard/status, /guard/reset ────┘◀──────── /joint_states ──────┘

Fault schedule (one cycle = 16 s at 30 Hz, repeats):
    3.0 s   action spike on elbow          -> CLAMP
    6.0 s   wrist_1 driven past its limit  -> CLAMP
    9.0 s   wrist_3 encoder slips +1.2 rad -> HALT (out of distribution)
    13.0 s  NaN in policy output           -> HALT (non-finite action)

While the guard is latched the policy pauses, as a real controller would. With
--auto-reset N the script plays the operator and publishes a reset N seconds after
each halt; with --auto-reset 0 reset manually:

    ros2 topic pub --once /guard/reset std_msgs/String "{data: 'your-name'}"

USAGE (after `robofleet guard demo` has written guard_demo/envelope.yaml):
    terminal 1: python3 -m robofleet_nexus.adapters.ros2_guard_node \\
                    --robot-id arm_demo --envelope guard_demo/envelope.yaml \\
                    --baseline guard_demo/baseline.json
    terminal 2: python3 -m robofleet_nexus.adapters.ros2_fake_arm
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from typing import Any

import numpy as np

from robofleet_nexus.guard.scenario import HZ, JOINTS, nominal

logger = logging.getLogger("robofleet.fake_arm")

CYCLE_STEPS = int(16 * HZ)
SPIKE_STEP = int(3 * HZ)
DRIFT_STEPS = range(int(6 * HZ), int(6 * HZ) + 5)
SLIP_STEPS = range(int(9 * HZ), int(9 * HZ) + 15)
NAN_STEP = int(13 * HZ)

ELBOW, WRIST_1, WRIST_2, WRIST_3 = 2, 3, 4, 5


def scripted_action(step: int, policy_t: int) -> tuple[list[float], str | None]:
    """Policy output for this tick, and a label if a fault was injected."""
    s = step % CYCLE_STEPS
    action = nominal(policy_t / HZ)
    label = None
    if s == SPIKE_STEP:
        action[ELBOW] += 0.8
        label = "FAULT: one-step spike on elbow (+0.8 rad)"
    elif s in DRIFT_STEPS:
        action[WRIST_1] = 0.4
        label = "FAULT: policy driving wrist_1 past its limit" if s == DRIFT_STEPS[0] else ""
    elif s == NAN_STEP:
        action[WRIST_2] = math.nan
        label = "FAULT: NaN in policy output (wrist_2)"
    return action.tolist(), label


def encoder_slip(step: int) -> float:
    return 1.2 if step % CYCLE_STEPS in SLIP_STEPS else 0.0


class GuardStatusTracker:
    """
    Tracks the guard's state from /guard/status and /guard/reset for the fake policy.

    The guard publishes a status on every held step while latched, so a status
    published just before a reset can arrive just after it. Each status carries
    the guard's reset count; anything older than the resets seen here is stale.
    """

    def __init__(self) -> None:
        self.paused = False
        self.resets_seen = 0
        self._last_clamp_step = -2

    def on_status(self, status: dict[str, Any]) -> tuple[str, str] | None:
        """Update state; return (level, message) to log, or None."""
        if int(status.get("resets", 0)) < self.resets_seen:
            return None
        verdict = status.get("verdict", "")
        reason = (status.get("violations") or ["?"])[0]
        if status.get("latched"):
            if self.paused:
                return None
            self.paused = True
            return "error", f"GUARD HALT: {reason}; policy paused, holding position"
        if verdict == "clamp":
            step = int(status.get("step", 0))
            first_of_run = step != self._last_clamp_step + 1
            self._last_clamp_step = step
            if first_of_run:
                return "warning", f"GUARD CLAMP: {reason}"
        return None

    def on_reset(self, operator: str) -> str | None:
        if not operator.strip():
            return None
        self.resets_seen += 1
        was_paused, self.paused = self.paused, False
        return f"Guard reset by '{operator}'; policy resuming" if was_paused else None


def run_fake_arm(args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
    except ImportError as exc:
        logger.error("ROS2 not found: %s\nRun: source /opt/ros/jazzy/setup.bash", exc)
        sys.exit(1)

    class FakeArm(Node):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__("robofleet_fake_arm")
            self._state = nominal(0.0)
            self._step = 0
            self._policy_t = 0
            self._tracker = GuardStatusTracker()
            self._halted_at: float | None = None

            self._state_pub = self.create_publisher(JointState, args.state_topic, 10)
            self._policy_pub = self.create_publisher(JointState, args.policy_topic, 10)
            self._reset_pub = self.create_publisher(String, "/guard/reset", 10)
            self.create_subscription(JointState, args.command_topic, self._on_command, 10)
            self.create_subscription(String, "/guard/status", self._on_status, 10)
            self.create_subscription(String, "/guard/reset", self._on_reset, 10)
            self.create_timer(1.0 / HZ, self._tick)
            self.get_logger().info(
                f"Fake arm running at {HZ:.0f} Hz; 16 s fault cycle; "
                f"auto-reset={'off' if args.auto_reset <= 0 else f'{args.auto_reset:.0f}s'}"
            )

        # robot side ──────────────────────────────────────────────────────
        def _on_command(self, msg: Any) -> None:
            by_name = dict(zip(msg.name, msg.position))
            if all(j in by_name for j in JOINTS):
                self._state = np.array([by_name[j] for j in JOINTS], dtype=np.float64)

        def _publish_state(self) -> None:
            reported = self._state.copy()
            slip = encoder_slip(self._step)
            reported[WRIST_3] += slip
            if slip and self._step % CYCLE_STEPS == SLIP_STEPS[0]:
                self.get_logger().warning("FAULT: wrist_3 encoder slipped (+1.2 rad)")
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = list(JOINTS)
            out.position = [float(v) for v in reported]
            self._state_pub.publish(out)

        # policy side ─────────────────────────────────────────────────────
        def _tick(self) -> None:
            self._publish_state()
            action, label = scripted_action(self._step, self._policy_t)
            if label:
                self.get_logger().warning(label)
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = list(JOINTS)
            out.position = action
            self._policy_pub.publish(out)

            self._step += 1
            if not self._tracker.paused:
                self._policy_t += 1
            self._maybe_auto_reset()

        # guard side ──────────────────────────────────────────────────────
        def _on_status(self, msg: Any) -> None:
            try:
                status = json.loads(msg.data)
            except json.JSONDecodeError:
                return
            event = self._tracker.on_status(status)
            if event is None:
                return
            level, text = event
            if level == "error":
                self._halted_at = time.monotonic()
                self.get_logger().error(text)
            else:
                self.get_logger().warning(text)

        def _on_reset(self, msg: Any) -> None:
            text = self._tracker.on_reset(str(msg.data))
            self._halted_at = None
            if text:
                self.get_logger().info(text)

        def _maybe_auto_reset(self) -> None:
            if args.auto_reset <= 0 or self._halted_at is None:
                return
            if time.monotonic() - self._halted_at >= args.auto_reset:
                self._halted_at = None
                reset = String()
                reset.data = "demo-operator"
                self._reset_pub.publish(reset)

    rclpy.init()
    node = FakeArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    p = argparse.ArgumentParser(description="Fake arm + scripted policy for RoboFleet Guard")
    p.add_argument("--policy-topic", default="/policy/joint_command")
    p.add_argument("--state-topic", default="/joint_states")
    p.add_argument("--command-topic", default="/joint_command")
    p.add_argument("--auto-reset", type=float, default=3.0,
                   help="Seconds after a halt before publishing an operator reset (0 = never)")
    run_fake_arm(p.parse_args())


if __name__ == "__main__":
    main()
