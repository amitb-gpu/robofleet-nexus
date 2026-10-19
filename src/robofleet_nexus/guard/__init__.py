"""
RoboFleet Guard: runtime safety for learned robot policies.

    from robofleet_nexus.guard import ActionGuard, SafetyEnvelope, ObservationBaseline

    baseline = ObservationBaseline.from_lerobot_dataset("~/robofleet_datasets/bot_001/...")
    envelope = SafetyEnvelope.from_baseline(baseline, control_hz=30)
    guard = ActionGuard(envelope, baseline=baseline)

    decision = guard.check(policy_action, observation=joint_positions)
    send_to_robot(decision.safe_action)
"""

from robofleet_nexus.guard.baseline import ObservationBaseline
from robofleet_nexus.guard.core import (
    ActionGuard,
    GuardDecision,
    GuardStats,
    Verdict,
    Violation,
    ViolationKind,
)
from robofleet_nexus.guard.envelope import JointLimit, SafetyEnvelope, WorkspaceBox

__all__ = [
    "ActionGuard",
    "GuardDecision",
    "GuardStats",
    "JointLimit",
    "ObservationBaseline",
    "SafetyEnvelope",
    "Verdict",
    "Violation",
    "ViolationKind",
    "WorkspaceBox",
]
