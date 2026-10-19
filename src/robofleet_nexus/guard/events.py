"""
Convert guard decisions into RoboFleet Nexus telemetry events.

Posting these to ``POST /telemetry`` puts guard interventions through the same
pipeline as all other robot telemetry: the audit chain, the diagnostics rules
(which raise findings for halts and clamps), AI root-cause analysis, and the
live dashboard event stream.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Literal

from robofleet_nexus.guard.core import GuardDecision, Verdict

EVENT_TYPE = "guard_intervention"

Source = Literal["ros2", "isaac_sim", "isaac_lab", "manual", "mock"]


def _finite_or_none(values: list[float]) -> list[float | None]:
    return [v if math.isfinite(v) else None for v in values]


def decision_to_event(decision: GuardDecision, source: Source = "ros2") -> dict[str, Any]:
    """Build a RobotEvent payload (as JSON-ready dict) for one guard decision."""
    severity = {
        Verdict.HALT: "critical",
        Verdict.CLAMP: "warning",
        Verdict.PASS: "info",
    }[decision.verdict]

    kinds = sorted({v.kind.value for v in decision.violations})
    summary = "; ".join(v.detail for v in decision.violations[:3]) or "no violations"
    message = f"Guard {decision.verdict.value.upper()} at step {decision.step}: {summary}"

    metrics: dict[str, float] = {
        "guard.halt": 1.0 if decision.verdict is Verdict.HALT else 0.0,
        "guard.clamp": 1.0 if decision.verdict is Verdict.CLAMP else 0.0,
        "guard.violations": float(len(decision.violations)),
        "guard.step": float(decision.step),
    }
    if decision.ood_score is not None:
        metrics["guard.ood_score"] = round(decision.ood_score, 3)

    return {
        "event_id": str(uuid.uuid4()),
        "robot_id": decision.robot_id,
        "source": source,
        "timestamp": decision.timestamp.isoformat(),
        "severity": severity,
        "event_type": EVENT_TYPE,
        "subsystem": "safety",
        "message": message[:500],
        "metrics": metrics,
        "metadata": {
            "verdict": decision.verdict.value,
            "violation_kinds": kinds,
            "violations": [v.model_dump(mode="json") for v in decision.violations],
            # JSON has no NaN/inf; a non-finite policy output is reported as null.
            "proposed_action": _finite_or_none(decision.proposed_action),
            "safe_action": (
                _finite_or_none(decision.safe_action)
                if decision.safe_action is not None else None
            ),
            "latched": decision.latched,
        },
    }


RESET_EVENT_TYPE = "guard_reset"


def reset_to_event(
    robot_id: str, operator: str, resets: int, source: Source = "ros2"
) -> dict[str, Any]:
    """Build a RobotEvent payload recording an operator reset of a halted guard."""
    return {
        "event_id": str(uuid.uuid4()),
        "robot_id": robot_id,
        "source": source,
        "severity": "info",
        "event_type": RESET_EVENT_TYPE,
        "subsystem": "safety",
        "message": f"Guard reset by operator '{operator}'; policy resumed",
        "metrics": {"guard.resets": float(resets)},
        "metadata": {"verdict": "reset", "operator": operator, "latched": False},
    }
