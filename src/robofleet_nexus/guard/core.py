"""
ActionGuard: the runtime safety layer between a learned policy and the robot.

Every control step, the policy proposes an action and the guard returns one of:

  PASS   the action is inside the envelope and is forwarded unchanged
  CLAMP  the action broke a soft limit (position range, rate limit); a corrected
         action inside the envelope is forwarded instead
  HALT   the step is unsafe to act on (non-finite output, effort limit, workspace
         breach, out-of-distribution observation, or repeated clamping); the robot
         holds its last safe command and the guard latches until an operator resets it

Every CLAMP, HALT and reset is written to the tamper-evident audit log, so a
safety reviewer can see exactly what the policy proposed, what the guard did, and why.
The guard never generates motion of its own: it only passes, narrows, or stops what
the policy proposed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from robofleet_nexus.audit.hash_chain import HashChainAuditLog
from robofleet_nexus.guard.baseline import ObservationBaseline
from robofleet_nexus.guard.envelope import SafetyEnvelope


class Verdict(str, Enum):
    PASS = "pass"
    CLAMP = "clamp"
    HALT = "halt"


class ViolationKind(str, Enum):
    NON_FINITE_ACTION = "non_finite_action"
    ACTION_SHAPE = "action_shape"
    POSITION_LIMIT = "position_limit"
    RATE_LIMIT = "rate_limit"
    EFFORT_LIMIT = "effort_limit"
    WORKSPACE = "workspace"
    OBSERVATION_OOD = "observation_ood"
    CLAMP_ESCALATION = "clamp_escalation"
    LATCHED = "latched"


HALTING_KINDS = {
    ViolationKind.NON_FINITE_ACTION,
    ViolationKind.ACTION_SHAPE,
    ViolationKind.EFFORT_LIMIT,
    ViolationKind.WORKSPACE,
    ViolationKind.CLAMP_ESCALATION,
    ViolationKind.LATCHED,
}


class Violation(BaseModel):
    kind: ViolationKind
    detail: str
    joint: str | None = None
    value: float | None = None
    limit: float | None = None


class GuardDecision(BaseModel):
    robot_id: str
    step: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verdict: Verdict
    proposed_action: list[float]
    safe_action: list[float] | None = Field(
        description="Command to send to the robot. On HALT this holds the last safe command."
    )
    violations: list[Violation] = Field(default_factory=list)
    ood_score: float | None = None
    latched: bool = False

    @property
    def intervened(self) -> bool:
        return self.verdict is not Verdict.PASS


class GuardStats(BaseModel):
    steps: int = 0
    passed: int = 0
    clamped: int = 0
    halted: int = 0
    resets: int = 0
    violations_by_kind: dict[str, int] = Field(default_factory=dict)


DecisionSink = Callable[[GuardDecision], None]


def _json_safe(value: Any) -> Any:
    """Replace NaN/inf with None so audit payloads survive a JSON round trip unchanged."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


class ActionGuard:
    """Checks a learned policy's actions for one robot against a safety envelope."""

    def __init__(
        self,
        envelope: SafetyEnvelope,
        robot_id: str | None = None,
        baseline: ObservationBaseline | None = None,
        audit_log: HashChainAuditLog | None = None,
        on_intervention: DecisionSink | None = None,
    ) -> None:
        if baseline is not None and baseline.dim != len(envelope.joints):
            raise ValueError(
                f"baseline has {baseline.dim} dimensions but envelope has "
                f"{len(envelope.joints)} joints"
            )
        self.envelope = envelope
        self.robot_id = robot_id or envelope.robot_id or "robot"
        self.baseline = baseline
        self.audit_log = audit_log
        self.on_intervention = on_intervention
        self.stats = GuardStats()

        self._step = 0
        self._last_safe: list[float] | None = None
        self._consecutive_clamps = 0
        self._latched_reason: str | None = None

    # ── State ─────────────────────────────────────────────────────────────

    @property
    def latched(self) -> bool:
        return self._latched_reason is not None

    @property
    def latched_reason(self) -> str | None:
        return self._latched_reason

    def reset(self, operator: str, note: str = "") -> None:
        """Clear a latched halt. Requires a named operator; the reset is audited."""
        if not operator.strip():
            raise ValueError("an operator name is required to reset a halted guard")
        previous = self._latched_reason
        self._latched_reason = None
        self._consecutive_clamps = 0
        self.stats.resets += 1
        if self.audit_log is not None:
            self.audit_log.append(
                event_type="guard.reset",
                actor=f"operator:{operator}",
                payload={
                    "robot_id": self.robot_id,
                    "step": self._step,
                    "cleared_reason": previous,
                    "note": note,
                },
            )

    # ── The check ─────────────────────────────────────────────────────────

    def check(
        self,
        proposed_action: Sequence[float],
        observation: Sequence[float] | None = None,
        effort: Sequence[float] | None = None,
        position_xyz: Sequence[float] | None = None,
    ) -> GuardDecision:
        """
        Check one proposed action.

        Args:
            proposed_action: Policy output, one joint position target per envelope joint.
            observation:     Current joint positions (used for OOD scoring and as the
                             hold command before any safe action exists).
            effort:          Current joint efforts, if the robot reports them.
            position_xyz:    Current base or end-effector position for the workspace check.
        """
        self._step += 1
        proposed = [float(a) for a in proposed_action]
        violations: list[Violation] = []
        ood_score: float | None = None

        if observation is not None and self._last_safe is None:
            self._last_safe = [float(o) for o in observation]

        # 1. Latched: nothing moves until an operator resets.
        if self.latched:
            violations.append(
                Violation(
                    kind=ViolationKind.LATCHED,
                    detail=f"guard halted earlier ({self._latched_reason}); awaiting operator reset",
                )
            )
            return self._finish(Verdict.HALT, proposed, self._last_safe, violations, None)

        # 2. Malformed policy output.
        n = len(self.envelope.joints)
        if len(proposed) != n:
            violations.append(
                Violation(
                    kind=ViolationKind.ACTION_SHAPE,
                    detail=f"policy returned {len(proposed)} values, envelope has {n} joints",
                )
            )
        elif not all(math.isfinite(a) for a in proposed):
            bad = [self.envelope.joints[i].name for i, a in enumerate(proposed)
                   if not math.isfinite(a)]
            violations.append(
                Violation(
                    kind=ViolationKind.NON_FINITE_ACTION,
                    detail=f"policy returned NaN/inf for {', '.join(bad)}",
                )
            )

        # 3. Out-of-distribution observation.
        if observation is not None and self.baseline is not None:
            ood_score, dim_name = self.baseline.score(observation)
            if ood_score > self.envelope.ood_threshold:
                violations.append(
                    Violation(
                        kind=ViolationKind.OBSERVATION_OOD,
                        detail=(
                            f"observation is {ood_score:.1f} std devs from training data "
                            f"on {dim_name}"
                        ),
                        joint=dim_name,
                        value=ood_score,
                        limit=self.envelope.ood_threshold,
                    )
                )

        # 4. Effort (possible contact or collision).
        if effort is not None:
            for limit, value in zip(self.envelope.joints, effort):
                if limit.max_effort is not None and abs(value) > limit.max_effort:
                    violations.append(
                        Violation(
                            kind=ViolationKind.EFFORT_LIMIT,
                            detail=f"{limit.name} effort {value:.2f} exceeds {limit.max_effort:.2f}",
                            joint=limit.name,
                            value=float(value),
                            limit=limit.max_effort,
                        )
                    )

        # 5. Workspace.
        if position_xyz is not None and self.envelope.workspace is not None:
            xyz = [float(v) for v in position_xyz] + [0.0] * (3 - len(position_xyz))
            if not self.envelope.workspace.contains(xyz[0], xyz[1], xyz[2]):
                violations.append(
                    Violation(
                        kind=ViolationKind.WORKSPACE,
                        detail=f"position ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) "
                               "is outside the workspace",
                    )
                )

        if self._halts(violations):
            return self._halt(proposed, violations, ood_score)

        # 6. Soft limits: clamp to position range, then rate-limit against the last
        #    safe command.
        safe = list(proposed)
        for i, limit in enumerate(self.envelope.joints):
            if safe[i] < limit.position_min or safe[i] > limit.position_max:
                bound = limit.position_min if safe[i] < limit.position_min else limit.position_max
                violations.append(
                    Violation(
                        kind=ViolationKind.POSITION_LIMIT,
                        detail=f"{limit.name} target {safe[i]:.3f} outside "
                               f"[{limit.position_min:.3f}, {limit.position_max:.3f}]",
                        joint=limit.name,
                        value=safe[i],
                        limit=bound,
                    )
                )
                safe[i] = bound

        if self._last_safe is not None:
            for i, max_step in enumerate(self.envelope.max_step):
                delta = safe[i] - self._last_safe[i]
                if abs(delta) > max_step:
                    name = self.envelope.joints[i].name
                    violations.append(
                        Violation(
                            kind=ViolationKind.RATE_LIMIT,
                            detail=f"{name} step {delta:+.3f} exceeds ±{max_step:.3f} per step",
                            joint=name,
                            value=delta,
                            limit=max_step,
                        )
                    )
                    safe[i] = self._last_safe[i] + math.copysign(max_step, delta)

        clamped = any(v.kind in (ViolationKind.POSITION_LIMIT, ViolationKind.RATE_LIMIT)
                      for v in violations)

        if clamped:
            self._consecutive_clamps += 1
            if self._consecutive_clamps >= self.envelope.max_consecutive_clamps:
                violations.append(
                    Violation(
                        kind=ViolationKind.CLAMP_ESCALATION,
                        detail=f"policy needed correcting {self._consecutive_clamps} steps in "
                               "a row; it is persistently pushing past the envelope",
                        value=float(self._consecutive_clamps),
                        limit=float(self.envelope.max_consecutive_clamps),
                    )
                )
                return self._halt(proposed, violations, ood_score)
        else:
            self._consecutive_clamps = 0

        # An OOD observation with halt_on_ood disabled stays in `violations` as a
        # warning: the action is forwarded, and the warning is audited.
        verdict = Verdict.CLAMP if clamped else Verdict.PASS
        self._last_safe = safe
        return self._finish(verdict, proposed, safe, violations, ood_score)

    # ── Internals ─────────────────────────────────────────────────────────

    def _halts(self, violations: list[Violation]) -> bool:
        for v in violations:
            if v.kind in HALTING_KINDS:
                return True
            if v.kind is ViolationKind.OBSERVATION_OOD and self.envelope.halt_on_ood:
                return True
        return False

    def _halt(
        self, proposed: list[float], violations: list[Violation], ood_score: float | None
    ) -> GuardDecision:
        halting = [
            v for v in violations
            if v.kind in HALTING_KINDS or v.kind is ViolationKind.OBSERVATION_OOD
        ]
        self._latched_reason = halting[0].kind.value if halting else "halt"
        return self._finish(Verdict.HALT, proposed, self._last_safe, violations, ood_score)

    def _finish(
        self,
        verdict: Verdict,
        proposed: list[float],
        safe: list[float] | None,
        violations: list[Violation],
        ood_score: float | None,
    ) -> GuardDecision:
        decision = GuardDecision(
            robot_id=self.robot_id,
            step=self._step,
            verdict=verdict,
            proposed_action=proposed,
            safe_action=list(safe) if safe is not None else None,
            violations=violations,
            ood_score=ood_score,
            latched=self.latched,
        )
        self._record(decision)
        return decision

    def _record(self, decision: GuardDecision) -> None:
        self.stats.steps += 1
        if decision.verdict is Verdict.PASS:
            self.stats.passed += 1
            if not decision.violations:
                return
        elif decision.verdict is Verdict.CLAMP:
            self.stats.clamped += 1
        else:
            self.stats.halted += 1
        for v in decision.violations:
            key = v.kind.value
            self.stats.violations_by_kind[key] = self.stats.violations_by_kind.get(key, 0) + 1

        # A latched guard repeats the same halt every step; audit the first one only.
        repeat = (
            decision.verdict is Verdict.HALT
            and len(decision.violations) == 1
            and decision.violations[0].kind is ViolationKind.LATCHED
        )
        event = "warning" if decision.verdict is Verdict.PASS else decision.verdict.value
        if self.audit_log is not None and not repeat:
            self.audit_log.append(
                event_type=f"guard.{event}",
                actor="action_guard",
                payload=_json_safe(decision.model_dump(mode="json")),
            )
        if self.on_intervention is not None and not repeat:
            self.on_intervention(decision)
