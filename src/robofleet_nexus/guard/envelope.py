"""
Safety envelope for learned robot policies.

The envelope is the set of hard limits a policy's output must stay inside before
it reaches the robot: joint position ranges, per-step rate limits, effort limits,
and an optional Cartesian workspace box.

An envelope can be written by hand (YAML) or derived from the demonstrations a
policy was trained on (see ``SafetyEnvelope.from_baseline``), so a team gets a
reasonable starting envelope without writing one.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from robofleet_nexus.guard.baseline import ObservationBaseline


class JointLimit(BaseModel):
    """Hard limits for one joint."""

    name: str
    position_min: float
    position_max: float
    max_velocity: float = Field(
        gt=0,
        description="Maximum commanded speed (units/s). Enforced as a per-step rate limit.",
    )
    max_effort: float | None = Field(
        default=None,
        description="Observed effort above this halts the policy (possible collision).",
    )

    @model_validator(mode="after")
    def _check_range(self) -> JointLimit:
        if self.position_min >= self.position_max:
            raise ValueError(
                f"joint {self.name}: position_min must be below position_max"
            )
        return self


class WorkspaceBox(BaseModel):
    """Axis-aligned box the robot base or end effector must stay inside."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = float("-inf")
    z_max: float = float("inf")

    def contains(self, x: float, y: float, z: float = 0.0) -> bool:
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


class SafetyEnvelope(BaseModel):
    """All limits the guard enforces for one robot."""

    robot_id: str | None = None
    control_hz: float = Field(default=30.0, gt=0, description="Policy control rate.")
    joints: list[JointLimit] = Field(min_length=1)
    workspace: WorkspaceBox | None = None
    max_consecutive_clamps: int = Field(
        default=15,
        ge=1,
        description="Clamping this many steps in a row escalates to a halt.",
    )
    ood_threshold: float = Field(
        default=6.0,
        gt=0,
        description="Observation z-score above which the observation is out of distribution.",
    )
    halt_on_ood: bool = True

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def max_step(self) -> list[float]:
        """Largest allowed change per control step, per joint."""
        return [j.max_velocity / self.control_hz for j in self.joints]

    # ── Loading and saving ────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path | str) -> SafetyEnvelope:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"{path}: envelope file must contain a mapping")
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False),
            encoding="utf-8",
        )
        return out

    # ── Derivation from demonstrations ────────────────────────────────────

    @classmethod
    def from_baseline(
        cls,
        baseline: ObservationBaseline,
        control_hz: float = 30.0,
        position_margin: float = 0.10,
        velocity_margin: float = 0.25,
        robot_id: str | None = None,
    ) -> SafetyEnvelope:
        """
        Derive a starting envelope from the demonstrations a policy was trained on.

        Position limits are the observed range widened by ``position_margin`` of the
        range on each side. Velocity limits are the fastest observed per-step change
        (scaled to units/s) widened by ``velocity_margin``. Treat the result as a
        starting point to review, not a certified limit.
        """
        joints: list[JointLimit] = []
        for i, name in enumerate(baseline.names):
            lo, hi = baseline.minimum[i], baseline.maximum[i]
            span = max(hi - lo, 1e-3)
            step = max(baseline.max_step[i], 1e-4)
            joints.append(
                JointLimit(
                    name=name,
                    position_min=lo - position_margin * span,
                    position_max=hi + position_margin * span,
                    max_velocity=step * control_hz * (1.0 + velocity_margin),
                )
            )
        return cls(robot_id=robot_id, control_hz=control_hz, joints=joints)
