"""
Self-contained guard demo: no ROS2, GPU, or trained policy required.

1. Generates demonstration episodes for a 6-joint arm and exports them as a
   LeRobot v3 dataset (the same format RoboFleet Nexus exports from fleet telemetry).
2. Fits an observation baseline from that dataset and derives a safety envelope.
3. Runs a scripted "policy" through the guard with four injected faults:
     - a one-step action spike            -> CLAMP (rate limit)
     - a drift past a joint limit         -> CLAMP (position + rate limit)
     - a slipped wrist encoder           -> HALT  (out-of-distribution), operator reset
     - a NaN in the policy output         -> HALT  (non-finite action)
4. Verifies the tamper-evident audit chain.

While the guard is halted the policy is paused, as a real controller would pause it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from robofleet_nexus.audit.hash_chain import HashChainAuditLog
from robofleet_nexus.guard.baseline import ObservationBaseline
from robofleet_nexus.guard.core import ActionGuard, GuardDecision
from robofleet_nexus.guard.envelope import SafetyEnvelope
from robofleet_nexus.guard.scenario import HZ, JOINTS, nominal
from robofleet_nexus.lerobot.exporter import export_dataset
from robofleet_nexus.lerobot.schema import LeRobotEpisode, LeRobotFrame

__all__ = ["HZ", "JOINTS", "DemoResult", "nominal", "run_demo"]

SPIKE_STEP = 80
DRIFT_STEPS = range(150, 155)
ENCODER_FAULT_STEPS = range(230, 245)
RESET_STEP = 245
NAN_STEP = 330
TOTAL_STEPS = 340


@dataclass
class DemoResult:
    output_dir: Path
    envelope: SafetyEnvelope
    baseline: ObservationBaseline
    guard: ActionGuard
    audit: HashChainAuditLog
    interventions: list[GuardDecision] = field(default_factory=list)
    resets: list[int] = field(default_factory=list)
    chain_ok: bool = False


def make_demonstrations(n_episodes: int = 6, frames: int = 300, seed: int = 7) -> list[
    NDArray[np.float64]
]:
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        phase = rng.uniform(0, 2 * math.pi, size=len(JOINTS))
        traj = np.vstack([nominal(i / HZ, phase) for i in range(frames)])
        episodes.append(traj + rng.normal(0, 0.005, size=traj.shape))
    return episodes


def export_demonstrations(episodes: list[NDArray[np.float64]], out_dir: Path) -> Path:
    start = datetime(2026, 10, 18, tzinfo=UTC)
    lerobot_eps = []
    for e, traj in enumerate(episodes):
        frames = [
            LeRobotFrame(
                timestamp=i / HZ, frame_index=i, episode_index=e, task_index=0,
                action=traj[i].tolist(), observation_state=traj[i].tolist(),
                observation_velocity=[0.0] * len(JOINTS),
                observation_effort=[0.0] * len(JOINTS),
                observation_pose=[0.0, 0.0, 0.0, 0.0], observation_battery=[100.0],
                done=i == len(traj) - 1,
            )
            for i in range(len(traj))
        ]
        t0 = start + timedelta(minutes=e)
        lerobot_eps.append(
            LeRobotEpisode(e, "arm_demo", "pick_and_place", t0,
                           t0 + timedelta(seconds=len(traj) / HZ), frames)
        )
    return export_dataset(lerobot_eps, out_dir, robot_id="arm_demo",
                          task="pick_and_place", fps=HZ)


def run_demo(
    output_dir: Path | str,
    on_intervention: Callable[[GuardDecision], None] | None = None,
    on_step: Callable[[int], None] | None = None,
) -> DemoResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = export_demonstrations(make_demonstrations(), out / "demonstrations")
    baseline = ObservationBaseline.from_lerobot_dataset(dataset)
    baseline = baseline.model_copy(update={"names": list(JOINTS)})
    envelope = SafetyEnvelope.from_baseline(baseline, control_hz=HZ, robot_id="arm_demo")
    baseline.save(out / "baseline.json")
    envelope.to_yaml(out / "envelope.yaml")

    audit = HashChainAuditLog()
    interventions: list[GuardDecision] = []

    def sink(decision: GuardDecision) -> None:
        interventions.append(decision)
        if on_intervention is not None:
            on_intervention(decision)

    guard = ActionGuard(envelope, baseline=baseline, audit_log=audit, on_intervention=sink)
    result = DemoResult(out, envelope, baseline, guard, audit, interventions)

    rng = np.random.default_rng(11)
    policy_t = 0  # policy clock; frozen while the guard is halted
    hi = np.array([j.position_max for j in envelope.joints])
    commanded = nominal(0.0)

    for step in range(TOTAL_STEPS):
        if on_step is not None:
            on_step(step)
        if step == RESET_STEP and guard.latched:
            guard.reset(operator="demo-operator", note="encoder re-seated, readings nominal")
            result.resets.append(step)

        action = nominal(policy_t / HZ) + rng.normal(0, 0.002, size=len(JOINTS))
        observation = commanded.copy()

        if step == SPIKE_STEP:
            action[2] += 0.8
        if step in DRIFT_STEPS:
            action[3] = hi[3] + 0.3
        if step in ENCODER_FAULT_STEPS:
            observation[5] += 1.2
        if step == NAN_STEP:
            action[4] = math.nan

        decision = guard.check(action.tolist(), observation=observation.tolist())
        if decision.safe_action is not None:
            commanded = np.asarray(decision.safe_action)
        if not guard.latched:
            policy_t += 1

    (out / "audit.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in audit.records()), encoding="utf-8"
    )
    result.chain_ok, _ = audit.verify()
    return result
