from __future__ import annotations

import math
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from robofleet_nexus.audit.hash_chain import AuditRecord, HashChainAuditLog
from robofleet_nexus.guard import (
    ActionGuard,
    JointLimit,
    ObservationBaseline,
    SafetyEnvelope,
    Verdict,
    ViolationKind,
    WorkspaceBox,
)
from robofleet_nexus.guard.events import decision_to_event


def make_envelope(**overrides: object) -> SafetyEnvelope:
    data: dict[str, object] = {
        "robot_id": "arm_01",
        "control_hz": 10.0,
        "joints": [
            JointLimit(name="j0", position_min=-1.0, position_max=1.0, max_velocity=1.0,
                       max_effort=20.0),
            JointLimit(name="j1", position_min=-1.0, position_max=1.0, max_velocity=1.0),
        ],
        "max_consecutive_clamps": 3,
    }
    data.update(overrides)
    return SafetyEnvelope.model_validate(data)


def make_baseline() -> ObservationBaseline:
    rng = np.random.default_rng(0)
    episodes = [rng.normal(0.0, 0.1, size=(200, 2)) for _ in range(3)]
    return ObservationBaseline.fit(episodes, names=["j0", "j1"])


# ── Verdicts ───────────────────────────────────────────────────────────────

def test_in_envelope_action_passes_unchanged() -> None:
    guard = ActionGuard(make_envelope())
    d = guard.check([0.05, -0.05], observation=[0.0, 0.0])
    assert d.verdict is Verdict.PASS
    assert d.safe_action == [0.05, -0.05]
    assert d.violations == []


def test_position_limit_is_clamped() -> None:
    guard = ActionGuard(make_envelope(control_hz=1.0))  # max step 1.0, so only range binds
    d = guard.check([1.5, 0.0], observation=[0.9, 0.0])
    assert d.verdict is Verdict.CLAMP
    assert d.safe_action == [1.0, 0.0]
    assert d.violations[0].kind is ViolationKind.POSITION_LIMIT


def test_rate_limit_caps_step_size() -> None:
    guard = ActionGuard(make_envelope())  # max step = 1.0 / 10 Hz = 0.1
    d = guard.check([0.5, 0.0], observation=[0.0, 0.0])
    assert d.verdict is Verdict.CLAMP
    assert d.safe_action is not None
    assert d.safe_action[0] == pytest.approx(0.1)
    assert d.violations[0].kind is ViolationKind.RATE_LIMIT


def test_nan_action_halts_and_holds_last_safe_command() -> None:
    guard = ActionGuard(make_envelope())
    guard.check([0.05, 0.0], observation=[0.0, 0.0])
    d = guard.check([math.nan, 0.0], observation=[0.05, 0.0])
    assert d.verdict is Verdict.HALT
    assert d.safe_action == [0.05, 0.0]
    assert d.latched
    assert d.violations[0].kind is ViolationKind.NON_FINITE_ACTION


def test_wrong_action_shape_halts() -> None:
    guard = ActionGuard(make_envelope())
    d = guard.check([0.0, 0.0, 0.0], observation=[0.0, 0.0])
    assert d.verdict is Verdict.HALT
    assert d.violations[0].kind is ViolationKind.ACTION_SHAPE


def test_effort_limit_halts() -> None:
    guard = ActionGuard(make_envelope())
    d = guard.check([0.0, 0.0], observation=[0.0, 0.0], effort=[25.0, 0.0])
    assert d.verdict is Verdict.HALT
    assert d.violations[0].kind is ViolationKind.EFFORT_LIMIT


def test_workspace_breach_halts() -> None:
    env = make_envelope(workspace=WorkspaceBox(x_min=-1, x_max=1, y_min=-1, y_max=1))
    guard = ActionGuard(env)
    d = guard.check([0.0, 0.0], observation=[0.0, 0.0], position_xyz=[1.5, 0.0])
    assert d.verdict is Verdict.HALT
    assert d.violations[0].kind is ViolationKind.WORKSPACE


def test_repeated_clamping_escalates_to_halt() -> None:
    guard = ActionGuard(make_envelope())
    verdicts = [guard.check([0.9, 0.0], observation=[0.0, 0.0]).verdict for _ in range(3)]
    assert verdicts == [Verdict.CLAMP, Verdict.CLAMP, Verdict.HALT]
    assert guard.latched_reason == ViolationKind.CLAMP_ESCALATION.value


def test_clamp_counter_resets_after_a_clean_step() -> None:
    guard = ActionGuard(make_envelope())
    guard.check([0.5, 0.0], observation=[0.0, 0.0])           # clamp → 0.1
    guard.check([0.15, 0.0], observation=[0.1, 0.0])          # clean
    guard.check([0.9, 0.0], observation=[0.15, 0.0])          # clamp
    d = guard.check([0.9, 0.0], observation=[0.25, 0.0])      # clamp (2nd in a row)
    assert d.verdict is Verdict.CLAMP


# ── OOD ────────────────────────────────────────────────────────────────────

def test_ood_observation_halts() -> None:
    guard = ActionGuard(make_envelope(), baseline=make_baseline())
    d = guard.check([0.0, 0.0], observation=[0.0, 0.0])
    assert d.verdict is Verdict.PASS
    d = guard.check([0.0, 0.0], observation=[0.95, 0.0])
    assert d.verdict is Verdict.HALT
    assert d.violations[0].kind is ViolationKind.OBSERVATION_OOD
    assert d.ood_score is not None and d.ood_score > 6


def test_ood_warning_only_when_halt_disabled() -> None:
    audit = HashChainAuditLog()
    guard = ActionGuard(make_envelope(halt_on_ood=False), baseline=make_baseline(),
                        audit_log=audit)
    d = guard.check([0.95, 0.0], observation=[0.95, 0.0])
    assert d.verdict is Verdict.PASS
    assert d.safe_action == [0.95, 0.0]
    assert d.violations[0].kind is ViolationKind.OBSERVATION_OOD
    assert [r.event_type for r in audit.records()] == ["guard.warning"]


def test_baseline_dimension_mismatch_rejected() -> None:
    baseline = ObservationBaseline.fit([np.zeros((5, 3))])
    with pytest.raises(ValueError):
        ActionGuard(make_envelope(), baseline=baseline)


# ── Latch, reset, audit ────────────────────────────────────────────────────

def test_halt_latches_until_operator_reset() -> None:
    audit = HashChainAuditLog()
    guard = ActionGuard(make_envelope(), audit_log=audit)
    guard.check([math.inf, 0.0], observation=[0.0, 0.0])
    for _ in range(5):
        d = guard.check([0.0, 0.0], observation=[0.0, 0.0])
        assert d.verdict is Verdict.HALT
        assert d.violations[0].kind is ViolationKind.LATCHED

    with pytest.raises(ValueError):
        guard.reset(operator="  ")
    guard.reset(operator="amit", note="encoder reseated")
    assert guard.check([0.0, 0.0], observation=[0.0, 0.0]).verdict is Verdict.PASS

    # The first halt and the reset are audited; the repeated latched steps are not.
    assert [r.event_type for r in audit.records()] == ["guard.halt", "guard.reset"]
    assert audit.records()[1].actor == "operator:amit"
    assert guard.stats.halted == 6 and guard.stats.resets == 1


def test_intervention_callback_receives_clamps_and_halts() -> None:
    seen: list[Verdict] = []
    guard = ActionGuard(make_envelope(), on_intervention=lambda d: seen.append(d.verdict))
    guard.check([0.0, 0.0], observation=[0.0, 0.0])
    guard.check([0.5, 0.0])
    guard.check([math.nan, 0.0])
    assert seen == [Verdict.CLAMP, Verdict.HALT]


def test_audit_chain_verifies_and_detects_tampering() -> None:
    audit = HashChainAuditLog()
    guard = ActionGuard(make_envelope(), audit_log=audit)
    guard.check([0.5, 0.0], observation=[0.0, 0.0])
    guard.check([math.nan, 0.0])
    assert audit.verify() == (True, None)

    records = audit.records()
    tampered = records[0].model_copy(
        update={"payload": {**records[0].payload, "verdict": "pass"}}
    )
    assert HashChainAuditLog.verify_records([tampered, records[1]]) == (False, 0)
    assert HashChainAuditLog.verify_records([records[1]]) == (False, 0)
    assert isinstance(records[0], AuditRecord)


# ── Baseline and envelope derivation ───────────────────────────────────────

def test_baseline_roundtrip_and_envelope_derivation(tmp_path: Path) -> None:
    traj = np.column_stack([np.linspace(-0.5, 0.5, 101), np.zeros(101)])
    baseline = ObservationBaseline.fit([traj], names=["shoulder", "elbow"])
    assert baseline.max_step[0] == pytest.approx(0.01)

    path = baseline.save(tmp_path / "baseline.json")
    loaded = ObservationBaseline.load(path)
    assert loaded == baseline

    env = SafetyEnvelope.from_baseline(loaded, control_hz=30.0)
    assert env.joint_names == ["shoulder", "elbow"]
    assert env.joints[0].position_min == pytest.approx(-0.6)
    assert env.joints[0].position_max == pytest.approx(0.6)
    assert env.joints[0].max_velocity == pytest.approx(0.01 * 30 * 1.25)

    yaml_path = env.to_yaml(tmp_path / "envelope.yaml")
    assert SafetyEnvelope.from_yaml(yaml_path) == env


def test_baseline_from_exported_lerobot_dataset(tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    from robofleet_nexus.lerobot.exporter import export_dataset
    from robofleet_nexus.lerobot.schema import LeRobotEpisode, LeRobotFrame

    start = datetime(2026, 10, 18, tzinfo=UTC)
    frames = [
        LeRobotFrame(
            timestamp=i / 10, frame_index=i, episode_index=0, task_index=0,
            action=[0.01 * i, -0.01 * i], observation_state=[0.01 * i, -0.01 * i],
            observation_velocity=[0.0, 0.0], observation_effort=[0.0, 0.0],
            observation_pose=[0.0, 0.0, 0.0, 0.0], observation_battery=[90.0],
            done=i == 19,
        )
        for i in range(20)
    ]
    episode = LeRobotEpisode(0, "arm_01", "reach", start, start + timedelta(seconds=2), frames)
    out = export_dataset([episode], tmp_path / "ds", robot_id="arm_01", task="reach")

    baseline = ObservationBaseline.from_lerobot_dataset(out)
    assert baseline.num_frames == 20
    assert baseline.names == ["joint_0", "joint_1"]
    assert baseline.maximum[0] == pytest.approx(0.19)


# ── Telemetry integration ──────────────────────────────────────────────────

def test_guard_halt_event_creates_critical_finding() -> None:
    from robofleet_nexus.api.main import app

    guard = ActionGuard(make_envelope())
    decision = guard.check([math.nan, 0.0], observation=[0.0, 0.0])
    event = decision_to_event(decision, source="mock")

    response = TestClient(app).post("/telemetry", json=event)
    assert response.status_code == 200
    titles = [f["title"] for f in response.json()]
    assert "Policy halted by safety guard" in titles


def test_audit_survives_json_roundtrip_with_nan_halt() -> None:
    import json

    audit = HashChainAuditLog()
    guard = ActionGuard(make_envelope(), audit_log=audit)
    guard.check([math.nan, 0.0], observation=[0.0, 0.0])
    reloaded = [
        AuditRecord.model_validate(json.loads(r.model_dump_json())) for r in audit.records()
    ]
    assert HashChainAuditLog.verify_records(reloaded) == (True, None)


# ── ROS2 node helpers (no rclpy needed) ────────────────────────────────────

def test_ordered_values_matches_by_name_and_position() -> None:
    from robofleet_nexus.adapters.ros2_guard_node import ordered_values

    assert ordered_values(["b", "a"], [2.0, 1.0], ["a", "b"]) == [1.0, 2.0]
    assert ordered_values(["a"], [1.0], ["a", "b"]) is None
    assert ordered_values([], [1.0, 2.0], ["a", "b"]) == [1.0, 2.0]
    assert ordered_values([], [1.0], ["a", "b"]) is None
    assert ordered_values(["a", "b"], [], ["a", "b"]) is None


# ── End-to-end demo ────────────────────────────────────────────────────────

def test_demo_hits_every_fault_and_keeps_audit_intact(tmp_path: Path) -> None:
    from robofleet_nexus.guard.demo import run_demo

    result = run_demo(tmp_path)
    kinds = {v.kind for d in result.interventions for v in d.violations}
    assert {
        ViolationKind.RATE_LIMIT,
        ViolationKind.POSITION_LIMIT,
        ViolationKind.OBSERVATION_OOD,
        ViolationKind.NON_FINITE_ACTION,
    } <= kinds
    assert ViolationKind.CLAMP_ESCALATION not in kinds
    assert result.guard.stats.resets == 1
    assert result.chain_ok
    for name in ("baseline.json", "envelope.yaml", "audit.jsonl"):
        assert (tmp_path / name).exists()


def test_cli_guard_demo_and_verify(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from robofleet_nexus.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["guard", "demo", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "intact" in result.output
    result = runner.invoke(app, ["guard", "verify-audit", str(tmp_path / "audit.jsonl")])
    assert result.exit_code == 0, result.output


# ── ROS2 adapters: import footprint and closed-loop behaviour (no rclpy) ────

def test_ros2_adapters_import_without_pandas_or_pyarrow() -> None:
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pyarrow'] = None; sys.modules['pandas'] = None\n"
        "import robofleet_nexus.adapters.ros2_guard_node\n"
        "import robofleet_nexus.adapters.ros2_fake_arm\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_fake_arm_schedule_through_guard(tmp_path: Path) -> None:
    """Replays the fake arm's ROS2 loop in-process against the demo envelope."""
    from robofleet_nexus.adapters.ros2_fake_arm import (
        CYCLE_STEPS,
        WRIST_3,
        encoder_slip,
        scripted_action,
    )
    from robofleet_nexus.guard.demo import run_demo
    from robofleet_nexus.guard.scenario import HZ, nominal

    demo = run_demo(tmp_path)
    guard = ActionGuard(demo.envelope, baseline=demo.baseline)

    state = nominal(0.0)
    policy_t, halted_at = 0, None
    auto_reset_steps = int(3 * HZ)
    for step in range(CYCLE_STEPS + auto_reset_steps + 5):
        observed = state.copy()
        observed[WRIST_3] += encoder_slip(step)
        action, _ = scripted_action(step, policy_t)
        d = guard.check(action, observation=observed.tolist())
        if d.safe_action is not None:
            state = np.asarray(d.safe_action)
        if guard.latched and halted_at is None:
            halted_at = step
        if halted_at is not None and step - halted_at >= auto_reset_steps:
            guard.reset(operator="demo-operator")
            halted_at = None
        if not guard.latched:
            policy_t += 1

    kinds = guard.stats.violations_by_kind
    assert kinds.get("rate_limit", 0) > 0
    assert kinds.get("position_limit", 0) > 0
    assert kinds.get("observation_ood") == 1  # the slip doesn't re-halt after the reset
    assert kinds.get("non_finite_action") == 1
    assert "clamp_escalation" not in kinds
    assert guard.stats.resets == 2


def test_fake_arm_pauses_on_latch_and_resumes_only_on_guard_reset() -> None:
    from robofleet_nexus.adapters.ros2_fake_arm import GuardStatusTracker

    t = GuardStatusTracker()
    halt = {"verdict": "halt", "latched": True, "step": 10,
            "violations": ["observation is 9.8 std devs from training data on wrist_3"]}
    held = {"verdict": "halt", "latched": True, "step": 11,
            "violations": ["guard halted earlier (observation_ood); awaiting operator reset"]}
    reset = {"verdict": "reset", "latched": False, "operator": "demo-operator"}

    event = t.on_status(halt)
    assert event is not None and event[0] == "halt" and t.paused
    assert "9.8 std devs" in event[1]
    assert t.on_status(held) == ("still_halted", "") and t.paused
    assert t.on_status(reset) == (
        "resumed", "Guard reset by 'demo-operator'; policy resuming"
    )
    assert not t.paused

    # A fresh halt after the reset is reported.
    nan_halt = {"verdict": "halt", "latched": True, "step": 40,
                "violations": ["policy returned NaN/inf for wrist_2"]}
    event = t.on_status(nan_halt)
    assert event is not None and event[0] == "halt" and "NaN" in event[1]


def test_fake_arm_started_against_already_latched_guard() -> None:
    from robofleet_nexus.adapters.ros2_fake_arm import GuardStatusTracker

    t = GuardStatusTracker()
    held = {"verdict": "halt", "latched": True, "step": 900,
            "violations": ["guard halted earlier (observation_ood); awaiting operator reset"]}
    event = t.on_status(held)
    assert event is not None and event[0] == "halt" and t.paused


def test_fake_arm_logs_first_clamp_of_each_run() -> None:
    from robofleet_nexus.adapters.ros2_fake_arm import GuardStatusTracker

    t = GuardStatusTracker()

    def clamp(step: int, why: str) -> dict[str, object]:
        return {"verdict": "clamp", "latched": False, "step": step, "violations": [why]}

    assert t.on_status(clamp(90, "elbow")) == ("clamp", "GUARD CLAMP: elbow")
    assert t.on_status(clamp(91, "elbow")) is None
    assert t.on_status(clamp(180, "wrist_1")) == ("clamp", "GUARD CLAMP: wrist_1")
