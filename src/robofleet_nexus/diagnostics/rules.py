from __future__ import annotations
from uuid import uuid4
from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent, RobotSeverity


def evaluate_event(event: RobotEvent) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []

    def finding(severity, title, explanation, evidence, actions):
        return DiagnosticFinding(
            finding_id=str(uuid4()),
            robot_id=event.robot_id,
            severity=severity,
            title=title,
            explanation=explanation,
            evidence=evidence,
            recommended_actions=actions,
        )

    gpu_temp     = event.metrics.get("gpu_temp_c")
    motor_temp   = event.metrics.get("motor_temp_c")
    packet_loss  = event.metrics.get("packet_loss_pct")
    battery_pct  = event.metrics.get("battery_pct")

    # ── GPU temperature ───────────────────────────────────────────────────
    if gpu_temp is not None and gpu_temp >= 85:
        findings.append(finding(
            RobotSeverity.warning,
            "Elevated GPU temperature",
            "GPU temperature crossed the warning threshold.",
            [f"gpu_temp_c={gpu_temp}"],
            ["Check simulation workload placement.",
             "Inspect GPU utilization and cooling.",
             "Consider rescheduling non-critical simulation jobs."],
        ))

    # ── Motor temperature ─────────────────────────────────────────────────
    if motor_temp is not None and motor_temp >= 90:
        findings.append(finding(
            RobotSeverity.critical,
            "Motor thermal risk",
            "Motor temperature crossed the critical threshold.",
            [f"motor_temp_c={motor_temp}"],
            ["Stop non-essential motion.",
             "Request operator review.",
             "Inspect actuator telemetry and recent command history."],
        ))

    # ── Network degradation ───────────────────────────────────────────────
    if packet_loss is not None and packet_loss >= 5:
        findings.append(finding(
            RobotSeverity.warning,
            "Network degradation detected",
            "Packet loss may affect control loop stability or telemetry freshness.",
            [f"packet_loss_pct={packet_loss}"],
            ["Check ROS2 DDS/network path.",
             "Inspect host networking and QoS settings.",
             "Compare telemetry delay against command latency."],
        ))

    # ── Battery critical ──────────────────────────────────────────────────
    if battery_pct is not None and battery_pct <= 15:
        findings.append(finding(
            RobotSeverity.critical,
            "Battery critically low",
            f"Battery at {battery_pct:.1f}% — immediate return to charging station required.",
            [f"battery_pct={battery_pct}"],
            ["Abort current mission immediately.",
             "Navigate to nearest charging station.",
             "Alert fleet operator."],
        ))
    elif battery_pct is not None and battery_pct <= 25:
        findings.append(finding(
            RobotSeverity.warning,
            "Battery low",
            f"Battery at {battery_pct:.1f}% — return to charging station soon.",
            [f"battery_pct={battery_pct}"],
            ["Complete current task and return to charge.",
             "Defer non-essential operations."],
        ))

    # ── Joint effort ──────────────────────────────────────────────────────
    for key, val in event.metrics.items():
        if key.endswith(".eff") and val >= 40.0:
            joint = key.replace(".eff", "")
            findings.append(finding(
                RobotSeverity.warning,
                f"High joint effort: {joint}",
                f"Joint {joint} is drawing excessive effort ({val:.1f} Nm), "
                "indicating possible obstruction or mechanical resistance.",
                [f"{key}={val}"],
                [f"Inspect {joint} for mechanical obstruction.",
                 "Check for joint calibration drift.",
                 "Review recent motion commands for this joint."],
            ))

    return findings
