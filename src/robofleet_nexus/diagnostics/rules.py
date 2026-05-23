from __future__ import annotations

from uuid import uuid4

from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent, RobotSeverity


def evaluate_event(event: RobotEvent) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []

    gpu_temp = event.metrics.get("gpu_temp_c")
    motor_temp = event.metrics.get("motor_temp_c")
    packet_loss = event.metrics.get("packet_loss_pct")

    if gpu_temp is not None and gpu_temp >= 85:
        findings.append(
            DiagnosticFinding(
                finding_id=str(uuid4()),
                robot_id=event.robot_id,
                severity=RobotSeverity.warning,
                title="Elevated GPU temperature",
                explanation="GPU temperature crossed the warning threshold.",
                evidence=[f"gpu_temp_c={gpu_temp}"],
                recommended_actions=[
                    "Check simulation workload placement.",
                    "Inspect GPU utilization and cooling.",
                    "Consider rescheduling non-critical simulation jobs.",
                ],
            )
        )

    if motor_temp is not None and motor_temp >= 90:
        findings.append(
            DiagnosticFinding(
                finding_id=str(uuid4()),
                robot_id=event.robot_id,
                severity=RobotSeverity.critical,
                title="Motor thermal risk",
                explanation="Motor temperature crossed the critical threshold.",
                evidence=[f"motor_temp_c={motor_temp}"],
                recommended_actions=[
                    "Stop non-essential motion.",
                    "Request operator review.",
                    "Inspect actuator telemetry and recent command history.",
                ],
            )
        )

    if packet_loss is not None and packet_loss >= 5:
        findings.append(
            DiagnosticFinding(
                finding_id=str(uuid4()),
                robot_id=event.robot_id,
                severity=RobotSeverity.warning,
                title="Network degradation detected",
                explanation="Packet loss may affect control loop stability or telemetry freshness.",
                evidence=[f"packet_loss_pct={packet_loss}"],
                recommended_actions=[
                    "Check ROS2 DDS/network path.",
                    "Inspect host networking and QoS settings.",
                    "Compare telemetry delay against command latency.",
                ],
            )
        )

    return findings
