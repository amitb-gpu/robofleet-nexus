from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RobotSeverity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class RobotEvent(BaseModel):
    event_id: str
    robot_id: str
    source: Literal["ros2", "isaac_sim", "isaac_lab", "manual", "mock"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: RobotSeverity = RobotSeverity.info
    event_type: str
    subsystem: str
    message: str
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticFinding(BaseModel):
    finding_id: str
    robot_id: str
    severity: RobotSeverity
    title: str
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True
