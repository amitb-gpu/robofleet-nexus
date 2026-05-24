from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from anthropic import AsyncAnthropic

from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent

logger = logging.getLogger("robofleet.rca")

_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are an expert robotics diagnostics engineer embedded in RoboFleet Nexus,
a production robotics orchestration platform. You perform root cause analysis on robot fleet
diagnostic findings in real time.

You receive:
- A list of active diagnostic findings (structured)
- Recent telemetry events from the robot (metrics, subsystem, message)
- Robot ID and timestamp

You must respond with ONLY a valid JSON object — no preamble, no markdown, no explanation outside the JSON.

Response schema:
{
  "summary": "One sentence describing the overall robot health situation",
  "root_causes": [
    {
      "rank": 1,
      "cause": "Concise description of the root cause",
      "confidence": "high|medium|low",
      "subsystem": "which subsystem is affected",
      "evidence": ["specific metric or event supporting this cause"]
    }
  ],
  "recommended_actions": [
    {
      "priority": "immediate|soon|monitor",
      "action": "Specific actionable step for the operator or system",
      "rationale": "Why this action addresses the root cause"
    }
  ],
  "requires_human_approval": true,
  "risk_level": "critical|high|medium|low",
  "estimated_resolution_time": "e.g. 5 minutes, 1 hour, unknown"
}

Be precise and robotics-specific. Reference actual metric values from the evidence.
Prioritise safety — if there is any risk of hardware damage or injury, say so explicitly.
"""


async def run_rca(
    robot_id: str,
    findings: list[DiagnosticFinding],
    recent_events: list[RobotEvent],
) -> dict:
    """
    Run LLM-powered root cause analysis on a set of diagnostic findings.
    Returns a structured RCA dict ready to broadcast to the dashboard.
    """
    if not findings:
        return {}

    findings_payload = [
        {
            "finding_id": f.finding_id,
            "severity": f.severity.value,
            "title": f.title,
            "explanation": f.explanation,
            "evidence": f.evidence,
            "recommended_actions": f.recommended_actions,
        }
        for f in findings
    ]

    # Last 10 events as context — most recent first
    events_payload = [
        {
            "event_type": e.event_type,
            "subsystem": e.subsystem,
            "message": e.message,
            "severity": e.severity.value,
            "metrics": e.metrics,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in reversed(recent_events[-10:])
    ]

    user_message = json.dumps({
        "robot_id": robot_id,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "active_findings": findings_payload,
        "recent_telemetry_context": events_payload,
    }, indent=2)

    try:
        response = await _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        rca = json.loads(raw)
        rca["rca_id"] = str(uuid4())
        rca["robot_id"] = robot_id
        rca["timestamp"] = datetime.now(timezone.utc).isoformat()
        rca["findings_analyzed"] = len(findings)
        rca["input_tokens"] = response.usage.input_tokens
        rca["output_tokens"] = response.usage.output_tokens

        logger.info(
            "RCA complete for %s — risk=%s findings=%d tokens=%d+%d",
            robot_id, rca.get("risk_level", "?"), len(findings),
            response.usage.input_tokens, response.usage.output_tokens,
        )
        return rca

    except json.JSONDecodeError as exc:
        logger.error("RCA JSON parse error: %s\nRaw: %s", exc, raw[:300])
        return {"error": "RCA response parse failed", "robot_id": robot_id}
    except Exception as exc:
        logger.exception("RCA failed for %s: %s", robot_id, exc)
        return {"error": str(exc), "robot_id": robot_id}
