from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from anthropic import AsyncAnthropic

from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent

logger = logging.getLogger("robofleet.rca")

# ── Spend guard ───────────────────────────────────────────────────────────────
# Hard cap: max RCA calls per day across all robots.
# At ~2500 tokens/call and $3/MTok that's roughly $0.75/day max.
_MAX_RCA_CALLS_PER_DAY = 25
_rca_call_count: int = 0
_rca_call_date: str = ""

def _check_spend_guard() -> bool:
    """Returns True if the call is allowed, False if daily cap is reached."""
    import datetime
    global _rca_call_count, _rca_call_date
    today = datetime.date.today().isoformat()
    if _rca_call_date != today:
        _rca_call_date = today
        _rca_call_count = 0
    if _rca_call_count >= _MAX_RCA_CALLS_PER_DAY:
        logger.warning(
            "RCA daily cap reached (%d calls). Skipping to protect API credits. "
            "Reset tomorrow or raise _MAX_RCA_CALLS_PER_DAY in rca/agent.py.",
            _MAX_RCA_CALLS_PER_DAY,
        )
        return False
    _rca_call_count += 1
    logger.info("RCA call %d/%d today", _rca_call_count, _MAX_RCA_CALLS_PER_DAY)
    return True

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

    if not _check_spend_guard():
        return {"error": "daily_cap_reached", "robot_id": robot_id}

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

    # Remove this block once API credits are confirmed
    if not _client.api_key or False:  # flip to False when credits clear
        mock = {
            "summary": f"Robot {robot_id} has {len(findings)} active finding(s) requiring attention.",
            "root_causes": [{"rank": 1, "cause": findings[0].title, "confidence": "high",
                "subsystem": "power", "evidence": [e for e in findings[0].evidence]}],
            "recommended_actions": [{"priority": "immediate",
                "action": a, "rationale": "Diagnostic rule triggered."} 
                for a in findings[0].recommended_actions[:2]],
            "requires_human_approval": True,
            "risk_level": findings[0].severity.value if findings[0].severity.value != "warning" else "high",
            "estimated_resolution_time": "10 minutes",
        }
        mock["rca_id"] = str(__import__("uuid").uuid4())
        mock["robot_id"] = robot_id
        mock["timestamp"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        mock["findings_analyzed"] = len(findings)
        mock["input_tokens"] = 0
        mock["output_tokens"] = 0
        logger.info("MOCK RCA for %s — risk=%s findings=%d", robot_id, mock["risk_level"], len(findings))
        return mock

    try:
        response = await _client.messages.create(
            model="claude-sonnet-4-6",
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
