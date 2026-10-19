from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from anthropic import AsyncAnthropic

from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotEvent

logger = logging.getLogger("robofleet.rca")

# ── Spend guard ───────────────────────────────────────────────────────────────
# Hard cap: max RCA calls per day across all robots.
# At ~2500 tokens/call and $3/MTok that's roughly $0.75/day max.
#
# The count is stored on disk so restarting the API (or running it with --reload)
# does not reset it. Override with ROBOFLEET_RCA_MAX_PER_DAY and
# ROBOFLEET_RCA_USAGE_FILE.
_MAX_RCA_CALLS_PER_DAY = 25
_DEFAULT_USAGE_FILE = Path.home() / ".robofleet" / "rca_usage.json"


def _usage_file() -> Path:
    return Path(os.environ.get("ROBOFLEET_RCA_USAGE_FILE", str(_DEFAULT_USAGE_FILE)))


def _max_calls_per_day() -> int:
    try:
        return int(os.environ.get("ROBOFLEET_RCA_MAX_PER_DAY", _MAX_RCA_CALLS_PER_DAY))
    except ValueError:
        return _MAX_RCA_CALLS_PER_DAY


def _read_usage(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _check_spend_guard() -> bool:
    """Returns True if the call is allowed (and counts it), False if the daily cap is reached."""
    import datetime
    today = datetime.date.today().isoformat()
    path = _usage_file()
    usage = _read_usage(path)
    count = int(usage.get("count", 0)) if usage.get("date") == today else 0
    cap = _max_calls_per_day()
    if count >= cap:
        logger.warning(
            "RCA daily cap reached (%d/%d calls, counted in %s). Skipping to protect "
            "API credits. Resets tomorrow; raise ROBOFLEET_RCA_MAX_PER_DAY to allow more.",
            count, cap, path,
        )
        return False
    count += 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"date": today, "count": count}), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        # If the count can't be recorded, refuse rather than risk unbounded spend.
        logger.error("Cannot record RCA usage in %s (%s); skipping RCA.", path, exc)
        return False
    logger.warning("RCA call %d/%d today (real API call)", count, cap)
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


def _mock_rca(robot_id: str, findings: list) -> dict:
    """Deterministic mock RCA — zero API cost. Used when RCA is disabled."""
    import uuid
    from datetime import datetime, timezone
    if not findings:
        return {}
    f0 = findings[0]
    risk = f0.severity.value if f0.severity.value != "warning" else "high"
    return {
        "summary": f"Robot {robot_id} has {len(findings)} active finding(s) requiring attention.",
        "root_causes": [{"rank": 1, "cause": f0.title, "confidence": "high",
            "subsystem": "power", "evidence": list(f0.evidence)}],
        "recommended_actions": [{"priority": "immediate", "action": a,
            "rationale": "Diagnostic rule triggered."}
            for a in f0.recommended_actions[:2]],
        "requires_human_approval": True,
        "risk_level": risk,
        "estimated_resolution_time": "10 minutes",
        "rca_id": str(uuid.uuid4()),
        "robot_id": robot_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "findings_analyzed": len(findings),
        "input_tokens": 0,
        "output_tokens": 0,
        "mock": True,
    }


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

    import os
    if os.environ.get("ROBOFLEET_RCA_ENABLED", "0") != "1":
        logger.info("RCA disabled (set ROBOFLEET_RCA_ENABLED=1 to enable). Using mock.")
        return _mock_rca(robot_id, findings)
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
