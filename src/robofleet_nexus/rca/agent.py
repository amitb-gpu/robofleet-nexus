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
- The robot's distinct active findings, each with how many times it has occurred
- Recent telemetry events from the robot (metrics, subsystem, message)
- Robot ID and timestamp

Report your analysis by calling the report_rca tool. Be brief and specific:
- summary: one sentence, at most 40 words
- at most 3 root causes, each with at most 3 short evidence items
- at most 4 recommended actions

Be precise and robotics-specific. Reference actual metric values from the evidence.
Prioritise safety — if there is any risk of hardware damage or injury, say so explicitly.
Findings titled "...by safety guard" come from a runtime guard between a learned policy and
the robot: a halt means the guard stopped the policy and is holding the last safe command.
"""

# Forcing this tool makes the model return a parsed object instead of free text, so a
# stray character can't break JSON parsing.
RCA_TOOL = {
    "name": "report_rca",
    "description": "Report the root cause analysis for the robot's active findings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "root_causes": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "cause": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "subsystem": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                    },
                    "required": ["rank", "cause", "confidence", "subsystem", "evidence"],
                },
            },
            "recommended_actions": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "priority": {"type": "string", "enum": ["immediate", "soon", "monitor"]},
                        "action": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["priority", "action", "rationale"],
                },
            },
            "requires_human_approval": {"type": "boolean"},
            "risk_level": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
            "estimated_resolution_time": {"type": "string"},
        },
        "required": [
            "summary", "root_causes", "recommended_actions",
            "requires_human_approval", "risk_level", "estimated_resolution_time",
        ],
    },
}

_MAX_FINDINGS_PER_RCA = 8
_RCA_MAX_TOKENS = 2048


def _condense_findings(findings: list[DiagnosticFinding]) -> list[tuple[DiagnosticFinding, int]]:
    """
    Latest finding per title with its occurrence count, newest first, at most
    _MAX_FINDINGS_PER_RCA. Keeps the request size bounded however long a robot
    has been reporting the same problem.
    """
    latest: dict[str, DiagnosticFinding] = {}
    counts: dict[str, int] = {}
    for f in findings:
        latest.pop(f.title, None)
        latest[f.title] = f
        counts[f.title] = counts.get(f.title, 0) + 1
    newest_first = list(reversed(latest.values()))[:_MAX_FINDINGS_PER_RCA]
    return [(f, counts[f.title]) for f in newest_first]



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

    condensed = _condense_findings(findings)
    findings_payload = [
        {
            "severity": f.severity.value,
            "title": f.title,
            "occurrences": n,
            "explanation": f.explanation,
            "evidence": f.evidence[:5],
            "recommended_actions": f.recommended_actions,
        }
        for f, n in condensed
    ]

    # Last 10 events as context — most recent first
    events_payload = [
        {
            "event_type": e.event_type,
            "subsystem": e.subsystem,
            "message": e.message[:300],
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

    if not _client.api_key:
        logger.warning("RCA enabled but ANTHROPIC_API_KEY is not set. Using mock.")
        return _mock_rca(robot_id, findings)

    try:
        response = await _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=_RCA_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[RCA_TOOL],
            tool_choice={"type": "tool", "name": "report_rca"},
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as exc:
        logger.exception("RCA failed for %s: %s", robot_id, exc)
        return {"error": str(exc), "robot_id": robot_id}

    usage = response.usage
    if response.stop_reason == "max_tokens":
        logger.error(
            "RCA for %s hit the %d-token output limit; result discarded (tokens=%d+%d).",
            robot_id, _RCA_MAX_TOKENS, usage.input_tokens, usage.output_tokens,
        )
        return {"error": "rca_truncated", "robot_id": robot_id}

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None or not isinstance(tool_use.input, dict):
        logger.error("RCA for %s returned no report_rca call (stop=%s).",
                     robot_id, response.stop_reason)
        return {"error": "rca_no_tool_call", "robot_id": robot_id}

    rca = dict(tool_use.input)
    rca["rca_id"] = str(uuid4())
    rca["robot_id"] = robot_id
    rca["timestamp"] = datetime.now(timezone.utc).isoformat()
    rca["findings_analyzed"] = len(condensed)
    rca["input_tokens"] = usage.input_tokens
    rca["output_tokens"] = usage.output_tokens

    logger.warning(
        "RCA complete for %s — risk=%s findings=%d tokens=%d+%d",
        robot_id, rca.get("risk_level", "?"), len(condensed),
        usage.input_tokens, usage.output_tokens,
    )
    return rca
