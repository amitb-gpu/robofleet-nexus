from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cap: int):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ROBOFLEET_RCA_USAGE_FILE", str(tmp_path / "usage.json"))
    monkeypatch.setenv("ROBOFLEET_RCA_MAX_PER_DAY", str(cap))
    import robofleet_nexus.rca.agent as agent

    return importlib.reload(agent)


def test_daily_cap_survives_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent(monkeypatch, tmp_path, cap=3)
    assert [agent._check_spend_guard() for _ in range(2)] == [True, True]

    agent = importlib.reload(agent)  # simulates an API restart
    assert agent._check_spend_guard() is True
    assert agent._check_spend_guard() is False


def test_new_day_resets_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = _agent(monkeypatch, tmp_path, cap=1)
    (tmp_path / "usage.json").write_text('{"date": "2000-01-01", "count": 99}')
    assert agent._check_spend_guard() is True
    assert agent._check_spend_guard() is False


def test_unwritable_usage_file_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setenv("ROBOFLEET_RCA_USAGE_FILE", str(blocker / "usage.json"))
    import robofleet_nexus.rca.agent as agent

    assert importlib.reload(agent)._check_spend_guard() is False


# ── RCA request shape and response handling (fake client, no network) ─────────

class _Block:
    def __init__(self, type_: str, input_: dict | None = None) -> None:
        self.type = type_
        self.input = input_


class _Usage:
    input_tokens = 1200
    output_tokens = 400


class _Response:
    def __init__(self, stop_reason: str, content: list[_Block]) -> None:
        self.stop_reason = stop_reason
        self.content = content
        self.usage = _Usage()


class _FakeMessages:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    api_key = "test"

    def __init__(self, response: _Response) -> None:
        self.messages = _FakeMessages(response)


def _findings(n: int):  # type: ignore[no-untyped-def]
    from robofleet_nexus.telemetry.schemas import DiagnosticFinding, RobotSeverity

    titles = ["Policy halted by safety guard", "Policy actions clamped by safety guard"]
    return [
        DiagnosticFinding(finding_id=str(i), robot_id="arm_demo",
                          severity=RobotSeverity.critical, title=titles[i % 2],
                          explanation="x", evidence=["a", "b"], recommended_actions=["r"])
        for i in range(n)
    ]


REPORT = {
    "summary": "wrist_3 encoder slip caused an out-of-distribution halt.",
    "root_causes": [{"rank": 1, "cause": "encoder slip", "confidence": "high",
                     "subsystem": "kinematics", "evidence": ["9.8 std devs"]}],
    "recommended_actions": [{"priority": "immediate", "action": "reseat encoder",
                             "rationale": "restores observation"}],
    "requires_human_approval": True,
    "risk_level": "critical",
    "estimated_resolution_time": "15 minutes",
}


def _enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, response: _Response):  # type: ignore[no-untyped-def]
    agent = _agent(monkeypatch, tmp_path, cap=25)
    monkeypatch.setenv("ROBOFLEET_RCA_ENABLED", "1")
    client = _FakeClient(response)
    monkeypatch.setattr(agent, "_client", client)
    return agent, client


def test_rca_uses_forced_tool_and_bounded_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio
    import json

    agent, client = _enabled(monkeypatch, tmp_path,
                             _Response("tool_use", [_Block("tool_use", REPORT)]))
    rca = asyncio.run(agent.run_rca("arm_demo", _findings(400), []))

    assert rca["summary"] == REPORT["summary"]
    assert rca["input_tokens"] == 1200 and rca["findings_analyzed"] == 2
    call = client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "report_rca"}
    sent = json.loads(call["messages"][0]["content"])  # type: ignore[index]
    assert [f["occurrences"] for f in sent["active_findings"]] == [200, 200]


def test_rca_truncated_output_is_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    agent, _ = _enabled(monkeypatch, tmp_path,
                        _Response("max_tokens", [_Block("tool_use", {"summary": "cut"})]))
    assert asyncio.run(agent.run_rca("arm_demo", _findings(2), []))["error"] == "rca_truncated"


def test_rca_disabled_uses_mock_without_calling_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    agent, client = _enabled(monkeypatch, tmp_path, _Response("tool_use", []))
    monkeypatch.setenv("ROBOFLEET_RCA_ENABLED", "0")
    rca = asyncio.run(agent.run_rca("arm_demo", _findings(2), []))
    assert rca["mock"] is True and client.messages.calls == []
