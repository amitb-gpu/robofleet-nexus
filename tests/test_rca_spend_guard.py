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
