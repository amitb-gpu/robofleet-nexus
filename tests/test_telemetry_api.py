from fastapi.testclient import TestClient

from robofleet_nexus.api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_gpu_temperature_event_creates_finding() -> None:
    response = client.post(
        "/telemetry",
        json={
            "event_id": "evt-test-001",
            "robot_id": "robot-alpha",
            "source": "mock",
            "severity": "warning",
            "event_type": "thermal.telemetry",
            "subsystem": "gpu",
            "message": "GPU temperature elevated during simulation batch",
            "metrics": {
                "gpu_temp_c": 88,
                "packet_loss_pct": 1.2,
            },
            "metadata": {
                "host": "ganymede",
                "gpu": "L40S",
                "workflow": "isaac_sim_batch",
            },
        },
    )

    assert response.status_code == 200
    findings = response.json()

    assert len(findings) == 1
    assert findings[0]["title"] == "Elevated GPU temperature"
    assert findings[0]["severity"] == "warning"
    assert findings[0]["robot_id"] == "robot-alpha"
    assert "gpu_temp_c=88.0" in findings[0]["evidence"]
