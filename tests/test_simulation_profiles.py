from fastapi.testclient import TestClient

from robofleet_nexus.api.main import app


client = TestClient(app)


def test_simulation_profiles_endpoint() -> None:
    response = client.get("/simulations/profiles")

    assert response.status_code == 200

    profiles = response.json()
    names = {profile["name"] for profile in profiles}

    assert "ci_mock" in names
    assert "laptop_dev" in names
    assert "workstation_l40s" in names
    assert "production_gpu_node" in names
