from app.main import create_app
from fastapi.testclient import TestClient


def test_app_starts_and_reports_liveness():
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
