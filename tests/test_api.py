import pytest
from fastapi.testclient import TestClient

from ticket_router.api import create_app


def test_valid_request_and_health(app_factory):
    with TestClient(app_factory()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        response = client.post("/predict", json={"subject": "Payment", "description": "card charged twice"})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_category"] == "Billing"
    assert 0 <= body["confidence"] <= 1
    assert body["routing_decision"] == "auto_route"
    assert body["assigned_team"] == "Billing"


def test_manual_review_has_no_assigned_team(app_factory):
    with TestClient(app_factory(threshold=1.0)) as client:
        body = client.post("/predict", json={"subject": "help", "description": "something went wrong"}).json()
    assert body["routing_decision"] == "manual_review"
    assert body["assigned_team"] is None
    assert body["predicted_category"]


@pytest.mark.parametrize("payload", [
    {"subject": "", "description": ""},
    {"subject": " \t", "description": "\n\u00a0"},
    {"subject": "help"},
    {"subject": None, "description": "help"},
    {"subject": 42, "description": "help"},
    {"subject": "x" * 501, "description": "help"},
    {"subject": "help", "description": "x" * 10001},
    {"subject": "help", "description": "help", "threshold": 0},
])
def test_invalid_requests(app_factory, payload):
    with TestClient(app_factory()) as client:
        assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("subject,description", [("", "reset password"), ("reset password", "")])
def test_one_nonblank_field_is_enough(app_factory, subject, description):
    with TestClient(app_factory()) as client:
        assert client.post("/predict", json={"subject": subject, "description": description}).status_code == 200


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Run: python -m ticket_router.train"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
