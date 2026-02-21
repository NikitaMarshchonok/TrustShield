from fastapi.testclient import TestClient

from trustshield.serving.app import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_predict() -> None:
    payload = {
        "message_text": "Urgent transfer, click this link now",
        "country": "NG",
        "payment_attempts": 5,
        "account_age_days": 1,
        "device_reuse_count": 5,
        "chargeback_history": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in {"allow", "review", "block"}
    assert isinstance(body["reasons"], list)
