from fastapi.testclient import TestClient

from trustshield.serving.app import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_monitoring_summary_missing_report() -> None:
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "missing"}


def test_metrics_latest_endpoint() -> None:
    response = client.get("/metrics/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "missing"}


def test_policy_simulation_latest_endpoint() -> None:
    response = client.get("/policy/simulation/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "missing"}


def test_error_analysis_latest_endpoint() -> None:
    response = client.get("/error-analysis/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "missing"}


def test_reports_status_endpoint() -> None:
    response = client.get("/reports/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "reports" in body
    assert "metrics" in body["reports"]


def test_monitoring_dashboard_endpoint() -> None:
    response = client.get("/monitoring/dashboard")
    assert response.status_code in {200, 404}


def test_policy_reset_endpoint() -> None:
    response = client.post("/policy/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict() -> None:
    payload = {
        "message_text": "Urgent transfer, click this link now",
        "country": "NG",
        "device_id": "device_0001",
        "ip_id": "ip_0001",
        "card_id": "card_0001",
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
    assert isinstance(body["components"], dict)
    assert isinstance(body["feature_contributions"], dict)
    assert isinstance(body["policy_triggers"], list)
    assert body["explanation_method"] in {"shap", "linear_coef", "fallback", "none"}


def test_explain() -> None:
    payload = {
        "message_text": "Please click and send otp urgently",
        "country": "RU",
        "device_id": "device_0002",
        "ip_id": "ip_0002",
        "card_id": "card_0002",
        "payment_attempts": 4,
        "account_age_days": 2,
        "device_reuse_count": 4,
        "chargeback_history": 1,
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "text_score" in body["components"]
    assert "tabular_score" in body["components"]
    assert "feature_contributions" in body
    assert body["explanation_method"] in {"shap", "linear_coef", "fallback", "none"}
