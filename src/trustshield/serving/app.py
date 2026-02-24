from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from trustshield.models import explain_event
from trustshield.serving.policy import decide, init_policy_state, load_policy, reset_policy_state
from trustshield.serving.schemas import PredictRequest, PredictResponse


class HeuristicFallbackModel:
    def predict(self, payload: dict[str, Any]) -> float:
        score = 0.05
        text = payload["message_text"].lower()
        if any(token in text for token in ("urgent", "otp", "click", "outside platform", "transfer")):
            score += 0.35
        score += min(payload["payment_attempts"] * 0.06, 0.25)
        score += 0.2 if payload["account_age_days"] < 7 else 0.0
        score += 0.18 if payload["chargeback_history"] == 1 else 0.0
        score += 0.14 if payload["device_reuse_count"] >= 4 else 0.0
        return float(min(score, 0.99))


def _load_model_bundle(path: str = "reports/artifacts/model_bundle.joblib") -> dict[str, Any] | None:
    artifact = Path(path)
    if artifact.exists():
        return joblib.load(artifact)
    return None


app = FastAPI(title="TrustShield API", version="0.1.0")
policy_cfg = load_policy()
bundle = _load_model_bundle()
fallback = HeuristicFallbackModel()
policy_state = init_policy_state()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "policy_loaded": bool(policy_cfg),
    }


@app.get("/monitoring/summary")
def monitoring_summary() -> dict[str, Any]:
    report_path = Path("reports/monitoring.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make monitor` to generate report."}
    return {"status": "ok", "report": json.loads(report_path.read_text(encoding="utf-8"))}


@app.get("/monitoring/dashboard", response_class=HTMLResponse)
def monitoring_dashboard() -> HTMLResponse:
    dashboard_path = Path("reports/dashboard.html")
    if not dashboard_path.exists():
        return HTMLResponse(
            content="<h3>Dashboard missing. Run `make dashboard` first.</h3>",
            status_code=404,
        )
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"), status_code=200)


@app.post("/policy/reset")
def policy_reset() -> dict[str, str]:
    reset_policy_state(policy_state)
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    payload = req.model_dump()
    if bundle is not None:
        model_output = explain_event(bundle, payload)
        score = model_output["risk_score"]
        model_reasons = model_output["model_reasons"]
        feature_contributions = model_output["feature_contributions"]
        explanation_method = model_output.get("explanation_method", "none")
        components = {
            "text_score": round(float(model_output["text_score"]), 4),
            "tabular_score": round(float(model_output["tabular_score"]), 4),
            "graph_max_entity_fraud_rate": round(float(model_output["graph_max_entity_fraud_rate"]), 4),
            "graph_max_entity_pagerank": round(float(model_output["graph_max_entity_pagerank"]), 8),
        }
    else:
        score = fallback.predict(payload)
        model_reasons = []
        feature_contributions = {}
        explanation_method = "fallback"
        components = {"text_score": round(score, 4), "tabular_score": round(score, 4)}
    decision, reasons, policy_triggers = decide(score, payload, policy_cfg, state=policy_state)
    return PredictResponse(
        risk_score=round(score, 4),
        decision=decision,
        reasons=reasons,
        model_reasons=model_reasons,
        components=components,
        feature_contributions=feature_contributions,
        policy_triggers=policy_triggers,
        explanation_method=explanation_method,
    )


@app.post("/explain", response_model=PredictResponse)
def explain(req: PredictRequest) -> PredictResponse:
    return predict(req)
