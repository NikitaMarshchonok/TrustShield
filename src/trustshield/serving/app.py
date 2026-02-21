from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI

from trustshield.models import score_event
from trustshield.serving.policy import decide, load_policy
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "policy_loaded": bool(policy_cfg),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    payload = req.model_dump()
    if bundle is not None:
        score = score_event(bundle, payload)
    else:
        score = fallback.predict(payload)
    decision, reasons = decide(score, payload, policy_cfg)
    return PredictResponse(risk_score=round(score, 4), decision=decision, reasons=reasons)


@app.post("/explain", response_model=PredictResponse)
def explain(req: PredictRequest) -> PredictResponse:
    return predict(req)
