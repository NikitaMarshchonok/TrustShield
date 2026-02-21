from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    message_text: str = Field(..., min_length=1)
    country: str = Field(default="US")
    payment_attempts: int = Field(default=1, ge=0)
    account_age_days: int = Field(default=30, ge=0)
    device_reuse_count: int = Field(default=1, ge=0)
    chargeback_history: int = Field(default=0, ge=0, le=1)


class PredictResponse(BaseModel):
    risk_score: float
    decision: str
    reasons: list[str]
