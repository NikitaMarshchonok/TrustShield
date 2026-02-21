from __future__ import annotations

from typing import Any


def extract_reason_flags(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    text = str(payload.get("message_text", "")).lower()
    if any(token in text for token in ("otp", "urgent", "click", "transfer", "outside platform")):
        reasons.append("suspicious_message_pattern")

    if int(payload.get("payment_attempts", 0)) >= 4:
        reasons.append("high_payment_attempts")

    if int(payload.get("account_age_days", 0)) < 7:
        reasons.append("new_account")

    if int(payload.get("device_reuse_count", 0)) >= 4:
        reasons.append("high_device_reuse")

    if int(payload.get("chargeback_history", 0)) == 1:
        reasons.append("prior_chargeback")

    if str(payload.get("country", "")).upper() in {"NG", "RU"}:
        reasons.append("high_risk_country")

    return reasons
