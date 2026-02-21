from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trustshield.preprocessing import normalize_text


def score_event(model_bundle: dict[str, Any], payload: dict[str, Any]) -> float:
    model = model_bundle["model"]
    text_vectorizer = model_bundle["text_vectorizer"]
    country_encoder = model_bundle["country_encoder"]

    text = normalize_text(str(payload.get("message_text", "")))
    country = str(payload.get("country", "UNK")).upper()
    payment_attempts = float(payload.get("payment_attempts", 0))
    account_age_days = float(payload.get("account_age_days", 0))
    device_reuse_count = float(payload.get("device_reuse_count", 0))
    chargeback_history = float(payload.get("chargeback_history", 0))

    x_text = text_vectorizer.transform([text]).toarray()
    country_encoded = country_encoder.transform([[country]])

    x_num = np.array(
        [[payment_attempts, account_age_days, device_reuse_count, chargeback_history]],
        dtype=float,
    )

    features = np.hstack([x_text, country_encoded, x_num])
    score = float(model.predict_proba(pd.DataFrame(features))[0][1])
    return score
