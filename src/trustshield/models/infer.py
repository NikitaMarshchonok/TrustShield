from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trustshield.features import graph_features_for_payload
from trustshield.preprocessing import normalize_text


def explain_event(model_bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    text_model = model_bundle["text_model"]
    tabular_model = model_bundle["tabular_model"]
    text_vectorizer = model_bundle["text_vectorizer"]
    country_encoder = model_bundle["country_encoder"]
    graph_stats = model_bundle["graph_stats"]
    weights = model_bundle["ensemble_weights"]
    top_ngrams = set(model_bundle.get("top_ngrams", []))

    text = normalize_text(str(payload.get("message_text", "")))
    country = str(payload.get("country", "UNK")).upper()
    payment_attempts = float(payload.get("payment_attempts", 0))
    account_age_days = float(payload.get("account_age_days", 0))
    device_reuse_count = float(payload.get("device_reuse_count", 0))
    chargeback_history = float(payload.get("chargeback_history", 0))

    x_text = text_vectorizer.transform([text]).toarray()
    country_encoded = country_encoder.transform([[country]])
    graph_features = graph_features_for_payload(payload, graph_stats)

    x_num = np.array(
        [
            [
                payment_attempts,
                account_age_days,
                device_reuse_count,
                chargeback_history,
                graph_features["graph_device_id_degree"],
                graph_features["graph_ip_id_degree"],
                graph_features["graph_card_id_degree"],
                graph_features["graph_max_entity_fraud_rate"],
                graph_features["graph_mean_entity_degree"],
            ]
        ],
        dtype=float,
    )

    tabular_features = np.hstack([country_encoded, x_num])

    text_score = float(text_model.predict_proba(pd.DataFrame(x_text))[0][1])
    tabular_score = float(tabular_model.predict_proba(pd.DataFrame(tabular_features))[0][1])
    risk_score = float(weights["text"] * text_score + weights["tabular"] * tabular_score)

    message_tokens = set(text.split())
    model_reasons = sorted(ng for ng in top_ngrams if ng in message_tokens)[:5]

    return {
        "risk_score": risk_score,
        "text_score": text_score,
        "tabular_score": tabular_score,
        "graph_max_entity_fraud_rate": graph_features["graph_max_entity_fraud_rate"],
        "model_reasons": model_reasons,
    }


def score_event(model_bundle: dict[str, Any], payload: dict[str, Any]) -> float:
    return float(explain_event(model_bundle, payload)["risk_score"])
