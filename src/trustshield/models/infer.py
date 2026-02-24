from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trustshield.features import graph_features_for_payload
from trustshield.preprocessing import normalize_text


def _top_text_matches(text: str, ngrams: set[str], limit: int = 5) -> list[str]:
    text_tokens = text.split()
    matched: list[str] = []
    for ng in sorted(ngrams):
        ng_tokens = ng.split()
        n = len(ng_tokens)
        if n == 1 and ng_tokens[0] in text_tokens:
            matched.append(ng)
        elif n > 1:
            for i in range(max(len(text_tokens) - n + 1, 0)):
                if text_tokens[i : i + n] == ng_tokens:
                    matched.append(ng)
                    break
        if len(matched) >= limit:
            break
    return matched


def _tabular_explain(
    model: Any, feature_names: list[str], feature_row: np.ndarray
) -> tuple[dict[str, float], str]:
    try:
        import shap  # type: ignore

        explainer = shap.LinearExplainer(model, feature_row, feature_perturbation="interventional")
        shap_values = explainer.shap_values(feature_row)
        values = np.asarray(shap_values)[0]
        raw = {feature_names[i]: float(values[i]) for i in range(min(len(values), len(feature_names)))}
        top = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        return ({name: round(value, 4) for name, value in top}, "shap")
    except Exception:
        coef = model.coef_[0]
        contrib_values = feature_row[0] * coef
        raw = {feature_names[i]: float(contrib_values[i]) for i in range(min(len(coef), len(feature_names)))}
        top = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        return ({name: round(value, 4) for name, value in top}, "linear_coef")


def explain_event(model_bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    text_model = model_bundle["text_model"]
    tabular_model = model_bundle["tabular_model"]
    text_vectorizer = model_bundle["text_vectorizer"]
    country_encoder = model_bundle["country_encoder"]
    graph_stats = model_bundle["graph_stats"]
    weights = model_bundle["ensemble_weights"]
    top_ngrams = set(model_bundle.get("top_ngrams", []))
    tabular_feature_names = model_bundle.get("tabular_feature_names", [])

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
                graph_features["graph_device_id_pagerank"],
                graph_features["graph_ip_id_pagerank"],
                graph_features["graph_card_id_pagerank"],
                graph_features["graph_device_id_component_size"],
                graph_features["graph_ip_id_component_size"],
                graph_features["graph_card_id_component_size"],
                graph_features["graph_max_entity_fraud_rate"],
                graph_features["graph_mean_entity_degree"],
                graph_features["graph_max_entity_pagerank"],
                graph_features["graph_min_component_size"],
            ]
        ],
        dtype=float,
    )

    tabular_features = np.hstack([country_encoded, x_num])

    text_score = float(text_model.predict_proba(pd.DataFrame(x_text))[0][1])
    tabular_score = float(tabular_model.predict_proba(pd.DataFrame(tabular_features))[0][1])
    risk_score = float(weights["text"] * text_score + weights["tabular"] * tabular_score)

    model_reasons = _top_text_matches(text, top_ngrams, limit=5)
    feature_contributions: dict[str, float] = {}
    explanation_method = "none"
    if tabular_feature_names:
        feature_contributions, explanation_method = _tabular_explain(
            tabular_model, tabular_feature_names, tabular_features
        )

    return {
        "risk_score": risk_score,
        "text_score": text_score,
        "tabular_score": tabular_score,
        "graph_max_entity_fraud_rate": graph_features["graph_max_entity_fraud_rate"],
        "graph_max_entity_pagerank": graph_features["graph_max_entity_pagerank"],
        "model_reasons": model_reasons,
        "feature_contributions": feature_contributions,
        "explanation_method": explanation_method,
    }


def score_event(model_bundle: dict[str, Any], payload: dict[str, Any]) -> float:
    return float(explain_event(model_bundle, payload)["risk_score"])
