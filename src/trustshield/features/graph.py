from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ENTITY_COLS = ["device_id", "ip_id", "card_id"]


def build_graph_stats(train_df: pd.DataFrame, target_col: str = "is_fraud") -> dict[str, Any]:
    stats: dict[str, Any] = {"entity_degree": {}, "entity_fraud_rate": {}}
    for col in ENTITY_COLS:
        degree_map = train_df[col].value_counts().astype(float).to_dict()
        fraud_map = train_df.groupby(col)[target_col].mean().astype(float).to_dict()
        stats["entity_degree"][col] = degree_map
        stats["entity_fraud_rate"][col] = fraud_map

    stats["global_degree_mean"] = float(
        np.mean([train_df[col].value_counts().mean() for col in ENTITY_COLS])
    )
    stats["global_fraud_rate"] = float(train_df[target_col].mean())
    return stats


def enrich_with_graph_features(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    for col in ENTITY_COLS:
        degree_map = stats["entity_degree"][col]
        fraud_map = stats["entity_fraud_rate"][col]
        out[f"graph_{col}_degree"] = out[col].map(degree_map).fillna(stats["global_degree_mean"])
        out[f"graph_{col}_fraud_rate"] = out[col].map(fraud_map).fillna(stats["global_fraud_rate"])

    out["graph_max_entity_fraud_rate"] = out[
        [f"graph_{col}_fraud_rate" for col in ENTITY_COLS]
    ].max(axis=1)
    out["graph_mean_entity_degree"] = out[[f"graph_{col}_degree" for col in ENTITY_COLS]].mean(axis=1)
    return out


def graph_features_for_payload(payload: dict[str, Any], stats: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for col in ENTITY_COLS:
        value = str(payload.get(col, f"unknown_{col}"))
        degree_map = stats["entity_degree"][col]
        fraud_map = stats["entity_fraud_rate"][col]
        features[f"graph_{col}_degree"] = float(degree_map.get(value, stats["global_degree_mean"]))
        features[f"graph_{col}_fraud_rate"] = float(fraud_map.get(value, stats["global_fraud_rate"]))

    features["graph_max_entity_fraud_rate"] = float(
        max(features[f"graph_{col}_fraud_rate"] for col in ENTITY_COLS)
    )
    features["graph_mean_entity_degree"] = float(
        np.mean([features[f"graph_{col}_degree"] for col in ENTITY_COLS])
    )
    return features
