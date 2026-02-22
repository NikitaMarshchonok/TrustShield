from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
try:
    import networkx as nx
except Exception:
    nx = None


ENTITY_COLS = ["device_id", "ip_id", "card_id"]
NODE_COLS = ["user_id", "device_id", "ip_id", "card_id", "merchant_id"]


def _node_name(col: str, value: str) -> str:
    return f"{col}:{value}"


def build_graph_stats(train_df: pd.DataFrame, target_col: str = "is_fraud") -> dict[str, Any]:
    stats: dict[str, Any] = {
        "entity_degree": {},
        "entity_fraud_rate": {},
        "entity_pagerank": {},
        "entity_component_size": {},
    }

    node_pagerank: dict[str, float] = {}
    node_component_size: dict[str, float] = {}
    if nx is not None:
        graph = nx.Graph()
        for row in train_df[NODE_COLS].to_dict(orient="records"):
            row_nodes = [_node_name(col, str(row[col])) for col in NODE_COLS]
            for i in range(1, len(row_nodes)):
                graph.add_edge(row_nodes[0], row_nodes[i])

        if graph.number_of_nodes() > 0:
            node_pagerank = nx.pagerank(graph, alpha=0.85)
            for component in nx.connected_components(graph):
                comp_size = float(len(component))
                for node in component:
                    node_component_size[node] = comp_size

    for col in ENTITY_COLS:
        degree_map = train_df[col].value_counts().astype(float).to_dict()
        fraud_map = train_df.groupby(col)[target_col].mean().astype(float).to_dict()
        stats["entity_degree"][col] = degree_map
        stats["entity_fraud_rate"][col] = fraud_map
        if node_pagerank:
            stats["entity_pagerank"][col] = {
                value: float(node_pagerank.get(_node_name(col, value), 0.0)) for value in degree_map.keys()
            }
            stats["entity_component_size"][col] = {
                value: float(node_component_size.get(_node_name(col, value), 1.0)) for value in degree_map.keys()
            }
        else:
            degree_values = np.array(list(degree_map.values()), dtype=float)
            fallback_pr = degree_values / max(float(degree_values.sum()), 1.0)
            stats["entity_pagerank"][col] = {
                value: float(fallback_pr[idx]) for idx, value in enumerate(degree_map.keys())
            }
            stats["entity_component_size"][col] = {value: 1.0 for value in degree_map.keys()}

    stats["global_degree_mean"] = float(
        np.mean([train_df[col].value_counts().mean() for col in ENTITY_COLS])
    )
    stats["global_fraud_rate"] = float(train_df[target_col].mean())
    stats["global_pagerank_mean"] = float(
        np.mean(
            [np.mean(list(stats["entity_pagerank"][col].values()) or [0.0]) for col in ENTITY_COLS]
        )
    )
    stats["global_component_size_mean"] = float(
        np.mean(
            [np.mean(list(stats["entity_component_size"][col].values()) or [1.0]) for col in ENTITY_COLS]
        )
    )
    return stats


def enrich_with_graph_features(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    for col in ENTITY_COLS:
        degree_map = stats["entity_degree"][col]
        fraud_map = stats["entity_fraud_rate"][col]
        pagerank_map = stats["entity_pagerank"][col]
        component_map = stats["entity_component_size"][col]
        out[f"graph_{col}_degree"] = out[col].map(degree_map).fillna(stats["global_degree_mean"])
        out[f"graph_{col}_fraud_rate"] = out[col].map(fraud_map).fillna(stats["global_fraud_rate"])
        out[f"graph_{col}_pagerank"] = out[col].map(pagerank_map).fillna(stats["global_pagerank_mean"])
        out[f"graph_{col}_component_size"] = out[col].map(component_map).fillna(
            stats["global_component_size_mean"]
        )

    out["graph_max_entity_fraud_rate"] = out[
        [f"graph_{col}_fraud_rate" for col in ENTITY_COLS]
    ].max(axis=1)
    out["graph_mean_entity_degree"] = out[[f"graph_{col}_degree" for col in ENTITY_COLS]].mean(axis=1)
    out["graph_max_entity_pagerank"] = out[[f"graph_{col}_pagerank" for col in ENTITY_COLS]].max(axis=1)
    out["graph_min_component_size"] = out[
        [f"graph_{col}_component_size" for col in ENTITY_COLS]
    ].min(axis=1)
    return out


def graph_features_for_payload(payload: dict[str, Any], stats: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for col in ENTITY_COLS:
        value = str(payload.get(col, f"unknown_{col}"))
        degree_map = stats["entity_degree"][col]
        fraud_map = stats["entity_fraud_rate"][col]
        pagerank_map = stats["entity_pagerank"][col]
        component_map = stats["entity_component_size"][col]
        features[f"graph_{col}_degree"] = float(degree_map.get(value, stats["global_degree_mean"]))
        features[f"graph_{col}_fraud_rate"] = float(fraud_map.get(value, stats["global_fraud_rate"]))
        features[f"graph_{col}_pagerank"] = float(
            pagerank_map.get(value, stats["global_pagerank_mean"])
        )
        features[f"graph_{col}_component_size"] = float(
            component_map.get(value, stats["global_component_size_mean"])
        )

    features["graph_max_entity_fraud_rate"] = float(
        max(features[f"graph_{col}_fraud_rate"] for col in ENTITY_COLS)
    )
    features["graph_mean_entity_degree"] = float(
        np.mean([features[f"graph_{col}_degree"] for col in ENTITY_COLS])
    )
    features["graph_max_entity_pagerank"] = float(
        max(features[f"graph_{col}_pagerank"] for col in ENTITY_COLS)
    )
    features["graph_min_component_size"] = float(
        min(features[f"graph_{col}_component_size"] for col in ENTITY_COLS)
    )
    return features
