import pandas as pd

from trustshield.features.graph import (
    build_graph_stats,
    enrich_with_graph_features,
    graph_features_for_payload,
)


def test_graph_stats_and_payload_features() -> None:
    train_df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "device_id": "d1",
                "ip_id": "i1",
                "card_id": "c1",
                "merchant_id": "m1",
                "is_fraud": 1,
            },
            {
                "user_id": "u2",
                "device_id": "d2",
                "ip_id": "i2",
                "card_id": "c2",
                "merchant_id": "m1",
                "is_fraud": 0,
            },
        ]
    )
    stats = build_graph_stats(train_df, target_col="is_fraud")

    payload = {
        "device_id": "d1",
        "ip_id": "i1",
        "card_id": "c1",
    }
    features = graph_features_for_payload(payload, stats)
    assert "graph_device_id_degree" in features
    assert "graph_max_entity_pagerank" in features
    assert "graph_min_component_size" in features


def test_enrich_with_graph_features_columns() -> None:
    base = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "device_id": "d1",
                "ip_id": "i1",
                "card_id": "c1",
                "merchant_id": "m1",
            }
        ]
    )
    train_df = base.copy()
    train_df["is_fraud"] = [1]
    stats = build_graph_stats(train_df, target_col="is_fraud")
    out = enrich_with_graph_features(base, stats)
    assert "graph_device_id_pagerank" in out.columns
    assert "graph_min_component_size" in out.columns
