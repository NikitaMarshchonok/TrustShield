from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from trustshield.features import build_graph_stats, enrich_with_graph_features
from trustshield.ingestion import generate_synthetic_events
from trustshield.preprocessing import normalize_text, validate_events


def _load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.9) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = recall[precision >= target_precision]
    return float(valid.max()) if len(valid) else 0.0


def _maybe_log_mlflow(cfg: dict, metrics: dict[str, float], artifact_path: Path) -> None:
    mlflow_cfg = cfg.get("mlflow", {})
    if not mlflow_cfg.get("enabled", True):
        return
    try:
        import mlflow
    except Exception:
        print("MLflow is not installed or unavailable. Skipping MLflow logging.")
        return

    tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "trustshield")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="train_ensemble"):
        mlflow.log_params(
            {
                "n_samples": cfg["dataset"]["n_samples"],
                "random_state": cfg["dataset"]["random_state"],
                "max_features_tfidf": cfg["model"]["max_features_tfidf"],
                "logreg_c": cfg["model"]["c"],
            }
        )
        mlflow.log_metrics(metrics)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))


def train() -> dict:
    cfg = _load_training_config(Path("configs/training.yaml"))
    n_samples = int(cfg["dataset"]["n_samples"])
    random_state = int(cfg["dataset"]["random_state"])
    max_features_tfidf = int(cfg["model"]["max_features_tfidf"])
    c = float(cfg["model"]["c"])

    df = generate_synthetic_events(n_samples=n_samples, random_state=random_state)
    df["message_text"] = df["message_text"].map(normalize_text)
    validate_events(df)

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["is_fraud", "event_id"]),
        df["is_fraud"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["is_fraud"],
    )

    train_df_with_target = x_train.copy()
    train_df_with_target["is_fraud"] = y_train.to_numpy()
    graph_stats = build_graph_stats(train_df_with_target, target_col="is_fraud")
    x_train = enrich_with_graph_features(x_train, graph_stats)
    x_test = enrich_with_graph_features(x_test, graph_stats)

    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features_tfidf)
    x_text_train = text_vectorizer.fit_transform(x_train["message_text"]).toarray()
    x_text_test = text_vectorizer.transform(x_test["message_text"]).toarray()
    text_model = LogisticRegression(C=c, max_iter=1000, n_jobs=None)
    text_model.fit(pd.DataFrame(x_text_train), y_train)

    country_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_country_train = country_encoder.fit_transform(x_train[["country"]])
    x_country_test = country_encoder.transform(x_test[["country"]])

    num_cols = [
        "payment_attempts",
        "account_age_days",
        "device_reuse_count",
        "chargeback_history",
        "graph_device_id_degree",
        "graph_ip_id_degree",
        "graph_card_id_degree",
        "graph_max_entity_fraud_rate",
        "graph_mean_entity_degree",
    ]
    x_num_train = x_train[num_cols].to_numpy(dtype=float)
    x_num_test = x_test[num_cols].to_numpy(dtype=float)

    x_train_all = np.hstack([x_country_train, x_num_train])
    x_test_all = np.hstack([x_country_test, x_num_test])

    tabular_model = LogisticRegression(C=c, max_iter=1000, n_jobs=None)
    tabular_model.fit(pd.DataFrame(x_train_all), y_train)

    y_score_text = text_model.predict_proba(pd.DataFrame(x_text_test))[:, 1]
    y_score_tabular = tabular_model.predict_proba(pd.DataFrame(x_test_all))[:, 1]
    ensemble_weights = {"text": 0.45, "tabular": 0.55}
    y_score = ensemble_weights["text"] * y_score_text + ensemble_weights["tabular"] * y_score_tabular
    pr_auc = average_precision_score(y_test, y_score)
    recall_at_90p = _recall_at_precision(y_test.to_numpy(), y_score, target_precision=0.9)

    artifact_path = Path(cfg["output"]["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    ngram_names = np.array(text_vectorizer.get_feature_names_out())
    text_coef = text_model.coef_[0]
    top_idx = np.argsort(text_coef)[-20:]
    top_ngrams = [str(ngram_names[i]) for i in top_idx]

    bundle = {
        "text_model": text_model,
        "tabular_model": tabular_model,
        "text_vectorizer": text_vectorizer,
        "country_encoder": country_encoder,
        "graph_stats": graph_stats,
        "ensemble_weights": ensemble_weights,
        "top_ngrams": top_ngrams,
        "metrics": {
            "pr_auc": float(pr_auc),
            "recall_at_precision_0_90": float(recall_at_90p),
        },
        "meta": {"num_cols": num_cols},
    }
    joblib.dump(bundle, artifact_path)

    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        (
            "{\n"
            f'  "pr_auc": {pr_auc:.6f},\n'
            f'  "recall_at_precision_0_90": {recall_at_90p:.6f}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    print(f"Saved model bundle to {artifact_path}")
    print(f"PR-AUC: {pr_auc:.4f} | Recall@P>=0.90: {recall_at_90p:.4f}")
    _maybe_log_mlflow(
        cfg,
        {"pr_auc": float(pr_auc), "recall_at_precision_0_90": float(recall_at_90p)},
        artifact_path,
    )

    return bundle


if __name__ == "__main__":
    train()
