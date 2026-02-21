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

from trustshield.ingestion import generate_synthetic_events
from trustshield.preprocessing import normalize_text


def _load_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.9) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = recall[precision >= target_precision]
    return float(valid.max()) if len(valid) else 0.0


def train() -> dict:
    cfg = _load_training_config(Path("configs/training.yaml"))
    n_samples = int(cfg["dataset"]["n_samples"])
    random_state = int(cfg["dataset"]["random_state"])
    max_features_tfidf = int(cfg["model"]["max_features_tfidf"])
    c = float(cfg["model"]["c"])

    df = generate_synthetic_events(n_samples=n_samples, random_state=random_state)
    df["message_text"] = df["message_text"].map(normalize_text)

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["is_fraud", "event_id"]),
        df["is_fraud"],
        test_size=0.2,
        random_state=random_state,
        stratify=df["is_fraud"],
    )

    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features_tfidf)
    x_text_train = text_vectorizer.fit_transform(x_train["message_text"]).toarray()
    x_text_test = text_vectorizer.transform(x_test["message_text"]).toarray()

    country_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_country_train = country_encoder.fit_transform(x_train[["country"]])
    x_country_test = country_encoder.transform(x_test[["country"]])

    num_cols = ["payment_attempts", "account_age_days", "device_reuse_count", "chargeback_history"]
    x_num_train = x_train[num_cols].to_numpy(dtype=float)
    x_num_test = x_test[num_cols].to_numpy(dtype=float)

    x_train_all = np.hstack([x_text_train, x_country_train, x_num_train])
    x_test_all = np.hstack([x_text_test, x_country_test, x_num_test])

    model = LogisticRegression(C=c, max_iter=1000, n_jobs=None)
    model.fit(pd.DataFrame(x_train_all), y_train)

    y_score = model.predict_proba(pd.DataFrame(x_test_all))[:, 1]
    pr_auc = average_precision_score(y_test, y_score)
    recall_at_90p = _recall_at_precision(y_test.to_numpy(), y_score, target_precision=0.9)

    artifact_path = Path(cfg["output"]["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "text_vectorizer": text_vectorizer,
        "country_encoder": country_encoder,
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

    return bundle


if __name__ == "__main__":
    train()
