from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trustshield.ingestion import generate_synthetic_events
from trustshield.models import score_event
from trustshield.preprocessing import normalize_text


def generate_error_analysis_report(threshold: float = 0.5) -> dict:
    artifact_path = Path("reports/artifacts/model_bundle.joblib")
    if not artifact_path.exists():
        raise FileNotFoundError("Model artifact is missing. Run `make train` first.")
    bundle = joblib.load(artifact_path)

    holdout = generate_synthetic_events(n_samples=800, random_state=123)
    holdout["message_text"] = holdout["message_text"].map(normalize_text)

    scores = np.array([score_event(bundle, row) for row in holdout.to_dict(orient="records")])
    preds = (scores >= threshold).astype(int)
    truth = holdout["is_fraud"].to_numpy(dtype=int)

    frame = holdout.copy()
    frame["score"] = scores
    frame["pred"] = preds

    fp = frame[(frame["pred"] == 1) & (truth == 0)].sort_values("score", ascending=False).head(10)
    fn = frame[(frame["pred"] == 0) & (truth == 1)].sort_values("score", ascending=True).head(10)

    report = {
        "threshold": threshold,
        "counts": {
            "false_positives": int(((preds == 1) & (truth == 0)).sum()),
            "false_negatives": int(((preds == 0) & (truth == 1)).sum()),
            "samples": int(len(frame)),
        },
        "top_false_positives": fp[
            ["message_text", "country", "payment_attempts", "account_age_days", "score"]
        ].to_dict(orient="records"),
        "top_false_negatives": fn[
            ["message_text", "country", "payment_attempts", "account_age_days", "score"]
        ].to_dict(orient="records"),
    }

    out_path = Path("reports/error_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Error analysis report saved to {out_path}")
    return report


if __name__ == "__main__":
    generate_error_analysis_report()
