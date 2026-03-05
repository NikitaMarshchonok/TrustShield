from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from trustshield.evaluation.metrics import cost_saved_metric
from trustshield.ingestion import generate_synthetic_events
from trustshield.models import score_event
from trustshield.preprocessing import normalize_text


def generate_cost_report(block_threshold: float = 0.75) -> dict:
    artifact_path = Path("reports/artifacts/model_bundle.joblib")
    if not artifact_path.exists():
        raise FileNotFoundError("Model artifact is missing. Run `make train` first.")
    bundle = joblib.load(artifact_path)

    holdout = generate_synthetic_events(n_samples=1000, random_state=321)
    holdout["message_text"] = holdout["message_text"].map(normalize_text)

    scores = np.array([score_event(bundle, row) for row in holdout.to_dict(orient="records")])
    y_true = holdout["is_fraud"].to_numpy(dtype=int)
    y_block = (scores >= block_threshold).astype(int)

    saved = cost_saved_metric(y_true, y_block, fraud_cost=100.0, review_cost=2.0)
    report = {
        "block_threshold": block_threshold,
        "n_samples": int(len(holdout)),
        "blocked_events": int(y_block.sum()),
        "estimated_cost_saved": float(saved),
        "assumptions": {"fraud_cost": 100.0, "review_cost": 2.0},
    }

    out_path = Path("reports/cost_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Cost report saved to {out_path}")
    return report


if __name__ == "__main__":
    generate_cost_report()
