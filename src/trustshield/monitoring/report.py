from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from trustshield.ingestion import generate_synthetic_events
from trustshield.models import score_event


def _load_policy() -> dict:
    with Path("configs/policy.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_monitoring_report() -> dict:
    policy = _load_policy()
    bundle_path = Path("reports/artifacts/model_bundle.joblib")
    if not bundle_path.exists():
        raise FileNotFoundError("Model artifact is missing. Run `make train` first.")

    bundle = joblib.load(bundle_path)
    recent = generate_synthetic_events(n_samples=600, random_state=7)
    baseline = generate_synthetic_events(n_samples=600, random_state=42)

    recent_scores = np.array([score_event(bundle, row) for row in recent.to_dict(orient="records")])
    baseline_scores = np.array([score_event(bundle, row) for row in baseline.to_dict(orient="records")])

    score_shift = float(abs(recent_scores.mean() - baseline_scores.mean()))
    alert_threshold = float(policy["monitoring"]["score_shift_alert"])

    report = {
        "baseline_score_mean": float(baseline_scores.mean()),
        "recent_score_mean": float(recent_scores.mean()),
        "score_shift_abs": score_shift,
        "alert": bool(score_shift > alert_threshold),
        "alert_threshold": alert_threshold,
    }

    output_path = Path("reports/monitoring.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Monitoring report saved to {output_path}")
    return report


if __name__ == "__main__":
    generate_monitoring_report()
