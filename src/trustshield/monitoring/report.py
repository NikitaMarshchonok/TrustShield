from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from trustshield.ingestion import generate_synthetic_events
from trustshield.models import score_event
from trustshield.preprocessing import normalize_text


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
    recent["message_text"] = recent["message_text"].map(normalize_text)
    baseline["message_text"] = baseline["message_text"].map(normalize_text)

    recent_scores = np.array([score_event(bundle, row) for row in recent.to_dict(orient="records")])
    baseline_scores = np.array([score_event(bundle, row) for row in baseline.to_dict(orient="records")])
    recent_pr_auc = float(average_precision_score(recent["is_fraud"], recent_scores))
    baseline_pr_auc = float(average_precision_score(baseline["is_fraud"], baseline_scores))

    score_shift = float(abs(recent_scores.mean() - baseline_scores.mean()))
    fraud_rate_shift = float(abs(recent["is_fraud"].mean() - baseline["is_fraud"].mean()))
    payment_attempts_shift = float(
        abs(recent["payment_attempts"].mean() - baseline["payment_attempts"].mean())
    )
    account_age_shift = float(abs(recent["account_age_days"].mean() - baseline["account_age_days"].mean()))
    device_reuse_shift = float(
        abs(recent["device_reuse_count"].mean() - baseline["device_reuse_count"].mean())
    )
    alert_threshold = float(policy["monitoring"]["score_shift_alert"])
    quality_drop_ratio = float(policy["monitoring"]["quality_drop_ratio_alert"])
    latency_p95_alert_ms = float(policy["monitoring"]["latency_p95_ms_alert"])

    latencies_ms = []
    for row in recent.head(250).to_dict(orient="records"):
        started = time.perf_counter()
        _ = score_event(bundle, row)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    latency_p95_ms = float(np.percentile(latencies_ms, 95))

    report = {
        "baseline_score_mean": float(baseline_scores.mean()),
        "recent_score_mean": float(recent_scores.mean()),
        "score_shift_abs": score_shift,
        "baseline_pr_auc": baseline_pr_auc,
        "recent_pr_auc": recent_pr_auc,
        "fraud_rate_shift_abs": fraud_rate_shift,
        "feature_shifts": {
            "payment_attempts_mean_shift_abs": payment_attempts_shift,
            "account_age_days_mean_shift_abs": account_age_shift,
            "device_reuse_count_mean_shift_abs": device_reuse_shift,
        },
        "latency_p95_ms": latency_p95_ms,
        "alert": bool(
            score_shift > alert_threshold
            or recent_pr_auc < (baseline_pr_auc * quality_drop_ratio)
            or latency_p95_ms > latency_p95_alert_ms
        ),
        "alert_thresholds": {
            "score_shift_abs": alert_threshold,
            "quality_drop_ratio": quality_drop_ratio,
            "latency_p95_ms": latency_p95_alert_ms,
        },
    }

    output_path = Path("reports/monitoring.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Monitoring report saved to {output_path}")
    return report


if __name__ == "__main__":
    generate_monitoring_report()
