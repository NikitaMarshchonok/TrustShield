from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from trustshield.evaluation.cost_report import generate_cost_report
from trustshield.evaluation.error_analysis import generate_error_analysis_report
from trustshield.evaluation.policy_simulation import run_policy_simulation
from trustshield.models import explain_event
from trustshield.monitoring.dashboard import build_dashboard_html
from trustshield.monitoring.report import generate_monitoring_report
from trustshield.serving.policy import (
    decide,
    init_policy_state,
    load_policy,
    policy_state_summary,
    reset_policy_state,
)
from trustshield.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    ReportsGenerateRequest,
)


class HeuristicFallbackModel:
    def predict(self, payload: dict[str, Any]) -> float:
        score = 0.05
        text = payload["message_text"].lower()
        if any(token in text for token in ("urgent", "otp", "click", "outside platform", "transfer")):
            score += 0.35
        score += min(payload["payment_attempts"] * 0.06, 0.25)
        score += 0.2 if payload["account_age_days"] < 7 else 0.0
        score += 0.18 if payload["chargeback_history"] == 1 else 0.0
        score += 0.14 if payload["device_reuse_count"] >= 4 else 0.0
        return float(min(score, 0.99))


def _load_model_bundle(path: str = "reports/artifacts/model_bundle.joblib") -> dict[str, Any] | None:
    artifact = Path(path)
    if artifact.exists():
        return joblib.load(artifact)
    return None


app = FastAPI(title="TrustShield API", version="0.1.0")
policy_cfg = load_policy()
bundle = _load_model_bundle()
fallback = HeuristicFallbackModel()
policy_runtime_state = init_policy_state()
serving_stats_lock = threading.Lock()
serving_stats: dict[str, Any] = {
    "total_requests": 0,
    "predict_requests": 0,
    "batch_requests": 0,
    "predicted_items": 0,
    "decision_counts": {"allow": 0, "review": 0, "block": 0},
    "latency_ms_window": [],
}
LATENCY_WINDOW_SIZE = 500


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "policy_loaded": bool(policy_cfg),
    }


@app.get("/serving/stats", tags=["serving"])
def serving_stats_snapshot() -> dict[str, Any]:
    with serving_stats_lock:
        snapshot = {
            "total_requests": int(serving_stats["total_requests"]),
            "predict_requests": int(serving_stats["predict_requests"]),
            "batch_requests": int(serving_stats["batch_requests"]),
            "predicted_items": int(serving_stats["predicted_items"]),
            "decision_counts": dict(serving_stats["decision_counts"]),
        }
    return {"status": "ok", "stats": snapshot}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = int(round((len(sorted_values) - 1) * q))
    pos = max(0, min(pos, len(sorted_values) - 1))
    return float(sorted_values[pos])


@app.get("/serving/latency", tags=["serving"])
def serving_latency_snapshot() -> dict[str, Any]:
    with serving_stats_lock:
        values = list(serving_stats["latency_ms_window"])
    if not values:
        return {"status": "ok", "count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "latest_ms": 0.0}
    return {
        "status": "ok",
        "count": len(values),
        "p50_ms": round(_percentile(values, 0.50), 4),
        "p95_ms": round(_percentile(values, 0.95), 4),
        "latest_ms": round(float(values[-1]), 4),
    }


@app.post("/serving/stats/reset", tags=["serving"])
def serving_stats_reset() -> dict[str, str]:
    with serving_stats_lock:
        serving_stats["total_requests"] = 0
        serving_stats["predict_requests"] = 0
        serving_stats["batch_requests"] = 0
        serving_stats["predicted_items"] = 0
        serving_stats["decision_counts"] = {"allow": 0, "review": 0, "block": 0}
        serving_stats["latency_ms_window"] = []
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def health_ready() -> dict[str, Any]:
    checks = {
        "policy_loaded": bool(policy_cfg),
        "model_loaded": bundle is not None,
        "model_artifact_exists": Path("reports/artifacts/model_bundle.joblib").exists(),
        "training_metrics_exists": Path("reports/metrics.json").exists(),
    }
    ready = all(checks.values())
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@app.get("/openapi/tags-summary", tags=["health"])
def openapi_tags_summary() -> dict[str, Any]:
    tags_map: dict[str, int] = {}
    for route in app.routes:
        if not hasattr(route, "tags"):
            continue
        route_tags = getattr(route, "tags", None) or []
        for tag in route_tags:
            tags_map[str(tag)] = tags_map.get(str(tag), 0) + 1
    return {
        "status": "ok",
        "tags_count": len(tags_map),
        "tags": {k: tags_map[k] for k in sorted(tags_map.keys())},
    }


@app.get("/model/info", tags=["model"])
def model_info() -> dict[str, Any]:
    artifact_path = Path("reports/artifacts/model_bundle.joblib")
    if bundle is None:
        return {"status": "missing", "message": "Run `make train` to generate model artifact."}

    metrics = bundle.get("metrics", {})
    top_ngrams = bundle.get("top_ngrams", [])
    tabular_feature_names = bundle.get("tabular_feature_names", [])
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": str(bundle.get("model_version", "unknown")),
        "artifact_updated_at_epoch": int(artifact_path.stat().st_mtime) if artifact_path.exists() else None,
        "ensemble_weights": bundle.get("ensemble_weights", {}),
        "metrics": metrics,
        "text_top_ngrams_count": len(top_ngrams),
        "tabular_feature_count": len(tabular_feature_names),
    }


@app.get("/monitoring/summary", tags=["monitoring"])
def monitoring_summary() -> dict[str, Any]:
    report_path = Path("reports/monitoring.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make monitor` to generate report."}
    return {"status": "ok", "report": json.loads(report_path.read_text(encoding="utf-8"))}


@app.get("/latency/latest", tags=["monitoring"])
def latency_latest() -> dict[str, Any]:
    report_path = Path("reports/monitoring.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make monitor` to generate latency metrics."}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    latency_p95_ms = report.get("latency_p95_ms")
    threshold_ms = policy_cfg.get("monitoring", {}).get("latency_p95_ms_alert")
    if latency_p95_ms is None or threshold_ms is None:
        return {"status": "missing", "message": "Latency fields are missing in monitoring config/report."}

    latency_status = "ok" if float(latency_p95_ms) <= float(threshold_ms) else "warn"
    return {
        "status": "ok",
        "latency_status": latency_status,
        "latency_p95_ms": float(latency_p95_ms),
        "threshold_ms": float(threshold_ms),
    }


@app.get("/alerts/latest", tags=["monitoring"])
def alerts_latest() -> dict[str, Any]:
    report_path = Path("reports/monitoring.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make monitor` to generate alert sources."}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    active_alerts: list[dict[str, Any]] = []

    if bool(report.get("alert", False)):
        active_alerts.append(
            {
                "source": "monitoring",
                "type": "drift_or_quality_or_latency",
                "severity": "high",
            }
        )

    latency_p95_ms = report.get("latency_p95_ms")
    latency_threshold = policy_cfg.get("monitoring", {}).get("latency_p95_ms_alert")
    if latency_p95_ms is not None and latency_threshold is not None and float(latency_p95_ms) > float(latency_threshold):
        active_alerts.append(
            {
                "source": "latency",
                "type": "latency_p95_budget_exceeded",
                "severity": "medium",
                "value": float(latency_p95_ms),
                "threshold": float(latency_threshold),
            }
        )

    return {
        "status": "ok",
        "has_alerts": bool(active_alerts),
        "alerts_count": len(active_alerts),
        "alerts": active_alerts,
    }


@app.get("/quality/latest", tags=["monitoring"])
def quality_latest() -> dict[str, Any]:
    report_path = Path("reports/monitoring.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make monitor` to generate quality metrics."}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline_pr_auc = report.get("baseline_pr_auc")
    recent_pr_auc = report.get("recent_pr_auc")
    quality_ratio_threshold = policy_cfg.get("monitoring", {}).get("quality_drop_ratio_alert")
    if baseline_pr_auc is None or recent_pr_auc is None or quality_ratio_threshold is None:
        return {"status": "missing", "message": "Quality fields are missing in monitoring config/report."}

    ratio = float(recent_pr_auc) / max(float(baseline_pr_auc), 1e-9)
    quality_status = "ok" if ratio >= float(quality_ratio_threshold) else "warn"
    return {
        "status": "ok",
        "quality_status": quality_status,
        "baseline_pr_auc": float(baseline_pr_auc),
        "recent_pr_auc": float(recent_pr_auc),
        "quality_ratio": ratio,
        "threshold_ratio": float(quality_ratio_threshold),
    }


@app.get("/drift/latest", tags=["monitoring"])
def drift_latest() -> dict[str, Any]:
    report_path = Path("reports/monitoring.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make monitor` to generate drift metrics."}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    score_shift_abs = report.get("score_shift_abs")
    feature_shifts = report.get("feature_shifts", {})
    score_shift_threshold = policy_cfg.get("monitoring", {}).get("score_shift_alert")
    if score_shift_abs is None or score_shift_threshold is None:
        return {"status": "missing", "message": "Drift fields are missing in monitoring config/report."}

    drift_status = "ok" if float(score_shift_abs) <= float(score_shift_threshold) else "warn"
    return {
        "status": "ok",
        "drift_status": drift_status,
        "score_shift_abs": float(score_shift_abs),
        "threshold_abs": float(score_shift_threshold),
        "feature_shifts": feature_shifts,
    }


@app.get("/decision-mix/latest", tags=["policy"])
def decision_mix_latest() -> dict[str, Any]:
    report_path = Path("reports/policy_simulation.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make policy-sim` to generate decision mix metrics."}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    decisions = report.get("decisions")
    if not isinstance(decisions, dict):
        return {"status": "missing", "message": "Decision fields are missing in policy simulation report."}

    allow = int(decisions.get("allow", 0))
    review = int(decisions.get("review", 0))
    block = int(decisions.get("block", 0))
    total = max(allow + review + block, 1)

    return {
        "status": "ok",
        "counts": {"allow": allow, "review": review, "block": block},
        "ratios": {
            "allow": round(allow / total, 4),
            "review": round(review / total, 4),
            "block": round(block / total, 4),
        },
        "review_precision_proxy": report.get("review_precision_proxy"),
        "block_precision_proxy": report.get("block_precision_proxy"),
        "n_events": report.get("n_events"),
    }


@app.get("/policy/triggers/latest", tags=["policy"])
def policy_triggers_latest() -> dict[str, Any]:
    report_path = Path("reports/policy_simulation.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make policy-sim` to generate trigger stats."}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    top_triggers = report.get("top_policy_triggers")
    if not isinstance(top_triggers, list):
        return {"status": "missing", "message": "Trigger fields are missing in policy simulation report."}

    normalized = []
    for item in top_triggers:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append({"trigger": str(item[0]), "count": int(item[1])})

    return {"status": "ok", "top_policy_triggers": normalized}


@app.get("/cost/latest", tags=["reports"])
def cost_latest() -> dict[str, Any]:
    report_path = Path("reports/cost_report.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make cost-report` to generate cost metrics."}
    return {"status": "ok", "report": json.loads(report_path.read_text(encoding="utf-8"))}


@app.get("/metrics/latest", tags=["model"])
def metrics_latest() -> dict[str, Any]:
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        return {"status": "missing", "message": "Run `make train` to generate metrics."}
    return {"status": "ok", "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))}


@app.get("/policy/simulation/latest", tags=["policy"])
def policy_simulation_latest() -> dict[str, Any]:
    report_path = Path("reports/policy_simulation.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make policy-sim` to generate report."}
    return {"status": "ok", "report": json.loads(report_path.read_text(encoding="utf-8"))}


@app.get("/error-analysis/latest", tags=["reports"])
def error_analysis_latest() -> dict[str, Any]:
    report_path = Path("reports/error_analysis.json")
    if not report_path.exists():
        return {"status": "missing", "message": "Run `make error-analysis` to generate report."}
    return {"status": "ok", "report": json.loads(report_path.read_text(encoding="utf-8"))}


@app.get("/reports/status", tags=["reports"])
def reports_status() -> dict[str, Any]:
    paths = {
        "metrics": Path("reports/metrics.json"),
        "monitoring": Path("reports/monitoring.json"),
        "policy_simulation": Path("reports/policy_simulation.json"),
        "error_analysis": Path("reports/error_analysis.json"),
        "cost_report": Path("reports/cost_report.json"),
        "dashboard": Path("reports/dashboard.html"),
    }
    status: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if path.exists():
            status[name] = {"exists": True, "updated_at_epoch": int(path.stat().st_mtime)}
        else:
            status[name] = {"exists": False, "updated_at_epoch": None}
    return {"status": "ok", "reports": status}


@app.get("/reports/timestamps", tags=["reports"])
def reports_timestamps() -> dict[str, Any]:
    report_paths = {
        "monitoring": Path("reports/monitoring.json"),
        "error_analysis": Path("reports/error_analysis.json"),
        "policy_simulation": Path("reports/policy_simulation.json"),
        "cost_report": Path("reports/cost_report.json"),
    }
    timestamps: dict[str, Any] = {}
    for name, path in report_paths.items():
        if not path.exists():
            timestamps[name] = {"generated_at_epoch": None, "file_updated_at_epoch": None}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamps[name] = {
            "generated_at_epoch": payload.get("generated_at_epoch"),
            "file_updated_at_epoch": int(path.stat().st_mtime),
        }
    return {"status": "ok", "reports": timestamps}


@app.get("/reports/missing", tags=["reports"])
def reports_missing() -> dict[str, Any]:
    report_specs = {
        "metrics": {"path": Path("reports/metrics.json"), "hint": "make train"},
        "monitoring": {"path": Path("reports/monitoring.json"), "hint": "make monitor"},
        "error_analysis": {"path": Path("reports/error_analysis.json"), "hint": "make error-analysis"},
        "policy_simulation": {"path": Path("reports/policy_simulation.json"), "hint": "make policy-sim"},
        "cost_report": {"path": Path("reports/cost_report.json"), "hint": "make cost-report"},
        "dashboard": {"path": Path("reports/dashboard.html"), "hint": "make dashboard"},
    }
    missing: list[dict[str, str]] = []
    for name, spec in report_specs.items():
        if not spec["path"].exists():
            missing.append({"report": name, "hint": str(spec["hint"])})

    return {
        "status": "ok",
        "all_present": len(missing) == 0,
        "missing_count": len(missing),
        "missing": missing,
    }


@app.get("/reports/staleness", tags=["reports"])
def reports_staleness(max_age_minutes: int = Query(default=120, ge=1, le=10_080)) -> dict[str, Any]:
    report_paths = {
        "metrics": Path("reports/metrics.json"),
        "monitoring": Path("reports/monitoring.json"),
        "error_analysis": Path("reports/error_analysis.json"),
        "policy_simulation": Path("reports/policy_simulation.json"),
        "cost_report": Path("reports/cost_report.json"),
        "dashboard": Path("reports/dashboard.html"),
    }
    now = time.time()
    max_age_seconds = float(max_age_minutes) * 60.0

    stale: list[dict[str, Any]] = []
    for name, path in report_paths.items():
        if not path.exists():
            continue
        age_seconds = now - path.stat().st_mtime
        if age_seconds > max_age_seconds:
            stale.append(
                {
                    "report": name,
                    "age_minutes": round(age_seconds / 60.0, 2),
                }
            )

    return {
        "status": "ok",
        "max_age_minutes": max_age_minutes,
        "stale_count": len(stale),
        "stale": stale,
    }


@app.get("/reports/overview", tags=["reports"])
def reports_overview() -> dict[str, Any]:
    metrics_path = Path("reports/metrics.json")
    monitoring_path = Path("reports/monitoring.json")
    cost_path = Path("reports/cost_report.json")
    policy_sim_path = Path("reports/policy_simulation.json")

    overview: dict[str, Any] = {"status": "ok", "kpis": {}, "sources": {}}

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        overview["kpis"]["pr_auc"] = metrics.get("pr_auc")
        overview["kpis"]["recall_at_precision_0_90"] = metrics.get("recall_at_precision_0_90")
    overview["sources"]["metrics"] = metrics_path.exists()

    if monitoring_path.exists():
        monitoring = json.loads(monitoring_path.read_text(encoding="utf-8"))
        overview["kpis"]["quality_ratio"] = (
            float(monitoring.get("recent_pr_auc", 0.0)) / max(float(monitoring.get("baseline_pr_auc", 1.0)), 1e-9)
        )
        overview["kpis"]["score_shift_abs"] = monitoring.get("score_shift_abs")
        overview["kpis"]["latency_p95_ms"] = monitoring.get("latency_p95_ms")
        overview["kpis"]["monitoring_alert"] = monitoring.get("alert")
    overview["sources"]["monitoring"] = monitoring_path.exists()

    if cost_path.exists():
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        overview["kpis"]["estimated_cost_saved"] = cost.get("estimated_cost_saved")
    overview["sources"]["cost_report"] = cost_path.exists()

    if policy_sim_path.exists():
        policy_sim = json.loads(policy_sim_path.read_text(encoding="utf-8"))
        overview["kpis"]["review_precision_proxy"] = policy_sim.get("review_precision_proxy")
        overview["kpis"]["block_precision_proxy"] = policy_sim.get("block_precision_proxy")
    overview["sources"]["policy_simulation"] = policy_sim_path.exists()

    return overview


@app.post("/reports/generate", tags=["reports"])
def reports_generate(req: ReportsGenerateRequest) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}

    if req.monitoring:
        try:
            report = generate_monitoring_report()
            results["monitoring"] = {"ok": True, "alert": report.get("alert", False)}
        except Exception as exc:
            results["monitoring"] = {"ok": False, "error": str(exc)}

    if req.error_analysis:
        try:
            report = generate_error_analysis_report()
            results["error_analysis"] = {"ok": True, "samples": report.get("counts", {}).get("samples")}
        except Exception as exc:
            results["error_analysis"] = {"ok": False, "error": str(exc)}

    if req.policy_simulation:
        try:
            report = run_policy_simulation()
            results["policy_simulation"] = {
                "ok": True,
                "n_events": report.get("n_events"),
                "decisions": report.get("decisions"),
            }
        except Exception as exc:
            results["policy_simulation"] = {"ok": False, "error": str(exc)}

    if req.dashboard:
        try:
            path = build_dashboard_html()
            results["dashboard"] = {"ok": True, "path": str(path)}
        except Exception as exc:
            results["dashboard"] = {"ok": False, "error": str(exc)}

    if req.cost_report:
        try:
            report = generate_cost_report()
            results["cost_report"] = {
                "ok": True,
                "estimated_cost_saved": report.get("estimated_cost_saved"),
            }
        except Exception as exc:
            results["cost_report"] = {"ok": False, "error": str(exc)}

    all_ok = all(entry.get("ok", False) for entry in results.values()) if results else True
    return {"status": "ok" if all_ok else "partial", "results": results}


@app.post("/reports/generate/all", tags=["reports"])
def reports_generate_all() -> dict[str, Any]:
    return reports_generate(
        ReportsGenerateRequest(
            monitoring=True,
            error_analysis=True,
            policy_simulation=True,
            cost_report=True,
            dashboard=True,
        )
    )


@app.get("/monitoring/dashboard", response_class=HTMLResponse, tags=["monitoring"])
def monitoring_dashboard() -> HTMLResponse:
    dashboard_path = Path("reports/dashboard.html")
    if not dashboard_path.exists():
        return HTMLResponse(
            content="<h3>Dashboard missing. Run `make dashboard` first.</h3>",
            status_code=404,
        )
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"), status_code=200)


@app.post("/policy/reset", tags=["policy"])
def policy_reset() -> dict[str, str]:
    reset_policy_state(policy_runtime_state)
    return {"status": "ok"}


@app.get("/policy/state", tags=["policy"])
def policy_state_snapshot() -> dict[str, Any]:
    return {"status": "ok", "state": policy_state_summary(policy_runtime_state)}


@app.get("/policy/config", tags=["policy"])
def policy_config() -> dict[str, Any]:
    return {
        "status": "ok",
        "score_thresholds": policy_cfg.get("score_thresholds", {}),
        "hard_rules": policy_cfg.get("hard_rules", {}),
        "rate_limits": policy_cfg.get("rate_limits", {}),
        "monitoring": policy_cfg.get("monitoring", {}),
    }


@app.post("/predict", response_model=PredictResponse, tags=["serving"])
def predict(req: PredictRequest) -> PredictResponse:
    started_at = time.perf_counter()
    payload = req.model_dump()
    if bundle is not None:
        model_output = explain_event(bundle, payload)
        model_version = str(model_output.get("model_version", "unknown"))
        score = model_output["risk_score"]
        model_reasons = model_output["model_reasons"]
        feature_contributions = model_output["feature_contributions"]
        explanation_method = model_output.get("explanation_method", "none")
        components = {
            "text_score": round(float(model_output["text_score"]), 4),
            "tabular_score": round(float(model_output["tabular_score"]), 4),
            "graph_max_entity_fraud_rate": round(float(model_output["graph_max_entity_fraud_rate"]), 4),
            "graph_max_entity_pagerank": round(float(model_output["graph_max_entity_pagerank"]), 8),
        }
    else:
        model_version = "fallback-heuristic"
        score = fallback.predict(payload)
        model_reasons = []
        feature_contributions = {}
        explanation_method = "fallback"
        components = {"text_score": round(score, 4), "tabular_score": round(score, 4)}
    decision, reasons, policy_triggers = decide(score, payload, policy_cfg, state=policy_runtime_state)
    with serving_stats_lock:
        serving_stats["total_requests"] += 1
        serving_stats["predict_requests"] += 1
        serving_stats["predicted_items"] += 1
        serving_stats["decision_counts"][decision] = serving_stats["decision_counts"].get(decision, 0) + 1
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        serving_stats["latency_ms_window"].append(float(elapsed_ms))
        if len(serving_stats["latency_ms_window"]) > LATENCY_WINDOW_SIZE:
            serving_stats["latency_ms_window"] = serving_stats["latency_ms_window"][-LATENCY_WINDOW_SIZE:]
    return PredictResponse(
        model_version=model_version,
        risk_score=round(score, 4),
        decision=decision,
        reasons=reasons,
        model_reasons=model_reasons,
        components=components,
        feature_contributions=feature_contributions,
        policy_triggers=policy_triggers,
        explanation_method=explanation_method,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["serving"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    with serving_stats_lock:
        serving_stats["total_requests"] += 1
        serving_stats["batch_requests"] += 1
    results = [predict(item) for item in req.items]
    return BatchPredictResponse(items=results)


@app.post("/explain", response_model=PredictResponse, tags=["serving"])
def explain(req: PredictRequest) -> PredictResponse:
    return predict(req)
