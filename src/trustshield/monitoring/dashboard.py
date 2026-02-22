from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dashboard_html() -> Path:
    monitoring = _read_json(Path("reports/monitoring.json"))
    errors = _read_json(Path("reports/error_analysis.json"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TrustShield Monitoring Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    .kpi {{ font-size: 20px; font-weight: 600; }}
    .muted {{ color: #6b7280; }}
    pre {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>TrustShield Dashboard</h1>
  <p class="muted">Generated from local reports artifacts.</p>

  <div class="card">
    <h2>Monitoring Snapshot</h2>
    <div class="kpi">Alert: {monitoring.get("alert", "n/a")}</div>
    <p>Recent PR-AUC: {monitoring.get("recent_pr_auc", "n/a")}</p>
    <p>Latency p95 (ms): {monitoring.get("latency_p95_ms", "n/a")}</p>
    <pre>{json.dumps(monitoring, indent=2)}</pre>
  </div>

  <div class="card">
    <h2>Error Analysis Snapshot</h2>
    <p>False positives: {errors.get("counts", {}).get("false_positives", "n/a")}</p>
    <p>False negatives: {errors.get("counts", {}).get("false_negatives", "n/a")}</p>
    <pre>{json.dumps(errors, indent=2)}</pre>
  </div>
</body>
</html>
"""
    out = Path("reports/dashboard.html")
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {out}")
    return out


if __name__ == "__main__":
    build_dashboard_html()
