# TrustShield

TrustShield is an end-to-end scam/fraud detection project for marketplace-like messaging and transaction flows.
It demonstrates text modeling, tabular + graph-derived risk features, policy-based decisions, and production-oriented ML engineering.

## MVP Scope

- Train an ensemble fraud model (`text + tabular + graph-derived` features)
- Serve predictions through FastAPI
- Return:
  - `risk_score` (0..1)
  - `reasons` (human-readable influential factors)
  - `decision` (`allow`, `review`, `block`)
- Include tests, Docker image, CI, and reproducible training commands

## Project Structure

```text
trustshield/
  README.md
  pyproject.toml
  Makefile
  data/                  # ignored raw data, tracked sample data
  configs/
  src/trustshield/
    ingestion/
    preprocessing/
    features/
    models/
    evaluation/
    serving/
    monitoring/
  tests/
  docker/
  .github/workflows/ci.yml
  reports/
```

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# optional: SHAP-based explainability
pip install -e ".[explain]"
```

### 2) Train

```bash
make train
```

Artifacts are saved to `reports/artifacts/model_bundle.joblib`.

### 3) Run API

```bash
make serve
```

Open Swagger at `http://127.0.0.1:8000/docs`.

### 4) Test

```bash
make test
```

### 5) Generate Reports Bundle

```bash
make reports-all
```

## API

- `GET /health` - service health and model loading status
- `POST /predict` - risk score, reasons, decision, score components
- `POST /explain` - explanation-focused output with top feature contributions and method
- `GET /metrics/latest` - latest training metrics snapshot
- `GET /policy/simulation/latest` - latest policy simulation report
- `GET /error-analysis/latest` - latest error analysis report
- `GET /reports/status` - availability and update time for all reports
- `GET /monitoring/summary` - latest drift/quality/latency report
- `GET /monitoring/dashboard` - rendered local HTML dashboard
- `POST /policy/reset` - reset in-memory policy counters

## Policy Engine

Decision policy combines model score, deterministic risk rules, and stateful rate-limits:

- `block`: score above block threshold or hard rule violation
- `review`: medium score or suspicious pattern
- `allow`: low score and no suspicious pattern
- rolling-window triggers for `user_id`, `device_id`, `ip_id`

Thresholds are in `configs/policy.yaml`.

## Monitoring (MVP)

Monitoring includes a lightweight report generator:

- score distribution summary
- feature shift checks
- quality check (`PR-AUC` baseline vs recent)
- inference latency budget (`p95` in ms)
- alert if thresholds are exceeded

Run:

```bash
make monitor
```

## Error Analysis

Generate top false positives/false negatives report:

```bash
make error-analysis
```

Output: `reports/error_analysis.json`

## Policy Simulation

Simulate policy decisions on a stream of events:

```bash
make policy-sim
```

Output: `reports/policy_simulation.json`

## Validation and Tracking

- Schema validation via `pandera` (with safe fallback checks if unavailable)
- MLflow run logging for training metrics and artifacts (`file:./mlruns` by default)
- Explainability via optional SHAP (`pip install -e ".[explain]"`), with linear fallback if not installed

Commands:

```bash
make validate
make train
```

## Next Iterations

- Add node2vec embeddings on top of current graph metrics (`degree`, `pagerank`, components)
- Integrate Evidently dashboards and alert routing
