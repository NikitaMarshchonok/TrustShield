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

### 5) Generate Monitoring + Error Analysis

```bash
make monitor
make error-analysis
make dashboard
```

## API

- `GET /health` - service health and model loading status
- `POST /predict` - risk score, reasons, decision
- `POST /explain` - explanation-focused output
- `GET /monitoring/summary` - latest drift/quality/latency report
- `GET /monitoring/dashboard` - rendered local HTML dashboard

## Policy Engine

Decision policy combines model score and deterministic risk rules:

- `block`: score above block threshold or hard rule violation
- `review`: medium score or suspicious pattern
- `allow`: low score and no suspicious pattern

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

## Validation and Tracking

- Schema validation via `pandera` (with safe fallback checks if unavailable)
- MLflow run logging for training metrics and artifacts (`file:./mlruns` by default)

Commands:

```bash
make validate
make train
```

## Next Iterations

- Upgrade graph block to full graph algorithms (`pagerank`, components, node2vec embeddings)
- Add SHAP + richer text explanations in `/explain`
- Integrate Evidently dashboards and alert routing
