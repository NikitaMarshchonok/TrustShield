# TrustShield

TrustShield is an end-to-end scam/fraud detection project for marketplace-like messaging and transaction flows.
It demonstrates text modeling, tabular risk features, policy-based decisions, and production-oriented ML engineering.

## MVP Scope

- Train a text + tabular fraud model on synthetic/semi-synthetic events
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

## API

- `GET /health` - service health and model loading status
- `POST /predict` - risk score, reasons, decision
- `POST /explain` - explanation-focused output

## Policy Engine

Decision policy combines model score and deterministic risk rules:

- `block`: score above block threshold or hard rule violation
- `review`: medium score or suspicious pattern
- `allow`: low score and no suspicious pattern

Thresholds are in `configs/policy.yaml`.

## Monitoring (MVP)

MVP includes a lightweight monitoring report generator:

- score distribution summary
- positive-rate shift check
- alert if shift exceeds threshold

Run:

```bash
make monitor
```

## Next Iterations

- Graph features (`degree`, `pagerank`, connected components, node embeddings)
- PR-AUC + Recall@FixedPrecision + cost-based evaluation dashboard
- MLflow experiment tracking and model registry
- Drift/quality dashboard with Evidently
