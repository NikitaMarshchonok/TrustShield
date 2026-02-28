.PHONY: install train serve test lint monitor validate error-analysis dashboard policy-sim reports-all

install:
	pip install -e ".[dev]"

train:
	python -m trustshield.models.train

serve:
	uvicorn trustshield.serving.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check src tests

monitor:
	python -m trustshield.monitoring.report

validate:
	python -m trustshield.tools.validate_data

error-analysis:
	python -m trustshield.evaluation.error_analysis

dashboard:
	python -m trustshield.monitoring.dashboard

policy-sim:
	python -m trustshield.evaluation.policy_simulation

reports-all: monitor error-analysis policy-sim dashboard
