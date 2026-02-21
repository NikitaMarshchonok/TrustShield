.PHONY: install train serve test lint monitor

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
