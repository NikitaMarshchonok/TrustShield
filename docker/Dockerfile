FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "trustshield.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
