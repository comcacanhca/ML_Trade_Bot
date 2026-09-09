FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/research/rf_mlflow:/app/vendor/scj

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "research/rf_mlflow/research_h4edge_feature_transforms_optuna.py", "--variants", "products", "--top-n", "10", "--n-trials", "50", "--max-train-rows", "1000000", "--enable-mlflow"]
