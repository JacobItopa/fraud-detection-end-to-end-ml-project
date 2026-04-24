# ── Fraud Detection API — Dockerfile ─────────────────────────────────────────
# Uses Python 3.11-slim for stable binary wheels with XGBoost/LightGBM.
# Cloud Run expects the app to listen on port 8080.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies (needed for some scientific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files the API needs (not raw data, notebooks, reports)
COPY src/ ./src/
COPY models/xgboost_tuned.pkl  ./models/xgboost_tuned.pkl
COPY models/robust_scaler.pkl  ./models/robust_scaler.pkl

# Cloud Run injects PORT env var; default to 8080
ENV PORT=8080
EXPOSE 8080

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
