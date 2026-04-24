"""
FastAPI application entry point.
Exposes health, single-predict and batch-predict endpoints.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    TransactionInput, BatchTransactionInput,
    PredictionOutput, BatchPredictionOutput, HealthResponse
)
from src.api.predictor import FraudPredictor
from src.api.config import MODEL_NAME, MODEL_VERSION, FRAUD_THRESHOLD

# ── Application lifespan (model loads once, not per request) ─────────────────
predictor: FraudPredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = FraudPredictor()
    yield
    # Cleanup on shutdown (nothing needed here)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud detection service powered by a tuned XGBoost classifier "
        "trained on the Synthetic Financial Fraud dataset. "
        "Submit a transaction and receive an instant fraud probability score."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import json
import logging
import os
from datetime import datetime

# ── Logging setup ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("prediction_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/predictions.jsonl")
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)

def log_prediction(transaction_dict: dict, prediction_dict: dict):
    """Log the request and response as a single JSON object."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request": transaction_dict,
        "response": prediction_dict
    }
    logger.info(json.dumps(log_entry))

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root health check — confirms the service is running."""
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        version=MODEL_VERSION,
        threshold=FRAUD_THRESHOLD,
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Lightweight liveness probe required by GCP Cloud Run.
    Returns 200 OK when the model is loaded and ready.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        version=MODEL_VERSION,
        threshold=FRAUD_THRESHOLD,
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    """
    Score a **single** transaction for fraud.

    - Returns `is_fraud=true` if the model probability exceeds the optimal threshold.
    - Returns the raw `fraud_probability` for downstream risk-scoring systems.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    
    tx_dict = transaction.model_dump()
    results = predictor.predict([tx_dict])
    
    # Log the prediction
    log_prediction(tx_dict, results[0])
    
    return PredictionOutput(**results[0])

@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(payload: BatchTransactionInput):
    """
    Score a **batch** of up to 1,000 transactions in one request.

    Useful for bulk nightly re-scoring or back-testing pipelines.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    transactions = [t.model_dump() for t in payload.transactions]
    results = predictor.predict(transactions)

    # Log each prediction in the batch
    for tx_dict, res_dict in zip(transactions, results):
        log_prediction(tx_dict, res_dict)

    predictions = [PredictionOutput(**r) for r in results]
    fraud_count = sum(1 for p in predictions if p.is_fraud)

    return BatchPredictionOutput(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
    )
