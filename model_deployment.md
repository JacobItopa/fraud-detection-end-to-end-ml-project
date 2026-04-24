# Step 7: Model Deployment (MLOps)

## 1. Save / Serialize the Model

All model artefacts were already serialized during Steps 4 and 6. The API loads the following at startup:

| File | Description |
|---|---|
| `models/xgboost_tuned.pkl` | Champion XGBoost classifier |
| `models/robust_scaler.pkl` | Fitted RobustScaler (from Step 4 preprocessing) |

The optimal decision threshold (`0.9983`) is centralised in `src/api/config.py` so it never gets hardcoded across files.

---

## 2. API — FastAPI REST Service

A production-grade REST API was built using **FastAPI** with three layers:

### `src/api/config.py`
- Stores the model path, scaler path, optimal threshold, and the **exact ordered feature column list** the scaler was fitted on.
- Ensures the inference pipeline is always deterministic and consistent with training.

### `src/api/schemas.py`
- **Pydantic v2** request/response models that auto-validate all inputs before they reach the model.
- Invalid or missing fields return a `422 Unprocessable Entity` response automatically.

### `src/api/predictor.py`
- `FraudPredictor` class loaded **once at startup** (not per request — prevents slow cold starts).
- Replicates the full preprocessing pipeline:
  1. Engineers `errorBalanceOrig` and `errorBalanceDest` features
  2. **Manually** one-hot encodes the `type` column (avoids `pd.get_dummies` breaking on single rows)
  3. Reorders columns to match the scaler's fitted feature order
  4. Applies `RobustScaler.transform()`
  5. Returns probability and binary label using the optimal threshold

### `src/api/main.py` — Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns model name, version, threshold |
| `GET` | `/health` | GCP Cloud Run liveness probe |
| `POST` | `/predict` | Single transaction fraud prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1,000 transactions) |

### Live Test Results ✅

**Fraud case** (TRANSFER — empty recipient account, full sender drain):
```json
{
  "is_fraud": true,
  "fraud_probability": 0.999992,
  "threshold_used": 0.9983,
  "model": "XGBoost (Tuned)",
  "version": "1.0.0"
}
```

**Legitimate case** (PAYMENT with normal balance flow):
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0,
  "threshold_used": 0.9983,
  "model": "XGBoost (Tuned)",
  "version": "1.0.0"
}
```

---

## 3. Containerization — Docker

### `Dockerfile`
- Base image: `python:3.11-slim` (stable XGBoost/LightGBM binary wheels; keeps image lean)
- Copies only what the API needs: `src/`, `models/xgboost_tuned.pkl`, `models/robust_scaler.pkl`
- Runs on port **8080** (GCP Cloud Run default)
- Optimised layer ordering — dependencies installed before source code for faster rebuilds

### `docker-compose.yml`
Allows the full production stack to be tested locally with a single command:
```bash
docker-compose up --build
# API available at http://localhost:8080
```

**To run locally without Docker:**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

---

## 4. Deployment — GCP Cloud Run

### `cloudbuild.yaml`
A GCP Cloud Build CI/CD pipeline that automatically:
1. Builds the Docker image on every GitHub push
2. Pushes the image to Google Container Registry (`gcr.io/YOUR_PROJECT_ID/fraud-detection-api`)
3. Deploys it to **Cloud Run** with 1Gi RAM, scaling from 0 to 10 instances

**Why Cloud Run?**
- Serverless — no servers to manage
- Scales to zero when idle (cost-efficient)
- GCP free tier: **2 million requests/month**
- Gets a live HTTPS URL automatically

**One-time deployment commands (when GCP billing is active):**
```bash
# 1. Authenticate
gcloud auth login

# 2. Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fraud-detection-api

# 3. Deploy
gcloud run deploy fraud-detection-api \
  --image gcr.io/YOUR_PROJECT_ID/fraud-detection-api \
  --platform managed --region us-central1 \
  --allow-unauthenticated --memory 1Gi
```

---

## 5. Saved Artifacts

| File | Description |
|---|---|
| `src/api/config.py` | Centralised config (threshold, paths, feature order) |
| `src/api/schemas.py` | Pydantic request/response validation schemas |
| `src/api/predictor.py` | FraudPredictor inference class |
| `src/api/main.py` | FastAPI application with all endpoints |
| `Dockerfile` | Production Docker image definition |
| `docker-compose.yml` | Local container testing configuration |
| `cloudbuild.yaml` | GCP Cloud Build CI/CD pipeline |
| `requirements.txt` | Updated with FastAPI, Uvicorn, Pydantic |

## 6. Next Steps
Proceed to **Step 8: Monitoring & Maintenance** — set up data drift detection, logging, and an automated retraining trigger.
