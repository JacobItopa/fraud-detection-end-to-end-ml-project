# 🚨 End-to-End Financial Fraud Detection Pipeline

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg)

## 📖 Overview
This repository contains a complete, production-ready Machine Learning pipeline for detecting fraudulent financial transactions in real-time. 

Built from scratch, the project encompasses the entire MLOps lifecycle: from Exploratory Data Analysis (EDA) on a heavily imbalanced 6.3 million row dataset, through feature engineering and hyperparameter tuning, to deploying a containerised, auto-scaling REST API on Google Cloud Platform (GCP).

---

## 📊 The Dataset & Business Context
The model is trained on the **Synthetic Financial Datasets For Fraud Detection** (derived from PaySim). 
- **Volume:** 6.36 million transactions.
- **Challenge:** Extreme class imbalance. Only **0.13%** of transactions are actually fraudulent.
- **Solution:** Instead of dropping data or relying entirely on synthetic oversampling (SMOTE), we utilized **Algorithm-Level Weighting** (`scale_pos_weight`) alongside optimal threshold tuning via Precision-Recall curves to maximize the model's ability to catch fraud without overwhelming investigators with False Positives.

---

## 🏆 Model Performance
Three models were evaluated: Logistic Regression (Baseline), LightGBM, and XGBoost.
**XGBoost** emerged as the champion model after RandomizedSearchCV tuning.

| Metric | Score (Tuned XGBoost) |
| :--- | :--- |
| **Optimal Threshold** | `0.9983` |
| **F1-Score** | `0.94` |
| **Precision** | `0.94` |
| **Recall (Sensitivity)** | `0.94` |
| **PR-AUC** | `0.9882` |

---

## 🏗️ Architecture & Features

### 1. Data Engineering (`src/features/build_features.py`)
- Engineered critical `errorBalanceOrig` and `errorBalanceDest` features to catch account balance mismatches.
- Manual one-hot encoding for transaction types to ensure deterministic feature alignment.
- Utilized `RobustScaler` to normalize features while remaining immune to extreme financial outliers.

### 2. The API (`src/api/`)
- Built with **FastAPI** for blazing fast, asynchronous inference.
- Strict **Pydantic v2** validation to ensure only correctly formatted financial data enters the model.
- Includes endpoints for both single real-time predictions (`/predict`) and bulk nightly rescoring (`/predict/batch`).

### 3. Comprehensive Testing (`tests/`)
- A **50-test pytest suite** covering unit tests for schemas, preprocessing logic, and the inference wrapper.
- End-to-end integration tests mimicking live API traffic.
- Isolated test fixtures prevent accidental corruption of production `.pkl` model artifacts.

### 4. Containerisation & CI/CD (`Dockerfile` & `cloudbuild.yaml`)
- Packaged in a lean `python:3.11-slim` Docker image.
- `docker-compose` configured for one-click local testing.
- Ready for serverless deployment on **GCP Cloud Run** via Google Cloud Build.

### 5. Monitoring & MLOps (`src/monitoring/detect_drift.py`)
- **Structured Logging:** All predictions are logged to `logs/predictions.jsonl` for auditing.
- **Data Drift:** Integrated **Evidently AI** to compare live production logs against the training baseline to automatically detect statistical drift in incoming financial behavior.

---

## 🚀 Quick Start (Local Setup)

### Option A: Using Docker (Recommended)
You can spin up the entire production API locally with a single command.
```bash
# Build and start the container
docker-compose up --build

# The API will be available at http://localhost:8080
# View interactive API documentation at http://localhost:8080/docs
```

### Option B: Local Python Environment
```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

---

## 💻 API Usage Examples

### 1. Health Check
```bash
curl -X GET http://localhost:8080/
```
**Response:**
```json
{
  "status": "ok",
  "model": "XGBoost (Tuned)",
  "version": "1.0.0",
  "threshold": 0.9983
}
```

### 2. Predict a Fraudulent Transaction
```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{
           "type": "TRANSFER",
           "amount": 50000.0,
           "oldbalanceOrg": 50000.0,
           "newbalanceOrig": 0.0,
           "oldbalanceDest": 0.0,
           "newbalanceDest": 0.0
         }'
```
**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.99999,
  "threshold_used": 0.9983,
  "model": "XGBoost (Tuned)",
  "version": "1.0.0"
}
```

---

## 📁 Directory Structure
```text
fraud_detection/
├── data/
│   ├── raw/                 # Original 6.3m row dataset (ignored in git)
│   └── processed/           # Transformed features ready for modeling
├── models/
│   ├── robust_scaler.pkl    # Serialized preprocessing scaler
│   └── xgboost_tuned.pkl    # Champion predictive model
├── notebooks/               # Jupyter notebooks for EDA and experiments
├── reports/                 # Markdown reports detailing model evaluation
├── src/
│   ├── api/                 # FastAPI application, Pydantic schemas, and Predictor
│   ├── data/                # Scripts to fetch and describe data
│   ├── features/            # Feature engineering pipeline (build_features.py)
│   ├── models/              # Training, evaluation, and tuning scripts
│   └── monitoring/          # Evidently AI data drift detection scripts
├── tests/
│   ├── unit/                # Isolated tests for schemas and logic
│   └── integration/         # End-to-end tests for the FastAPI service
├── Dockerfile               # Production container definition
├── docker-compose.yml       # Local container orchestration
├── cloudbuild.yaml          # GCP CI/CD deployment pipeline
└── pytest.ini               # Pytest configuration
```

---

## 📚 Detailed Project Documentation
For a deep dive into how this pipeline was built from scratch, refer to the step-by-step markdown logs generated during development:

### The MLOps Roadmap
- [`ml_project_steps.md`](./ml_project_steps.md) — The high-level master plan outlining all 8 phases of the project.
- [`ml_project_structure.md`](./ml_project_structure.md) — The initial blueprint for the cookiecutter directory structure.

### Phase-by-Phase Walkthroughs
- **Step 1:** [`problem_definition.md`](./problem_definition.md) — Definition of the business problem, goals, and success metrics.
- **Step 2:** [`data_collection.md`](./data_collection.md) — Details on dataset extraction and schema.
- **Step 3:** [`exploratory_data_analysis.md`](./exploratory_data_analysis.md) — Deep dive into the data, revealing the 0.13% extreme class imbalance. *(Also see [`reports/descriptive_statistics.md`](./reports/descriptive_statistics.md) for raw statistical output).*
- **Step 4:** [`data_preprocessing.md`](./data_preprocessing.md) — Rationale behind the feature engineering formulas, `RobustScaler` usage, and one-hot encoding logic.
- **Step 5:** [`model_selection_and_training.md`](./model_selection_and_training.md) — Initial baseline training comparing Logistic Regression, LightGBM, and XGBoost using algorithm-level weighting.
- **Step 6:** [`model_evaluation_and_tuning.md`](./model_evaluation_and_tuning.md) — Rigorous evaluation using PR-curves, threshold optimization (`0.9983`), and hyperparameter tuning that led to XGBoost becoming the champion.
- **Step 7:** [`model_deployment.md`](./model_deployment.md) — Documentation of the FastAPI implementation, Docker containerization, and GCP deployment prep.
- **Step 8:** [`reports/monitoring_and_maintenance.md`](./reports/monitoring_and_maintenance.md) — The strategy for CI/CD/CT retraining, JSON logging, and the Evidently AI drift detection pipeline.

### Testing Reports
- [`reports/testing_report.md`](./reports/testing_report.md) — A comprehensive breakdown of the 50-test `pytest` suite covering unit and integration testing for the API and model wrapper.

---

## ☁️ Deployment to GCP Cloud Run
This project is configured for automated Serverless deployment on GCP. Once GCP billing is active:
1. Connect your GitHub repository to Google Cloud Build.
2. The `cloudbuild.yaml` file will automatically trigger on pushes.
3. It will build the Docker image, push it to Google Container Registry, and deploy to Cloud Run (scaling from 0 to 10 instances based on traffic).
