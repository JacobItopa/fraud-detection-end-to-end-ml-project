# Step 8: Monitoring & Maintenance

Once a machine learning model is deployed, it immediately begins to degrade as real-world behaviors and patterns change. This document outlines the strategies implemented to maintain the Fraud Detection pipeline.

---

## 1. Structured Prediction Logging

**Implemented in:** `src/api/main.py`
Every time the `/predict` or `/predict/batch` endpoint is hit, the API automatically writes a structured JSON log to `logs/predictions.jsonl`.

### Why?
1. **Auditing & Compliance:** Financial institutions require records of algorithmic decisions. If a customer disputes a flagged transaction, the exact inputs and probability score can be retrieved.
2. **Data Collection for Retraining:** The raw requests form our "Current" dataset, which we can eventually label (via back-office investigations) and use to retrain the model.

**Log Format Example:**
```json
{
  "timestamp": "2026-04-24T12:00:00.000000",
  "request": {
    "type": "TRANSFER",
    "amount": 50000.0,
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  },
  "response": {
    "is_fraud": true,
    "fraud_probability": 0.99999,
    "threshold_used": 0.9983,
    "model": "XGBoost (Tuned)",
    "version": "1.0.0"
  }
}
```

---

## 2. Automated Data Drift Detection

**Implemented in:** `src/monitoring/detect_drift.py`

### What is Data Drift?
Data drift occurs when the statistical distribution of live, incoming data differs from the data the model was originally trained on. For example, if users suddenly start making much larger transactions on average, the `amount` feature has drifted.

### How it works
We use **Evidently AI** to compare a Reference Dataset (a static 10,000-row sample from our original training data) against our Current Dataset (extracted from `logs/predictions.jsonl`).

**Execution:**
```bash
python src/monitoring/detect_drift.py
```
This script generates an interactive HTML dashboard saved at `reports/drift_report.html` highlighting exactly which features have drifted using statistical tests (e.g., Wasserstein distance).

---

## 3. Retraining Strategy (CI/CD/CT)

If data drift is severe, or if we notice the model's F1-score dropping (Concept Drift), we trigger the **Continuous Training (CT)** pipeline.

### Retraining Triggers
1. **Scheduled:** Run the drift report weekly. If >30% of features drift, trigger a retraining evaluation.
2. **Performance Drop:** If back-office fraud investigators report the False Positive Rate (FPR) has climbed above a certain threshold over a 30-day window.

### The CT Pipeline Workflow
When a trigger is hit, the following automated or semi-automated pipeline executes:

1. **Data Aggregation:** The JSON logs from the last 3 months are joined with true labels provided by the fraud investigation team.
2. **Model Retraining:** 
   ```bash
   python src/features/build_features.py
   python src/models/train_model.py
   python src/models/tune_model.py
   ```
   A new `xgboost_tuned.pkl` and `robust_scaler.pkl` are generated.
3. **Automated Testing:** The CI pipeline runs `pytest` (which we configured in Step 7) against the new model to ensure backward compatibility and API schema validation.
4. **Deployment:** If tests pass, Cloud Build (via `cloudbuild.yaml`) builds a new Docker image and seamlessly deploys it to GCP Cloud Run.
