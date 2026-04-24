# Model Deployment Testing Report

## Overview
To ensure the production deployment of the Fraud Detection Machine Learning Pipeline is stable and robust, a comprehensive testing suite was implemented using **pytest**. The suite consists of **50 individual tests** categorized into unit tests and end-to-end integration tests.

The testing architecture is designed to validate all aspects of the pipeline, from input validation (Pydantic) to data engineering (Pandas/Scikit-learn), right through to the HTTP API layer (FastAPI).

---

## 1. Unit Testing Strategy

The unit tests run in isolation. They test logic without relying on the actual `.pkl` models or network connections, ensuring rapid feedback and protecting against accidental corruption.

### 1.1 API Schema Validation (`tests/unit/test_schemas.py`)
**Goal:** Verify that the FastAPI Pydantic models correctly reject bad data before it ever hits the predictive model.
- **Accepted Types:** Confirms the API only accepts the 5 allowed transaction types (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`).
- **Data Integrity:** Ensures transaction amounts and balances cannot be negative.
- **Edge Cases:** Validates that incomplete payloads (missing fields) immediately trigger a `422 Unprocessable Entity` response.
- **Output Validation:** Ensures the system response always correctly formats the model probability, threshold used, and model version metadata.

### 1.2 Feature Engineering Logic (`tests/unit/test_build_features.py`)
**Goal:** Verify that the mathematical transformations performed during Step 4 data preprocessing are strictly preserved.
- **Balance Error Formulas:** Directly validates that the engineered features compute correctly:
  - `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`
  - `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`
- **One-Hot Encoding:** Confirms that `TRANSFER` types trigger `type_TRANSFER=1` and that `CASH_IN` correctly acts as the dropped baseline (all `type_*` columns = 0).
- **Data Leakage Prevention:** Ensures columns that cause overfitting (like `step`, `nameOrig`, and `nameDest`) are successfully dropped.

### 1.3 Inference Pipeline Wrapper (`tests/unit/test_predictor.py`)
**Goal:** Verify the `FraudPredictor` class logic by mocking the XGBoost and RobustScaler binaries.
- **Feature Ordering:** Asserts that the final array fed into the XGBoost `.predict_proba()` method matches the **exact 11-column order** the scaler was trained on.
- **Threshold Application:** Confirms that the hardcoded optimal threshold (`0.9983`) correctly dictates the boolean `is_fraud` flag (e.g., a raw probability of `0.9982` correctly resolves to `False`).
- **Batching:** Validates that the predictor handles lists of dictionaries correctly when processing batch requests.

---

## 2. Integration Testing Strategy

The integration tests treat the API as a black box. They use FastAPI's `TestClient` to fire HTTP requests against the endpoints utilizing the **real trained models** and **real scalers**.

### 2.1 E2E API Functionality (`tests/integration/test_api.py`)
**Goal:** Ensure the REST endpoints are wired correctly and respond successfully to realistic inputs.
- **Health Checks (`GET /`, `GET /health`):** Verifies the liveness probes return HTTP 200 and expose the current model version and threshold.
- **Fraud Classification (`POST /predict`):** Fires a classic fraud pattern payload (a `TRANSFER` draining a sender's account) and asserts that the real XGBoost model flags it as `is_fraud: True` with a probability > 0.9983.
- **Legitimate Classification (`POST /predict`):** Fires a standard `PAYMENT` payload and asserts the model correctly classifies it as `is_fraud: False`.
- **Batch Processing (`POST /predict/batch`):** Submits multiple transactions simultaneously and verifies the API accurately aggregates total counts and fraud counts.

---

## 3. Key MLOps Safety Mechanisms Implemented

During the implementation of this suite, a critical MLOps safety guardrail was built:
**Model Protection Fixture:** A `mock_joblib_dump` fixture was added to `tests/conftest.py`. This ensures that when the test suite runs the preprocessing logic, it does **not** overwrite the production `robust_scaler.pkl` with a scaler trained only on dummy test data. This protects the integrity of the live container.

---

## 4. Test Execution Summary

The suite executes successfully in under 5 seconds, making it ideal for the CI/CD pipeline defined in `cloudbuild.yaml`.

```text
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.0.2
rootdir: C:\Users\user\Desktop\2026 PLAN AND PROJECT\projects\fraud_detection

tests/integration/test_api.py ................                           [ 32%]
tests/unit/test_build_features.py .........                              [ 50%]
tests/unit/test_predictor.py ..............                              [ 78%]
tests/unit/test_schemas.py ...........                                   [100%]

============================= 50 passed in 3.79s ==============================
```

**Conclusion:** The deployment architecture is fully tested and verified. The `FraudPredictor` accurately mirrors the training environment, and the API securely handles data parsing and validation.
