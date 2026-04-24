"""
Global configuration for the Fraud Detection API.
Centralises all paths, thresholds and model metadata.
"""

# ── Model artefacts ──────────────────────────────────────────────────────────
MODEL_PATH   = "models/xgboost_tuned.pkl"
SCALER_PATH  = "models/robust_scaler.pkl"
MODEL_NAME   = "XGBoost (Tuned)"
MODEL_VERSION = "1.0.0"

# ── Decision threshold ───────────────────────────────────────────────────────
# Optimal threshold derived from Step 6 PR-curve analysis (maximises F1).
# At this threshold: Precision=0.94, Recall=0.94, F1=0.94
FRAUD_THRESHOLD = 0.9983

# ── Feature engineering ──────────────────────────────────────────────────────
# Exact column order the RobustScaler was fitted on during Step 4.
# The predictor MUST produce features in this exact order.
FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

# Valid transaction types (type_CASH_IN is the dropped baseline after one-hot encoding)
VALID_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
