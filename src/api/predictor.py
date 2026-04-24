"""
FraudPredictor: loads model artefacts once at startup and handles
the full preprocessing → inference pipeline for single or batch transactions.
"""

import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.api.config import (
    MODEL_PATH, SCALER_PATH, FRAUD_THRESHOLD,
    FEATURE_COLUMNS, MODEL_NAME, MODEL_VERSION
)


class FraudPredictor:
    """
    Encapsulates model loading and the full inference pipeline.
    Instantiate once at app startup; call predict() per request.
    """

    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.model   = joblib.load(MODEL_PATH)
        print(f"Loading scaler from {SCALER_PATH}...")
        self.scaler  = joblib.load(SCALER_PATH)
        self.threshold = FRAUD_THRESHOLD
        print("FraudPredictor ready.")

    def _build_features(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Replicates the exact preprocessing pipeline from src/features/build_features.py
        for one or more transactions at inference time.
        """
        df = pd.DataFrame(transactions)

        # 1. Feature Engineering
        df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

        # 2. Manual One-Hot Encoding for `type`
        #    CASH_IN is the baseline (dropped). We create the remaining 4 dummy columns.
        for t in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
            df[f"type_{t}"] = (df["type"] == t).astype(int)

        # 3. Select and order columns to exactly match the scaler's fitted feature set
        df = df[FEATURE_COLUMNS]

        return df

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df)

    def predict(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the full pipeline and return predictions.

        Returns a list of dicts with keys:
            is_fraud, fraud_probability, threshold_used, model, version
        """
        features_df  = self._build_features(transactions)
        features_scaled = self._scale(features_df)

        probabilities = self.model.predict_proba(features_scaled)[:, 1]

        results = []
        for prob in probabilities:
            results.append({
                "is_fraud": bool(prob >= self.threshold),
                "fraud_probability": round(float(prob), 6),
                "threshold_used": self.threshold,
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
            })
        return results
