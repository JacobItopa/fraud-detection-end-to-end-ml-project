"""
Shared pytest fixtures available to all test modules.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ── Sample transaction data ──────────────────────────────────────────────────

@pytest.fixture
def fraud_transaction():
    """A known fraud pattern: TRANSFER that drains sender to zero."""
    return {
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

@pytest.fixture
def legit_transaction():
    """A typical legitimate PAYMENT transaction."""
    return {
        "type": "PAYMENT",
        "amount": 9839.64,
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

@pytest.fixture
def cash_in_transaction():
    """CASH_IN transaction — baseline one-hot category (all type_ columns = 0)."""
    return {
        "type": "CASH_IN",
        "amount": 5000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 15000.0,
        "oldbalanceDest": 20000.0,
        "newbalanceDest": 15000.0,
    }

@pytest.fixture
def mock_predictor():
    """A FraudPredictor with mocked model and scaler — no .pkl files needed."""
    with patch("src.api.predictor.joblib") as mock_joblib:
        mock_model  = MagicMock()
        mock_scaler = MagicMock()

        mock_joblib.load.side_effect = [mock_model, mock_scaler]
        mock_scaler.transform.return_value = np.zeros((1, 11))
        mock_model.predict_proba.return_value = np.array([[0.0001, 0.9999]])

        from src.api.predictor import FraudPredictor
        predictor = FraudPredictor()
        predictor.model   = mock_model
        predictor.scaler  = mock_scaler
        yield predictor
