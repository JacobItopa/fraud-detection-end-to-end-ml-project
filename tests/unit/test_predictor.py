"""
Unit tests for the FraudPredictor's feature engineering logic.
The model and scaler are mocked — no .pkl files required.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.api.config import FEATURE_COLUMNS


@pytest.fixture
def predictor(mock_predictor):
    return mock_predictor


class TestFeatureEngineering:
    """Tests that _build_features() produces the correct outputs."""

    def test_output_columns_match_feature_columns(self, predictor, fraud_transaction):
        df = predictor._build_features([fraud_transaction])
        assert list(df.columns) == FEATURE_COLUMNS

    def test_column_count_is_eleven(self, predictor, fraud_transaction):
        df = predictor._build_features([fraud_transaction])
        assert df.shape[1] == 11

    def test_error_balance_orig_calculated_correctly(self, predictor, fraud_transaction):
        """errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg"""
        df = predictor._build_features([fraud_transaction])
        tx = fraud_transaction
        expected = tx["newbalanceOrig"] + tx["amount"] - tx["oldbalanceOrg"]
        assert df["errorBalanceOrig"].iloc[0] == pytest.approx(expected)

    def test_error_balance_dest_calculated_correctly(self, predictor, fraud_transaction):
        """errorBalanceDest = oldbalanceDest + amount - newbalanceDest"""
        df = predictor._build_features([fraud_transaction])
        tx = fraud_transaction
        expected = tx["oldbalanceDest"] + tx["amount"] - tx["newbalanceDest"]
        assert df["errorBalanceDest"].iloc[0] == pytest.approx(expected)

    def test_transfer_one_hot_encoding(self, predictor, fraud_transaction):
        """TRANSFER should set type_TRANSFER=1, all others=0."""
        df = predictor._build_features([fraud_transaction])
        assert df["type_TRANSFER"].iloc[0] == 1
        assert df["type_CASH_OUT"].iloc[0] == 0
        assert df["type_DEBIT"].iloc[0] == 0
        assert df["type_PAYMENT"].iloc[0] == 0

    def test_cash_out_one_hot_encoding(self, predictor):
        tx = {
            "type": "CASH_OUT", "amount": 500.0,
            "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 1000.0, "newbalanceDest": 1500.0,
        }
        df = predictor._build_features([tx])
        assert df["type_CASH_OUT"].iloc[0] == 1
        assert df["type_TRANSFER"].iloc[0] == 0

    def test_cash_in_is_baseline_all_type_cols_zero(self, predictor, cash_in_transaction):
        """CASH_IN is the dropped baseline — all type_ dummy columns must be 0."""
        df = predictor._build_features([cash_in_transaction])
        for col in ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
            assert df[col].iloc[0] == 0, f"Expected {col}=0 for CASH_IN"

    def test_batch_of_multiple_transactions(self, predictor, fraud_transaction, legit_transaction):
        df = predictor._build_features([fraud_transaction, legit_transaction])
        assert df.shape == (2, 11)


class TestPredictMethod:

    def test_returns_list_of_dicts(self, predictor, fraud_transaction):
        results = predictor.predict([fraud_transaction])
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_result_has_required_keys(self, predictor, fraud_transaction):
        result = predictor.predict([fraud_transaction])[0]
        assert "is_fraud" in result
        assert "fraud_probability" in result
        assert "threshold_used" in result
        assert "model" in result

    def test_high_probability_flagged_as_fraud(self, predictor, fraud_transaction):
        predictor.model.predict_proba.return_value = np.array([[0.0001, 0.9999]])
        result = predictor.predict([fraud_transaction])[0]
        assert result["is_fraud"] is True

    def test_low_probability_not_flagged_as_fraud(self, predictor, legit_transaction):
        predictor.model.predict_proba.return_value = np.array([[0.9999, 0.0001]])
        result = predictor.predict([legit_transaction])[0]
        assert result["is_fraud"] is False

    def test_threshold_applied_correctly(self, predictor, fraud_transaction):
        """A probability just below the threshold should NOT be flagged."""
        predictor.model.predict_proba.return_value = np.array([[0.01, 0.9982]])
        result = predictor.predict([fraud_transaction])[0]
        assert result["is_fraud"] is False  # 0.9982 < FRAUD_THRESHOLD (0.9983)

    def test_scaler_is_called(self, predictor, fraud_transaction):
        predictor.predict([fraud_transaction])
        predictor.scaler.transform.assert_called_once()
