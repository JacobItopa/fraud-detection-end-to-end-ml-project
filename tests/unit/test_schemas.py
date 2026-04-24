"""
Unit tests for Pydantic request/response schemas.
Tests validation logic without touching any model files.
"""

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    TransactionInput, BatchTransactionInput,
    PredictionOutput, BatchPredictionOutput
)


class TestTransactionInput:

    def test_valid_transfer(self):
        t = TransactionInput(
            type="TRANSFER", amount=181.0,
            oldbalanceOrg=181.0, newbalanceOrig=0.0,
            oldbalanceDest=0.0, newbalanceDest=0.0
        )
        assert t.type == "TRANSFER"
        assert t.amount == 181.0

    def test_type_is_uppercased(self):
        """Validator should uppercase the type string."""
        t = TransactionInput(
            type="transfer", amount=100.0,
            oldbalanceOrg=100.0, newbalanceOrig=0.0,
            oldbalanceDest=0.0, newbalanceDest=0.0
        )
        assert t.type == "TRANSFER"

    def test_all_valid_types_accepted(self):
        valid_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        for tx_type in valid_types:
            t = TransactionInput(
                type=tx_type, amount=100.0,
                oldbalanceOrg=100.0, newbalanceOrig=0.0,
                oldbalanceDest=0.0, newbalanceDest=0.0
            )
            assert t.type == tx_type

    def test_invalid_type_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            TransactionInput(
                type="WIRE_TRANSFER", amount=100.0,
                oldbalanceOrg=100.0, newbalanceOrig=0.0,
                oldbalanceDest=0.0, newbalanceDest=0.0
            )
        assert "type" in str(exc_info.value).lower() or "WIRE_TRANSFER" in str(exc_info.value)

    def test_negative_amount_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TransactionInput(
                type="PAYMENT", amount=-50.0,
                oldbalanceOrg=100.0, newbalanceOrig=50.0,
                oldbalanceDest=0.0, newbalanceDest=0.0
            )

    def test_missing_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TransactionInput(type="PAYMENT", amount=100.0)  # missing balance fields

    def test_zero_amount_is_valid(self):
        """Edge case: zero-amount transactions should be accepted."""
        t = TransactionInput(
            type="DEBIT", amount=0.0,
            oldbalanceOrg=0.0, newbalanceOrig=0.0,
            oldbalanceDest=0.0, newbalanceDest=0.0
        )
        assert t.amount == 0.0


class TestBatchTransactionInput:

    def test_valid_batch(self):
        payload = {
            "transactions": [
                {"type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 100.0,
                 "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
                {"type": "PAYMENT", "amount": 50.0, "oldbalanceOrg": 200.0,
                 "newbalanceOrig": 150.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
            ]
        }
        batch = BatchTransactionInput(**payload)
        assert len(batch.transactions) == 2

    def test_empty_batch_raises_validation_error(self):
        with pytest.raises(ValidationError):
            BatchTransactionInput(transactions=[])


class TestPredictionOutput:

    def test_fraud_response(self):
        r = PredictionOutput(is_fraud=True, fraud_probability=0.999)
        assert r.is_fraud is True
        assert r.fraud_probability == 0.999

    def test_defaults_populated(self):
        r = PredictionOutput(is_fraud=False, fraud_probability=0.001)
        assert r.model == "XGBoost (Tuned)"
        assert r.version == "1.0.0"
        assert r.threshold_used == 0.9983
