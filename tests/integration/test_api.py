"""
Integration tests for the FastAPI application.
Uses FastAPI's TestClient to exercise the full request → predictor → response cycle
with the REAL trained model and scaler (.pkl files must exist).
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the duration of the module."""
    with TestClient(app) as c:
        yield c


# ── Health endpoints ─────────────────────────────────────────────────────────

class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_correct_keys(self, client):
        data = client.get("/").json()
        assert "status" in data
        assert "model" in data
        assert "version" in data
        assert "threshold" in data

    def test_root_status_is_ok(self, client):
        assert client.get("/").json()["status"] == "ok"

    def test_health_endpoint_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_threshold_value_is_correct(self, client):
        assert client.get("/").json()["threshold"] == 0.9983


# ── POST /predict ─────────────────────────────────────────────────────────────

class TestPredictEndpoint:

    def test_fraud_transaction_flagged(self, client):
        """Classic fraud pattern: TRANSFER draining sender's account to 0."""
        payload = {
            "type": "TRANSFER",
            "amount": 50000.0,
            "oldbalanceOrg": 50000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is True
        assert data["fraud_probability"] > 0.9983

    def test_legitimate_transaction_cleared(self, client):
        """Normal PAYMENT transaction should not be flagged."""
        payload = {
            "type": "PAYMENT",
            "amount": 9839.64,
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["is_fraud"] is False

    def test_response_contains_required_fields(self, client):
        payload = {
            "type": "PAYMENT", "amount": 100.0,
            "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        data = client.post("/predict", json=payload).json()
        for field in ["is_fraud", "fraud_probability", "threshold_used", "model", "version"]:
            assert field in data, f"Missing field: {field}"

    def test_probability_is_between_0_and_1(self, client):
        payload = {
            "type": "CASH_OUT", "amount": 5000.0,
            "oldbalanceOrg": 5000.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 10000.0, "newbalanceDest": 15000.0,
        }
        prob = client.post("/predict", json=payload).json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_invalid_transaction_type_returns_422(self, client):
        payload = {
            "type": "WIRE",  # Invalid type
            "amount": 100.0,
            "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_amount_returns_422(self, client):
        payload = {
            "type": "PAYMENT", "amount": -500.0,
            "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        }
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_field_returns_422(self, client):
        payload = {"type": "PAYMENT", "amount": 100.0}  # Missing balance fields
        assert client.post("/predict", json=payload).status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post("/predict", json={}).status_code == 422


# ── POST /predict/batch ───────────────────────────────────────────────────────

class TestBatchPredictEndpoint:

    def test_batch_returns_200(self, client):
        payload = {
            "transactions": [
                {"type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0,
                 "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
                {"type": "PAYMENT", "amount": 9839.64, "oldbalanceOrg": 170136.0,
                 "newbalanceOrig": 160296.36, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
            ]
        }
        assert client.post("/predict/batch", json=payload).status_code == 200

    def test_batch_response_counts_are_correct(self, client):
        payload = {
            "transactions": [
                {"type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0,
                 "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
                {"type": "PAYMENT", "amount": 9839.64, "oldbalanceOrg": 170136.0,
                 "newbalanceOrig": 160296.36, "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
            ]
        }
        data = client.post("/predict/batch", json=payload).json()
        assert data["total"] == 2
        assert "fraud_count" in data
        assert len(data["predictions"]) == 2

    def test_empty_batch_returns_422(self, client):
        assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
