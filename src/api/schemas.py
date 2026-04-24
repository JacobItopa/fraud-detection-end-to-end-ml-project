"""
Pydantic schemas for request and response validation.
FastAPI uses these to auto-validate inputs and generate OpenAPI docs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List
from src.api.config import VALID_TYPES, MODEL_NAME, MODEL_VERSION, FRAUD_THRESHOLD


class TransactionInput(BaseModel):
    """Schema for a single transaction prediction request."""

    type: str = Field(..., description="Transaction type", examples=["TRANSFER"])
    amount: float = Field(..., ge=0, description="Transaction amount in USD")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender's balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender's balance after transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Recipient's balance before transaction")
    newbalanceDest: float = Field(..., ge=0, description="Recipient's balance after transaction")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}")
        return v_upper

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "TRANSFER",
                    "amount": 181.0,
                    "oldbalanceOrg": 181.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                }
            ]
        }
    }


class BatchTransactionInput(BaseModel):
    """Schema for batch prediction requests."""
    transactions: List[TransactionInput] = Field(..., min_length=1, max_length=1000)


class PredictionOutput(BaseModel):
    """Schema for a single prediction response."""
    is_fraud: bool
    fraud_probability: float = Field(..., description="Raw model probability (0–1)")
    threshold_used: float = Field(default=FRAUD_THRESHOLD)
    model: str = Field(default=MODEL_NAME)
    version: str = Field(default=MODEL_VERSION)


class BatchPredictionOutput(BaseModel):
    """Schema for a batch prediction response."""
    predictions: List[PredictionOutput]
    total: int
    fraud_count: int


class HealthResponse(BaseModel):
    """Schema for the health check endpoint."""
    status: str
    model: str
    version: str
    threshold: float
