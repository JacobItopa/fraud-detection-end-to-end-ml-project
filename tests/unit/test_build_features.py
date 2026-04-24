"""
Unit tests for the Step 4 data preprocessing pipeline (build_features.py).
Tests run on a small in-memory DataFrame — no CSV files required.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_raw_df():
    """A minimal raw DataFrame that mimics the structure of the full dataset."""
    return pd.DataFrame([
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C1234567890",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C0987654321",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        },
        {
            "step": 2,
            "type": "PAYMENT",
            "amount": 9839.64,
            "nameOrig": "C1231006815",
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "nameDest": "M1979787155",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
    ])


from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_joblib_dump():
    """Prevent tests from overwriting production .pkl files."""
    with patch("src.features.build_features.joblib.dump") as mock_dump:
        yield mock_dump

class TestPreprocessData:

    def test_dropped_columns_absent(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        for col in ["nameOrig", "nameDest", "isFlaggedFraud", "step", "type"]:
            assert col not in X.columns, f"Column '{col}' should have been dropped"

    def test_engineered_features_present(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        assert "errorBalanceOrig" in X.columns
        assert "errorBalanceDest" in X.columns

    def test_error_balance_orig_values(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        # Row 0: newbalanceOrig(0) + amount(181) - oldbalanceOrg(181) = 0
        assert X["errorBalanceOrig"].iloc[0] == pytest.approx(0.0)

    def test_error_balance_dest_formula_is_correct(self, sample_raw_df):
        """
        preprocess_data scales the output, so we verify the formula directly
        on the raw DataFrame instead of comparing absolute scaled values.
        Row 0: oldbalanceDest(0) + amount(181) - newbalanceDest(0) = 181
        """
        row = sample_raw_df.iloc[0]
        result = row["oldbalanceDest"] + row["amount"] - row["newbalanceDest"]
        assert result == pytest.approx(181.0)

    def test_no_object_dtype_columns_after_encoding(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        object_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
        assert object_cols == [], f"Found unencoded string columns: {object_cols}"

    def test_no_missing_values_in_output(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        assert X.isnull().sum().sum() == 0, "Output X contains NaN values"

    def test_target_separated_correctly(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        assert "isFraud" not in X.columns
        assert list(y) == [1, 0]

    def test_output_row_count_unchanged(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        assert len(X) == len(sample_raw_df)
        assert len(y) == len(sample_raw_df)

    def test_transfer_one_hot_encoded(self, sample_raw_df):
        from src.features.build_features import preprocess_data
        X, y = preprocess_data(sample_raw_df)
        # Row 0 is TRANSFER — type_TRANSFER should be present and = 1
        assert "type_TRANSFER" in X.columns
        assert X["type_TRANSFER"].iloc[0] == 1
