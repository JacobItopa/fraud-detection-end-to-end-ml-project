import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

df = pd.read_csv("data/processed/processed_transactions.csv")
y = df["isFraud"]
X = df.drop(columns=["isFraud"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": joblib.load("models/baseline_logreg.pkl"),
    "XGBoost": joblib.load("models/xgboost_model.pkl"),
    "LightGBM": joblib.load("models/lightgbm_model.pkl"),
}

print("\n=== MODEL COMPARISON SUMMARY (Class 1: Fraud) ===")
for name, m in models.items():
    p = m.predict(X_test)
    pr = m.predict_proba(X_test)[:, 1]
    print(f"\n{name}:")
    print(f"  Fraud Precision: {precision_score(y_test, p):.4f}")
    print(f"  Fraud Recall:    {recall_score(y_test, p):.4f}")
    print(f"  Fraud F1:        {f1_score(y_test, p):.4f}")
    print(f"  PR AUC:          {average_precision_score(y_test, pr):.4f}")
