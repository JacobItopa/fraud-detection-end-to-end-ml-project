import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import time

def evaluate_model(model_name, y_true, y_pred, y_prob):
    print(f"\n--- {model_name} Evaluation ---")
    print(classification_report(y_true, y_pred))
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"PR AUC (Average Precision): {pr_auc:.4f}")
    return pr_auc

def main():
    start_time = time.time()
    
    data_path = 'data/processed/processed_transactions.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run build_features.py first.")
        return

    print("Loading preprocessed dataset (this may take a minute)...")
    
    # We heavily optimize memory during load by using efficient typing if needed, 
    # but for a standard 1GB file, pandas read_csv handles it fine in ~10 seconds.
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    y = df['isFraud']
    X = df.drop(columns=['isFraud'])

    print("\nSplitting data (80% Train, 20% Test) with Stratification...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training set: {X_train.shape[0]:,} rows")
    print(f"Testing set: {X_test.shape[0]:,} rows")
    
    # Calculate scale_pos_weight (Count of Negatives / Count of Positives)
    # Required for XGBoost and LightGBM to heavily penalize errors on Fraud class
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = float(neg_count / pos_count)
    print(f"Handling Imbalance. Calculated scale_pos_weight = {scale_weight:.2f}")

    # Ensure output directory exists
    os.makedirs('models', exist_ok=True)
    
    # --- 1. Baseline Model (Logistic Regression) ---
    print("\nTraining Baseline (Logistic Regression)...")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
    evaluate_model("Baseline (Logistic Regression)", y_test, y_pred_base, y_prob_base)
    joblib.dump(baseline_model, 'models/baseline_logreg.pkl')

    # --- 2. Primary Model (XGBoost) ---
    print("\nTraining Primary Model (XGBoost)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        scale_pos_weight=scale_weight, 
        tree_method='hist', # 'hist' is massively faster for multi-million row datasets
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities of class 1
    evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')

    # --- 3. Alternative Model (LightGBM) ---
    print("\nTraining Alternative Model (LightGBM)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    evaluate_model("LightGBM", y_test, y_pred_lgb, y_prob_lgb)
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')

    print(f"\nAll models saved successfully to 'models/'. Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
