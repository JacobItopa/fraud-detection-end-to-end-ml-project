import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os

def preprocess_data(df):
    """
    Apply data cleaning, feature engineering, categorical encoding, and scaling to the dataframe.
    """
    print("Starting data preprocessing...")
    df_clean = df.copy()
    
    # 1. Feature Engineering
    print("Engineering features (error balances)...")
    # In fraud detection, mismatches in expected balances frequently indicate fraudulent activity.
    df_clean['errorBalanceOrig'] = df_clean['newbalanceOrig'] + df_clean['amount'] - df_clean['oldbalanceOrg']
    df_clean['errorBalanceDest'] = df_clean['oldbalanceDest'] + df_clean['amount'] - df_clean['newbalanceDest']

    # 2. Data Cleaning / Feature Selection
    print("Dropping sparse and unique ID columns...")
    # 'nameOrig' and 'nameDest' are almost fully unique strings, not useful without graph methods.
    # 'isFlaggedFraud' is a naive rule-based flag that doesn't help tree models.
    # 'step' is dropped to prevent the model from overfitting on specific timeframes.
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    df_clean = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])

    # 3. Categorical Encoding
    print("One-hot encoding 'type' column...")
    df_clean = pd.get_dummies(df_clean, columns=['type'], drop_first=True)
    
    # Ensure boolean columns are integers
    for col in df_clean.select_dtypes(include=['bool']).columns:
        df_clean[col] = df_clean[col].astype(int)

    # 4. Separate X and y
    print("Separating features and target...")
    y = df_clean['isFraud']
    X = df_clean.drop(columns=['isFraud'])

    # 5. Feature Scaling
    print("Scaling numeric columns using RobustScaler...")
    # We use RobustScaler instead of StandardScaler because it isolates the extreme right-tail outliers.
    numeric_cols = X.columns
    scaler = RobustScaler()
    X_scaled_arr = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=numeric_cols, index=X.index)

    # Save the scaler artifact for inference later
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/robust_scaler.pkl')
    print("RobustScaler saved to 'models/robust_scaler.pkl'.")
    
    return X_scaled, y

if __name__ == "__main__":
    data_path = 'data/raw/Synthetic_Financial_datasets_log.csv'
    if not os.path.exists(data_path):
         print(f"Error: {data_path} not found. Please place dataset in data/raw/")
         exit(1)
         
    print(f"Loading raw data from {data_path} ...")
    df_raw = pd.read_csv(data_path)
    
    # Execute preprocessing pipeline
    X_processed, y_processed = preprocess_data(df_raw)
    
    print("\nPreprocessing complete. Final Data shapes:")
    print(f"Features (X): {X_processed.shape}")
    print(f"Target (y): {y_processed.shape}")
    
    # Save processed data for modeling step
    os.makedirs('data/processed', exist_ok=True)
    
    # For a dataset this size, CSV can be slow. Using Parquet would be faster, but let's stick to CSV for universal compatibility
    processed_path = 'data/processed/processed_transactions.csv'
    print(f"Saving fully processed dataset to {processed_path} (This may take a minute)...")
    
    # Stitch target back to features purely for saving
    X_processed['isFraud'] = y_processed.values
    X_processed.to_csv(processed_path, index=False)
    
    print("Done! Data is ready for model training.")
