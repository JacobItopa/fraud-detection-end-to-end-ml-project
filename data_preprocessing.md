# Step 4: Data Preprocessing Walkthrough

I have successfully initialized and fully implemented the **Data Preprocessing & Feature Engineering** pipeline across the entire synthetic fraud dataset.

## What Was Done

1. **Feature Engineering Strategy:** 
   Calculated new predictive features representing the delta/irregularity in customer balances: `errorBalanceOrig` and `errorBalanceDest`. Fraudsters often manipulate expected amounts; tracking these errors creates a highly influential mathematical signal.
   
2. **Data Cleaning:** 
   Filtered out raw identifier schemas (`nameOrig`, `nameDest`), dropped unhelpful rule-based flags (`isFlaggedFraud`), and stripped the time sequence tracker (`step`) to force the underlying model to identify pure transactional and mathematical patterns rather than merely overfitting to a specific timeframe block.

3. **Categorical Translation:** 
   One-hot encoded the transactional strings (`type` feature structurally splits into dummy binary variables like `type_CASH_OUT`, `type_TRANSFER`) so algebraic numerical models can consume them seamlessly.

4. **Robust Outlier Scaling:** 
   Passed all numerical columns strictly through `RobustScaler()`, isolating the massive 94 million dollar right-scale outlier transactions. The learned scaler state was serialized and permanently preserved inside `models/robust_scaler.pkl` so you can identically scale future streaming transaction items in production.

5. **Class Imbalance Strategy:** 
   Per your approval of **Option B (Algorithm-Level Weighting)**, we intentionally preserved the extreme mathematical imbalance during this structural rewrite. The preprocessed data is entirely unmodified class-wise, shifting the responsibility directly onto target weight algorithms (e.g. using `scale_pos_weight` mapping in XGBoost) during the actual training schema in Step 5.

## Testing & Verification
- You can find the exploratory sandbox validating these operations in **[notebooks/02-data-preprocessing.ipynb](file:///c:/Users/user/Desktop/2026%20PLAN%20AND%20PROJECT/projects/fraud_detection/notebooks/02-data-preprocessing.ipynb)**.
- The active modular processing pipeline **[build_features.py](file:///c:/Users/user/Desktop/2026%20PLAN%20AND%20PROJECT/projects/fraud_detection/src/features/build_features.py)** was successfully built and tested against the true 6 million raw row distribution, outputting the machine-ready artifact to `/data/processed/processed_transactions.csv`.
