# Step 5: Model Selection & Training

## 1. Data Splitting
* **Split Ratio:** 80% Training / 20% Testing on the fully processed 6.3M row dataset.
* **Stratification:** Used `stratify=y` to guarantee the fraud class (0.13%) was proportionally represented in both splits.
* **Results:**
  * Training set: ~5,090,096 rows
  * Testing set: ~1,272,524 rows

## 2. Baseline Established
Two baselines were established to measure model improvement against:
* **Naive Baseline:** Predict `0` (Legitimate) for every transaction → ~99.87% Accuracy but **0 fraud caught**.
* **ML Baseline (Logistic Regression):** Trained to capture a linear decision boundary.
  * Fraud Recall: **45.53%** — only catching about half of all fraudsters.
  * Confirmed our advanced models have a concrete benchmark to beat.

## 3. Algorithm Selection
Two advanced gradient boosting classifiers were selected, **both configured to handle the class imbalance via Algorithm-Level Weighting (Option B)**:

* **XGBoost** — High accuracy boosting model, optimized for large datasets via `tree_method='hist'`.
* **LightGBM** — Leaf-wise boosting model, typically faster than XGBoost on datasets >2M rows.

The class imbalance weight was calculated as:
```
scale_pos_weight = Count of Legitimate / Count of Fraudulent = ~774
```
This instructs both models to penalize errors on the minority fraud class ~774x more heavily.

## 4. Model Training Results

All models were trained and serialized to the `models/` directory.

| Model | Precision | Recall | F1-Score | PR AUC |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.8905 | 0.4553 | 0.6025 | 0.6464 |
| **XGBoost** | **0.8501** | **0.9939** | **0.9164** | **0.9868** |
| LightGBM | 0.0816 | 0.9270 | 0.1499 | 0.0623 |

## 5. Key Findings
* **XGBoost** is the clear winner at this stage, catching **99.39% of all fraudulent transactions** with an outstanding PR AUC of **0.9868**.
* **LightGBM** needs significant hyperparameter tuning — it is generating too many false positives (only 8.16% precision), effectively flagging ~11 innocent customers for every true fraudster.
* The Logistic Regression baseline only caught **45.53%** of fraud, confirming that our advanced tree models are learning complex non-linear patterns that a linear model cannot detect.

## 6. Saved Artifacts
| File | Description |
|---|---|
| `models/baseline_logreg.pkl` | Serialized Logistic Regression baseline |
| `models/xgboost_model.pkl` | Serialized XGBoost classifier |
| `models/lightgbm_model.pkl` | Serialized LightGBM classifier |
| `models/robust_scaler.pkl` | Fitted RobustScaler for inference |
| `src/models/train_model.py` | Full training pipeline script |
| `src/models/compare_models.py` | Model comparison metrics script |
| `notebooks/03-model-training.ipynb` | Interactive training and evaluation notebook |

## 7. Next Steps
Proceed to **Step 6: Model Evaluation & Hyperparameter Tuning** to:
* Tune LightGBM's `num_leaves`, `min_child_samples`, and decision threshold to fix its precision/recall imbalance.
* Run cross-validation on XGBoost to ensure its performance is consistent across folds.
* Use PR Curve visualization to pick the optimal classification threshold for production use.
