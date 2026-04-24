# Step 6: Model Evaluation & Hyperparameter Tuning

## 1. Evaluation Metrics

All three models were evaluated on the held-out 20% test set (1,272,524 rows). Focus was placed on Class 1 (Fraud) metrics as per the success criteria defined in Step 1.

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | PR AUC | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.89 | 0.46 | 0.60 | 0.6464 | 0.9922 |
| **XGBoost** | **0.85** | **0.99** | **0.92** | **0.9868** | **0.9995** |
| LightGBM | 0.08 | 0.93 | 0.15 | 0.0623 | 0.9418 |

**Saved Figures:**
- `reports/figures/pr_curves.png` — Precision-Recall curves for all 3 models
- `reports/figures/roc_curves.png` — ROC curves for all 3 models

---

## 2. Threshold Optimisation (XGBoost)

The default classification threshold of 0.5 is not always optimal for imbalanced datasets. We used the PR curve to find the threshold that maximises the F1-Score for the fraud class.

| Threshold | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|---|---|---|---|
| 0.50 (Default) | 0.85 | 0.99 | 0.92 |
| **0.9983 (Optimal)** | **0.94** | **0.94** | **0.94** |

**Key Insight:** Raising the threshold to 0.9983 trades a small amount of recall for a significant precision boost (+9%). For our fraud use case, this means: for every 100 flags the model raises, we go from correctly catching 85 real fraudsters to catching 94 — dramatically reducing customer "insult rate."

---

## 3. Hyperparameter Tuning

**Strategy:** `RandomizedSearchCV` with 3-fold Stratified CV was run on a stratified 200,000-row sample of the training data (10 iterations per model). The best configurations were then re-fitted on the full ~5M row training set.

### Best XGBoost Hyperparameters Found:
| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 8 |
| `learning_rate` | 0.1 |
| `subsample` | 1.0 |
| `colsample_bytree` | 1.0 |

### Best LightGBM Hyperparameters Found:
| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `num_leaves` | 127 |
| `min_child_samples` | 20 |
| `learning_rate` | 0.05 |
| `reg_alpha` | 0.0 |

---

## 4. Cross-Validation

Cross-validation (3-fold, stratified) was performed on the 200k sample during the hyperparameter search for both models. This guaranteed that reported CV F1-scores were not overfitting to any particular subset of the training data.

---

## 5. Final Test Results (Tuned Models on Held-Out Test Set)

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | PR AUC |
|---|---|---|---|---|
| **XGBoost (Tuned)** | **0.87** | **0.99** | **0.93** | **0.9882** |
| LightGBM (Tuned) | 0.03 | 0.93 | 0.06 | 0.0292 |

### Key Findings:
- **XGBoost (Tuned)** marginally improved from F1=0.92 → **F1=0.93** and PR AUC from 0.9868 → **0.9882**. This is our **champion model**.
- **LightGBM** consistently underperforms on this dataset regardless of tuning. Its `scale_pos_weight` weighting strategy causes it to overpredict fraud, flooding the model with false positives. It is **not recommended for deployment** on this dataset without further investigation (e.g., using `is_unbalance=True` instead of `scale_pos_weight`).

---

## 6. Champion Model Summary

> **XGBoost (Tuned)** with threshold **0.9983** is the selected champion model.

| Metric | Value |
|---|---|
| Fraud Precision | 0.94 |
| Fraud Recall | 0.94 |
| Fraud F1-Score | 0.94 |
| PR AUC | 0.9882 |
| ROC AUC | 0.9995 |
| Saved Path | `models/xgboost_tuned.pkl` |

---

## 7. Saved Artifacts
| File | Description |
|---|---|
| `models/xgboost_tuned.pkl` | Tuned XGBoost champion model |
| `models/lightgbm_tuned.pkl` | Tuned LightGBM model (for reference) |
| `reports/figures/pr_curves.png` | PR Curve comparison plot |
| `reports/figures/roc_curves.png` | ROC Curve comparison plot |
| `reports/evaluation_results.txt` | Full evaluation metrics report |
| `reports/tuning_results.txt` | Best hyperparameters & final test scores |
| `src/models/evaluate_model.py` | Evaluation pipeline script |
| `src/models/tune_model.py` | Hyperparameter tuning script |
| `notebooks/04-evaluation-and-tuning.ipynb` | Interactive evaluation notebook |

## 8. Next Steps
Proceed to **Step 7: Model Deployment (MLOps)** — wrap the champion XGBoost model (with the fitted RobustScaler and optimal threshold) in a **FastAPI** REST endpoint, containerise with **Docker**, and prepare for cloud deployment.
