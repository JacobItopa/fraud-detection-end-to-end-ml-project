import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    make_scorer,
    f1_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

SAMPLE_SIZE   = 200_000  # Rows used for hyperparameter search
RANDOM_STATE  = 42
CV_FOLDS      = 3
N_ITER        = 10        # Number of random combinations to try per model

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # ── Load & Split ────────────────────────────────────────────────────────────
    print("Loading processed data...")
    df = pd.read_csv('data/processed/processed_transactions.csv')
    y = df['isFraud']
    X = df.drop(columns=['isFraud'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Full training set: {X_train.shape[0]:,} rows")

    # ── Stratified sample for search ────────────────────────────────────────────
    print(f"\nSampling {SAMPLE_SIZE:,} rows for hyperparameter search...")
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
    print(f"Sample fraud rate: {y_sample.mean()*100:.4f}%")

    # Scale weight for imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = float(neg / pos)

    # Scorer: F1 on minority class
    f1_fraud = make_scorer(f1_score, pos_label=1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    # ── XGBoost Search ──────────────────────────────────────────────────────────
    print("\n[1/2] RandomizedSearchCV on XGBoost...")
    xgb_params = {
        'n_estimators':    [100, 200, 300],
        'max_depth':       [4, 6, 8],
        'learning_rate':   [0.05, 0.1, 0.2],
        'subsample':       [0.7, 0.8, 1.0],
        'colsample_bytree':[0.7, 0.8, 1.0],
    }
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        tree_method='hist',
        random_state=RANDOM_STATE,
        eval_metric='aucpr'
    )
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params,
        n_iter=N_ITER, cv=cv, scoring=f1_fraud,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    t0 = time.time()
    xgb_search.fit(X_sample, y_sample)
    print(f"XGBoost search complete in {(time.time()-t0)/60:.1f} min")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    print(f"Best CV F1 (fraud):  {xgb_search.best_score_:.4f}")
    results['XGBoost'] = xgb_search.best_params_

    # ── LightGBM Search ─────────────────────────────────────────────────────────
    print("\n[2/2] RandomizedSearchCV on LightGBM...")
    lgb_params = {
        'n_estimators':      [100, 200, 300],
        'num_leaves':        [31, 63, 127],
        'min_child_samples': [20, 50, 100],
        'learning_rate':     [0.05, 0.1, 0.2],
        'reg_alpha':         [0.0, 0.1, 0.5],
    }
    lgb_base = lgb.LGBMClassifier(
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
    )
    lgb_search = RandomizedSearchCV(
        lgb_base, lgb_params,
        n_iter=N_ITER, cv=cv, scoring=f1_fraud,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1
    )
    t0 = time.time()
    lgb_search.fit(X_sample, y_sample)
    print(f"LightGBM search complete in {(time.time()-t0)/60:.1f} min")
    print(f"Best LightGBM params: {lgb_search.best_params_}")
    print(f"Best CV F1 (fraud):   {lgb_search.best_score_:.4f}")
    results['LightGBM'] = lgb_search.best_params_

    # ── Refit on Full Training Set ───────────────────────────────────────────────
    print("\nRefitting tuned models on full training set (~5M rows)...")

    xgb_tuned = xgb.XGBClassifier(
        **xgb_search.best_params_,
        scale_pos_weight=scale_weight,
        tree_method='hist',
        random_state=RANDOM_STATE
    )
    xgb_tuned.fit(X_train, y_train)
    joblib.dump(xgb_tuned, 'models/xgboost_tuned.pkl')
    print("Saved: models/xgboost_tuned.pkl")

    lgb_tuned = lgb.LGBMClassifier(
        **lgb_search.best_params_,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE
    )
    lgb_tuned.fit(X_train, y_train)
    joblib.dump(lgb_tuned, 'models/lightgbm_tuned.pkl')
    print("Saved: models/lightgbm_tuned.pkl")

    # ── Final Test Evaluation ────────────────────────────────────────────────────
    print("\n========== FINAL TEST EVALUATION ==========")
    tuned_models = {
        'XGBoost (Tuned)':   xgb_tuned,
        'LightGBM (Tuned)':  lgb_tuned,
    }
    with open('reports/tuning_results.txt', 'w') as f:
        f.write("=== Step 6: Hyperparameter Tuning Results ===\n\n")
        f.write(f"Search sample size: {SAMPLE_SIZE:,} rows\n")
        f.write(f"CV folds: {CV_FOLDS} | Iterations per model: {N_ITER}\n\n")
        for name, params in results.items():
            f.write(f"Best {name} Params:\n{params}\n\n")
        f.write("\n=== Final Test Results (Held-out 20%) ===\n")

        for name, model in tuned_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            report = classification_report(y_test, y_pred)
            pr_auc = average_precision_score(y_test, y_prob)
            print(f"\n--- {name} ---")
            print(report)
            print(f"PR AUC: {pr_auc:.4f}")
            f.write(f"\n--- {name} ---\n{report}\nPR AUC: {pr_auc:.4f}\n")

    print("\nTuning complete. Results saved to reports/tuning_results.txt")

if __name__ == "__main__":
    main()
