import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script mode

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score
)

def load_data():
    print("Loading processed data...")
    df = pd.read_csv('data/processed/processed_transactions.csv')
    y = df['isFraud']
    X = df.drop(columns=['isFraud'])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Test set loaded: {X_test.shape[0]:,} rows")
    return X_test, y_test

def find_optimal_threshold(y_true, y_prob):
    """Find the threshold that maximises F1 on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]

def main():
    os.makedirs('reports/figures', exist_ok=True)

    X_test, y_test = load_data()

    models = {
        'Logistic Regression': joblib.load('models/baseline_logreg.pkl'),
        'XGBoost':             joblib.load('models/xgboost_model.pkl'),
        'LightGBM':            joblib.load('models/lightgbm_model.pkl'),
    }

    # ── 1. PR Curves ────────────────────────────────────────────────────────────
    print("\nPlotting PR Curves...")
    colors = ['#6c757d', '#0d6efd', '#fd7e14']
    fig, ax = plt.subplots(figsize=(9, 6))
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f'{name} (AP={pr_auc:.4f})', color=color, lw=2)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Models', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/pr_curves.png', dpi=150)
    plt.close()
    print("Saved: reports/figures/pr_curves.png")

    # ── 2. ROC Curves ───────────────────────────────────────────────────────────
    print("Plotting ROC Curves...")
    fig, ax = plt.subplots(figsize=(9, 6))
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})', color=color, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curves.png', dpi=150)
    plt.close()
    print("Saved: reports/figures/roc_curves.png")

    # ── 3. Optimal Threshold for XGBoost ────────────────────────────────────────
    print("\n--- XGBoost Threshold Optimisation ---")
    xgb_model = models['XGBoost']
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    best_thresh, best_f1 = find_optimal_threshold(y_test, y_prob_xgb)
    print(f"Default threshold (0.5):")
    y_pred_default = (y_prob_xgb >= 0.50).astype(int)
    print(classification_report(y_test, y_pred_default))

    print(f"Optimal threshold ({best_thresh:.4f}) — F1: {best_f1:.4f}:")
    y_pred_optimal = (y_prob_xgb >= best_thresh).astype(int)
    print(classification_report(y_test, y_pred_optimal))

    # Save results summary
    with open('reports/evaluation_results.txt', 'w') as f:
        f.write("=== Evaluation Results Summary ===\n\n")
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            f.write(f"\n--- {name} ---\n")
            f.write(f"ROC AUC:  {roc_auc_score(y_test, y_prob):.4f}\n")
            f.write(f"PR AUC:   {average_precision_score(y_test, y_prob):.4f}\n")
            f.write(classification_report(y_test, y_pred))
        f.write(f"\n--- XGBoost (Optimal Threshold: {best_thresh:.4f}) ---\n")
        f.write(classification_report(y_test, y_pred_optimal))

    print("\nEvaluation complete. Results saved to reports/evaluation_results.txt")

if __name__ == "__main__":
    main()
