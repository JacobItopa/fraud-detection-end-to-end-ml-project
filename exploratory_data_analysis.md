# Exploratory Data Analysis (EDA)

## 1. Descriptive Statistics
Descriptive statistics for the dataset were fully generated using the `src/data/generate_desc_stats.py` script. The full tabular results have been permanently saved in **`reports/descriptive_statistics.md`**.
*   **Key Finding:** The dataset is massive, containing 6,362,620 rows and 11 features.

## 2. Data Visualization
A dedicated testing ground for visualizations was created in **`notebooks/01-eda-initial.ipynb`**. This notebook contains code to intuitively visualize:
*   The raw class distribution of fraudulent vs. legitimate transactions (using a log scale).
*   The breakdown of fraud cases across different transaction types (e.g., `CASH_OUT`, `TRANSFER`).

## 3. Identified Issues
Initial exploration of the data explicitly revealed the following issues that must be addressed in the Data Preprocessing stage:
*   **Missing Values:** Excellent data hygiene; there are exactly **0** missing values across all columns.
*   **Extreme Class Imbalance:** The target variable (`isFraud`) is heavily skewed. Fraudulent transactions account for only **~0.13%** of the entire dataset. Specialized handling (like SMOTE, custom class weights, or under-sampling) will be strictly necessary.
*   **Heavy Outliers:** The `amount` feature exhibits massive variance. The average transaction is \$179,861, yet the maximum transaction spikes to \$92.4 million. These extreme right-skewed distributions must be scaled or capped to prevent the model from being overly sensitive to outliers.
