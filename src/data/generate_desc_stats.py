import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

print("Loading dataset...")
data_path = 'data/raw/Synthetic_Financial_datasets_log.csv'

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

df = pd.read_csv(data_path)
print("Dataset loaded. Generating statistics...")

with open("reports/descriptive_statistics.md", "w", encoding="utf-8") as f:
    f.write("# Exploratory Data Analysis: Descriptive Statistics\n\n")
    
    f.write("## 1. Dataset Shape\n")
    f.write(f"- **Rows:** {df.shape[0]:,}\n")
    f.write(f"- **Columns:** {df.shape[1]}\n\n")
    
    f.write("## 2. Column Data Types\n")
    f.write("```text\n")
    buf = io.StringIO()
    df.info(buf=buf)
    f.write(buf.getvalue())
    f.write("```\n\n")
    
    f.write("## 3. Missing Values\n")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({'Missing Values': missing})
    f.write(missing_df.to_markdown())
    f.write("\n\n")
    
    f.write("## 4. Summary Statistics (Numerical)\n")
    f.write(df.describe().T.to_markdown())
    f.write("\n\n")
    
    f.write("## 5. Summary Statistics (Categorical)\n")
    cat_df = df.select_dtypes(include=['object', 'category'])
    if not cat_df.empty:
        f.write(cat_df.describe().T.to_markdown())
    else:
        f.write("No categorical features found.\n")

print("Descriptive statistics successfully saved to reports/descriptive_statistics.md")

print("Generating visualizations...")

# Ensure reports/figures directory exists
os.makedirs("reports/figures", exist_ok=True)

# 1. Distribution of Fraudulent vs Legitimate Transactions
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='isFraud')
plt.title('Distribution of Fraudulent vs Legitimate Transactions')
plt.yscale('log') # Log scale because of high imbalance
plt.tight_layout()
plt.savefig('reports/figures/fraud_distribution.png')
plt.close()

# 2. Transaction Types by Fraud Status
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='type', hue='isFraud')
plt.title('Transaction Types by Fraud Status')
plt.yscale('log')
plt.tight_layout()
plt.savefig('reports/figures/transaction_types_by_fraud.png')
plt.close()

print("Visualizations successfully saved to reports/figures/")
