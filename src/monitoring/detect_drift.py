"""
Data Drift Detection using Evidently AI.
Compares historical training data (reference) against live production predictions (current).
Outputs an HTML report.
"""

import os
import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def load_live_data(logs_path="logs/predictions.jsonl"):
    """Loads JSONL prediction logs into a Pandas DataFrame."""
    if not os.path.exists(logs_path):
        print(f"No live logs found at {logs_path}")
        return pd.DataFrame()

    records = []
    with open(logs_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Flatten the request payload
                row = entry.get("request", {})
                # Add the model's prediction
                row["prediction"] = 1 if entry.get("response", {}).get("is_fraud") else 0
                records.append(row)
            except json.JSONDecodeError:
                pass
    
    return pd.DataFrame(records)

def detect_drift():
    print("Loading reference data (subset of training data)...")
    # In a real pipeline, we'd load a dedicated reference subset from GCP/S3.
    # Here, we grab 10,000 rows from the processed raw dataset for speed.
    try:
        ref_data = pd.read_csv("data/raw/Synthetic_Financial_datasets_log.csv", nrows=10000)
    except FileNotFoundError:
        print("Raw data not found. Run the extraction notebook first.")
        return

    # Clean the reference data exactly like the API schema expects
    ref_data["prediction"] = ref_data["isFraud"] # Pretend the actual labels are what the model predicted for reference
    
    print("Loading current live API data...")
    curr_data = load_live_data()
    
    if curr_data.empty:
        print("No live data to compare against. Try hitting the /predict endpoint first.")
        return

    # Ensure columns match for drift comparison
    common_cols = list(set(curr_data.columns).intersection(set(ref_data.columns)))
    ref_data = ref_data[common_cols]
    curr_data = curr_data[common_cols]

    print("Running Evidently Data Drift Report...")
    # Initialize the report with the Data Drift preset
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # Run the comparison
    drift_report.run(reference_data=ref_data, current_data=curr_data)
    
    # Save the report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    drift_report.save_html(report_path)
    
    print(f"✅ Drift report generated successfully: {report_path}")

if __name__ == "__main__":
    detect_drift()
