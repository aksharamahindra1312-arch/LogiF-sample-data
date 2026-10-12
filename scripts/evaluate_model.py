"""
CLI script to evaluate model performance on test datasets.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.test import evaluate_classification_metrics, evaluate_eta_metrics

def main():
    parser = argparse.ArgumentParser(description="Logi Freight — Model Evaluation CLI")
    parser.add_argument("--test_predictions", type=str, default="predictions.csv", help="Path to predictions file with ground truth")
    args = parser.parse_args()
    
    if not os.path.exists(args.test_predictions):
        print(f"File not found: {args.test_predictions}")
        return
        
    df = pd.read_csv(args.test_predictions)
    if 'Actual_Is_Delayed' in df.columns:
        valid_df = df.dropna(subset=['Actual_Is_Delayed'])
        y_true = valid_df['Actual_Is_Delayed'].values
        y_probs = (valid_df['Delay_Risk_Percentage'] / 100.0).values
        metrics = evaluate_classification_metrics(y_true, y_probs)
        print("Classification Metrics:")
        print(metrics)
    else:
        print("File does not contain ground truth column 'Actual_Is_Delayed'. Evaluation skipped.")

if __name__ == "__main__":
    main()
