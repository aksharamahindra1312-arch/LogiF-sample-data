"""
Script to evaluate Logi Freight (LF) model on unseen data: Akshare_May_Jun_Events.parquet
Generates classification metrics, confusion matrix, and ETA regression metrics.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Set paths
base_dir = r"C:\Users\raksh\.gemini\antigravity\scratch\Logi_Freight"
sys.path.append(base_dir)

from src.data_preprocessing import clean_event_stream
from src.snapshot_generator import generate_consignment_snapshots
from src.test import evaluate_classification_metrics, evaluate_eta_metrics

data_path = r"C:\Users\raksh\.gemini\antigravity\scratch\LogiF-sample-data\Akshare_May_Jun_Events.parquet"
models_dir = os.path.join(base_dir, "models")

print("=" * 80)
print("LOGI FREIGHT (LF) — EVALUATING ON UNSEEN DATA (May–June Events)")
print("=" * 80)
print(f"Data file: {data_path}")
print(f"Models dir: {models_dir}")

# 1. Load Model Artifacts
print("\n[1/4] Loading trained LF model artifacts...")
with open(os.path.join(models_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

clf_model = joblib.load(os.path.join(models_dir, "delay_classifier.joblib"))
calibrator = joblib.load(os.path.join(models_dir, "probability_calibrator.joblib"))
p10_model = joblib.load(os.path.join(models_dir, "eta_p10_regressor.joblib"))
p50_model = joblib.load(os.path.join(models_dir, "eta_p50_regressor.joblib"))
p90_model = joblib.load(os.path.join(models_dir, "eta_p90_regressor.joblib"))
encoder = joblib.load(os.path.join(models_dir, "ordinal_encoder.joblib"))

feature_cols = meta["feature_columns"]
categorical_cols = meta["categorical_cols"]
threshold = meta.get("best_threshold", 0.34)
print(f"Loaded successfully. Using decision threshold: {threshold}")

# 2. Load and Preprocess Unseen Data
print("\n[2/4] Reading parquet dataset and cleaning event stream...")
raw_df = pd.read_parquet(data_path)
print(f"Raw events: {len(raw_df):,} rows | Unique consignments: {raw_df['Consignment No'].nunique():,}")

clean_df = clean_event_stream(raw_df)
print(f"Cleaned events: {len(clean_df):,} rows")

# 3. Generate In-Transit Snapshots (Strict Point-in-Time, Zero Lookahead)
print("\n[3/4] Generating in-transit snapshots across consignment lifecycles...")
all_snapshots = []
for c_no, c_events in clean_df.groupby("Consignment No"):
    # Generate snapshots at 25%, 50%, 75%, 90% transit progress
    snaps = generate_consignment_snapshots(c_events, snapshot_ratios=[0.25, 0.50, 0.75, 0.90])
    all_snapshots.extend(snaps)

df_snaps = pd.DataFrame(all_snapshots)
print(f"Total in-transit snapshots generated: {len(df_snaps):,}")
print(f"Unique consignments evaluated: {df_snaps['Consignment_Number'].nunique():,}")

# Filter valid ground truth
target_col = "Target_Is_Delayed" if "Target_Is_Delayed" in df_snaps.columns else "Actual_Is_Delayed"
df_snaps = df_snaps.dropna(subset=[target_col])
df_snaps[target_col] = df_snaps[target_col].astype(int)
print(f"Evaluable snapshots with ground truth: {len(df_snaps):,}")
print(f"Actual Delayed: {df_snaps[target_col].sum():,} ({df_snaps[target_col].mean()*100:.1f}%)")
print(f"Actual On-Time: {(1 - df_snaps[target_col]).sum():,} ({(1 - df_snaps[target_col]).mean()*100:.1f}%)")

# 4. Feature Encoding and Inference
print("\n[4/4] Executing LF model inference...")
# Prepare feature matrix
X_df = df_snaps.copy()
for col in feature_cols:
    if col not in X_df.columns:
        X_df[col] = 0.0

# Encode categorical variables using trained encoder
encoded_cats = encoder.transform(X_df[categorical_cols].astype(str))
for i, col in enumerate(categorical_cols):
    X_df[col] = encoded_cats[:, i]

X_mat = X_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Predict probabilities and calibrated risk
raw_probs = clf_model.predict_proba(X_mat)[:, 1]
calibrated_probs = calibrator.predict(raw_probs)
preds = (calibrated_probs >= threshold).astype(int)

# ETA regression predictions
p10_preds = p10_model.predict(X_mat)
p50_preds = p50_model.predict(X_mat)
p90_preds = p90_model.predict(X_mat)

y_true = df_snaps[target_col].values
y_true_rem = df_snaps["Target_Remaining_Hours"].values if "Target_Remaining_Hours" in df_snaps else (df_snaps["Actual_Remaining_Hours"].values if "Actual_Remaining_Hours" in df_snaps else None)

# Calculate metrics
cls_metrics = evaluate_classification_metrics(y_true, calibrated_probs, threshold=threshold)

# Extract Confusion Matrix values
cm = cls_metrics["confusion_matrix"]
tn, fp = cm[0][0], cm[0][1]
fn, tp = cm[1][0], cm[1][1]

print("\n" + "=" * 80)
print("LOGI FREIGHT (LF) — PERFORMANCE REPORT ON UNSEEN DATA")
print("=" * 80)
print(f"Dataset:              Akshare_May_Jun_Events.parquet")
print(f"Total Snapshots:      {len(df_snaps):,}")
print(f"Unique Consignments:  {df_snaps['Consignment_Number'].nunique():,}")
print(f"Decision Threshold:   {threshold}")
print("-" * 80)
print(f"ACCURACY:             {cls_metrics['accuracy']:.2f}%")
print(f"PRECISION:            {cls_metrics['precision']:.2f}%")
print(f"RECALL (Delays):      {cls_metrics['recall']:.2f}%")
print(f"F1-SCORE:             {cls_metrics['f1']:.4f}")
print(f"ROC-AUC:              {cls_metrics['roc_auc']:.4f}")
print(f"PR-AUC:               {cls_metrics['pr_auc']:.4f}")
print(f"BRIER SCORE:          {cls_metrics['brier']:.4f}")
print("-" * 80)
print("CONFUSION MATRIX:")
print(f"  True Negative (On-Time Correct):  {tn:,}  ({tn/len(df_snaps)*100:.1f}%)")
print(f"  False Positive (False Alarms):    {fp:,}  ({fp/len(df_snaps)*100:.1f}%)")
print(f"  False Negative (Missed Delays):   {fn:,}  ({fn/len(df_snaps)*100:.1f}%)")
print(f"  True Positive (Delays Caught):    {tp:,}  ({tp/len(df_snaps)*100:.1f}%)")
print("-" * 80)

if y_true_rem is not None and len(y_true_rem) > 0 and not np.isnan(y_true_rem).all():
    valid_mask = ~np.isnan(y_true_rem)
    eta_metrics = evaluate_eta_metrics(y_true_rem[valid_mask], p10_preds[valid_mask], p50_preds[valid_mask], p90_preds[valid_mask])
    print(f"ETA MAE (Mean Error): {eta_metrics['mae']:.2f} hours")
    print(f"ETA MedAE (Median):   {eta_metrics['medae']:.2f} hours")
    print(f"ETA P10-P90 Coverage: {eta_metrics['p10_coverage']:.1f}% - {eta_metrics['p90_coverage']:.1f}%")
    print("-" * 80)

# Save results to JSON and CSV for downstream reporting
df_snaps["Predicted_Delayed"] = preds
df_snaps["Delay_Risk_Percentage"] = (calibrated_probs * 100.0).round(2)
df_snaps["Predicted_Remaining_Hours_P50"] = np.round(p50_preds, 2)

report_data = {
    "dataset": "Akshare_May_Jun_Events.parquet",
    "evaluation_timestamp": datetime.now().isoformat(),
    "total_snapshots": len(df_snaps),
    "unique_consignments": int(df_snaps["Consignment_Number"].nunique()),
    "classification_metrics": cls_metrics,
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
}

out_report_json = os.path.join(base_dir, "reports", "unseen_may_jun_evaluation_report.json")
os.makedirs(os.path.dirname(out_report_json), exist_ok=True)
with open(out_report_json, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

out_preds_csv = os.path.join(base_dir, "reports", "unseen_may_jun_predictions.csv")
df_snaps.to_csv(out_preds_csv, index=False)
print(f"Report saved to: {out_report_json}")
print(f"Predictions saved to: {out_preds_csv}")
print("=" * 80)
