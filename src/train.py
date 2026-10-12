"""
Model training workflow and production artifact generation.
Trains Delay Classifier, Probability Calibrator, Quantile ETA Regressors (P10/P50/P90),
Anomaly Detector, and Ordinal Encoder.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

from src.delay_model import build_delay_classifier
from src.calibration import fit_probability_calibrator
from src.eta_models import build_quantile_regressor
from src.anomaly_detection import build_anomaly_detector

def train_and_export_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                            feature_cols: list, categorical_cols: list, numeric_cols: list,
                            models_dir: str) -> dict:
    """Trains all multi-task models, tunes decision threshold, and saves artifacts to disk."""
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Fit Ordinal Encoder on Training set
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(train_df[categorical_cols].astype(str))
    joblib.dump(encoder, os.path.join(models_dir, 'ordinal_encoder.joblib'))
    
    def prepare_X(df):
        df_copy = df.copy()
        encoded = encoder.transform(df_copy[categorical_cols].astype(str))
        for i, col in enumerate(categorical_cols):
            df_copy[col] = encoded[:, i]
        return df_copy[feature_cols]
        
    X_train = prepare_X(train_df)
    y_train_delay = train_df['Target_Is_Delayed'].values
    y_train_rem = train_df['Target_Remaining_Hours'].values
    
    X_val = prepare_X(val_df)
    y_val_delay = val_df['Target_Is_Delayed'].values
    y_val_rem = val_df['Target_Remaining_Hours'].values
    
    X_test = prepare_X(test_df)
    y_test_delay = test_df['Target_Is_Delayed'].values
    y_test_rem = test_df['Target_Remaining_Hours'].values
    
    # 2. Train Delay Classifier (LightGBM)
    clf = build_delay_classifier({
        'n_estimators': 450,
        'learning_rate': 0.03,
        'num_leaves': 40,
        'max_depth': 8,
        'min_child_samples': 15,
        'subsample': 0.85,
        'colsample_bytree': 0.80,
        'random_state': 42
    })
    clf.fit(X_train, y_train_delay)
    joblib.dump(clf, os.path.join(models_dir, 'delay_classifier.joblib'))
    
    # 3. Fit Probability Calibrator (Isotonic Regression on Val set)
    val_raw_probs = clf.predict_proba(X_val)[:, 1]
    calibrator = fit_probability_calibrator(val_raw_probs, y_val_delay)
    joblib.dump(calibrator, os.path.join(models_dir, 'probability_calibrator.joblib'))
    
    # Find Best Decision Threshold on Validation set balancing accuracy and F1
    val_calib_probs = calibrator.predict(val_raw_probs)
    best_th = 0.35
    best_score = 0.0
    for th in np.arange(0.30, 0.70, 0.02):
        pred_bin = (val_calib_probs >= th).astype(int)
        acc_v = accuracy_score(y_val_delay, pred_bin)
        f1_v = f1_score(y_val_delay, pred_bin, zero_division=0)
        score = acc_v * 0.6 + f1_v * 0.4
        if score > best_score:
            best_score = score
            best_th = round(float(th), 2)
            
    # 4. Quantile Regressors (P10, P50, P90)
    p10 = build_quantile_regressor(0.10)
    p10.fit(X_train, y_train_rem)
    joblib.dump(p10, os.path.join(models_dir, 'eta_p10_regressor.joblib'))
    
    p50 = build_quantile_regressor(0.50)
    p50.fit(X_train, y_train_rem)
    joblib.dump(p50, os.path.join(models_dir, 'eta_p50_regressor.joblib'))
    
    p90 = build_quantile_regressor(0.90)
    p90.fit(X_train, y_train_rem)
    joblib.dump(p90, os.path.join(models_dir, 'eta_p90_regressor.joblib'))
    
    # 5. Anomaly Detector
    anom_cols = ['Effective_Speed_Kmph', 'Telemetry_Freshness_Min', 'Elapsed_Frac', 'Behind_Pace', 'Route_Progress_Ratio']
    detector = build_anomaly_detector()
    detector.fit(X_train[anom_cols])
    joblib.dump(detector, os.path.join(models_dir, 'anomaly_detector.joblib'))
    
    # 6. Benchmark Evaluation on Independent Test Set
    test_raw_probs = clf.predict_proba(X_test)[:, 1]
    test_calib_probs = calibrator.predict(test_raw_probs)
    test_preds = (test_calib_probs >= best_th).astype(int)
    
    acc = accuracy_score(y_test_delay, test_preds) * 100.0
    prec = precision_score(y_test_delay, test_preds, zero_division=0) * 100.0
    rec = recall_score(y_test_delay, test_preds, zero_division=0) * 100.0
    f1 = f1_score(y_test_delay, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test_delay, test_calib_probs)
    brier = brier_score_loss(y_test_delay, test_calib_probs)
    cm = confusion_matrix(y_test_delay, test_preds)
    
    p50_preds = p50.predict(X_test)
    mae_rem = mean_absolute_error(y_test_rem, p50_preds)
    medae_rem = float(np.median(np.abs(y_test_rem - p50_preds)))
    
    metadata = {
        'training_date': datetime.now().isoformat(),
        'feature_columns': feature_cols,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'best_threshold': best_th,
        'benchmark_results': {
            'classification': {
                'accuracy_pct': round(acc, 2),
                'precision_pct': round(prec, 2),
                'recall_pct': round(rec, 2),
                'f1_score': round(f1, 4),
                'roc_auc': round(roc_auc, 4),
                'brier_score': round(brier, 4),
                'true_on_time': int(cm[0, 0]),
                'false_delayed': int(cm[0, 1]),
                'false_on_time': int(cm[1, 0]),
                'true_delayed': int(cm[1, 1])
            },
            'eta_regression': {
                'mae_hours': round(mae_rem, 2),
                'medae_hours': round(medae_rem, 2)
            }
        }
    }
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"All models successfully trained and exported to: {models_dir}")
    print(f"Validation-tuned Decision Threshold: {best_th}")
    print(f"Independent Test Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    return metadata
