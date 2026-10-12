"""
Model evaluation and comprehensive benchmarking metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss,
    mean_absolute_error, median_absolute_error, mean_squared_error
)

def evaluate_classification_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.42) -> dict:
    """Calculates classification performance metrics."""
    preds = (y_probs >= threshold).astype(int)
    return {
        'accuracy': round(accuracy_score(y_true, preds) * 100, 2),
        'precision': round(precision_score(y_true, preds, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_true, preds, zero_division=0) * 100, 2),
        'f1': round(f1_score(y_true, preds, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_true, y_probs), 4) if len(np.unique(y_true)) > 1 else None,
        'pr_auc': round(average_precision_score(y_true, y_probs), 4) if len(np.unique(y_true)) > 1 else None,
        'brier': round(brier_score_loss(y_true, y_probs), 4),
        'confusion_matrix': confusion_matrix(y_true, preds).tolist()
    }

def evaluate_eta_metrics(y_true_rem: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict:
    """Calculates ETA quantile regression performance metrics."""
    return {
        'mae': round(mean_absolute_error(y_true_rem, p50), 2),
        'medae': round(median_absolute_error(y_true_rem, p50), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_true_rem, p50)), 2),
        'p10_coverage': round(np.mean(y_true_rem >= p10) * 100, 2),
        'p90_coverage': round(np.mean(y_true_rem <= p90) * 100, 2),
        'monotonicity_pct': round(np.mean((p10 <= p50) & (p50 <= p90)) * 100, 2)
    }
