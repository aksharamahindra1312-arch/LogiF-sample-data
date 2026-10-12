"""
Model validation and decision threshold optimization.
"""

import numpy as np
from sklearn.metrics import f1_score

def optimize_decision_threshold(val_calib_probs: np.ndarray, y_val: np.ndarray) -> float:
    """Finds the optimal F1 decision threshold on validation data."""
    best_t = 0.50
    best_f1 = 0.0
    for t in np.arange(0.20, 0.80, 0.02):
        score = f1_score(y_val, (val_calib_probs >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 2)
