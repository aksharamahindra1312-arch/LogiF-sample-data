"""
Probability calibration using Isotonic Regression.
"""

from sklearn.isotonic import IsotonicRegression
import numpy as np

def fit_probability_calibrator(val_raw_probs: np.ndarray, val_true_labels: np.ndarray) -> IsotonicRegression:
    """Fits an isotonic calibrator strictly on out-of-fold validation predictions."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_raw_probs, val_true_labels)
    return calibrator
