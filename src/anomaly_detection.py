"""
Unsupervised shipment movement anomaly detector.
"""

from sklearn.ensemble import IsolationForest

def build_anomaly_detector() -> IsolationForest:
    """Initializes the movement anomaly detector."""
    return IsolationForest(
        n_estimators=150,
        contamination=0.04,
        random_state=42,
        n_jobs=-1
    )
