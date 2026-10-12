"""
Delay classification model definition and training.
"""

import lightgbm as lgb
from typing import Dict, Any

def build_delay_classifier(params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
    """Initializes the LightGBM delay classifier."""
    default_params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 300,
        'learning_rate': 0.04,
        'num_leaves': 45,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    if params:
        default_params.update(params)
    return lgb.LGBMClassifier(**default_params)
