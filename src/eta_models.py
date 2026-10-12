"""
Quantile ETA regression models (P10, P50, P90).
"""

import lightgbm as lgb
from typing import Dict, Any

def build_quantile_regressor(alpha: float, params: Dict[str, Any] = None) -> lgb.LGBMRegressor:
    """Initializes a LightGBM quantile pinball loss regressor."""
    default_params = {
        'objective': 'quantile',
        'alpha': alpha,
        'n_estimators': 300,
        'learning_rate': 0.04,
        'num_leaves': 40,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)
