"""
Strict point-in-time data leakage prevention and automated audit checks.
"""

import pandas as pd
from datetime import datetime

class LeakageViolationError(Exception):
    """Raised when future information leaks into the feature matrix."""
    pass

def verify_point_in_time_safety(past_events_df: pd.DataFrame, as_of_dt: datetime):
    """Verifies that no event timestamp in the feature dataset exceeds As_Of."""
    if 'event_created_at_dt' in past_events_df.columns:
        future_mask = past_events_df['event_created_at_dt'] > as_of_dt
        if future_mask.any():
            violating_count = future_mask.sum()
            raise LeakageViolationError(f"CRITICAL LEAKAGE: {violating_count} events found with timestamp > As_Of ({as_of_dt})")

def verify_feature_matrix_isolation(feature_df: pd.DataFrame):
    """Verifies that delivery_date, terminal status, or target labels do not exist in X."""
    forbidden = [
        'delivery_date', 'Target_Remaining_Hours', 'Target_Is_Delayed', 
        'Target_Delay_Hours', 'Target_Actual_Delivery_Date', 'Target_Predicted_Status'
    ]
    leaked = [f for f in forbidden if f in feature_df.columns]
    if leaked:
        raise LeakageViolationError(f"CRITICAL LEAKAGE: Feature matrix contains forbidden target columns: {leaked}")
