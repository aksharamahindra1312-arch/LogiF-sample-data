"""
Mathematical definitions and ground-truth target calculations for supervised learning.
"""

from datetime import datetime
from typing import Dict, Any

def calculate_ground_truth_targets(dispatch_dt: datetime, delivery_dt: datetime, original_eta_dt: datetime, as_of_dt: datetime) -> Dict[str, Any]:
    """Calculates supervised future-outcome targets strictly using post-As_Of delivery events."""
    rem_hours = max(0.0, (delivery_dt - as_of_dt).total_seconds() / 3600.0)
    is_delayed = bool(delivery_dt > original_eta_dt)
    delay_hours = max(0.0, (delivery_dt - original_eta_dt).total_seconds() / 3600.0)
    
    p10_target = round(rem_hours * 0.82, 2)
    p50_target = round(rem_hours, 2)
    p90_target = round(rem_hours * 1.28 + (delay_hours * 0.5), 2)
    
    if not is_delayed:
        status = 'ON_TIME'
    elif delay_hours <= 8.0:
        status = 'DELAYED'
    else:
        status = 'SEVERELY_DELAYED'
        
    return {
        'Target_Remaining_Hours': round(rem_hours, 2),
        'Target_Actual_Delivery_Date': delivery_dt.isoformat(),
        'Target_Is_Delayed': int(is_delayed),
        'Target_Delay_Hours': round(delay_hours, 2),
        'Target_P10_Hours': p10_target,
        'Target_P50_Hours': p50_target,
        'Target_P90_Hours': p90_target,
        'Target_Predicted_Status': status
    }
