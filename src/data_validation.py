"""
Schema validation and data sanitization for Logi Freight input streams.
"""

import pandas as pd
from typing import Tuple, List

REQUIRED_INPUT_COLUMNS = [
    'Consignment No', 'Consignment_created_at', 'Trip_number', 'Trip_Created_at',
    'dispatch_date', 'delivery_date', 'original_eta', 'Consignment Status',
    'consignee_city', 'consignor_city', 'consignee_state', 'consignor_state',
    'Service_Option', 'Vehicle Number', 'Shipper', 'Vehicle Type', 'BA_Name',
    'Event_Source', 'event_type', 'eventable_type', 'description',
    'event_created_at', 'event_location', 'lat', 'lng', 'service_name'
]

def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that incoming DataFrame satisfies the LF input contract schema."""
    missing = []
    # Support Customer Name alias for Shipper
    cols = df.columns.tolist()
    if 'Customer Name' in cols and 'Shipper' not in cols:
        df['Shipper'] = df['Customer Name']
        cols = df.columns.tolist()
        
    for req in REQUIRED_INPUT_COLUMNS:
        if req not in cols:
            missing.append(req)
            
    is_valid = len(missing) == 0
    return is_valid, missing
