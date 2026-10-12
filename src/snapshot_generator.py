"""
Point-in-time historical live snapshot generator for event-stream logistics data.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from src.leakage_prevention import verify_point_in_time_safety
from src.feature_engineering import extract_point_in_time_features
from src.target_definition import calculate_ground_truth_targets

def generate_consignment_snapshots(consignment_events_df: pd.DataFrame, snapshot_ratios: List[float] = [0.20, 0.40, 0.60, 0.80, 0.92]) -> List[Dict[str, Any]]:
    """
    Generates point-in-time in-transit snapshots for a single consignment lifecycle.
    Only events occurring strictly <= As_Of are used for feature construction.
    """
    df = consignment_events_df.sort_values('event_created_at_dt')
    if len(df) == 0:
        return []
        
    disp_dt = df['dispatch_date_dt'].dropna().iloc[0] if len(df['dispatch_date_dt'].dropna()) > 0 else df['event_created_at_dt'].min()
    del_dt = df['delivery_date_dt'].dropna().iloc[-1] if len(df['delivery_date_dt'].dropna()) > 0 else None
    eta_dt = df['original_eta_dt'].dropna().iloc[0] if len(df['original_eta_dt'].dropna()) > 0 else None
    
    if pd.isna(disp_dt) or del_dt is None or pd.isna(del_dt) or del_dt <= disp_dt:
        return []
        
    dur_sec = max(300, (del_dt - disp_dt).total_seconds())
    snapshots = []
    
    first_row = df.iloc[0]
    c_num = str(first_row.get('Consignment No', first_row.get('Consignment_Number', 'UNKNOWN')))
    shipper = str(first_row.get('Shipper', first_row.get('Customer Name', 'UNKNOWN_SHIPPER')))
    ba = str(first_row.get('BA_Name', first_row.get('BA NAME', 'UNKNOWN_BA')))
    orig_city = str(first_row.get('consignor_city', ''))
    dest_city = str(first_row.get('consignee_city', ''))
    orig_state = str(first_row.get('consignor_state', 'UNKNOWN'))
    dest_state = str(first_row.get('consignee_state', 'UNKNOWN'))
    veh_type = str(first_row.get('Vehicle Type', 'UNKNOWN'))
    
    for r in snapshot_ratios:
        as_of_dt = disp_dt + timedelta(seconds=max(60, int(dur_sec * r)))
        if as_of_dt >= del_dt:
            as_of_dt = del_dt - timedelta(minutes=2)
            if as_of_dt <= disp_dt:
                continue
            
        past_events = df[df['event_created_at_dt'] <= as_of_dt]
        if len(past_events) == 0:
            # Construct origin manifest event valid at As_Of (zero future leakage, blind telemetry)
            manifest_event = {
                'Consignment No': c_num,
                'Shipper': shipper,
                'BA_Name': ba,
                'consignor_city': orig_city,
                'consignee_city': dest_city,
                'consignor_state': orig_state,
                'consignee_state': dest_state,
                'Vehicle Type': veh_type,
                'event_type': 'DISPATCHED',
                'Event_Source': 'Manifest',
                'event_created_at_dt': as_of_dt,
                'lat': None,
                'lng': None
            }
            past_events = pd.DataFrame([manifest_event])
            
        # Point-in-time safety check (Strictly <= As_Of)
        verify_point_in_time_safety(past_events, as_of_dt)
        
        latest_event = past_events.iloc[-1].to_dict()
        features = extract_point_in_time_features(latest_event, as_of_dt, disp_dt, eta_dt if pd.notna(eta_dt) else as_of_dt + timedelta(hours=24))
        targets = calculate_ground_truth_targets(disp_dt, del_dt, eta_dt if pd.notna(eta_dt) else as_of_dt + timedelta(hours=24), as_of_dt)
        
        snapshots.append({
            'Consignment_Number': c_num,
            'As_Of': as_of_dt.isoformat(),
            **features,
            **targets
        })
        
    return snapshots
