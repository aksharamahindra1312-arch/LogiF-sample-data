"""
Data preprocessing and cleaning pipeline for raw logistics event streams.
Separates physical movement events from administrative paperwork records.
"""

import pandas as pd
import numpy as np
from src.geo_resolver import resolve_location

ADMIN_KEYWORDS = [
    'draft', 'modified', 'changed', 'invoice', 'dms trigger',
    'eway bill', 'references', 'weight', 'emailed', 'lr content'
]

def clean_event_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans, parses timestamps, categorizes event types, and sanitizes GPS coordinates."""
    df_clean = df.copy()
    
    # Handle Column Aliases
    if 'Customer Name' in df_clean.columns and 'Shipper' not in df_clean.columns:
        df_clean['Shipper'] = df_clean['Customer Name']
    if 'BA NAME' in df_clean.columns and 'BA_Name' not in df_clean.columns:
        df_clean['BA_Name'] = df_clean['BA NAME']
    if 'Service' in df_clean.columns and 'Service_Option' not in df_clean.columns:
        df_clean['Service_Option'] = df_clean['Service']
    if 'service name' in df_clean.columns and 'service_name' not in df_clean.columns:
        df_clean['service_name'] = df_clean['service name']
    if 'Event_Source' not in df_clean.columns:
        df_clean['Event_Source'] = 'GPS'
        
    # Timestamp parsing
    for ts_col in ['event_created_at', 'dispatch_date', 'delivery_date', 'original_eta', 'Consignment_created_at', 'Trip_Created_at']:
        if ts_col in df_clean.columns:
            df_clean[ts_col + '_dt'] = pd.to_datetime(df_clean[ts_col], errors='coerce')
            
    # GPS validation & numeric conversion
    for coord in ['lat', 'lng', 'Latitude', 'Longitude']:
        if coord in df_clean.columns:
            df_clean[coord] = pd.to_numeric(df_clean[coord], errors='coerce')
            
    # Classify Physical vs Administrative Events
    desc_series = df_clean['description'].astype(str).str.lower() if 'description' in df_clean.columns else pd.Series('', index=df_clean.index)
    is_admin_desc = desc_series.apply(lambda d: any(k in d for k in ADMIN_KEYWORDS))
    
    event_type_series = df_clean['event_type'].astype(str) if 'event_type' in df_clean.columns else pd.Series('IN_TRANSIT', index=df_clean.index)
    df_clean['is_administrative'] = event_type_series.isin(['GENERAL', 'CREATED']) | is_admin_desc
    df_clean['is_physical'] = event_type_series.isin(['IN_TRANSIT', 'DISPATCHED', 'VEHICLE_REACHED', 'VEHICLE_LEFT', 'DELIVERED', 'REACHED_DESTINATION']) & (~df_clean['is_administrative'])
    
    # Drop rows without consignment identifier
    if 'Consignment No' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Consignment No'])
        df_clean['Consignment No'] = df_clean['Consignment No'].astype(str)
    elif 'Consignment_Number' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Consignment_Number'])
        df_clean['Consignment_Number'] = df_clean['Consignment_Number'].astype(str)
        df_clean['Consignment No'] = df_clean['Consignment_Number']
        
    # Chronological sort
    if 'event_created_at_dt' in df_clean.columns:
        df_clean = df_clean.sort_values(['Consignment No', 'event_created_at_dt'])
        
    return df_clean
