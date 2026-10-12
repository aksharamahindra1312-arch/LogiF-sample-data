"""
End-to-end batch processing pipeline for historical and live logistics datasets.
"""

import pandas as pd
from datetime import datetime, timedelta
from src.data_preprocessing import clean_event_stream
from src.inference import LogiFreightInferenceService

def process_live_dataset(input_data_path: str, models_dir: str, output_csv_path: str = None) -> pd.DataFrame:
    """Processes an incoming LF dataset and generates the 22-output contract for all shipments."""
    engine = LogiFreightInferenceService(models_dir)
    if input_data_path.endswith('.parquet'):
        df_raw = pd.read_parquet(input_data_path)
    else:
        df_raw = pd.read_csv(input_data_path)
        
    df_clean = clean_event_stream(df_raw)
    
    grouped = df_clean.groupby('Consignment No')
    results = []
    
    for c_id, c_events in grouped:
        # Separate physical movement events from administrative records
        non_deliv_events = c_events[~c_events['event_type'].isin(['DELIVERED', 'REACHED_DESTINATION'])]
        phys_events = non_deliv_events[non_deliv_events['is_physical']]
        
        if len(phys_events) > 0:
            snapshot_events = phys_events
        elif len(non_deliv_events) > 0:
            snapshot_events = non_deliv_events
        else:
            snapshot_events = c_events
            
        snapshot_row = snapshot_events.iloc[-1]
        latest = snapshot_row.to_dict()
        as_of_dt = snapshot_row.get('event_created_at_dt')
        if pd.isna(as_of_dt) or as_of_dt is None:
            as_of_dt = datetime.now()
            
        # Strictly hide ground truth future delivery date
        latest.pop('delivery_date', None)
        latest.pop('delivery_date_dt', None)
        latest['Consignment Status'] = 'in_transit'
        
        pred = engine.predict(latest, as_of_iso=as_of_dt.isoformat())
        results.append(pred)
        
    df_out = pd.DataFrame(results)
    if output_csv_path:
        df_out.to_csv(output_csv_path, index=False)
    return df_out
