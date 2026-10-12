"""
CLI script to train the Logi Freight multi-task model ensemble from processed datasets.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import clean_event_stream
from src.snapshot_generator import generate_consignment_snapshots
from src.train import train_and_export_models

def main():
    parser = argparse.ArgumentParser(description="Logi Freight — Model Training Workflow")
    parser.add_argument("--data_path", type=str, default=r"C:\Users\raksh\.gemini\antigravity\scratch\LogiF-sample-data\Akshare_3Months_Events.parquet", help="Path to raw or processed dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save trained model artifacts")
    args = parser.parse_args()
    
    print("=" * 80)
    print("LOGI FREIGHT — PRODUCTION MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Data Source:      {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Ingest and preprocess event streams
    print("Cleaning event stream and isolating physical transit telemetry...")
    df_raw = pd.read_parquet(args.data_path) if args.data_path.endswith('.parquet') else pd.read_csv(args.data_path)
    df_clean = clean_event_stream(df_raw)
    
    # Group by consignment to get metadata & chronological order
    c_meta = []
    grouped = df_clean.groupby('Consignment No')
    for c_no, c_events in grouped:
        first_row = c_events.iloc[0]
        disp_dt = first_row.get('dispatch_date_dt')
        deliv_dt = first_row.get('delivery_date_dt')
        eta_dt = first_row.get('original_eta_dt')
        if pd.isna(disp_dt) or pd.isna(deliv_dt) or pd.isna(eta_dt) or deliv_dt <= disp_dt:
            continue
            
        c_meta.append({
            'Consignment_Number': str(c_no),
            'disp_dt': disp_dt,
            'deliv_dt': deliv_dt,
            'eta_dt': eta_dt,
            'actual_delayed': int(deliv_dt > eta_dt),
            'orig_city': str(first_row.get('consignor_city', '')),
            'dest_city': str(first_row.get('consignee_city', '')),
            'shipper': str(first_row.get('Shipper', first_row.get('Customer Name', 'UNKNOWN'))),
            'ba': str(first_row.get('BA_Name', first_row.get('BA NAME', 'UNKNOWN')))
        })
        
    meta_df = pd.DataFrame(c_meta).sort_values('disp_dt').reset_index(drop=True)
    n_consignments = len(meta_df)
    n_train_c = int(n_consignments * 0.70)
    n_val_c = int(n_consignments * 0.15)
    
    train_c_ids = set(meta_df.iloc[:n_train_c]['Consignment_Number'])
    val_c_ids = set(meta_df.iloc[n_train_c:n_train_c+n_val_c]['Consignment_Number'])
    test_c_ids = set(meta_df.iloc[n_train_c+n_val_c:]['Consignment_Number'])
    
    print(f"Total Consignments: {n_consignments:,}")
    print(f"  Train: {len(train_c_ids):,} | Val: {len(val_c_ids):,} | Test: {len(test_c_ids):,}")
    
    # 2. Compute empirical Bayesian delay priors strictly from train fold
    train_meta = meta_df.iloc[:n_train_c]
    global_delay_prior = float(train_meta['actual_delayed'].mean())
    
    def compute_smoothed_priors(series, targets, m=5):
        stats = pd.DataFrame({'cat': series, 'target': targets}).groupby('cat')['target'].agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + m * global_delay_prior) / (stats['count'] + m)
        return smoothed.to_dict()
        
    shipper_priors = compute_smoothed_priors(train_meta['shipper'], train_meta['actual_delayed'], m=5)
    ba_priors = compute_smoothed_priors(train_meta['ba'], train_meta['actual_delayed'], m=5)
    corridor_priors = compute_smoothed_priors(train_meta['orig_city'] + '->' + train_meta['dest_city'], train_meta['actual_delayed'], m=3)
    
    priors_path = os.path.join(args.output_dir, 'historical_priors.json')
    with open(priors_path, 'w', encoding='utf-8') as f:
        json.dump({
            'global_delay_prior': global_delay_prior,
            'shipper_priors': shipper_priors,
            'ba_priors': ba_priors,
            'corridor_priors': corridor_priors
        }, f, indent=2)
    print(f"Exported empirical delay priors to: {priors_path}")
    
    # 3. Generate point-in-time multi-checkpoint lifecycle snapshots
    print("Generating point-in-time multi-checkpoint lifecycle snapshots...")
    all_snaps = []
    for c_no, c_events in grouped:
        snaps = generate_consignment_snapshots(c_events, snapshot_ratios=[0.20, 0.40, 0.60, 0.80, 0.92])
        all_snaps.extend(snaps)
        
    df_snaps = pd.DataFrame(all_snaps)
    train_df = df_snaps[df_snaps['Consignment_Number'].isin(train_c_ids)].copy()
    val_df = df_snaps[df_snaps['Consignment_Number'].isin(val_c_ids)].copy()
    test_df = df_snaps[df_snaps['Consignment_Number'].isin(test_c_ids)].copy()
    
    print(f"Total Snapshots: {len(df_snaps):,}")
    print(f"  Train Set      : {len(train_df):,} snapshots")
    print(f"  Validation Set : {len(val_df):,} snapshots")
    print(f"  Holdout Test   : {len(test_df):,} snapshots")
    
    # 4. Feature Definitions
    feature_cols = [
        'Stage', 'Route_Type', 'Shipper', 'BA_Name', 'Origin_City', 'Origin_State',
        'Dest_City', 'Dest_State', 'Vehicle_Type', 'Event_Source', 'Eventable_Type',
        'Traffic_Severity', 'Civic_Disruption_Severity', 'Journey_Stage_Group',
        'Latitude', 'Longitude', 'Has_GPS_Coordinates', 'Elapsed_Hours', 'Planned_Hours',
        'Elapsed_Frac', 'Remaining_KM', 'Traversed_KM', 'Total_Route_KM',
        'Corridor_Planned_Speed_Kmph', 'Route_Progress_Ratio', 'Required_Speed_Kmph',
        'Effective_Speed_Kmph', 'Behind_Pace', 'Origin_Dwell_Hours', 'Buffer_Hours',
        'Proximity_Buffer_Flag', 'Pings_Last_3H', 'Pings_Last_6H', 'Telemetry_Freshness_Min',
        'Hour_Of_Day', 'Day_Of_Week', 'Month', 'BA_Historical_Delay_Rate',
        'BA_Historical_Trips', 'Shipper_Historical_Delay_Rate', 'Corridor_Baseline_Hours',
        'Corridor_Historical_Delay_Rate', 'Weather_Rain_MM', 'Weather_Temp_C',
        'Weather_Wind_Kmph', 'Weather_Severe_Flag', 'Weather_Fog_Flag',
        'Traffic_Congestion_Ratio', 'Civic_Disruption_Alert'
    ]
    categorical_cols = [
        'Stage', 'Route_Type', 'Shipper', 'BA_Name', 'Origin_City', 'Origin_State',
        'Dest_City', 'Dest_State', 'Vehicle_Type', 'Event_Source', 'Eventable_Type',
        'Traffic_Severity', 'Civic_Disruption_Severity', 'Journey_Stage_Group'
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    
    # 5. Train and Export Models
    metadata = train_and_export_models(
        train_df, val_df, test_df,
        feature_cols, categorical_cols, numeric_cols,
        args.output_dir
    )
    
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
