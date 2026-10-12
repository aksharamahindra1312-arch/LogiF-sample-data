"""
CLI tool for running live predictions on incoming shipment snapshot files or datasets.
"""

import os
import sys
import json
import argparse
import pandas as pd

# Set path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import LogiFreightInferenceService
from src.pipeline import process_live_dataset

def main():
    parser = argparse.ArgumentParser(description="Logi Freight — Live In-Transit Inference Engine")
    parser.add_argument("--input", type=str, default="examples/sample_input.json", help="Path to input snapshot JSON or full CSV dataset")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output predictions CSV or JSON")
    parser.add_argument("--as_of", type=str, default=None, help="Evaluation timestamp ISO string (defaults to now)")
    parser.add_argument("--tomtom_key", type=str, default=None, help="Optional TomTom Traffic Flow API key")
    
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "models")
    
    print("=" * 80)
    print("LOGI FREIGHT — LIVE IN-TRANSIT PREDICTION SERVICE")
    print("=" * 80)
    
    if args.input.endswith(".csv"):
        print(f"Processing batch dataset: {args.input}")
        df_preds = process_live_dataset(args.input, models_dir, output_csv_path=args.output)
        print(f"Generated {len(df_preds)} predictions -> Saved to {args.output}")
    else:
        # Single JSON snapshot
        engine = LogiFreightInferenceService(models_dir)
        with open(args.input, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)
            
        result = engine.predict(snapshot_data, as_of_iso=args.as_of, tomtom_key=args.tomtom_key)
        print(json.dumps(result, indent=2))
        
        # Save output JSON if requested
        if args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as out_f:
                json.dump(result, out_f, indent=2)
        else:
            pd.DataFrame([result]).to_csv(args.output, index=False)
            print(f"Saved prediction to: {args.output}")

if __name__ == "__main__":
    main()
