"""
CLI script to validate incoming LF datasets against the required 26-column input schema.
"""

import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_validation import validate_input_schema

def main():
    parser = argparse.ArgumentParser(description="Logi Freight — Dataset Schema Validator")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV dataset to validate")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return
        
    df = pd.read_csv(args.input, nrows=50)
    is_valid, missing = validate_input_schema(df)
    
    print("=" * 80)
    print("DATASET SCHEMA VALIDATION")
    print("=" * 80)
    print(f"Target File: {args.input}")
    print(f"Valid Schema: {is_valid}")
    if not is_valid:
        print(f"Missing Required Columns: {missing}")
    else:
        print("All required 26 LF columns are present and valid.")
    print("=" * 80)

if __name__ == "__main__":
    main()
