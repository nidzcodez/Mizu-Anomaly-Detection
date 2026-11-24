"""
Generate test fixtures for MIZU Dashboard CSV upload validation.
"""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.data_loader import load_water_treatment_data

def main():
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw dataset
    df = load_water_treatment_data()
    # Drop Date_Parsed if present so it mirrors raw upload format
    if "Date_Parsed" in df.columns:
        df = df.drop(columns=["Date_Parsed"])
        
    print(f"Loaded raw data: {df.shape}")
    
    # 1. Valid sample (10 rows with all 38 continuous features + Date)
    valid_sample = df.head(10).copy()
    valid_path = fixtures_dir / "valid_sample.csv"
    valid_sample.to_csv(valid_path, index=False)
    print(f"Created {valid_path} with shape {valid_sample.shape}")
    
    # 2. Missing columns (drop 2 features: ZN-E and SED-S)
    missing_cols_df = valid_sample.drop(columns=["ZN-E", "SED-S"]).copy()
    missing_path = fixtures_dir / "missing_columns.csv"
    missing_cols_df.to_csv(missing_path, index=False)
    print(f"Created {missing_path} with shape {missing_cols_df.shape} (dropped ZN-E, SED-S)")
    
    # 3. With NaNs (10 rows where some values are NaN)
    with_nans_df = valid_sample.copy()
    # Artificially inject additional NaNs across several columns
    with_nans_df.loc[0, "Q-E"] = np.nan
    with_nans_df.loc[2, "PH-P"] = np.nan
    with_nans_df.loc[5, "RD-DBO-G"] = np.nan
    nans_path = fixtures_dir / "with_nans.csv"
    with_nans_df.to_csv(nans_path, index=False)
    print(f"Created {nans_path} with shape {with_nans_df.shape} (contains NaNs)")
    
    # 4. Single row (1 row with 38 features + Date, specifically D-13/3/90 which is consensus anomaly)
    anomaly_row = df[df["Date"] == "D-13/3/90"]
    if anomaly_row.empty:
        single_row = df.iloc[[10]].copy()
    else:
        single_row = anomaly_row.copy()
    single_path = fixtures_dir / "single_row.csv"
    single_row.to_csv(single_path, index=False)
    print(f"Created {single_path} with shape {single_row.shape}")

if __name__ == "__main__":
    main()
