"""
Script to fetch the UCI Water Treatment Plant dataset (ID: 106)
and save raw data and metadata locally.
"""
import os
import json
import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    print("Fetching Water Treatment Plant dataset from UCI repository (ID: 106)...")
    try:
        dataset = fetch_ucirepo(id=106)
    except Exception as e:
        print(f"ucimlrepo fetch failed: {e}. Falling back to direct URL...")
        # Direct fallback if ucimlrepo fails
        import urllib.request
        url_data = "https://archive.ics.uci.edu/static/public/106/water+treatment+plant.zip"
        # We will handle if needed
        raise e

    # Extract features and targets/metadata
    X = dataset.data.features
    y = dataset.data.targets
    metadata = dataset.metadata
    variables = dataset.variables

    os.makedirs("data/raw", exist_ok=True)

    print("\n--- DATASET METADATA ---")
    print(f"Name: {metadata.get('name')}")
    print(f"Number of instances: {metadata.get('num_instances')}")
    print(f"Number of features: {metadata.get('num_features')}")
    print(f"Has missing values: {metadata.get('has_missing_values')}")
    
    # Save combined raw dataframe
    df_raw = dataset.data.original
    raw_path = "data/raw/water-treatment.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path} with shape {df_raw.shape}")

    # Save metadata as json
    meta_dict = {
        "name": metadata.get("name"),
        "abstract": metadata.get("abstract"),
        "num_instances": metadata.get("num_instances"),
        "num_features": metadata.get("num_features"),
        "has_missing_values": metadata.get("has_missing_values"),
        "intro_paper": metadata.get("intro_paper"),
    }
    with open("data/raw/metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=2)

    # Save variable descriptions as csv
    if variables is not None:
        variables.to_csv("data/raw/variables.csv", index=False)
        print("Variables metadata saved to data/raw/variables.csv")

    # Detailed inspection
    print("\n--- HEAD OF RAW DATA ---")
    print(df_raw.head())

    print("\n--- COLUMN DATA TYPES ---")
    print(df_raw.dtypes)

    print("\n--- MISSING VALUE ANALYSIS ---")
    # In UCI raw data, '?' is often used for missing values, or already converted to NaN
    # Let's inspect string '?' occurrences vs np.nan
    qmark_counts = (df_raw == '?').sum()
    nan_counts = df_raw.isna().sum()
    missing_summary = pd.DataFrame({
        "data_type": df_raw.dtypes,
        "nan_count": nan_counts,
        "question_mark_count": qmark_counts,
        "total_missing": nan_counts + qmark_counts,
        "pct_missing": ((nan_counts + qmark_counts) / len(df_raw) * 100).round(2)
    })
    print(missing_summary[missing_summary["total_missing"] > 0])

    print("\nFetch and inspection completed successfully.")

if __name__ == "__main__":
    main()

