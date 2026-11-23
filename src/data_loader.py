"""
MIZU: Data Ingestion & Schema Management Module
===============================================
Author: MIZU Team
Purpose: Loads the raw UCI Water Treatment Plant dataset, enforces
         clean column schemas, parses date identifiers, and provides
         feature grouping for downstream analytics.

Why this file exists:
In real-world data science, we decouple data ingestion from data analysis.
If the raw data format, path, or file encoding changes in the future,
we only need to update this single file without breaking the rest of our pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# 1. Canonical Column Names (39 columns: 1 Date + 38 physical/chemical sensors)
COLUMN_NAMES: List[str] = [
    "Date",
    # Stage 1: Plant Influent / Input (-E)
    "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E",
    # Stage 2: Primary Settler (-P)
    "PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P",
    # Stage 3: Secondary Settler / Aeration Basin (-D)
    "PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D",
    # Stage 4: Plant Effluent / Discharged Water (-S)
    "PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S",
    # Stage 5: Plant Removal Efficiencies (RD-)
    "RD-DBO-P", "RD-SS-P", "RD-SED-P",
    "RD-DBO-S", "RD-DQO-S",
    "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G"
]


def get_column_names() -> List[str]:
    """Return a copy of the canonical 39 column names."""
    return list(COLUMN_NAMES)


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Categorize columns by physical treatment stage.
    
    Returns:
        dict: A dictionary mapping stage name to its list of feature names.
    """
    return {
        "influent": ["Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "SSV-E", "SED-E", "COND-E"],
        "primary": ["PH-P", "DBO-P", "SS-P", "SSV-P", "SED-P", "COND-P"],
        "secondary": ["PH-D", "DBO-D", "DQO-D", "SS-D", "SSV-D", "SED-D", "COND-D"],
        "effluent": ["PH-S", "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S", "COND-S"],
        "efficiency": [
            "RD-DBO-P", "RD-SS-P", "RD-SED-P",
            "RD-DBO-S", "RD-DQO-S",
            "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G"
        ]
    }


def load_water_treatment_data(
    filepath: Optional[str] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load the raw Water Treatment Plant dataset from disk.

    Steps performed:
    1. Locates the raw file at `data/raw/water-treatment.data` if no path is given.
    2. Maps missing value indicators ('?') to NumPy NaN.
    3. Assigns clear column names to all 39 variables.
    4. Drops completely blank/empty rows.
    5. Converts the 'Date' column into standard datetime objects if parse_dates=True.

    Args:
        filepath: Optional path to the raw data file.
        parse_dates: Whether to parse the 'Date' string into a datetime object.

    Returns:
        pd.DataFrame: Cleaned raw dataframe with 527 rows and 39 columns.
    """
    if filepath is None:
        # Default relative path from project root
        filepath = Path("data/raw/water-treatment.data")
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Please ensure data/raw/water-treatment.data exists by running scripts/download_data.py."
        )

    # Read CSV with '?' mapped to NaN
    df = pd.read_csv(
        filepath,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
        header=None
    )

    # Drop any completely empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Convert all 38 sensor columns to numeric float64
    numeric_cols = [c for c in COLUMN_NAMES if c != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse Date string (e.g., 'D-1/3/90' -> datetime 1990-03-01)
    if parse_dates:
        cleaned_dates = df["Date"].astype(str).str.replace("D-", "", regex=False).str.strip()
        df["Date_Parsed"] = pd.to_datetime(cleaned_dates, format="%d/%m/%y", errors="coerce")

    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean summary table of missing values across all columns.

    Args:
        df: The water treatment dataframe.

    Returns:
        pd.DataFrame: Table with columns ['column', 'missing_count', 'missing_pct']
                      sorted in descending order by missing count.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values
    })
    
    return summary.sort_values(by="missing_count", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Testing MIZU Data Loader...")
    data = load_water_treatment_data()
    print(f"Successfully loaded dataset with shape: {data.shape}")
    print(f"Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")
    missing = get_missing_summary(data)
    print("\nTop 5 columns with missing values:")
    print(missing.head(5).to_string(index=False))

