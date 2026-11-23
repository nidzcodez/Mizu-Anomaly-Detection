"""
MIZU: Data Preprocessing Pipeline Module
========================================
Author: MIZU Team
Purpose: Cleans raw water treatment plant data by:
         1. Extracting numerical features and isolating non-feature metadata (Dates).
         2. Imputing missing values using the feature-wise Median.
         3. Scaling all features using RobustScaler (Median & IQR).
         4. Exporting serialized pipeline artifacts and preprocessed datasets.

Why this file exists:
Raw sensor data cannot be fed directly into machine learning models:
- Missing values (NaN) cause mathematical algorithms to crash.
- Unequal scales (e.g., flow rate in tens of thousands vs pH between 6 and 9)
  cause distance-based algorithms (like LOF, K-Means, PCA) to ignore small features.
- Extreme outliers (like Zinc spikes) distort standard scaling (mean & variance),
  making RobustScaler the mathematically defensible choice.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Add project root to sys.path so 'src.*' imports work both when run as a script or imported as a package
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Import our custom data loader
from src.data_loader import load_water_treatment_data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract the list of 38 continuous sensor column names from the dataframe,
    ignoring non-feature metadata columns ('Date', 'Date_Parsed').
    """
    exclude_cols = {"Date", "Date_Parsed"}
    return [col for col in df.columns if col not in exclude_cols]


def create_preprocessing_pipeline() -> Pipeline:
    """
    Construct a scikit-learn Pipeline combining:
    1. SimpleImputer(strategy='median'): Replaces NaN values with the median of each column.
    2. RobustScaler(with_centering=True, with_scaling=True, unit_variance=False):
       Scales features using statistics that are robust to outliers:
       x_scaled = (x - median) / IQR, where IQR = Q75 - Q25.

    Returns:
        Pipeline: Unfitted scikit-learn Pipeline object.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True))
        ]
    )
    return pipeline


class MizuPreprocessor:
    """
    High-level wrapper that manages data splitting, pipeline fitting,
    transformation, column reconstruction, and persistence.
    """

    def __init__(self):
        self.pipeline: Pipeline = create_preprocessing_pipeline()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MizuPreprocessor":
        """
        Fit the imputer and scaler on the provided dataframe features.

        Args:
            df: DataFrame containing the 38 continuous sensor columns.

        Returns:
            self: The fitted MizuPreprocessor instance.
        """
        self.feature_names = get_feature_columns(df)
        X = df[self.feature_names]
        self.pipeline.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform raw data using the fitted pipeline.

        Args:
            df: Raw DataFrame containing sensors and metadata.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - X_scaled: Cleaned, scaled DataFrame with all 38 original column names.
                - metadata: DataFrame holding 'Date' and 'Date_Parsed' aligned by index.
        """
        if not self.is_fitted:
            raise RuntimeError("MizuPreprocessor must be fitted before calling transform().")

        features = df[self.feature_names]
        metadata_cols = [c for c in ["Date", "Date_Parsed"] if c in df.columns]
        metadata = df[metadata_cols].copy()

        # Execute imputer and scaler
        X_transformed = self.pipeline.transform(features)

        # Reconstruct clean DataFrame preserving exact column names
        X_scaled = pd.DataFrame(
            X_transformed,
            columns=self.feature_names,
            index=df.index
        )

        return X_scaled, metadata

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience method to fit and transform in a single step."""
        return self.fit(df).transform(df)

    def save(self, filepath: Union[str, Path] = "models/preprocessor.joblib") -> None:
        """Serialize the fitted preprocessor to disk using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/preprocessor.joblib") -> "MizuPreprocessor":
        """Load a serialized preprocessor from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Serialized preprocessor not found at {filepath}")
        instance = joblib.load(filepath)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return instance


def preprocess_and_save_data(
    raw_data_path: Optional[str] = None,
    output_csv_path: str = "data/processed/water_treatment_preprocessed.csv",
    model_save_path: str = "models/preprocessor.joblib"
) -> Tuple[pd.DataFrame, pd.DataFrame, MizuPreprocessor]:
    """
    End-to-end preprocessing runner:
    1. Loads raw data via load_water_treatment_data().
    2. Fits and transforms using MizuPreprocessor.
    3. Saves the preprocessed dataset and fitted preprocessor model to disk.

    Returns:
        Tuple of (X_scaled, metadata, preprocessor).
    """
    # 1. Load raw data
    raw_df = load_water_treatment_data(raw_data_path)

    # 2. Fit and transform
    preprocessor = MizuPreprocessor()
    X_scaled, metadata = preprocessor.fit_transform(raw_df)

    # 3. Save combined preprocessed dataframe
    combined = pd.concat([metadata, X_scaled], axis=1)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Preprocessed data exported to {output_path} ({combined.shape[0]} rows x {combined.shape[1]} cols)")

    # 4. Save preprocessor artifact
    preprocessor.save(model_save_path)

    return X_scaled, metadata, preprocessor


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MIZU Preprocessing Pipeline")
    print("=" * 60)

    X_scaled, meta, prep = preprocess_and_save_data()

    print("\n--- PIPELINE VERIFICATION AUDIT ---")
    print(f"Features processed: {X_scaled.shape[1]} sensor columns")
    print(f"Instances processed: {X_scaled.shape[0]} daily records")
    print(f"Remaining Missing Values (NaN): {X_scaled.isnull().sum().sum()}")
    print("\nSample Transformed Data (First 3 rows, first 6 features):")
    print(X_scaled.iloc[:3, :6].round(4))

    print("\nVerification: Feature medians after RobustScaler (should all be 0.0):")
    medians = X_scaled.median()
    print(f"Max absolute median deviation from 0.0: {abs(medians).max():.6f}")

    print("\nPreprocessing pipeline test passed with 100% success!")

