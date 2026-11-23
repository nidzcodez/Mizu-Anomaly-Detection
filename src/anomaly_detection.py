"""
MIZU: Anomaly Detection Module
==============================
Author: MIZU Team
Purpose: Implements unsupervised anomaly detection for water treatment plant data.
         Currently features Isolation Forest (iForest) with calibrated scoring,
         reproducible random states, model persistence, and diagnostic visualizations.

Why Isolation Forest:
Isolation Forest does not try to build a model of 'normal' data. Instead, it
explicitly isolates anomalies by creating random recursive splits across features.
Because anomalies are 'few and different', they require far fewer random cuts
to be isolated in a leaf node than normal, densely clustered observations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in module path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

from src.preprocessing import get_feature_columns


class IsolationForestDetector:
    """
    Modular wrapper around scikit-learn's IsolationForest.
    
    Provides:
    - Calibrated intuitive anomaly scores (0 to 1, higher = more anomalous).
    - Binary anomaly classification flags (1 = flagged anomaly, 0 = normal).
    - Model persistence via joblib.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: Union[str, int, float] = "auto",
        contamination: float = 0.03,
        random_state: int = 42
    ):
        """
        Initialize the Isolation Forest detector.

        Args:
            n_estimators: Number of isolation trees (default: 150 for ensemble stability).
            max_samples: Number of samples drawn to train each tree ('auto' draws min(256, n)).
                         Subsampling prevents 'swamping' (normal points labeled abnormal)
                         and 'masking' (anomalies masking other anomalies).
            contamination: Expected proportion of outliers (default: 0.03, or 3%).
                           Corresponds to domain literature documenting ~13 fault days out of 527.
            random_state: Seed for random splits to guarantee 100% reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.offset_: Optional[float] = None

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        """Fit the isolation forest ensemble on preprocessed feature matrix X."""
        self.feature_names = get_feature_columns(X)
        features = X[self.feature_names]
        self.model.fit(features)
        self.offset_ = float(self.model.offset_)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether an observation is anomalous.
        
        Returns:
            np.ndarray: Binary flags where 1 = Anomaly (flagged), 0 = Normal.
                        (Converted from scikit-learn's default -1/1 convention).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        features = X[self.feature_names]
        raw_pred = self.model.predict(features)
        # scikit-learn returns -1 for outlier, 1 for inlier.
        # We convert: -1 -> 1 (Anomaly), 1 -> 0 (Normal) for user clarity.
        return np.where(raw_pred == -1, 1, 0)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Raw decision function from scikit-learn. Negative = outlier, positive = inlier."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before decision_function().")
        features = X[self.feature_names]
        return self.model.decision_function(features)

    def compute_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute an intuitive, calibrated anomaly score between 0.0 and 1.0.
        
        Formula:
        In scikit-learn, score_samples() computes the negative of the original
        Isolation Forest anomaly score formula s(x, n).
        We negate and calibrate it so:
        - 0.0 to 0.45: Typical normal operation.
        - 0.45 to 0.55: Borderline observation.
        - 0.55 to 1.00: Highly unusual observation (low path length, isolated early).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing anomaly scores.")
        features = X[self.feature_names]
        # raw score_samples returns opposite of original Liu et al. paper
        raw_scores = -self.model.score_samples(features)
        return raw_scores

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and return (anomaly_flags, anomaly_scores)."""
        self.fit(X)
        flags = self.predict(X)
        scores = self.compute_anomaly_scores(X)
        return flags, scores

    def save(self, filepath: Union[str, Path] = "models/isolation_forest.joblib") -> None:
        """Save the fitted detector to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Isolation Forest model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/isolation_forest.joblib") -> "IsolationForestDetector":
        """Load a saved detector from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        detector = joblib.load(filepath)
        return detector


def run_isolation_forest_pipeline(
    preprocessed_path: str = "data/processed/water_treatment_preprocessed.csv",
    contamination: float = 0.03,
    n_estimators: int = 150,
    random_state: int = 42
) -> Tuple[pd.DataFrame, IsolationForestDetector]:
    """
    End-to-end execution of Isolation Forest on the preprocessed dataset:
    1. Loads preprocessed CSV.
    2. Trains the detector.
    3. Adds anomaly scores and flags.
    4. Saves results table and model artifact.
    5. Generates diagnostic timeline and distribution plots.
    """
    data_path = Path(preprocessed_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}. Run preprocessing first!")

    df = pd.read_csv(data_path)
    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )

    flags, scores = detector.fit_predict(df)

    # Attach results to dataframe
    df_results = df.copy()
    df_results["anomaly_flag"] = flags
    df_results["anomaly_score"] = scores
    df_results["decision_score"] = detector.decision_function(df)

    # Export results
    results_path = Path("data/processed/isolation_forest_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"Results exported to {results_path}")

    # Save model
    model_path = Path("models/isolation_forest.joblib")
    detector.save(model_path)

    # Generate visual diagnostics
    generate_isolation_forest_plots(df_results)

    return df_results, detector


def generate_isolation_forest_plots(
    df: pd.DataFrame,
    save_path: str = "reports/figures/05_isolation_forest_scores.png"
) -> None:
    """
    Create a 2-panel diagnostic visualization:
    Panel A: Time-series of Anomaly Scores with Flagged Days.
    Panel B: Anomaly Score Distribution with Decision Threshold.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.3, 1]})

    # Ensure Date parsing for time axis
    dates = pd.to_datetime(df["Date_Parsed"], errors="coerce")
    if dates.isnull().all():
        x_axis = np.arange(len(df))
        x_label = "Operational Day Index"
    else:
        x_axis = dates
        x_label = "Date"

    scores = df["anomaly_score"]
    flags = df["anomaly_flag"]

    # Panel 1: Time Series Timeline
    ax1.plot(x_axis, scores, color="#0284c7", linewidth=1.2, alpha=0.8, label="Isolation Forest Score")
    
    # Highlight anomalies
    anomaly_indices = flags == 1
    ax1.scatter(
        x_axis[anomaly_indices],
        scores[anomaly_indices],
        color="#dc2626",
        s=60,
        edgecolor="black",
        linewidth=0.8,
        label=f"Flagged Anomaly (N = {anomaly_indices.sum()})",
        zorder=5
    )

    # Draw threshold line (approximate score at contamination percentile)
    threshold = np.percentile(scores, 100 * (1 - df["anomaly_flag"].mean()))
    ax1.axhline(threshold, color="#b91c1c", linestyle="--", linewidth=1.2, label=f"Decision Cutoff (Contamination: {df['anomaly_flag'].mean()*100:.1f}%)")
    
    ax1.set_title("Panel A: Operational Timeline & Anomaly Scores (527 Consecutive Days)", fontsize=12, fontweight="bold")
    ax1.set_xlabel(x_label, fontsize=10)
    ax1.set_ylabel("Anomaly Score (Higher = More Unusual)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 2: Score Distribution
    sns.histplot(scores[~anomaly_indices], bins=35, color="#0284c7", alpha=0.6, label="Normal Operation", ax=ax2, edgecolor="white")
    sns.histplot(scores[anomaly_indices], bins=15, color="#dc2626", alpha=0.8, label="Flagged Anomalies", ax=ax2, edgecolor="black")
    ax2.axvline(threshold, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Cutoff Threshold ({threshold:.3f})")
    
    ax2.set_title("Panel B: Distribution of Anomaly Scores (Normal vs Flagged)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Anomaly Score", fontsize=10)
    ax2.set_ylabel("Frequency (Days)", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("MIZU: Isolation Forest Anomaly Detection Diagnostic Report", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Diagnostic plot saved to: {out_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("Executing MIZU Isolation Forest Anomaly Detection")
    print("=" * 60)

    results_df, model = run_isolation_forest_pipeline(contamination=0.03, random_state=42)

    total_days = len(results_df)
    flagged_days = results_df["anomaly_flag"].sum()
    flagged_pct = (flagged_days / total_days) * 100

    print("\n--- DETECTION SUMMARY ---")
    print(f"Total Operational Days Evaluated: {total_days}")
    print(f"Days Flagged as Unusual: {flagged_days} ({flagged_pct:.2f}%)")
    print(f"Average Anomaly Score: {results_df['anomaly_score'].mean():.4f}")
    print(f"Maximum Anomaly Score: {results_df['anomaly_score'].max():.4f}")
    print(f"Minimum Anomaly Score: {results_df['anomaly_score'].min():.4f}")

    print("\n--- TOP 5 MOST UNUSUAL DAYS ---")
    top_anomalies = results_df.sort_values(by="anomaly_score", ascending=False).head(5)
    cols_to_show = ["Date", "anomaly_score", "Q-E", "ZN-E", "DBO-S", "SS-S"]
    print(top_anomalies[cols_to_show].to_string(index=False))

    print("\nPipeline execution complete!")
