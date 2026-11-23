"""
MIZU: Local Outlier Factor (LOF) Anomaly Detection Module
=========================================================
Author: MIZU Team
Purpose: Implements Local Outlier Factor (LOF) for detecting density-based
         local anomalies across water treatment plant operations.

How LOF Works:
Unlike Isolation Forest (which isolates anomalies using global random cuts),
LOF looks at the LOCAL density around an observation compared to the density
of its k-nearest neighbors.

An observation is flagged if its local reachability density is significantly
lower than the density of its neighbors (LOF >> 1.0).
This allows LOF to catch subtle anomalies inside low-density clusters that
global algorithms might overlook.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root in module path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

from src.preprocessing import get_feature_columns


class LOFDetector:
    """
    Modular wrapper around scikit-learn's LocalOutlierFactor in Novelty Detection mode.

    Key Features:
    - Runs with novelty=True so it can score both historical records and future streaming data.
    - Computes genuine Local Outlier Factor (LOF) ratios (LOF ~ 1.0 = normal, LOF >> 1.0 = anomaly).
    - Uses binary anomaly flags (1 = Flagged Anomaly, 0 = Normal).
    - Serializes model to disk via joblib.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.03,
        metric: str = "euclidean",
        novelty: bool = True
    ):
        """
        Initialize the Local Outlier Factor detector.

        Args:
            n_neighbors: Number of neighbors to consider when calculating local density (k).
                         Default: 20. A standard rule of thumb: large enough to smooth out noise,
                         small enough to retain sensitivity to local clusters.
            contamination: Expected proportion of outliers in the data (default: 0.03 or 3%).
            metric: Distance metric to compute neighborhood (default: 'euclidean').
            novelty: Must be True to allow predict() and decision_function() on data.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=self.novelty,
            n_jobs=-1
        )
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.offset_: Optional[float] = None

    def fit(self, X: pd.DataFrame) -> "LOFDetector":
        """Fit the LOF neighborhood structure on preprocessed features X."""
        self.feature_names = get_feature_columns(X)
        features = X[self.feature_names].values
        self.model.fit(features)
        self.offset_ = float(self.model.offset_)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether observations are local outliers.
        
        Returns:
            np.ndarray: 1 = Flagged Local Outlier, 0 = Normal Inlier.
                        (Converted from scikit-learn's default -1/1 convention).
        """
        if not self.is_fitted:
            raise RuntimeError("LOF model must be fitted before predict().")
        features = X[self.feature_names].values
        raw_pred = self.model.predict(features)
        # scikit-learn returns -1 for outlier, 1 for inlier.
        # We convert: -1 -> 1 (Anomaly), 1 -> 0 (Normal).
        return np.where(raw_pred == -1, 1, 0)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Shifted opposite of LOF score.
        Negative values = outliers, Positive values = inliers.
        """
        if not self.is_fitted:
            raise RuntimeError("LOF model must be fitted before decision_function().")
        features = X[self.feature_names].values
        return self.model.decision_function(features)

    def compute_lof_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the actual Local Outlier Factor (LOF) score for each observation.

        In scikit-learn, score_samples() returns the negative of the true LOF.
        By negating it (-score_samples), we obtain the true physical LOF ratio:
        - LOF ~ 1.0: Observation has similar density to its 20 neighbors (Normal).
        - LOF < 1.0: Observation is in a denser region than its neighbors (Deep inlier).
        - LOF > 1.5 - 2.0: Observation has substantially lower density than its neighbors (Outlier).
        """
        if not self.is_fitted:
            raise RuntimeError("LOF model must be fitted before computing LOF scores.")
        features = X[self.feature_names].values
        raw_lof = -self.model.score_samples(features)
        return raw_lof

    def fit_predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit model and return (anomaly_flags, lof_scores)."""
        self.fit(X)
        flags = self.predict(X)
        scores = self.compute_lof_scores(X)
        return flags, scores

    def save(self, filepath: Union[str, Path] = "models/lof_model.joblib") -> None:
        """Serialize fitted LOF model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"LOF model saved successfully to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/lof_model.joblib") -> "LOFDetector":
        """Load serialized LOF model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        return joblib.load(filepath)


def run_lof_pipeline(
    preprocessed_path: str = "data/processed/water_treatment_preprocessed.csv",
    n_neighbors: int = 20,
    contamination: float = 0.03
) -> Tuple[pd.DataFrame, LOFDetector]:
    """
    End-to-end execution of Local Outlier Factor on preprocessed data:
    1. Loads preprocessed CSV.
    2. Trains the LOF detector.
    3. Calculates LOF scores and flags.
    4. Exports results to data/processed/lof_results.csv.
    5. Saves model to models/lof_model.joblib.
    6. Generates diagnostic timeline and distribution plots.
    """
    data_path = Path(preprocessed_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at {data_path}. Run preprocessing first!")

    df = pd.read_csv(data_path)
    detector = LOFDetector(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True
    )

    flags, scores = detector.fit_predict(df)

    df_results = df.copy()
    df_results["lof_flag"] = flags
    df_results["lof_score"] = scores
    df_results["decision_score"] = detector.decision_function(df)

    # Save results
    results_path = Path("data/processed/lof_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"LOF results exported to: {results_path}")

    # Save model artifact
    model_path = Path("models/lof_model.joblib")
    detector.save(model_path)

    # Generate visual diagnostics
    generate_lof_plots(df_results, detector)

    return df_results, detector


def generate_lof_plots(
    df: pd.DataFrame,
    detector: LOFDetector,
    save_path: str = "reports/figures/06_lof_scores.png"
) -> None:
    """
    Generate 2-panel diagnostic visualization for Local Outlier Factor:
    Panel A: Time-series of LOF Scores across operational timeline.
    Panel B: Score distribution showing local density threshold.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.3, 1]})

    dates = pd.to_datetime(df["Date_Parsed"], errors="coerce")
    if dates.isnull().all():
        x_axis = np.arange(len(df))
        x_label = "Operational Day Index"
    else:
        x_axis = dates
        x_label = "Date"

    scores = df["lof_score"]
    flags = df["lof_flag"]
    threshold = -detector.model.offset_

    # Panel A: Timeline
    ax1.plot(x_axis, scores, color="#059669", linewidth=1.2, alpha=0.8, label="Local Outlier Factor (LOF)")
    
    anomaly_mask = flags == 1
    ax1.scatter(
        x_axis[anomaly_mask],
        scores[anomaly_mask],
        color="#dc2626",
        s=60,
        edgecolor="black",
        linewidth=0.8,
        label=f"Flagged Local Outlier (N = {anomaly_mask.sum()})",
        zorder=5
    )

    ax1.axhline(threshold, color="#b91c1c", linestyle="--", linewidth=1.2, label=f"Density Cutoff Threshold ({threshold:.2f})")
    ax1.set_title(f"Panel A: LOF Operational Timeline (k = {detector.n_neighbors} Neighbors)", fontsize=12, fontweight="bold")
    ax1.set_xlabel(x_label, fontsize=10)
    ax1.set_ylabel("LOF Score (1.0 = Normal, >>1.0 = Low Density Outlier)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Distribution
    sns.histplot(scores[~anomaly_mask], bins=35, color="#059669", alpha=0.6, label="Normal Local Density", ax=ax2, edgecolor="white")
    sns.histplot(scores[anomaly_mask], bins=15, color="#dc2626", alpha=0.8, label="Flagged Local Outliers", ax=ax2, edgecolor="black")
    ax2.axvline(threshold, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Cutoff Threshold ({threshold:.2f})")

    ax2.set_title("Panel B: Distribution of Local Outlier Factor Scores", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Local Outlier Factor (LOF)", fontsize=10)
    ax2.set_ylabel("Frequency (Days)", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("MIZU: Local Outlier Factor (LOF) Anomaly Detection Diagnostic Report", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Diagnostic plot saved to: {out_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("Executing MIZU Local Outlier Factor (LOF) Pipeline")
    print("=" * 60)

    results_df, model = run_lof_pipeline(n_neighbors=20, contamination=0.03)

    total_days = len(results_df)
    flagged_days = results_df["lof_flag"].sum()
    flagged_pct = (flagged_days / total_days) * 100

    print("\n--- LOF DETECTION SUMMARY ---")
    print(f"Total Operational Days Evaluated: {total_days}")
    print(f"Neighborhood Size (k): {model.n_neighbors}")
    print(f"Decision Threshold (-offset_): {-model.model.offset_:.4f}")
    print(f"Days Flagged as Local Outliers: {flagged_days} ({flagged_pct:.2f}%)")
    print(f"Average LOF Score: {results_df['lof_score'].mean():.4f}")
    print(f"Maximum LOF Score: {results_df['lof_score'].max():.4f}")
    print(f"Minimum LOF Score: {results_df['lof_score'].min():.4f}")

    print("\n--- TOP 5 MOST EXTREME LOCAL OUTLIERS ---")
    top_anomalies = results_df.sort_values(by="lof_score", ascending=False).head(5)
    cols_to_show = ["Date", "lof_score", "Q-E", "ZN-E", "DBO-S", "SS-S"]
    print(top_anomalies[cols_to_show].to_string(index=False))

    print("\nLOF execution complete!")

