"""
MIZU: Operational Clustering Module (K-Means)
==============================================
Author: MIZU Team
Purpose: Identifies operational regimes and recurring patterns in water treatment plant
         telemetry using K-Means clustering.

Why Clustering in MIZU:
Anomaly detection (Isolation Forest, LOF) tells operators WHEN an individual observation
is unusual. Clustering tells operators WHAT distinct operational regimes exist in the plant
(e.g., canonical baseline operation, sub-optimal/degraded operation, extreme stress regimes).

Evaluation Heuristics:
1. Elbow Method (Inertia): Measures within-cluster sum of squared distances.
2. Silhouette Score: Measures how well-separated and cohesive clusters are (-1 to +1).
3. Davies-Bouldin Index: Ratio of within-cluster scatter to between-cluster separation (lower = better).
4. Domain Interpretability: The mathematically highest silhouette (e.g., K=2) often merely isolates
   a tiny outlier group without revealing operational regimes. We balance math with operational utility.
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from src.preprocessing import get_feature_columns


class MizuKMeansClusterer:
    """
    Production-ready wrapper around scikit-learn's KMeans clustering algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300
    ):
        """
        Initialize the K-Means clusterer.

        Args:
            n_clusters: Number of clusters (K).
            random_state: Seed for reproducible centroid initialization.
            n_init: Number of initializations to run for stability (default: 10).
            max_iter: Maximum number of iterations per run (default: 300).
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "MizuKMeansClusterer":
        """Fit K-Means centroids on preprocessed features X."""
        self.feature_names = get_feature_columns(X)
        features = X[self.feature_names].values
        self.model.fit(features)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign closest cluster label (0 to K-1) for observations."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        features = X[self.feature_names].values
        return self.model.predict(features)

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit centroids and return assigned cluster labels."""
        self.fit(X)
        return self.predict(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Compute distances from each observation to all cluster centroids."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling transform().")
        features = X[self.feature_names].values
        return self.model.transform(features)

    def get_centroids(self) -> pd.DataFrame:
        """Return cluster centroids in a readable DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inspecting centroids.")
        return pd.DataFrame(
            self.model.cluster_centers_,
            columns=self.feature_names,
            index=[f"Cluster_{i}" for i in range(self.n_clusters)]
        )

    def save(self, filepath: Union[str, Path] = "models/kmeans_model.joblib") -> None:
        """Serialize fitted model to disk via joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"K-Means model saved successfully to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/kmeans_model.joblib") -> "MizuKMeansClusterer":
        """Load serialized K-Means model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        return joblib.load(filepath)


def evaluate_cluster_range(
    X: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate multiple cluster counts across Inertia, Silhouette, and Davies-Bouldin metrics.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        results.append({
            "k": k,
            "inertia": round(float(inertia), 2),
            "silhouette": round(float(sil), 4),
            "davies_bouldin": round(float(db), 4),
            "calinski_harabasz": round(float(ch), 2)
        })
    return pd.DataFrame(results)


def generate_clustering_evaluation_plots(
    eval_df: pd.DataFrame,
    selected_k: int = 4,
    save_path: Union[str, Path] = "reports/figures/08_kmeans_evaluation.png"
) -> None:
    """
    Generate 3-panel diagnostic plot for cluster count selection:
    Panel A: Elbow Curve (Inertia vs K).
    Panel B: Silhouette Score vs K.
    Panel C: Davies-Bouldin Index vs K.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    k_vals = eval_df["k"]

    # 1. Elbow Curve
    ax1.plot(k_vals, eval_df["inertia"], marker="o", color="#2563eb", linewidth=2, markersize=7)
    ax1.axvline(selected_k, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Selected K = {selected_k}")
    ax1.set_title("Elbow Method (Inertia)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)

    # 2. Silhouette Score
    ax2.plot(k_vals, eval_df["silhouette"], marker="s", color="#059669", linewidth=2, markersize=7)
    ax2.axvline(selected_k, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Selected K = {selected_k}")
    ax2.set_title("Silhouette Score (Separation & Cohesion)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax2.set_ylabel("Mean Silhouette Coefficient", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    # 3. Davies-Bouldin Index (Lower is Better)
    ax3.plot(k_vals, eval_df["davies_bouldin"], marker="^", color="#d97706", linewidth=2, markersize=7)
    ax3.axvline(selected_k, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Selected K = {selected_k}")
    ax3.set_title("Davies-Bouldin Index (Lower = Better)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax3.set_ylabel("Davies-Bouldin Index", fontsize=10)
    ax3.legend(loc="upper left", fontsize=9)

    plt.suptitle("MIZU: K-Means Multi-Criteria Cluster Selection Diagnostics (K=2 to K=8)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Cluster evaluation plot saved to: {out_file}")


def generate_cluster_profile_plots(
    df_results: pd.DataFrame,
    save_path: Union[str, Path] = "reports/figures/09_kmeans_cluster_profiles.png"
) -> None:
    """
    Generate diagnostic visual profiles of the discovered clusters:
    Panel A: Distribution of Operational Days Across Clusters.
    Panel B: Key Sensor Mean Deviations per Cluster.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 2]})

    # Panel A: Cluster Size Bar Chart
    counts = df_results["cluster"].value_counts().sort_index()
    cluster_labels = [f"Cluster {i}" for i in counts.index]
    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]

    bars = ax1.bar(cluster_labels, counts.values, color=colors[:len(counts)], edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Number of Operational Days", fontsize=10, fontweight="bold")
    ax1.set_title("Panel A: Operational Regime Cluster Sizes", fontsize=11, fontweight="bold")

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df_results)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")

    ax1.set_ylim(0, max(counts.values) * 1.18)

    # Panel B: Profile of Key Physical Features (RobustScaler Units: 0.0 = median)
    key_features = [
        "Q-E",       # Flow Rate
        "ZN-E",      # Zinc
        "DBO-E",     # Influent BOD
        "SS-E",      # Influent Solids
        "DBO-S",     # Effluent BOD
        "SS-S",      # Effluent Solids
        "RD-DBO-G",  # Global BOD Removal Efficiency
        "RD-SS-G"    # Global Solids Removal Efficiency
    ]

    cluster_means = df_results.groupby("cluster")[key_features].mean()
    cluster_means.index = [f"Cluster {i}" for i in cluster_means.index]

    # Transpose for grouped bar plot
    df_profile = cluster_means.T

    df_profile.plot(kind="bar", ax=ax2, color=colors[:len(counts)], edgecolor="black", alpha=0.85, width=0.8)
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Mean Standardized Deviation (RobustScaler Units)", fontsize=10, fontweight="bold")
    ax2.set_title("Panel B: Sensor Reading Profile Across Discovered Clusters (0.0 = Plant Median)", fontsize=11, fontweight="bold")
    ax2.set_xticklabels(key_features, rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax2.legend(title="Operating Regime", fontsize=9, title_fontsize=9)

    plt.suptitle("MIZU: Operational Cluster Characterization & Sensor Profiles", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Cluster profile plot saved to: {out_file}")


def run_clustering_pipeline(
    preprocessed_path: Union[str, Path] = "data/processed/water_treatment_preprocessed.csv",
    selected_k: int = 4
) -> Tuple[pd.DataFrame, MizuKMeansClusterer, pd.DataFrame]:
    """
    Main clustering execution pipeline:
    1. Loads preprocessed data.
    2. Evaluates cluster counts K=2 through K=8.
    3. Trains final MizuKMeansClusterer with selected_k.
    4. Saves model to models/kmeans_model.joblib.
    5. Saves results table to data/processed/kmeans_results.csv.
    6. Generates diagnostic evaluation and profile plots.
    """
    data_path = Path(preprocessed_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}. Run preprocessing first!")

    df = pd.read_csv(data_path)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values

    # 1. Multi-Criteria Evaluation Range
    eval_df = evaluate_cluster_range(X, k_range=range(2, 9), random_state=42)

    # 2. Train Final Model
    clusterer = MizuKMeansClusterer(n_clusters=selected_k, random_state=42, n_init=10)
    labels = clusterer.fit_predict(df)

    # 3. Build Results DataFrame
    df_results = df.copy()
    df_results["cluster"] = labels

    # Compute distance to assigned centroid
    distances = clusterer.transform(df)
    min_distances = np.min(distances, axis=1)
    df_results["distance_to_centroid"] = min_distances

    # 4. Save Artifacts
    out_csv = Path("data/processed/kmeans_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"Clustering results exported to: {out_csv}")

    model_path = Path("models/kmeans_model.joblib")
    clusterer.save(model_path)

    # 5. Generate Visual Diagnostics
    generate_clustering_evaluation_plots(eval_df, selected_k=selected_k)
    generate_cluster_profile_plots(df_results)

    return df_results, clusterer, eval_df


if __name__ == "__main__":
    print("=" * 60)
    print("Executing MIZU Operational K-Means Clustering Pipeline")
    print("=" * 60)

    results_df, model, eval_metrics = run_clustering_pipeline(selected_k=4)

    print("\n--- CLUSTER SELECTION METRICS (K=2 to K=8) ---")
    print(eval_metrics.to_string(index=False))

    print(f"\n--- SELECTED MODEL (K = {model.n_clusters}) SUMMARY ---")
    counts = results_df["cluster"].value_counts().sort_index()
    for cluster_id, count in counts.items():
        pct = (count / len(results_df)) * 100
        print(f"Cluster {cluster_id}: {count} days ({pct:.2f}%)")

    print("\nClustering pipeline execution complete!")

