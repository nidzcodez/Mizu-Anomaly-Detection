"""
MIZU: Dimensionality Reduction Module (Principal Component Analysis - PCA)
==========================================================================
Author: MIZU Team
Purpose: Projects the 38-dimensional scaled water-treatment operational space
         onto 2 orthogonal principal components for visualization, clustering audit,
         and geometric inspection.

Why PCA in MIZU:
Human eyes and 2D computer screens cannot visualize 38 dimensions simultaneously.
PCA finds the orthogonal directions (principal components) along which the data
varies the most.
- PC1 captures the maximum operational variance across all sensors.
- PC2 is orthogonal (perpendicular) to PC1 and captures the second highest variance.

Important Boundary:
PCA is a linear transformation and dimensionality reduction technique.
It is NOT an anomaly detection model and does not certify or replace laboratory testing.
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
from sklearn.decomposition import PCA

from src.preprocessing import get_feature_columns


class MizuPCA:
    """
    Wrapper around scikit-learn's PCA with feature tracking, loadings extraction,
    and serialization.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        Initialize the PCA transformer.

        Args:
            n_components: Number of principal components to project onto (default: 2).
            random_state: Random state for deterministic SVD solver initialization.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "MizuPCA":
        """Fit PCA on preprocessed 38 continuous sensor columns."""
        self.feature_names = get_feature_columns(X)
        features = X[self.feature_names].values
        self.model.fit(features)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Project features onto fitted principal components."""
        if not self.is_fitted:
            raise RuntimeError("PCA model must be fitted before calling transform().")
        features = X[self.feature_names].values
        return self.model.transform(features)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Return explained variance ratio for each component."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to check explained variance.")
        return self.model.explained_variance_ratio_

    def get_loadings(self) -> pd.DataFrame:
        """
        Return feature loadings (eigenvectors):
        How much each original sensor contributes to each principal component.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to extract loadings.")
        columns = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(
            self.model.components_.T,
            index=self.feature_names,
            columns=columns
        )

    def save(self, filepath: Union[str, Path] = "models/pca_model.joblib") -> None:
        """Serialize fitted PCA model to disk via joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted PCA model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"PCA model saved successfully to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/pca_model.joblib") -> "MizuPCA":
        """Load serialized PCA model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        return joblib.load(filepath)


def generate_pca_cluster_plot(
    df_pca: pd.DataFrame,
    evr: np.ndarray,
    save_path: Union[str, Path] = "reports/figures/10_pca_clusters.png"
) -> None:
    """
    Generate 2D PCA scatter plot colored by K-Means cluster assignments.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Cluster styling
    cluster_meta = {
        0: {"label": "Cluster 0: Baseline Stable Operation", "color": "#2563eb", "marker": "o", "size": 45, "alpha": 0.6},
        2: {"label": "Cluster 2: Sub-Optimal / Degraded Clarification", "color": "#d97706", "marker": "s", "size": 85, "alpha": 0.85},
        1: {"label": "Cluster 1: Hydraulic Clarifier Washout (D-13/3/90)", "color": "#dc2626", "marker": "^", "size": 130, "alpha": 1.0},
        3: {"label": "Cluster 3: Biological Oxidation Collapse", "color": "#7c3aed", "marker": "D", "size": 130, "alpha": 1.0}
    }

    # Plot baseline first, then outlier clusters
    for c_id in [0, 2, 1, 3]:
        if c_id not in df_pca["cluster"].values:
            continue
        mask = df_pca["cluster"] == c_id
        meta = cluster_meta[c_id]
        ax.scatter(
            df_pca.loc[mask, "PC1"],
            df_pca.loc[mask, "PC2"],
            label=f"{meta['label']} (N = {mask.sum()})",
            color=meta["color"],
            marker=meta["marker"],
            s=meta["size"],
            alpha=meta["alpha"],
            edgecolors="black" if c_id != 0 else "none",
            linewidth=0.8,
            zorder=3 if c_id != 0 else 2
        )

    # Reference gridlines through origin
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate the extreme points
    extreme_points = df_pca[df_pca["cluster"].isin([1, 3])]
    for _, row in extreme_points.iterrows():
        ax.annotate(
            f"{row['Date']}\n(C{row['cluster']})",
            xy=(row["PC1"], row["PC2"]),
            xytext=(row["PC1"] + 0.8, row["PC2"] + 0.3),
            fontsize=8.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", alpha=0.9)
        )

    # Labels with exact variance explained
    pc1_pct = evr[0] * 100
    pc2_pct = evr[1] * 100
    total_pct = (evr[0] + evr[1]) * 100

    ax.set_xlabel(f"Principal Component 1 (PC1) — {pc1_pct:.2f}% Variance Explained (Effluent Clarification & Settleable Solids)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 (PC2) — {pc2_pct:.2f}% Variance Explained (Influent Particulate Burden)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"MIZU: 2D Principal Component Projection with Operational K-Means Clusters\n(Total Retained Operational Variance: {total_pct:.2f}%)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.legend(loc="upper left", frameon=True, fontsize=9.5, title="Operational Regime", title_fontsize=10)

    plt.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PCA cluster scatter plot saved to: {out_file}")


def generate_pca_loadings_plot(
    loadings: pd.DataFrame,
    evr: np.ndarray,
    save_path: Union[str, Path] = "reports/figures/11_pca_feature_loadings.png",
    top_n: int = 10
) -> None:
    """
    Generate 2-panel visualization of feature contributions (loadings) for PC1 and PC2.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Top contributors for PC1
    pc1_top = loadings["PC1"].abs().sort_values(ascending=False).head(top_n).index
    pc1_vals = loadings.loc[pc1_top, "PC1"]
    colors_pc1 = ["#2563eb" if v >= 0 else "#ef4444" for v in pc1_vals]

    bars1 = ax1.barh(pc1_top, pc1_vals, color=colors_pc1, edgecolor="black", alpha=0.85, height=0.6)
    ax1.set_xlabel("Feature Loading (Eigenvector Weight)", fontsize=10, fontweight="bold")
    ax1.set_title(f"Panel A: Top {top_n} Contributors to PC1 ({evr[0]*100:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax1.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.invert_yaxis()

    for bar in bars1:
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 0.02 if w >= 0 else -0.02
        ax1.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")

    # Top contributors for PC2
    pc2_top = loadings["PC2"].abs().sort_values(ascending=False).head(top_n).index
    pc2_vals = loadings.loc[pc2_top, "PC2"]
    colors_pc2 = ["#059669" if v >= 0 else "#ef4444" for v in pc2_vals]

    bars2 = ax2.barh(pc2_top, pc2_vals, color=colors_pc2, edgecolor="black", alpha=0.85, height=0.6)
    ax2.set_xlabel("Feature Loading (Eigenvector Weight)", fontsize=10, fontweight="bold")
    ax2.set_title(f"Panel B: Top {top_n} Contributors to PC2 ({evr[1]*100:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax2.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.invert_yaxis()

    for bar in bars2:
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 0.02 if w >= 0 else -0.02
        ax2.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")

    plt.suptitle("MIZU: Principal Component Feature Loadings (Physical Interpretation of Latent Axes)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PCA feature loadings plot saved to: {out_file}")


def run_pca_pipeline(
    preprocessed_path: Union[str, Path] = "data/processed/water_treatment_preprocessed.csv",
    kmeans_path: Union[str, Path] = "data/processed/kmeans_results.csv",
    n_components: int = 2
) -> Tuple[pd.DataFrame, MizuPCA, Dict[str, float]]:
    """
    Main PCA pipeline execution:
    1. Loads preprocessed features.
    2. Merges K-Means cluster labels.
    3. Fits PCA(n_components=2).
    4. Projects 38D space to 2D (PC1, PC2).
    5. Saves models/pca_model.joblib.
    6. Saves data/processed/pca_results.csv.
    7. Generates reports/figures/10_pca_clusters.png and 11_pca_feature_loadings.png.
    """
    path_prep = Path(preprocessed_path)
    path_km = Path(kmeans_path)

    if not path_prep.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {path_prep}.")
    if not path_km.exists():
        raise FileNotFoundError(f"K-Means results not found at {path_km}.")

    df_prep = pd.read_csv(path_prep)
    df_km = pd.read_csv(path_km)

    # Verify date alignment
    if not (df_prep["Date"] == df_km["Date"]).all():
        raise ValueError("Date alignment mismatch between preprocessed data and K-Means results!")

    # Fit PCA
    pca = MizuPCA(n_components=n_components, random_state=42)
    X_projected = pca.fit_transform(df_prep)

    evr = pca.get_explained_variance_ratio()
    metrics = {
        "pc1_explained_variance": float(round(evr[0], 4)),
        "pc2_explained_variance": float(round(evr[1], 4)),
        "cumulative_explained_variance": float(round(np.sum(evr), 4)),
        "retained_variance_percentage": float(round(np.sum(evr) * 100, 2))
    }

    # Build exported DataFrame
    df_pca = pd.DataFrame({
        "Date": df_prep["Date"],
        "Date_Parsed": df_prep.get("Date_Parsed", df_prep["Date"]),
        "PC1": X_projected[:, 0],
        "PC2": X_projected[:, 1],
        "cluster": df_km["cluster"],
        "distance_to_centroid": df_km["distance_to_centroid"]
    })

    # Save CSV
    out_csv = Path("data/processed/pca_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pca.to_csv(out_csv, index=False)
    print(f"PCA results exported to: {out_csv}")

    # Save model
    model_path = Path("models/pca_model.joblib")
    pca.save(model_path)

    # Generate plots
    generate_pca_cluster_plot(df_pca, evr)
    loadings = pca.get_loadings()
    generate_pca_loadings_plot(loadings, evr)

    return df_pca, pca, metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Executing MIZU Principal Component Analysis (PCA) Pipeline")
    print("=" * 60)

    results_df, model, stats = run_pca_pipeline()

    print("\n--- PCA VARIANCE EXPLAINED ---")
    print(f"PC1 Variance: {stats['pc1_explained_variance']:.4f} ({stats['pc1_explained_variance']*100:.2f}%)")
    print(f"PC2 Variance: {stats['pc2_explained_variance']:.4f} ({stats['pc2_explained_variance']*100:.2f}%)")
    print(f"Cumulative Variance Retained: {stats['cumulative_explained_variance']:.4f} ({stats['retained_variance_percentage']:.2f}%)")
    print(f"Dimensionality Reduction: 38 Dimensions -> 2 Dimensions ({((38-2)/38)*100:.1f}% compression)")

    print("\n--- TOP 3 LOADINGS FOR PC1 ---")
    loadings = model.get_loadings()
    print(loadings["PC1"].abs().sort_values(ascending=False).head(3))

    print("\n--- TOP 3 LOADINGS FOR PC2 ---")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(3))

    print("\nPCA pipeline execution complete!")

