"""
MIZU: Model Comparison & Unsupervised Evaluation Module
======================================================
Author: MIZU Team
Purpose: Compares anomaly detection models (Isolation Forest vs Local Outlier Factor)
         without using ground-truth labels.
         
Why this file exists:
In real-world wastewater treatment monitoring, ground-truth labels do NOT exist in production.
Therefore, calling agreement between models 'accuracy' is mathematically invalid and deceptive.
Instead, we evaluate:
1. Algorithmic Concordance: How much do two distinct mathematical paradigms agree?
2. Jaccard Similarity: The intersection over union of flagged anomaly sets.
3. Orthogonal Coverage: What does each model catch that the other misses?
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in module path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_align_results(
    iforest_path: Union[str, Path] = "data/processed/isolation_forest_results.csv",
    lof_path: Union[str, Path] = "data/processed/lof_results.csv"
) -> pd.DataFrame:
    """
    Load results from both models and strictly verify date and row alignment.
    
    Args:
        iforest_path: Path to Isolation Forest results CSV.
        lof_path: Path to LOF results CSV.

    Returns:
        pd.DataFrame: Merged DataFrame with aligned scores and flags.
    """
    path_if = Path(iforest_path)
    path_lof = Path(lof_path)

    if not path_if.exists():
        raise FileNotFoundError(f"Isolation Forest results not found at {path_if}.")
    if not path_lof.exists():
        raise FileNotFoundError(f"LOF results not found at {path_lof}.")

    df_if = pd.read_csv(path_if)
    df_lof = pd.read_csv(path_lof)

    # Verification: identical row count
    if len(df_if) != len(df_lof):
        raise ValueError(f"Row count mismatch! iForest has {len(df_if)}, LOF has {len(df_lof)}.")

    # Verification: identical sequential dates
    if not (df_if["Date"] == df_lof["Date"]).all():
        raise ValueError("Date index mismatch between Isolation Forest and LOF results!")

    # Build clean combined dataframe
    df_comp = pd.DataFrame({
        "Date": df_if["Date"],
        "Date_Parsed": df_if.get("Date_Parsed", df_if["Date"]),
        "iforest_score": df_if["anomaly_score"],
        "iforest_flag": df_if["anomaly_flag"].astype(int),
        "iforest_decision": df_if.get("decision_score", np.nan),
        "lof_score": df_lof["lof_score"],
        "lof_flag": df_lof["lof_flag"].astype(int),
        "lof_decision": df_lof.get("decision_score", np.nan)
    })

    # Assign consensus status
    conditions = [
        (df_comp["iforest_flag"] == 1) & (df_comp["lof_flag"] == 1),
        (df_comp["iforest_flag"] == 1) & (df_comp["lof_flag"] == 0),
        (df_comp["iforest_flag"] == 0) & (df_comp["lof_flag"] == 1),
        (df_comp["iforest_flag"] == 0) & (df_comp["lof_flag"] == 0)
    ]
    choices = [
        "Both Flagged",
        "Only Isolation Forest",
        "Only LOF",
        "Both Normal"
    ]
    df_comp["consensus_status"] = np.select(conditions, choices, default="Unknown")

    return df_comp


def compute_concordance_metrics(df_comp: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Calculate unsupervised agreement metrics between Isolation Forest and LOF.
    
    Metrics computed:
    - Confusion counts: Both Flagged, Only iForest, Only LOF, Both Normal.
    - Jaccard Similarity: |A ∩ B| / |A ∪ B|.
    - Dice Similarity Coefficient: 2|A ∩ B| / (|A| + |B|).
    - Overall Observation Agreement: (Both Flagged + Both Normal) / N.
    """
    if_flags = df_comp["iforest_flag"] == 1
    lof_flags = df_comp["lof_flag"] == 1

    both = (if_flags & lof_flags).sum()
    only_if = (if_flags & (~lof_flags)).sum()
    only_lof = ((~if_flags) & lof_flags).sum()
    both_norm = ((~if_flags) & (~lof_flags)).sum()
    total = len(df_comp)

    union = (if_flags | lof_flags).sum()
    jaccard = both / union if union > 0 else 0.0
    dice = (2.0 * both) / (if_flags.sum() + lof_flags.sum()) if (if_flags.sum() + lof_flags.sum()) > 0 else 0.0
    overall_agreement = (both + both_norm) / total

    return {
        "total_observations": int(total),
        "iforest_total_flagged": int(if_flags.sum()),
        "lof_total_flagged": int(lof_flags.sum()),
        "both_flagged_count": int(both),
        "only_iforest_count": int(only_if),
        "only_lof_count": int(only_lof),
        "both_normal_count": int(both_norm),
        "union_count": int(union),
        "jaccard_similarity": float(round(jaccard, 4)),
        "dice_similarity": float(round(dice, 4)),
        "overall_agreement_rate": float(round(overall_agreement, 4))
    }


def generate_comparison_plots(
    df_comp: pd.DataFrame,
    metrics: Dict[str, Union[int, float]],
    save_path: Union[str, Path] = "reports/figures/07_model_comparison.png"
) -> None:
    """
    Create a publication-quality 3-panel visualization comparing both models:
    Panel A: Categorical Concordance Breakdown (Overlap counts).
    Panel B: Dual-Score Scatter Plot (iForest Score vs LOF Score).
    Panel C: Operational Timeline Barcode (Timeline of flagged dates).
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.3, 0.8], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # -------------------------------------------------------------
    # Panel A: Overlap & Consensus Breakdown
    # -------------------------------------------------------------
    categories = [
        "Flagged by Both Models",
        "Only Isolation Forest",
        "Only Local Outlier Factor",
        "Classified Normal by Both"
    ]
    counts = [
        metrics["both_flagged_count"],
        metrics["only_iforest_count"],
        metrics["only_lof_count"],
        metrics["both_normal_count"]
    ]
    colors = ["#dc2626", "#ea580c", "#059669", "#3b82f6"]

    bars = ax1.barh(categories, counts, color=colors, edgecolor="black", alpha=0.85, height=0.55)
    ax1.set_xscale("log")
    ax1.set_xlabel("Count of Operational Days (Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_title("Panel A: Anomaly Detection Consensus & Overlap (N = 527 Days)", fontsize=12, fontweight="bold")

    for bar, count in zip(bars, counts):
        width = bar.get_width()
        pct = (count / metrics["total_observations"]) * 100
        ax1.text(width * 1.15, bar.get_y() + bar.get_height() / 2, f"{count} days ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")

    ax1.set_xlim(1, 1000)
    ax1.invert_yaxis()

    # -------------------------------------------------------------
    # Panel B: Dual-Score Scatter Plot
    # -------------------------------------------------------------
    status_palette = {
        "Both Flagged": "#dc2626",
        "Only Isolation Forest": "#ea580c",
        "Only LOF": "#059669",
        "Both Normal": "#94a3b8"
    }

    # Plot normal points first
    normal_mask = df_comp["consensus_status"] == "Both Normal"
    ax2.scatter(
        df_comp.loc[normal_mask, "iforest_score"],
        df_comp.loc[normal_mask, "lof_score"],
        c="#94a3b8",
        alpha=0.4,
        s=35,
        label=f"Both Normal ({metrics['both_normal_count']})"
    )

    # Plot abnormal points on top
    for status in ["Only Isolation Forest", "Only LOF", "Both Flagged"]:
        mask = df_comp["consensus_status"] == status
        count = mask.sum()
        ax2.scatter(
            df_comp.loc[mask, "iforest_score"],
            df_comp.loc[mask, "lof_score"],
            c=status_palette[status],
            alpha=0.9,
            s=80,
            edgecolors="black",
            linewidth=0.8,
            label=f"{status} ({count})",
            zorder=5
        )

    # Draw median reference lines
    iforest_cutoff = df_comp.loc[df_comp["iforest_flag"] == 1, "iforest_score"].min()
    lof_cutoff = df_comp.loc[df_comp["lof_flag"] == 1, "lof_score"].min()

    ax2.axvline(iforest_cutoff, color="#ea580c", linestyle="--", linewidth=1.2, label=f"iForest Cutoff ({iforest_cutoff:.3f})")
    ax2.axhline(lof_cutoff, color="#059669", linestyle="--", linewidth=1.2, label=f"LOF Cutoff ({lof_cutoff:.3f})")

    ax2.set_yscale("log")
    ax2.set_xlabel("Isolation Forest Anomaly Score (Higher = Globally Isolated)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Local Outlier Factor Score (Log Scale, Higher = Low Local Density)", fontsize=10, fontweight="bold")
    ax2.set_title("Panel B: Multi-Model Decision Space & Concordance Quadrants", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel C: Operational Timeline Barcode
    # -------------------------------------------------------------
    dates = pd.to_datetime(df_comp["Date_Parsed"], errors="coerce")
    x_timeline = np.arange(len(df_comp)) if dates.isnull().all() else dates

    # Strip 1: Isolation Forest
    ax3.scatter(
        x_timeline[df_comp["iforest_flag"] == 1],
        np.ones((df_comp["iforest_flag"] == 1).sum()),
        color="#ea580c",
        s=45,
        marker="|",
        linewidths=3,
        label=f"Isolation Forest Flags (N = {metrics['iforest_total_flagged']})"
    )

    # Strip 2: LOF
    ax3.scatter(
        x_timeline[df_comp["lof_flag"] == 1],
        np.zeros((df_comp["lof_flag"] == 1).sum()),
        color="#059669",
        s=45,
        marker="|",
        linewidths=3,
        label=f"Local Outlier Factor Flags (N = {metrics['lof_total_flagged']})"
    )

    # Highlight common flags
    ax3.scatter(
        x_timeline[df_comp["consensus_status"] == "Both Flagged"],
        np.full((df_comp["consensus_status"] == "Both Flagged").sum(), 0.5),
        color="#dc2626",
        s=55,
        marker="o",
        edgecolors="black",
        label=f"Consensus Anomalies (N = {metrics['both_flagged_count']})"
    )

    ax3.set_yticks([0, 0.5, 1])
    ax3.set_yticklabels(["LOF", "Both Flagged", "iForest"], fontsize=9, fontweight="bold")
    ax3.set_ylim(-0.4, 1.4)
    ax3.set_xlabel("Operational Timeline (1990 - 1991)", fontsize=10, fontweight="bold")
    ax3.set_title("Panel C: Operational Timeline Barcode of Model Flags", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("MIZU: Dual-Model Unsupervised Evaluation & Concordance Architecture", fontsize=14, fontweight="bold", y=0.995)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved successfully to: {out_path}")


def run_evaluation_pipeline(
    iforest_path: str = "data/processed/isolation_forest_results.csv",
    lof_path: str = "data/processed/lof_results.csv",
    output_csv: str = "data/processed/anomaly_comparison_results.csv"
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Main evaluation pipeline orchestrator:
    1. Loads and aligns both models' predictions.
    2. Computes unsupervised concordance metrics and Jaccard similarity.
    3. Exports merged comparison table to CSV.
    4. Generates publication-grade multi-panel visualization.
    """
    df_comp = load_and_align_results(iforest_path, lof_path)
    metrics = compute_concordance_metrics(df_comp)

    # Export merged comparison table
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(out_csv, index=False)
    print(f"Comparison table exported to: {out_csv}")

    # Generate visualization
    generate_comparison_plots(df_comp, metrics)

    return df_comp, metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Executing MIZU Model Comparison & Unsupervised Evaluation")
    print("=" * 60)

    df_results, eval_metrics = run_evaluation_pipeline()

    print("\n--- CONCORDANCE & SIMILARITY AUDIT ---")
    print(f"Total Operational Days: {eval_metrics['total_observations']}")
    print(f"Isolation Forest Flags: {eval_metrics['iforest_total_flagged']}")
    print(f"Local Outlier Factor Flags: {eval_metrics['lof_total_flagged']}")
    print(f"Total Unique Days Flagged (|A ∪ B|): {eval_metrics['union_count']}")
    print(f"Days Flagged by BOTH Models (|A ∩ B|): {eval_metrics['both_flagged_count']}")
    print(f"Days Flagged ONLY by Isolation Forest: {eval_metrics['only_iforest_count']}")
    print(f"Days Flagged ONLY by LOF: {eval_metrics['only_lof_count']}")
    print(f"Days Classified Normal by Both: {eval_metrics['both_normal_count']}")
    print(f"\nJaccard Similarity Coefficient: {eval_metrics['jaccard_similarity']:.4f} ({eval_metrics['jaccard_similarity']*100:.2f}%)")
    print(f"Dice Similarity (F1 Overlap): {eval_metrics['dice_similarity']:.4f} ({eval_metrics['dice_similarity']*100:.2f}%)")
    print(f"Raw Observation Agreement Rate: {eval_metrics['overall_agreement_rate']:.4f} ({eval_metrics['overall_agreement_rate']*100:.2f}%)")

    print("\n--- THE 9 HIGH-CONFIDENCE CONSENSUS ANOMALIES ---")
    consensus_df = df_results[df_results["consensus_status"] == "Both Flagged"].sort_values(
        by="iforest_score", ascending=False
    )
    cols = ["Date", "iforest_score", "lof_score", "consensus_status"]
    print(consensus_df[cols].to_string(index=False))

    print("\nEvaluation pipeline complete!")
