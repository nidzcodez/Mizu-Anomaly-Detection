"""
MIZU: Standalone Exploratory Data Analysis Runner
=================================================
Executes all EDA steps, generates and saves publication-quality figures,
and prints key statistical summaries.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in module path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import load_water_treatment_data, get_feature_groups, get_missing_summary

def main():
    print("=" * 60)
    print("MIZU: Running Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # Aesthetics
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data_path = project_root / "data" / "raw" / "water-treatment.data"
    df = load_water_treatment_data(data_path)
    print(f"\n1. Data Loaded: {df.shape[0]} days x {df.shape[1]} columns")
    print(f"   Date range: {df['Date_Parsed'].min().date()} to {df['Date_Parsed'].max().date()}")

    # 2. Missing Value Plot
    print("\n2. Generating Missing Values Plot...")
    missing_df = get_missing_summary(df)
    missing_only = missing_df[missing_df["missing_count"] > 0].copy()

    plt.figure(figsize=(10, 8))
    bars = plt.barh(missing_only["column"], missing_only["missing_pct"], color="#2563eb", edgecolor="#1d4ed8", alpha=0.85)
    plt.xlabel("Missing Values (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Sensor / Operational Feature", fontsize=11, fontweight="bold")
    plt.title("MIZU: Missing Value Rate Across All Operational Features (Total: 527 Days)", fontsize=13, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9, color="#1e293b")

    plt.xlim(0, 15)
    plt.tight_layout()
    fig1_path = figures_dir / "01_missing_values.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"   Saved: {fig1_path}")

    # 3. Sensor Distributions & Outliers
    print("\n3. Generating Sensor Distributions Plot...")
    key_sensors = [
        ("Q-E", "Influent Flow Rate (m³/day)", "#0284c7"),
        ("ZN-E", "Influent Zinc Concentration (mg/L)", "#ea580c"),
        ("DBO-S", "Effluent Biochemical Oxygen Demand (mg/L)", "#16a34a"),
        ("SS-S", "Effluent Suspended Solids (mg/L)", "#dc2626")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (col, title, color) in enumerate(key_sensors):
        ax = axes[idx]
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color=color, bins=30, alpha=0.6, edgecolor="white")
        median_val = data.median()
        p95_val = data.quantile(0.95)
        ax.axvline(median_val, color="black", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.1f}")
        ax.axvline(p95_val, color="purple", linestyle=":", linewidth=1.5, label=f"95th Pct: {p95_val:.1f}")
        ax.set_title(f"{title} [{col}]", fontsize=11, fontweight="bold")
        ax.set_xlabel("Measured Value")
        ax.set_ylabel("Count of Days")
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("MIZU: Distribution and Outlier Analysis of Key Operational Parameters", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = figures_dir / "02_sensor_distributions.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"   Saved: {fig2_path}")

    # 4. Stage-by-Stage Progression Plot
    print("\n4. Generating Treatment Progression Plot...")
    stages = ["Influent (-E)", "Primary Settler (-P)", "Biological Stage (-D)", "Effluent (-S)"]
    bod_stages = [df["DBO-E"].median(), df["DBO-P"].median(), df["DBO-D"].median(), df["DBO-S"].median()]
    ss_stages = [df["SS-E"].median(), df["SS-P"].median(), df["SS-D"].median(), df["SS-S"].median()]

    x = np.arange(len(stages))
    width = 0.35

    plt.figure(figsize=(11, 6))
    bars1 = plt.bar(x - width/2, bod_stages, width, label="BOD (Organic Pollution, DBO)", color="#2563eb", alpha=0.85)
    bars2 = plt.bar(x + width/2, ss_stages, width, label="Suspended Solids (Dirt Particles, SS)", color="#059669", alpha=0.85)

    plt.ylabel("Median Concentration (mg/L)", fontsize=11, fontweight="bold")
    plt.title("MIZU: Pollution Reduction Across Plant Treatment Stages", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, stages, fontsize=10, fontweight="bold")
    plt.legend(fontsize=10)

    for bar in bars1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f}", ha="center", fontsize=9, fontweight="bold", color="#1e3a8a")
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f}", ha="center", fontsize=9, fontweight="bold", color="#064e3b")

    bod_reduction = ((bod_stages[0] - bod_stages[-1]) / bod_stages[0]) * 100
    ss_reduction = ((ss_stages[0] - ss_stages[-1]) / ss_stages[0]) * 100

    plt.ylim(0, 290)
    plt.annotate(
        f"Overall BOD Reduction: {bod_reduction:.1f}%\nOverall Solids Reduction: {ss_reduction:.1f}%",
        xy=(2.5, 200),
        bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", ec="#94a3b8", lw=1.5),
        fontsize=10,
        fontweight="bold"
    )

    plt.tight_layout()
    fig3_path = figures_dir / "03_treatment_progression.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"   Saved: {fig3_path}")

    # 5. Correlation Heatmap
    print("\n5. Generating Correlation Heatmap...")
    selected_cols = [
        "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "COND-E",
        "DBO-S", "DQO-S", "SS-S", "RD-DBO-G", "RD-SS-G"
    ]
    corr_matrix = df[selected_cols].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    plt.title("MIZU: Correlation Matrix of Key Plant Features", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig4_path = figures_dir / "04_correlation_heatmap.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"   Saved: {fig4_path}")

    print("\n" + "=" * 60)
    print("EDA Execution Complete! All figures saved in reports/figures/")
    print("=" * 60)

if __name__ == "__main__":
    main()

