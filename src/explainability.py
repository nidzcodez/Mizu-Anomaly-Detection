"""
MIZU: Explainability & Feature-Level Anomaly Attribution Module
==============================================================
Author: MIZU Team
Purpose: Identifies which specific physical and chemical sensor measurements
         most strongly influenced an anomaly flag using model-agnostic
         feature ablation.

How Feature Ablation Works:
1. Start with an observation flagged by Isolation Forest or Local Outlier Factor.
2. Measure its baseline anomaly score.
3. For each of the 38 sensor features, temporarily replace its value with 0.0
   (which corresponds to the historical median under RobustScaler).
4. Recalculate the model's anomaly score.
5. Measure the score change:
   Delta Score = Baseline Score - Ablated Score
   A positive Delta Score means that replacing the sensor reading with normal operating
   median reduced the anomaly score towards normal. The larger the drop, the more
   influential that sensor was in driving the anomaly flag.
6. Rank sensors by their score reduction impact.

Important Scientific Caution:
Feature ablation measures mathematical sensitivity of the model to individual inputs.
It does NOT prove mechanical or chemical causation in the plant.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root in module path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.anomaly_detection import IsolationForestDetector
from src.lof_detection import LOFDetector
from src.data_loader import load_water_treatment_data
from src.preprocessing import get_feature_columns


# Human-understandable metadata dictionary for all 38 continuous sensors
SENSOR_METADATA: Dict[str, Dict[str, str]] = {
    "Q-E": {"name": "Influent Flow Rate", "stage": "Influent (-E)", "unit": "m³/day"},
    "ZN-E": {"name": "Influent Zinc Concentration", "stage": "Influent (-E)", "unit": "mg/L"},
    "PH-E": {"name": "Influent pH", "stage": "Influent (-E)", "unit": "pH units"},
    "DBO-E": {"name": "Influent Biological Oxygen Demand (BOD)", "stage": "Influent (-E)", "unit": "mg/L"},
    "DQO-E": {"name": "Influent Chemical Oxygen Demand (COD)", "stage": "Influent (-E)", "unit": "mg/L"},
    "SS-E": {"name": "Influent Suspended Solids", "stage": "Influent (-E)", "unit": "mg/L"},
    "SSV-E": {"name": "Influent Volatile Suspended Solids", "stage": "Influent (-E)", "unit": "%"},
    "SED-E": {"name": "Influent Settleable Solids", "stage": "Influent (-E)", "unit": "mL/L"},
    "COND-E": {"name": "Influent Electrical Conductivity", "stage": "Influent (-E)", "unit": "µS/cm"},

    "PH-P": {"name": "Primary Settler pH", "stage": "Primary (-P)", "unit": "pH units"},
    "DBO-P": {"name": "Primary Settler BOD", "stage": "Primary (-P)", "unit": "mg/L"},
    "SS-P": {"name": "Primary Settler Suspended Solids", "stage": "Primary (-P)", "unit": "mg/L"},
    "SSV-P": {"name": "Primary Settler Volatile Solids", "stage": "Primary (-P)", "unit": "%"},
    "SED-P": {"name": "Primary Settler Settleable Solids", "stage": "Primary (-P)", "unit": "mL/L"},
    "COND-P": {"name": "Primary Settler Electrical Conductivity", "stage": "Primary (-P)", "unit": "µS/cm"},

    "PH-D": {"name": "Secondary Settler pH", "stage": "Secondary (-D)", "unit": "pH units"},
    "DBO-D": {"name": "Secondary Settler BOD", "stage": "Secondary (-D)", "unit": "mg/L"},
    "DQO-D": {"name": "Secondary Settler COD", "stage": "Secondary (-D)", "unit": "mg/L"},
    "SS-D": {"name": "Secondary Settler Suspended Solids", "stage": "Secondary (-D)", "unit": "mg/L"},
    "SSV-D": {"name": "Secondary Settler Volatile Solids", "stage": "Secondary (-D)", "unit": "%"},
    "SED-D": {"name": "Secondary Settler Settleable Solids", "stage": "Secondary (-D)", "unit": "mL/L"},
    "COND-D": {"name": "Secondary Settler Electrical Conductivity", "stage": "Secondary (-D)", "unit": "µS/cm"},

    "PH-S": {"name": "Effluent pH", "stage": "Effluent (-S)", "unit": "pH units"},
    "DBO-S": {"name": "Effluent Biological Oxygen Demand (BOD)", "stage": "Effluent (-S)", "unit": "mg/L"},
    "DQO-S": {"name": "Effluent Chemical Oxygen Demand (COD)", "stage": "Effluent (-S)", "unit": "mg/L"},
    "SS-S": {"name": "Effluent Suspended Solids", "stage": "Effluent (-S)", "unit": "mg/L"},
    "SSV-S": {"name": "Effluent Volatile Suspended Solids", "stage": "Effluent (-S)", "unit": "%"},
    "SED-S": {"name": "Effluent Settleable Solids", "stage": "Effluent (-S)", "unit": "mL/L"},
    "COND-S": {"name": "Effluent Electrical Conductivity", "stage": "Effluent (-S)", "unit": "µS/cm"},

    "RD-DBO-P": {"name": "Primary Settler BOD Removal Efficiency", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-SS-P": {"name": "Primary Settler Solids Removal Efficiency", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-SED-P": {"name": "Primary Settler Settleable Solids Removal", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-DBO-S": {"name": "Secondary Settler BOD Removal Efficiency", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-DQO-S": {"name": "Secondary Settler COD Removal Efficiency", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-DBO-G": {"name": "Global Plant BOD Removal Efficiency", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-DQO-G": {"name": "Global Plant COD Removal Efficiency", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-SS-G": {"name": "Global Plant Suspended Solids Removal", "stage": "Efficiency (RD-)", "unit": "%"},
    "RD-SED-G": {"name": "Global Plant Settleable Solids Removal", "stage": "Efficiency (RD-)", "unit": "%"}
}


class AnomalyExplainer:
    """
    Provides feature ablation attribution for Isolation Forest and Local Outlier Factor.
    """

    def __init__(
        self,
        iforest_detector: Optional[IsolationForestDetector] = None,
        lof_detector: Optional[LOFDetector] = None,
        preprocessed_data_path: Union[str, Path] = "data/processed/water_treatment_preprocessed.csv",
        raw_data_path: Optional[str] = None
    ):
        """
        Initialize the Anomaly Explainer with trained models and datasets.
        """
        self.iforest = iforest_detector or IsolationForestDetector.load("models/isolation_forest.joblib")
        self.lof = lof_detector or LOFDetector.load("models/lof_model.joblib")

        self.df_preprocessed = pd.read_csv(preprocessed_data_path)
        self.raw_df = load_water_treatment_data(raw_data_path)
        self.feature_names = get_feature_columns(self.df_preprocessed)

    def explain_observation(
        self,
        date: str,
        model_type: str = "iforest",
        top_n: int = 5
    ) -> Tuple[pd.DataFrame, Dict[str, Union[str, float]]]:
        """
        Explain why an observation was flagged by ablating features one by one.

        Args:
            date: Operational Date identifier (e.g. 'D-13/3/90').
            model_type: 'iforest' or 'lof'.
            top_n: Number of top influential features to return.

        Returns:
            Tuple: (top_features_df, metadata_dict)
        """
        model_key = model_type.lower().strip()
        if model_key not in {"iforest", "isolation_forest", "lof", "local_outlier_factor"}:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'iforest' or 'lof'.")

        matches = self.df_preprocessed[self.df_preprocessed["Date"] == date]
        if matches.empty:
            raise KeyError(f"Date '{date}' not found in preprocessed dataset.")

        idx = matches.index[0]
        sample = self.df_preprocessed.loc[idx:idx, self.feature_names].copy()
        raw_row = self.raw_df.loc[idx]

        # Calculate baseline score using model-specific methods
        if model_key in {"iforest", "isolation_forest"}:
            model_name = "Isolation Forest"
            base_score = float(self.iforest.compute_anomaly_scores(sample)[0])
            score_func = lambda s: float(self.iforest.compute_anomaly_scores(s)[0])
        else:
            model_name = "Local Outlier Factor"
            base_score = float(self.lof.compute_lof_scores(sample)[0])
            score_func = lambda s: float(self.lof.compute_lof_scores(s)[0])

        # Feature ablation loop
        records = []
        for feature in self.feature_names:
            sample_ablated = sample.copy()
            # Replace scaled feature with 0.0 (the historical median in RobustScaler space)
            sample_ablated[feature] = 0.0

            ablated_score = score_func(sample_ablated)
            # Positive drop means the feature pushed the observation toward an anomaly
            score_change = base_score - ablated_score

            meta = SENSOR_METADATA.get(feature, {"name": feature, "stage": "Unknown", "unit": ""})
            raw_val = raw_row[feature] if feature in raw_row else np.nan

            records.append({
                "feature_code": feature,
                "sensor_name": meta["name"],
                "stage": meta["stage"],
                "unit": meta["unit"],
                "raw_value": round(float(raw_val), 2) if pd.notnull(raw_val) else np.nan,
                "scaled_value": round(float(sample[feature].iloc[0]), 4),
                "base_score": round(base_score, 4),
                "ablated_score": round(ablated_score, 4),
                "score_change": float(score_change),
                "score_drop": round(float(score_change), 6)
            })

        df_ablation = pd.DataFrame(records)
        # Rank by largest positive score drop (features that made it anomalous)
        df_sorted = df_ablation.sort_values(by="score_change", ascending=False).reset_index(drop=True)
        top_df = df_sorted.head(top_n).copy()

        metadata = {
            "date": date,
            "model_name": model_name,
            "base_score": base_score,
            "top_feature": top_df.iloc[0]["sensor_name"],
            "top_feature_code": top_df.iloc[0]["feature_code"],
            "top_score_drop": top_df.iloc[0]["score_drop"]
        }

        return top_df, metadata


def plot_anomaly_explanation(
    top_df: pd.DataFrame,
    metadata: Dict[str, Union[str, float]],
    save_path: Union[str, Path] = "reports/figures/12_anomaly_explanation.png"
) -> None:
    """
    Generate horizontal bar chart showing the top 5 influential features for an anomaly.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Construct readable labels with original physical units
    labels = []
    for _, row in top_df.iterrows():
        unit_str = f" {row['unit']}" if row['unit'] else ""
        raw_str = f"Raw: {row['raw_value']:.1f}{unit_str}" if pd.notnull(row['raw_value']) else "Raw: N/A"
        scaled_str = f"Scaled: {row['scaled_value']:+.2f}x IQR"
        labels.append(f"{row['sensor_name']} [{row['feature_code']}]\n({raw_str} | {scaled_str})")

    # Invert to put #1 at the top
    y_pos = np.arange(len(top_df))[::-1]
    scores = top_df["score_drop"].values

    colors = ["#dc2626", "#ea580c", "#d97706", "#2563eb", "#059669"]

    bars = ax.barh(y_pos, scores, color=colors[:len(scores)], edgecolor="black", alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")

    for bar, val in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + (val * 0.02), bar.get_y() + bar.get_height()/2, f"Δ Score: -{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel(f"Score Reduction When Feature is Ablated to Normal Median (Base {metadata['model_name']} Score: {metadata['base_score']:.4f})", fontsize=10, fontweight="bold")
    ax.set_title(
        f"MIZU: Feature Attribution for Flagged Operational Day [{metadata['date']}]\nModel: {metadata['model_name']} — Top {len(top_df)} Influential Sensor Features",
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Explanation plot saved successfully to: {out_file}")


def run_explainability_tests() -> Dict[str, pd.DataFrame]:
    """
    Execute tests across three distinct operational archetypes:
    1. Consensus Anomaly (Flagged by both models: D-13/3/90).
    2. Isolation Forest Only (Storm surge: D-12/8/90).
    3. LOF Only (Toxic Zinc spike: D-7/2/91).
    """
    explainer = AnomalyExplainer()
    test_cases = [
        {"date": "D-13/3/90", "model": "iforest", "desc": "Consensus Anomaly (Both Models) - Evaluated on iForest"},
        {"date": "D-13/3/90", "model": "lof", "desc": "Consensus Anomaly (Both Models) - Evaluated on LOF"},
        {"date": "D-12/8/90", "model": "iforest", "desc": "Only Isolation Forest (Hydraulic Storm Surge)"},
        {"date": "D-7/2/91", "model": "lof", "desc": "Only Local Outlier Factor (Toxic Zinc Shock Event)"}
    ]

    results = {}
    print("=" * 70)
    print("MIZU: Feature-Level Anomaly Attribution Test Suite")
    print("=" * 70)

    for tc in test_cases:
        top_features, meta = explainer.explain_observation(tc["date"], model_type=tc["model"], top_n=5)
        results[f"{tc['date']}_{tc['model']}"] = top_features

        print(f"\n--- TEST: {tc['desc']} ---")
        print(f"Date: {meta['date']} | Model: {meta['model_name']} | Base Score: {meta['base_score']:.4f}")
        print("Top 5 Influential Sensors:")
        show_cols = ["feature_code", "sensor_name", "raw_value", "unit", "scaled_value", "score_drop"]
        print(top_features[show_cols].to_string(index=False))

    # Generate and save diagnostic plot for the primary consensus anomaly D-13/3/90
    top_df_plot, meta_plot = explainer.explain_observation("D-13/3/90", model_type="iforest", top_n=5)
    plot_anomaly_explanation(top_df_plot, meta_plot, save_path="reports/figures/12_anomaly_explanation.png")

    return results


if __name__ == "__main__":
    run_explainability_tests()
    print("\nExplainability test suite completed successfully!")

