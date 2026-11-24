"""
MIZU: Intelligent Water Systems Monitoring Dashboard
===================================================
A production-grade decision-support application for unsupervised
anomaly detection, clustering, and explainability in wastewater treatment plants.

Dataset: UCI Water Treatment Plant (ID: 106)
Frameworks: Streamlit, Pandas, Plotly, Scikit-Learn
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# MIZU core modules
from src.data_loader import (
    COLUMN_NAMES,
    get_column_names,
    get_feature_groups,
    get_missing_summary,
    load_water_treatment_data,
)
from src.preprocessing import MizuPreprocessor, get_feature_columns
from src.anomaly_detection import IsolationForestDetector
from src.lof_detection import LOFDetector
from src.clustering import MizuKMeansClusterer
from src.dimensionality_reduction import MizuPCA
from src.explainability import AnomalyExplainer, SENSOR_METADATA


# =============================================================================
# 1. CONSTANTS & METADATA
# =============================================================================

# Cluster operational regime taxonomy
REGIME_METADATA = {
    0: {
        "name": "Regime 0: Baseline Normal Operations",
        "description": "Stable biological and physical clarification. Standard influent flow and organic loading with high removal efficiencies across all treatment stages.",
        "color": "#0ea5e9"
    },
    1: {
        "name": "Regime 1: Secondary Settler Solids Washout",
        "description": "Elevated effluent settleable solids and suspended solids. Biological clarification impaired, causing sub-optimal final sedimentation.",
        "color": "#f59e0b"
    },
    2: {
        "name": "Regime 2: Hydraulic & Organic Surge",
        "description": "Severe volumetric inflow spike or industrial organic slug. Elevated influent COD and primary solids with stressed downstream stages.",
        "color": "#8b5cf6"
    },
    3: {
        "name": "Regime 3: Critical Solids Separation Breakdown",
        "description": "Extreme physical separation failure in both primary and secondary sedimentation stages. Severe plant performance collapse.",
        "color": "#ef4444"
    }
}

CUSTOM_CSS = """
<style>
    /* Global Typography and Spacing */
    .main {
        color: #1e293b;
    }
    
    /* Title and Subtitle */
    .mizu-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .mizu-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }
    .mizu-subtitle {
        font-size: 1.05rem;
        color: #0284c7;
        font-weight: 500;
    }
    
    /* Regulatory Callout Banner */
    .regulatory-banner {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 0.85rem 1.2rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #94a3b8;
    }
</style>
"""


# =============================================================================
# 2. CACHED MODEL & DATA LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading trained MIZU ML artifacts...")
def load_all_models() -> Tuple[
    MizuPreprocessor,
    IsolationForestDetector,
    LOFDetector,
    MizuKMeansClusterer,
    MizuPCA,
    AnomalyExplainer
]:
    """Load all fitted scikit-learn models and explainer from disk."""
    preprocessor = MizuPreprocessor.load("models/preprocessor.joblib")
    iforest = IsolationForestDetector.load("models/isolation_forest.joblib")
    lof = LOFDetector.load("models/lof_model.joblib")
    kmeans = MizuKMeansClusterer.load("models/kmeans_model.joblib")
    pca = MizuPCA.load("models/pca_model.joblib")
    explainer = AnomalyExplainer(iforest_detector=iforest, lof_detector=lof)
    return preprocessor, iforest, lof, kmeans, pca, explainer


@st.cache_data(show_spinner="Loading historical baseline dataset...")
def load_baseline_raw_data() -> pd.DataFrame:
    """Load the historical 527-record UCI Water Treatment dataset."""
    df = load_water_treatment_data()
    return df


# =============================================================================
# 3. PIPELINE INFERENCE ENGINE
# =============================================================================

def run_mizu_inference(
    df_raw: pd.DataFrame,
    preprocessor: MizuPreprocessor,
    iforest: IsolationForestDetector,
    lof: LOFDetector,
    kmeans: MizuKMeansClusterer,
    pca: MizuPCA
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Execute full inference pipeline on raw or uploaded sensor records:
    1. Verify presence of 38 continuous sensor columns.
    2. Handle missing/null values via fitted median imputer.
    3. Scale features via fitted RobustScaler.
    4. Predict Isolation Forest flags & scores.
    5. Predict Novelty LOF flags & scores.
    6. Predict K-Means operational cluster regimes.
    7. Project into 2D PCA subspace.
    """
    required_features = preprocessor.feature_names
    missing_cols = [col for col in required_features if col not in df_raw.columns]
    if missing_cols:
        return None, None, missing_cols

    # Clean numeric features
    df_clean = df_raw.copy()
    for col in required_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Preserve or generate Date identifier
    if "Date" not in df_clean.columns:
        df_clean["Date"] = [f"Record_{i+1:04d}" for i in range(len(df_clean))]
    else:
        df_clean["Date"] = df_clean["Date"].astype(str)

    # Scale using pre-fitted preprocessor (Strictly in inference mode - NO retraining)
    X_scaled, _ = preprocessor.transform(df_clean)

    # 1. Isolation Forest inference
    iforest_flags = iforest.predict(X_scaled)
    iforest_scores = iforest.compute_anomaly_scores(X_scaled)

    # 2. Local Outlier Factor inference (novelty mode)
    lof_flags = lof.predict(X_scaled)
    lof_scores = lof.compute_lof_scores(X_scaled)

    # 3. K-Means clustering inference
    clusters = kmeans.predict(X_scaled)

    # 4. PCA 2D projection inference
    pca_coords = pca.transform(X_scaled)

    # Combine into unified results dataframe
    results = df_clean.copy()
    results["iforest_flag"] = iforest_flags
    results["iforest_score"] = np.round(iforest_scores, 4)
    results["lof_flag"] = lof_flags
    results["lof_score"] = np.round(lof_scores, 4)
    results["cluster"] = clusters
    results["regime_name"] = [REGIME_METADATA.get(c, {}).get("name", f"Cluster {c}") for c in clusters]
    results["PC1"] = np.round(pca_coords[:, 0], 4)
    results["PC2"] = np.round(pca_coords[:, 1], 4)

    # Categorize concordance
    def get_consensus_label(row):
        if row["iforest_flag"] == 1 and row["lof_flag"] == 1:
            return "Consensus Anomaly"
        elif row["iforest_flag"] == 1:
            return "iForest Only"
        elif row["lof_flag"] == 1:
            return "LOF Only"
        else:
            return "Normal Baseline"

    results["concordance"] = results.apply(get_consensus_label, axis=1)

    return results, X_scaled, None


# =============================================================================
# 4. APPLICATION ENTRYPOINT
# =============================================================================

def main():
    st.set_page_config(
        page_title="MIZU | Intelligent Water Systems Monitoring",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    preprocessor, iforest, lof, kmeans, pca, explainer = load_all_models()

    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "baseline"

    if "active_results" not in st.session_state:
        raw_baseline = load_baseline_raw_data()
        results, X_scaled, _ = run_mizu_inference(
            raw_baseline, preprocessor, iforest, lof, kmeans, pca
        )
        st.session_state["active_results"] = results
        st.session_state["active_scaled"] = X_scaled
        st.session_state["active_raw"] = raw_baseline
        st.session_state["active_name"] = "UCI Baseline Dataset (527 Historical Records)"

    # Sidebar
    with st.sidebar:
        st.markdown("### 🌊 MIZU System")
        st.caption("Intelligent Water Systems Monitoring")
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            options=[
                "1. Executive Overview",
                "2. Data Ingestion & Audit",
                "3. Anomaly Concordance",
                "4. Operational Regimes (PCA)",
                "5. Explainability Attribution",
                "6. Architecture & Compliance"
            ],
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("#### Active Telemetry Source")
        st.info(f"**Dataset:**\n{st.session_state['active_name']}\n\n**Observations:** {len(st.session_state['active_results'])} rows")
        
        if st.session_state["data_source"] == "uploaded":
            if st.button("↺ Restore Baseline Dataset", use_container_width=True):
                raw_baseline = load_baseline_raw_data()
                results, X_scaled, _ = run_mizu_inference(
                    raw_baseline, preprocessor, iforest, lof, kmeans, pca
                )
                st.session_state["active_results"] = results
                st.session_state["active_scaled"] = X_scaled
                st.session_state["active_raw"] = raw_baseline
                st.session_state["active_name"] = "UCI Baseline Dataset (527 Historical Records)"
                st.session_state["data_source"] = "baseline"
                st.rerun()

        st.markdown("---")
        st.markdown("#### Operational Calibration")
        st.caption("• Contamination rate: **3.0%** (~16 events)")
        st.caption("• LOF Neighborhood (k): **20 neighbors**")
        st.caption("• Operational Regimes: **K = 4 clusters**")
        st.caption("• PCA Retained Variance: **81.83%**")
        
        st.markdown("---")
        st.caption("MIZU v1.0.0 | Environmental Decision-Support")

    # Header & Regulatory Disclaimer
    st.markdown("""
    <div class="mizu-header">
        <div class="mizu-title">MIZU</div>
        <div class="mizu-subtitle">Intelligent Water Systems Monitoring & Unsupervised Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="regulatory-banner">
        <strong>⚖️ Regulatory & Engineering Compliance Notice:</strong><br>
        MIZU is an unsupervised algorithmic decision-support tool designed for secondary operational screening.
        It does <strong>NOT</strong> certify potable drinking water compliance, detect microbiological pathogens (e.g. <em>E. coli</em>, protozoan cysts, enteric viruses), or trigger automated physical actuation in the treatment plant.
        All anomaly scores and feature attributions indicate mathematical deviation from baseline operating distributions and must be verified by accredited laboratory assays.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PAGE 1: EXECUTIVE OVERVIEW
    # -------------------------------------------------------------------------
    if page == "1. Executive Overview":
        st.markdown("### 📊 Executive Overview & System Status")
        st.markdown(
            "Real-time operational summary comparing two complementary unsupervised anomaly detection models: "
            "**Isolation Forest** (global isolation depth) and **Local Outlier Factor** (local k-nearest neighbor density ratio)."
        )

        df_res = st.session_state["active_results"]
        total_obs = len(df_res)
        n_iforest = int(df_res["iforest_flag"].sum())
        n_lof = int(df_res["lof_flag"].sum())
        n_consensus = int((df_res["concordance"] == "Consensus Anomaly").sum())
        n_normal = int((df_res["concordance"] == "Normal Baseline").sum())

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Observations</div>
                <div class="metric-value">{total_obs}</div>
                <div class="metric-subtitle">Evaluated Records</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Consensus Flags</div>
                <div class="metric-value" style="color: #ef4444;">{n_consensus}</div>
                <div class="metric-subtitle">{n_consensus/total_obs*100:.2f}% (Both Models)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">iForest Flags</div>
                <div class="metric-value" style="color: #f97316;">{n_iforest}</div>
                <div class="metric-subtitle">{n_iforest/total_obs*100:.2f}% Flagged</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">LOF Flags</div>
                <div class="metric-value" style="color: #eab308;">{n_lof}</div>
                <div class="metric-subtitle">{n_lof/total_obs*100:.2f}% Flagged</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Normal Baseline</div>
                <div class="metric-value" style="color: #10b981;">{n_normal}</div>
                <div class="metric-subtitle">{n_normal/total_obs*100:.2f}% Stable</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Anomaly Score Timeline
        st.markdown("#### Dual-Model Anomaly Timeline")
        st.caption("Comparative anomaly scores over time. Consensus anomalies indicate high-confidence systemic events.")

        fig_timeline = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Isolation Forest Anomaly Score (Calibrated: 0 = Normal, 1 = Extreme Anomaly)",
                "Local Outlier Factor (LOF Score: ~1.0 = Normal Density, >1.5 = Low Density Outlier)"
            )
        )

        fig_timeline.add_trace(
            go.Scatter(
                x=df_res["Date"],
                y=df_res["iforest_score"],
                mode="lines+markers",
                name="iForest Score",
                line=dict(color="#0284c7", width=1.5),
                marker=dict(size=4),
                hovertemplate="Date: %{x}<br>iForest Score: %{y:.4f}<extra></extra>"
            ),
            row=1, col=1
        )
        consensus_df = df_res[df_res["concordance"] == "Consensus Anomaly"]
        fig_timeline.add_trace(
            go.Scatter(
                x=consensus_df["Date"],
                y=consensus_df["iforest_score"],
                mode="markers",
                name="Consensus Anomaly",
                marker=dict(color="#ef4444", size=9, symbol="diamond"),
                hovertemplate="Consensus Anomaly<br>Date: %{x}<br>Score: %{y:.4f}<extra></extra>"
            ),
            row=1, col=1
        )

        fig_timeline.add_trace(
            go.Scatter(
                x=df_res["Date"],
                y=df_res["lof_score"],
                mode="lines+markers",
                name="LOF Score",
                line=dict(color="#0d9488", width=1.5),
                marker=dict(size=4),
                hovertemplate="Date: %{x}<br>LOF Score: %{y:.4f}<extra></extra>"
            ),
            row=2, col=1
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=consensus_df["Date"],
                y=consensus_df["lof_score"],
                mode="markers",
                name="Consensus Anomaly (LOF)",
                marker=dict(color="#ef4444", size=9, symbol="diamond"),
                showlegend=False,
                hovertemplate="Consensus Anomaly<br>Date: %{x}<br>LOF: %{y:.4f}<extra></extra>"
            ),
            row=2, col=1
        )

        fig_timeline.update_layout(
            height=520,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        fig_timeline.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig_timeline.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Flagged Observations Table
        st.markdown("#### High-Priority Flagged Operational Days")
        anomalies_only = df_res[df_res["concordance"] != "Normal Baseline"].copy()
        if anomalies_only.empty:
            st.success("No anomalous records detected in the active telemetry stream.")
        else:
            display_cols = [
                "Date", "concordance", "iforest_score", "lof_score", "regime_name",
                "Q-E", "ZN-E", "SED-S", "RD-DBO-G", "RD-SED-G"
            ]
            present_cols = [c for c in display_cols if c in anomalies_only.columns]
            table_df = anomalies_only[present_cols].sort_values(
                by=["iforest_score", "lof_score"], ascending=[False, False]
            )
            st.dataframe(
                table_df.style.format({
                    "iforest_score": "{:.4f}",
                    "lof_score": "{:.4f}",
                    "Q-E": "{:.0f}",
                    "ZN-E": "{:.2f}",
                    "SED-S": "{:.2f}",
                    "RD-DBO-G": "{:.1f}%",
                    "RD-SED-G": "{:.1f}%"
                }),
                use_container_width=True
            )

    # -------------------------------------------------------------------------
    # PAGE 2: DATA INGESTION & AUDIT
    # -------------------------------------------------------------------------
    elif page == "2. Data Ingestion & Audit":
        st.markdown("### 📥 Data Ingestion, Schema Validation & Sensor Audit")
        st.markdown(
            "Upload new wastewater telemetry CSV files to run automated inference. "
            "The system enforces strict schema validation across all 38 continuous sensor streams."
        )

        col_upload, col_sample = st.columns([2, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Water Treatment Sensor CSV",
                type=["csv"],
                help="Upload a CSV file containing the 38 required continuous sensor columns."
            )

        with col_sample:
            st.markdown("##### Quick Test Fixtures")
            st.caption(
                "Ready-made test CSV files available in `tests/fixtures/`:\n"
                "- `valid_sample.csv` (10 valid rows)\n"
                "- `single_row.csv` (1 row test)\n"
                "- `with_nans.csv` (Missing value test)\n"
                "- `missing_columns.csv` (Schema failure test)"
            )

        # Process uploaded file
        if uploaded_file is not None:
            try:
                raw_upload = pd.read_csv(uploaded_file)
                st.session_state["uploaded_raw"] = raw_upload

                results_up, scaled_up, missing_cols = run_mizu_inference(
                    raw_upload, preprocessor, iforest, lof, kmeans, pca
                )

                if missing_cols:
                    st.error(
                        f"⚠️ **Schema Validation Failure**: Uploaded CSV is missing {len(missing_cols)} "
                        f"required continuous sensor columns:\n\n`{missing_cols}`\n\n"
                        "Inference halted. Pre-trained pipeline artifacts require all 38 continuous features."
                    )
                else:
                    st.success(
                        f"✅ **Schema Validation Passed**: Successfully parsed {len(raw_upload)} observations "
                        f"across all 38 required continuous sensor columns."
                    )
                    st.session_state["active_results"] = results_up
                    st.session_state["active_scaled"] = scaled_up
                    st.session_state["active_raw"] = raw_upload
                    st.session_state["active_name"] = f"Uploaded File: {uploaded_file.name} ({len(raw_upload)} rows)"
                    st.session_state["data_source"] = "uploaded"

            except Exception as e:
                st.error(f"Error reading uploaded CSV file: {str(e)}")

        st.markdown("---")

        df_active = st.session_state["active_results"]
        st.markdown(f"#### Active Sensor Telemetry: {st.session_state['active_name']}")

        tab_all, tab_missing, tab_stages, tab_download = st.tabs([
            "Data Preview", "Missing Value Audit", "Treatment Stages", "Export Results"
        ])

        with tab_all:
            st.dataframe(df_active.head(50), use_container_width=True)

        with tab_missing:
            st.markdown("##### Feature Missingness Profile")
            st.caption(
                "Missing values are imputed using the column-wise historical Median via SimpleImputer. "
                "Biochemical Oxygen Demand ($DBO_5$) exhibits the highest missing rate due to the physical 5-day laboratory incubation duration."
            )
            missing_summary = get_missing_summary(st.session_state["active_raw"])
            st.dataframe(missing_summary.head(15), use_container_width=True)

        with tab_stages:
            stages = get_feature_groups()
            selected_stage = st.selectbox(
                "Select Physical Plant Treatment Stage:",
                options=list(stages.keys()),
                format_func=lambda s: {
                    "influent": "Stage 1: Plant Influent / Raw Intake (-E)",
                    "primary": "Stage 2: Primary Settler Sedimentation (-P)",
                    "secondary": "Stage 3: Biological Aeration & Secondary Clarification (-D)",
                    "effluent": "Stage 4: Treated Plant Discharge / Effluent (-S)",
                    "efficiency": "Stage 5: Removal Efficiencies (RD-)"
                }.get(s, s)
            )
            stage_cols = ["Date"] + [c for c in stages[selected_stage] if c in df_active.columns]
            st.dataframe(df_active[stage_cols].head(30), use_container_width=True)

        with tab_download:
            st.markdown("##### Download Processed Telemetry")
            csv_bytes = df_active.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Inference Results (CSV)",
                data=csv_bytes,
                file_name="mizu_telemetry_inference_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    # -------------------------------------------------------------------------
    # PAGE 3: ANOMALY CONCORDANCE
    # -------------------------------------------------------------------------
    elif page == "3. Anomaly Concordance":
        st.markdown("### 🚨 Dual-Model Anomaly Detection & Concordance")
        st.markdown(
            "Because ground-truth fault labels are unavailable in production wastewater systems, "
            "MIZU utilizes **unsupervised concordance** to establish high-confidence anomaly consensus."
        )

        df_res = st.session_state["active_results"]

        cat_counts = df_res["concordance"].value_counts()
        n_both = cat_counts.get("Consensus Anomaly", 0)
        n_iforest_only = cat_counts.get("iForest Only", 0)
        n_lof_only = cat_counts.get("LOF Only", 0)
        n_norm = cat_counts.get("Normal Baseline", 0)

        union_flags = n_both + n_iforest_only + n_lof_only
        jaccard = (n_both / union_flags) if union_flags > 0 else 1.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Consensus Anomalies", f"{n_both}", delta=f"{n_both/len(df_res)*100:.2f}% (Both Models)", delta_color="inverse")
        with c2:
            st.metric("iForest Exclusive", f"{n_iforest_only}", help="Flagged by Isolation Forest, normal under LOF")
        with c3:
            st.metric("LOF Exclusive", f"{n_lof_only}", help="Flagged by LOF, normal under Isolation Forest")
        with c4:
            st.metric("Jaccard Similarity", f"{jaccard*100:.1f}%", help="Intersection over Union of flagged anomaly sets")

        st.markdown("<br>", unsafe_allow_html=True)

        # Cross-Model Scatter Plot
        st.markdown("#### Cross-Model Concordance Distribution")
        st.caption("Scatter mapping comparing Isolation Forest Anomaly Score vs Local Outlier Factor (LOF).")

        color_map = {
            "Consensus Anomaly": "#ef4444",
            "iForest Only": "#f97316",
            "LOF Only": "#eab308",
            "Normal Baseline": "#94a3b8"
        }

        fig_scatter = px.scatter(
            df_res,
            x="iforest_score",
            y="lof_score",
            color="concordance",
            color_discrete_map=color_map,
            hover_data=["Date", "Q-E", "ZN-E", "SED-S", "regime_name"],
            labels={
                "iforest_score": "Isolation Forest Anomaly Score",
                "lof_score": "Local Outlier Factor (LOF)",
                "concordance": "Concordance Status"
            },
            title="Model Score Concordance Space"
        )

        fig_scatter.add_vline(x=0.55, line_width=1.5, line_dash="dash", line_color="#f97316", annotation_text="iForest Borderline")
        fig_scatter.add_hline(y=1.5, line_width=1.5, line_dash="dash", line_color="#eab308", annotation_text="LOF Density Boundary")

        fig_scatter.update_layout(
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### Why Do Models Disagree on Specific Dates?")
        col_agree, col_disagree = st.columns(2)
        with col_agree:
            st.markdown("""
            **1. Consensus Anomalies (e.g., `D-13/3/90`)**
            - Severe physical breakdown affecting both global isolation depth and local density.
            - Example: Clarification failure causing effluent settleable solids (`SED-S`) to spike to $3.5\ mL/L$ (historical median: $0.02\ mL/L$).
            - Both algorithms independently flag these records with maximum confidence.
            """)
        with col_disagree:
            st.markdown("""
            **2. Model Divergence (e.g., `D-12/8/90` vs `D-7/2/91`)**
            - **Isolation Forest Only (`D-12/8/90`)**: Catch global dimensional extremes (such as massive inflow storm surges, `Q-E` > $50,000\ m^3/day$) that slice easily across random axis cuts.
            - **LOF Only (`D-7/2/91`)**: Catches local density collapse (such as toxic Zinc spikes, `ZN-E` = $10.0\ mg/L$) where the point is isolated from its 20 nearest operational neighbors, even if global tree depth is moderate.
            """)

    # -------------------------------------------------------------------------
    # PAGE 4: OPERATIONAL REGIMES (PCA)
    # -------------------------------------------------------------------------
    elif page == "4. Operational Regimes (PCA)":
        st.markdown("### 🔄 Operational Regimes & Dimensionality Reduction")
        st.markdown(
            "Unsupervised clustering (**K-Means**, K=4) identifies plant operating regimes, "
            "while **Principal Component Analysis (PCA)** projects the 38-dimensional sensor space "
            "into 2 interpretable operational axes."
        )

        df_res = st.session_state["active_results"]

        col_pca, col_summary = st.columns([2, 1])

        with col_pca:
            fig_pca = px.scatter(
                df_res,
                x="PC1",
                y="PC2",
                color="regime_name",
                symbol="concordance",
                symbol_map={
                    "Consensus Anomaly": "diamond",
                    "iForest Only": "triangle-up",
                    "LOF Only": "triangle-down",
                    "Normal Baseline": "circle"
                },
                color_discrete_map={
                    REGIME_METADATA[0]["name"]: REGIME_METADATA[0]["color"],
                    REGIME_METADATA[1]["name"]: REGIME_METADATA[1]["color"],
                    REGIME_METADATA[2]["name"]: REGIME_METADATA[2]["color"],
                    REGIME_METADATA[3]["name"]: REGIME_METADATA[3]["color"]
                },
                hover_data=["Date", "concordance", "SED-S", "Q-E", "ZN-E"],
                labels={
                    "PC1": "Principal Component 1 (77.24% Variance Explained)",
                    "PC2": "Principal Component 2 (4.59% Variance Explained)",
                    "regime_name": "Operating Regime",
                    "concordance": "Anomaly Status"
                },
                title="2D PCA Projection of Wastewater Treatment Operational Regimes"
            )
            fig_pca.update_traces(marker=dict(size=8, opacity=0.85))
            fig_pca.update_layout(
                height=540,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            fig_pca.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
            fig_pca.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
            st.plotly_chart(fig_pca, use_container_width=True)

        with col_summary:
            st.markdown("##### PCA Variance Budget")
            st.markdown("""
            - **PC1 (77.24%)**: Dominated by *Clarification & Effluent Solids* (`SED-S`, `RD-SED-G`). High positive PC1 indicates clarification failure.
            - **PC2 (4.59%)**: Dominated by *Influent Suspended Solids* (`SS-E`, `SS-P`). Represents raw intake solids loading.
            - **Cumulative Explained Variance**: **81.83%**
            """)

            st.markdown("##### Active Regime Distribution")
            cluster_counts = df_res["cluster"].value_counts().sort_index()
            for c_id in sorted(REGIME_METADATA.keys()):
                count = cluster_counts.get(c_id, 0)
                pct = count / len(df_res) * 100
                st.markdown(
                    f"**Cluster {c_id}**: {count} days ({pct:.1f}%)\n"
                    f"<small style='color: #64748b;'>{REGIME_METADATA[c_id]['name']}</small>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("#### Operational Regime Taxonomy")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {REGIME_METADATA[0]['color']};">
                <div class="metric-title">Cluster 0 (Baseline)</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                    Baseline Normal Operations
                </div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    {REGIME_METADATA[0]['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {REGIME_METADATA[1]['color']};">
                <div class="metric-title">Cluster 1 (Impaired)</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                    Solids Washout
                </div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    {REGIME_METADATA[1]['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {REGIME_METADATA[2]['color']};">
                <div class="metric-title">Cluster 2 (Hydraulic)</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                    Hydraulic / Organic Surge
                </div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    {REGIME_METADATA[2]['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {REGIME_METADATA[3]['color']};">
                <div class="metric-title">Cluster 3 (Critical)</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                    Separation Breakdown
                </div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    {REGIME_METADATA[3]['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PAGE 5: EXPLAINABILITY ATTRIBUTION
    # -------------------------------------------------------------------------
    elif page == "5. Explainability Attribution":
        st.markdown("### 🔍 Model-Agnostic Feature Ablation Explainability")
        st.markdown(
            "When an anomaly is flagged, plant operators need to know **which specific physical sensors** "
            "most strongly influenced the decision. MIZU uses **Feature Ablation**: temporarily replacing "
            "each sensor value with normal operating median ($0.0$ in RobustScaler space) and measuring "
            "the resulting reduction in anomaly score ($\Delta \text{Score} = \text{Base Score} - \text{Ablated Score}$)."
        )

        df_res = st.session_state["active_results"]
        scaled_df = st.session_state["active_scaled"]
        raw_df = st.session_state["active_raw"]

        anomalies_only = df_res[df_res["concordance"] != "Normal Baseline"]

        col_select1, col_select2, col_select3 = st.columns([2, 1, 1])

        with col_select1:
            date_options = list(anomalies_only["Date"]) if not anomalies_only.empty else list(df_res["Date"])
            def format_date_option(d):
                row = df_res[df_res["Date"] == d].iloc[0]
                return f"{d} — [{row['concordance']}] (iForest: {row['iforest_score']:.3f}, LOF: {row['lof_score']:.2f})"

            selected_date = st.selectbox(
                "Select Observation to Explain:",
                options=date_options,
                format_func=format_date_option
            )

        with col_select2:
            model_choice = st.radio(
                "Attribution Algorithm:",
                options=["iforest", "lof"],
                format_func=lambda m: "Isolation Forest" if m == "iforest" else "Local Outlier Factor (LOF)",
                horizontal=True
            )

        with col_select3:
            top_n = st.slider("Top Influential Features:", min_value=3, max_value=8, value=5)

        if selected_date:
            idx = df_res[df_res["Date"] == selected_date].index[0]
            sample_scaled = scaled_df.loc[idx:idx]
            sample_raw = raw_df.loc[idx] if idx < len(raw_df) else None

            with st.spinner("Computing feature ablation attribution..."):
                top_df, metadata = explainer.explain_sample(
                    sample=sample_scaled,
                    raw_row=sample_raw,
                    sample_id=selected_date,
                    model_type=model_choice,
                    top_n=top_n
                )

            st.markdown("<br>", unsafe_allow_html=True)

            fig_bar = px.bar(
                top_df,
                x="score_reduction",
                y="sensor_name",
                orientation="h",
                color="stage",
                text="score_reduction",
                color_discrete_map={
                    "Influent (-E)": "#0284c7",
                    "Primary (-P)": "#0d9488",
                    "Secondary (-D)": "#8b5cf6",
                    "Effluent (-S)": "#f59e0b",
                    "Efficiency (RD-)": "#ef4444"
                },
                labels={
                    "score_reduction": "Score Reduction Towards Normal (Δ Score)",
                    "sensor_name": "Sensor Measurement",
                    "stage": "Treatment Stage"
                },
                title=f"Top {top_n} Influential Sensors for {selected_date} ({metadata['model_name']})"
            )
            fig_bar.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Stage: %{customdata[0]}<br>Δ Score: %{x:.6f}<br>Raw Value: %{customdata[1]} %{customdata[2]}<extra></extra>",
                customdata=top_df[["stage", "raw_value", "unit"]].values
            )
            fig_bar.update_layout(
                height=400,
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            fig_bar.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### Detailed Sensor Contribution Metrics")
            display_attribution = top_df[[
                "feature_code", "sensor_name", "stage", "raw_value", "unit",
                "scaled_value", "original_score", "ablated_score", "score_reduction"
            ]].copy()

            st.dataframe(
                display_attribution.style.format({
                    "raw_value": "{:.2f}",
                    "scaled_value": "{:+.3f} IQR",
                    "original_score": "{:.4f}",
                    "ablated_score": "{:.4f}",
                    "score_reduction": "{:.6f}"
                }),
                use_container_width=True
            )

            st.info(
                f"💡 **Physical vs Mathematical Interpretation**: Neutralizing **{metadata['top_feature']}** "
                f"(`{metadata['top_feature_code']}`) reduces the {metadata['model_name']} score by "
                f"**{metadata['top_score_reduction']:.6f}**. This indicates that the model's decision "
                f"boundary was mathematically sensitive to this measurement. It does **not** physically prove "
                f"chemical failure without laboratory verification."
            )

    # -------------------------------------------------------------------------
    # PAGE 6: ARCHITECTURE & COMPLIANCE
    # -------------------------------------------------------------------------
    elif page == "6. Architecture & Compliance":
        st.markdown("### 📐 Methodology, Architecture & Compliance")

        st.markdown("#### 1. End-to-End Pipeline Architecture")
        st.markdown("""
        ```
        Raw Sensor CSV (38 Continuous Sensors)
                 │
                 ▼
        SimpleImputer (Strategy = Column Median)
                 │   ↳ Handles 5-day BOD (DBO) lab incubation delays
                 ▼
        RobustScaler (x_scaled = (x - median) / IQR)
                 │   ↳ Protects geometry against heavy metal spikes (e.g., Zinc)
                 ▼
        ┌────────────────────────┬────────────────────────┬────────────────────────┐
        ▼                        ▼                        ▼                        ▼
        Isolation Forest         Local Outlier Factor     K-Means Clustering       PCA (2D Projection)
        (n=150, contam=3.0%)     (k=20, novelty=True)     (K=4 regimes)            (Retained Var=81.83%)
        Global tree depth        Local density ratio      Operational baselines    2D Visual Geometry
        └────────────────────────┼────────────────────────┴────────────────────────┘
                                 │
                                 ▼
                     Unsupervised Concordance
                     (Consensus Anomaly Detection)
                                 │
                                 ▼
                     Feature-Level Explainability
                     (Model-Agnostic Feature Ablation)
        ```
        """)

        st.markdown("---")
        st.markdown("#### 2. Key Mathematical & Engineering Design Choices")

        q1, q2 = st.columns(2)
        with q1:
            st.markdown("""
            **Why RobustScaler instead of StandardScaler?**
            - Wastewater datasets experience massive, legitimate shock loads (e.g. industrial Zinc spikes up to $33.5\ mg/L$ vs median of $1.5\ mg/L$).
            - In `StandardScaler`, extreme outliers inflate the sample variance and pull the mean, artificially shrinking normal operational variation.
            - `RobustScaler` uses the **Median** and **Interquartile Range ($IQR = Q_{75} - Q_{25}$)**, which are mathematically immune to extreme outlier distortion.

            **Why Novelty LOF (`novelty=True`)?**
            - Standard LOF (`novelty=False`) operates in transductive mode and cannot score incoming streaming telemetry or user-uploaded CSV files.
            - With `novelty=True`, scikit-learn builds a KD-Tree of the training points, allowing real-time `.predict()` and `.decision_function()` calls on new test points.
            """)

        with q2:
            st.markdown("""
            **Why K=4 Clusters instead of K=2?**
            - K=2 achieved an artificially high silhouette score of $0.939$ by isolating 3 extreme anomaly days into one cluster and lumping 524 normal days into another.
            - K=4 reveals the genuine operational structure: normal baseline (510 days), secondary clarifier solids washout (14 days), hydraulic overload (1 day), and critical separation failure (2 days).

            **Why Feature Ablation over SHAP/LIME?**
            - SHAP and LIME rely on local linear approximations or sample permutations that struggle with unsupervised distance-based ensembles.
            - Feature ablation is model-agnostic, exact, deterministic, and intuitive: setting a feature to $0.0$ in RobustScaler space physically represents replacing that reading with the historical median.
            """)

        st.markdown("---")
        st.markdown("#### 3. Regulatory Disclaimers & Operational Safeguards")
        st.markdown("""
        1. **Potable Water Boundary**: MIZU monitors industrial and municipal wastewater-treatment processes. It is strictly forbidden from certifying potable drinking water compliance under EPA/WHO drinking water guidelines.
        2. **Microbiological Limitations**: The UCI dataset contains physical and chemical bulk parameters (pH, solids, flow rate, oxygen demand). It does NOT monitor biological pathogens (*Cryptosporidium*, *Giardia*, *Legionella*, *Norovirus*).
        3. **Human-in-the-Loop Governance**: This software is an advisory decision-support interface. All flagged anomalies require physical sampling and laboratory validation by certified wastewater operators prior to process adjustments.
        """)


if __name__ == "__main__":
    main()
